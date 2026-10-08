// rtseg.hpp — real-time class-agnostic object masks for a fixed camera (C++ port of
// python/certskip/segment.py::RealtimeSegmenter).  See THEORY.md §7.
//
// Per frame: background warped by the global sub-pixel jitter, per-patch illumination
// offset, studentised residual with texture-based shadow suppression, then the
// minimiser of the time-varying energy
//     E_t(u) = <f_t,u> + lam * sum w_t |grad u| + (mu/2)||u - uhat_t||^2,  u in [0,1]
// is TRACKED: uhat_t = previous solution with each object advected by its velocity,
// K primal-dual (Chambolle-Pock) correction steps on the active set only, and the
// KKT residual evaluated everywhere as a certificate for the frozen region.
#pragma once
#include "certskip.hpp"
#include <unordered_map>
#include <array>
#include <chrono>
#include <thread>
#include <functional>
#include <algorithm>
#include <cmath>
#include <vector>

namespace certskip {
namespace rt {

// Run fn(begin, end) over [0, n) split across hardware threads (spawned per call; fine at a few calls per frame).
inline void parallel_for(size_t n, const std::function<void(size_t, size_t)>& fn, int nthreads = 0, size_t min_items = 64) {
    if (nthreads <= 0) nthreads = int(std::max(1u, std::thread::hardware_concurrency()));
    if (nthreads == 1 || n < min_items) { fn(0, n); return; }
    std::vector<std::thread> th; const size_t chunk = (n + nthreads - 1) / nthreads;
    for (int t = 0; t < nthreads; ++t) { const size_t b = t * chunk, e = std::min(n, b + chunk); if (b >= e) break; th.emplace_back(fn, b, e); }
    for (auto& t : th) t.join();
}

struct RtParams {
    int patch = 16;
    double z0 = 3.0, gamma = 0.6, lam = 1.2, tau_edge = 6.0, mu = 0.5;
    int steps = 3, min_area = 40, halo = 0, fit_iters = 2;
    double alpha_bg = 0.02, alpha_fg = 5e-5, sigma_floor = 1.5;   // objects are not absorbed while they are objects
    bool shadows = true, certify = false;
    double kkt_tol = 0.5, act_delta0 = 40.0, delta_max = 1.0;
    int close_r = 2, min_side = 8; double max_aspect = 5.0;
    double rho = 0.6, tau_img = 12.0, snap_eps = 36.0; int snap_r = 3;
    int threads = 0;           // 0 = hardware concurrency
    double act_z = 3.0;        // active-set threshold on the studentised multi-scale score
    int max_shift = 48;        // largest per-frame object displacement that is predicted/advected (px)
    double ghost_contrast = 6.0;   // components whose contour contrast (in sigma units) is below this are ghosts
    double alpha_ghost = 0.25;     // absorption rate for ghost pixels
};

struct RtStats { double active_frac = 0, kkt_frozen_max = 0, kkt_active_max = 0, dy = 0, dx = 0; int n_objects = 0, pred_shift = 0; double stage_ms[12] = {0}; int n_score = 0, n_near = 0, n_kkt = 0, n_patches = 0; };

// (2r+1)-box mean with edge-replicated padding, via an integral image of the padded image.
inline void box_mean_edge(const double* src, int H, int W, int r, double* dst, std::vector<double>& I) {
    const int Hp = H + 2 * r, Wp = W + 2 * r, k = 2 * r + 1; const double inv = 1.0 / (double(k) * k);
    I.assign(size_t(Hp + 1) * (Wp + 1), 0.0);
    for (int y = 0; y < Hp; ++y) {
        const int sy = std::min(std::max(y - r, 0), H - 1); double row = 0;
        for (int x = 0; x < Wp; ++x) {
            const int sx = std::min(std::max(x - r, 0), W - 1); row += src[size_t(sy) * W + sx];
            I[size_t(y + 1) * (Wp + 1) + x + 1] = I[size_t(y) * (Wp + 1) + x + 1] + row;
        }
    }
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
        const size_t a = size_t(y) * (Wp + 1) + x, b = size_t(y + k) * (Wp + 1) + x;
        dst[size_t(y) * W + x] = (I[b + k] - I[a + k] - I[b] + I[a]) * inv;
    }
}

inline void dilate4(const uint8_t* m, int H, int W, int r, uint8_t* out, std::vector<uint8_t>& tmp) {
    const size_t N = size_t(H) * W; std::copy(m, m + N, out); tmp.resize(N);
    for (int it = 0; it < r; ++it) {
        std::copy(out, out + N, tmp.begin());
        parallel_for(size_t(H), [&](size_t y0, size_t y1) {
            for (size_t y = y0; y < y1; ++y) for (int x = 0; x < W; ++x) {
                const size_t i = y * W + x; if (tmp[i]) continue;
                if ((y > 0 && tmp[i - W]) || (y + 1 < size_t(H) && tmp[i + W]) || (x > 0 && tmp[i - 1]) || (x < W - 1 && tmp[i + 1])) out[i] = 1;
            }
        });
    }
}

inline void erode4(const uint8_t* m, int H, int W, int r, uint8_t* out, std::vector<uint8_t>& tmp) {
    const size_t N = size_t(H) * W; std::copy(m, m + N, out); tmp.resize(N);
    for (int it = 0; it < r; ++it) {
        std::copy(out, out + N, tmp.begin());
        parallel_for(size_t(H), [&](size_t y0, size_t y1) {
            for (size_t y = y0; y < y1; ++y) for (int x = 0; x < W; ++x) {
                const size_t i = y * W + x; if (!tmp[i]) continue;
                if (y == 0 || y + 1 == size_t(H) || x == 0 || x == W - 1 || !tmp[i - W] || !tmp[i + W] || !tmp[i - 1] || !tmp[i + 1]) out[i] = 0;
            }
        });
    }
}

// Fill holes of the mask inside one bounding box: background pixels not 4-connected to the box border.
inline void fill_holes_box(uint8_t* m, int H, int W, int y0, int y1, int x0, int x1, std::vector<uint8_t>& reach, std::vector<int32_t>& q) {
    y0 = std::max(y0 - 1, 0); x0 = std::max(x0 - 1, 0); y1 = std::min(y1 + 1, H - 1); x1 = std::min(x1 + 1, W - 1);
    const int h = y1 - y0 + 1, w = x1 - x0 + 1; reach.assign(size_t(h) * w, 0); q.clear();
    auto push = [&](int ly, int lx) { const size_t li = size_t(ly) * w + lx; if (!m[size_t(y0 + ly) * W + x0 + lx] && !reach[li]) { reach[li] = 1; q.push_back(int32_t(li)); } };
    for (int lx = 0; lx < w; ++lx) { push(0, lx); push(h - 1, lx); }
    for (int ly = 0; ly < h; ++ly) { push(ly, 0); push(ly, w - 1); }
    for (size_t qi = 0; qi < q.size(); ++qi) { const int li = q[qi], ly = li / w, lx = li % w; if (ly > 0) push(ly - 1, lx); if (ly < h - 1) push(ly + 1, lx); if (lx > 0) push(ly, lx - 1); if (lx < w - 1) push(ly, lx + 1); }
    for (int ly = 0; ly < h; ++ly) for (int lx = 0; lx < w; ++lx) { const size_t gi = size_t(y0 + ly) * W + x0 + lx; if (!m[gi] && !reach[size_t(ly) * w + lx]) m[gi] = 1; }
}

// Pixels within distance r of the mask boundary inside one bounding box (dilate_r & ~erode_r), appended as global indices.
inline void band_box(const uint8_t* m, int H, int W, int y0, int y1, int x0, int x1, int r, std::vector<int32_t>& band, std::vector<uint8_t>& a, std::vector<uint8_t>& b) {
    y0 = std::max(y0 - r, 0); x0 = std::max(x0 - r, 0); y1 = std::min(y1 + r, H - 1); x1 = std::min(x1 + r, W - 1);
    const int h = y1 - y0 + 1, w = x1 - x0 + 1; const size_t n = size_t(h) * w;
    a.resize(n); b.resize(n); std::vector<uint8_t> dil(n), ero(n);
    for (int ly = 0; ly < h; ++ly) for (int lx = 0; lx < w; ++lx) dil[size_t(ly) * w + lx] = ero[size_t(ly) * w + lx] = m[size_t(y0 + ly) * W + x0 + lx];
    for (int it = 0; it < r; ++it) {
        std::copy(dil.begin(), dil.end(), a.begin()); std::copy(ero.begin(), ero.end(), b.begin());
        for (int ly = 0; ly < h; ++ly) for (int lx = 0; lx < w; ++lx) {
            const size_t li = size_t(ly) * w + lx;
            if (!a[li] && ((ly > 0 && a[li - w]) || (ly < h - 1 && a[li + w]) || (lx > 0 && a[li - 1]) || (lx < w - 1 && a[li + 1]))) dil[li] = 1;
            if (b[li] && (ly == 0 || ly == h - 1 || lx == 0 || lx == w - 1 || !b[li - w] || !b[li + w] || !b[li - 1] || !b[li + 1])) ero[li] = 0;
        }
    }
    for (int ly = 0; ly < h; ++ly) for (int lx = 0; lx < w; ++lx) { const size_t li = size_t(ly) * w + lx; if (dil[li] && !ero[li]) band.push_back(int32_t(size_t(y0 + ly) * W + x0 + lx)); }
}

class RtSegmenter {
public:
    RtSegmenter(int height, int width, RtParams p)
        : H_(height), W_(width), P_(p.patch), prm_(p), gh_(height / p.patch), gw_(width / p.patch) {
        const size_t N = size_t(H_) * W_;
        B_.assign(N, 0); var_.assign(N, prm_.sigma_floor * prm_.sigma_floor); u_.assign(N, 0); py_.assign(N, 0); px_.assign(N, 0);
        gxB_.assign(N, 0); gyB_.assign(N, 0); mask_.assign(N, 0); labels_.assign(N, 0);
        kkt_active_.assign(size_t(gh_) * gw_, 0); rp_.assign(size_t(gh_) * gw_, 0.0);
        F_.assign(N, 0); Bw_.assign(N, 0); e0_.assign(N, 0); e_.assign(N, 0); zs_.assign(N, 0); f_.assign(N, 0); w_.assign(N, 0);
        uhat_.assign(N, 0); act_.assign(N, 0); t1_.assign(N, 0); t2_.assign(N, 0); t3_.assign(N, 0); t4_.assign(N, 0); t5_.assign(N, 0);
        m8a_.assign(N, 0); m8b_.assign(N, 0); m8c_.assign(N, 0); m8c2_.assign(N, 0); A_.assign(N, 0.0); sgn_.assign(N, 0.0);
    }
    int height() const { return H_; } int width() const { return W_; }
    const std::vector<uint8_t>& mask() const { return mask_; }
    const std::vector<int32_t>& labels() const { return labels_; }
    const RtStats& stats() const { return stats_; }
    bool initialised() const { return have_bg_; }

    // Background from the temporal median of n frames (uint8, n*H*W), noise from the MAD.
    void init_background(const uint8_t* frames, int n) {
        const size_t N = size_t(H_) * W_; std::vector<double> v(n);
        for (size_t i = 0; i < N; ++i) {
            for (int k = 0; k < n; ++k) v[k] = frames[size_t(k) * N + i];
            std::nth_element(v.begin(), v.begin() + n / 2, v.end()); const double med = v[n / 2];
            B_[i] = med;
            for (int k = 0; k < n; ++k) v[k] = std::fabs(double(frames[size_t(k) * N + i]) - med);
            std::nth_element(v.begin(), v.begin() + n / 2, v.end());
            const double s = std::max(1.4826 * v[n / 2], prm_.sigma_floor); var_[i] = s * s;
        }
        std::fill(u_.begin(), u_.end(), 0.0); gradients(); have_bg_ = true;
    }

    void step(const uint8_t* frame) {
        const int H = H_, W = W_, P = P_, gh = gh_, gw = gw_; const size_t N = size_t(H) * W, n = size_t(gh) * gw;
        ++t_;
        auto tick = std::chrono::steady_clock::now(); int stage = 0;
        auto lap = [&](void) { auto now = std::chrono::steady_clock::now(); stats_.stage_ms[stage++] += std::chrono::duration<double, std::milli>(now - tick).count(); tick = now; };
        parallel_for(N, [&](size_t a, size_t b) { for (size_t i = a; i < b; ++i) F_[i] = frame[i]; });
        if (!have_bg_) { init_background(frame, 1); return; }
        // ---- 2. global jitter and background warp (full frame, cheap)
        double gdy = 0, gdx = 0; global_shift(gdy, gdx); stats_.dy = gdy; stats_.dx = gdx;
        if (gdy != 0 || gdx != 0) warp_full(gdy, gdx); else std::copy(B_.begin(), B_.end(), Bw_.begin());
        parallel_for(N, [&](size_t a, size_t b) { for (size_t i = a; i < b; ++i) e0_[i] = F_[i] - Bw_[i]; });
        lap();  // 0: jitter fit + warp
        // ---- illumination offset per patch from background pixels, interpolated across occluded patches
        dilate4(mask_.data(), H, W, 2, m8a_.data(), m8c_);
        std::vector<double> off(n, 0.0); std::vector<uint8_t> ok(n, 0);
        for (int gy = 0; gy < gh; ++gy) for (int gx = 0; gx < gw; ++gx) {
            double s = 0; int c = 0;
            for (int y = 0; y < P; ++y) for (int x = 0; x < P; ++x) { const size_t i = size_t(gy * P + y) * W + gx * P + x; if (!m8a_[i]) { s += e0_[i]; ++c; } }
            const size_t gi = size_t(gy) * gw + gx; if (c >= P * P / 4) { off[gi] = s / c; ok[gi] = 1; }
        }
        for (int pass = 0; pass < 8; ++pass) {
            bool any = false; std::vector<double> off2 = off; std::vector<uint8_t> ok2 = ok;
            for (int gy = 0; gy < gh; ++gy) for (int gx = 0; gx < gw; ++gx) {
                const size_t gi = size_t(gy) * gw + gx; if (ok[gi]) continue; any = true;
                double s = 0; int c = 0;
                const int ny[4] = {std::max(gy - 1, 0), std::min(gy + 1, gh - 1), gy, gy}, nx[4] = {gx, gx, std::max(gx - 1, 0), std::min(gx + 1, gw - 1)};
                for (int k = 0; k < 4; ++k) { const size_t gj = size_t(ny[k]) * gw + nx[k]; if (ok[gj]) { s += off[gj]; ++c; } }
                if (c) { off2[gi] = s / c; ok2[gi] = 1; }
            }
            off = off2; ok = ok2; if (!any) break;
        }
        for (int gy = 0; gy < gh; ++gy) for (int gx = 0; gx < gw; ++gx) {
            const double o = off[size_t(gy) * gw + gx];
            for (int y = 0; y < P; ++y) for (int x = 0; x < P; ++x) { const size_t i = size_t(gy * P + y) * W + gx * P + x; e_[i] = e0_[i] - o; }
        }
        for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) if (y >= gh * P || x >= gw * P) e_[size_t(y) * W + x] = e0_[size_t(y) * W + x];
        lap();  // 1: illumination
        // ---- prediction: advect each object by its velocity
        std::copy(u_.begin(), u_.end(), uhat_.begin()); stats_.pred_shift = 0;
        plab_.assign(labels_.begin(), labels_.end());
        parallel_for(N, [&](size_t a, size_t b) { for (size_t i = a; i < b; ++i) sgn_[i] = std::min(6.0, std::max(-6.0, e_[i] / std::sqrt(var_[i]))); });
        if (!vel_.empty()) {
            std::unordered_map<int, std::vector<int32_t>> pix;
            for (size_t i = 0; i < N; ++i) if (labels_[i]) pix[labels_[i]].push_back(int32_t(i));
            std::vector<double> moved;
            for (auto& kv : vel_) {
                const int dy = int(std::lround(kv.second.first)), dx = int(std::lround(kv.second.second));
                if ((dy == 0 && dx == 0) || std::abs(dy) > prm_.max_shift || std::abs(dx) > prm_.max_shift) continue;
                auto it = pix.find(kv.first); if (it == pix.end()) continue;
                moved.assign(it->second.size(), 0.0); for (size_t k = 0; k < it->second.size(); ++k) moved[k] = A_[it->second[k]];
                for (int32_t i : it->second) { uhat_[i] = 0.0; A_[i] = 0.0; if (plab_[i] == kv.first) plab_[i] = 0; }
                for (size_t k = 0; k < it->second.size(); ++k) { const int32_t i = it->second[k]; const int y = i / W + dy, x = i % W + dx;
                    if (y >= 0 && y < H && x >= 0 && x < W) { const size_t j = size_t(y) * W + x; uhat_[j] = std::max(uhat_[j], u_[i]); plab_[j] = kv.first; A_[j] = moved[k]; } }
                ++stats_.pred_shift;
            }
        }
        parallel_for(N, [&](size_t a, size_t b) { for (size_t i = a; i < b; ++i) A_[i] = prm_.rho * A_[i] + sgn_[i]; });      // Lagrangian evidence accumulation
        lap();  // 2: prediction + accumulation
        // ---- active set: coherent evidence (multi-scale score on e), object neighbourhoods, non-stationary patches, halo
        for (size_t i = 0; i < N; ++i) m8b_[i] = (mask_[i] || uhat_[i] > 0.5) ? 1 : 0;
        dilate4(m8b_.data(), H, W, 4, m8a_.data(), m8c_);
        std::vector<uint8_t> actp(n, 0);
        parallel_for(n, [&](size_t g0, size_t g1) {
            std::vector<double> ep(size_t(P) * P), S(size_t(P + 1) * (P + 1));
            for (size_t gi = g0; gi < g1; ++gi) {
                const int gy = int(gi / gw), gx = int(gi % gw); bool near = false;
                for (int y = 0; y < P; ++y) for (int x = 0; x < P; ++x) { const size_t i = size_t(gy * P + y) * W + gx * P + x; ep[size_t(y) * P + x] = sgn_[i]; if (m8a_[i]) near = true; }
                // studentised multi-scale score: box mean of z over (2r+1)^2 pixels has std 1/(2r+1) under the noise model
                double score = 0; for (int r = 1; r <= 3; ++r) score = std::max(score, detail::box_max_abs_mean(ep.data(), P, r, S.data()) * (2.0 * r + 1));
                const bool sc = score >= prm_.act_z * 2.5;   // 2.5: max over positions of the box field
                actp[gi] = (sc ? 1 : 0) | (near ? 2 : 0) | (kkt_active_[gi] ? 4 : 0);
            }
        });
        stats_.n_score = stats_.n_near = stats_.n_kkt = 0; stats_.n_patches = int(n);
        for (size_t gi = 0; gi < n; ++gi) { if (actp[gi] & 1) ++stats_.n_score; if (actp[gi] & 2) ++stats_.n_near; if (actp[gi] & 4) ++stats_.n_kkt; actp[gi] = actp[gi] ? 1 : 0; }
        for (int h = 0; h < prm_.halo; ++h) {
            std::vector<uint8_t> a2 = actp;
            for (int gy = 0; gy < gh; ++gy) for (int gx = 0; gx < gw; ++gx) {
                const size_t gi = size_t(gy) * gw + gx; if (actp[gi]) continue;
                if ((gy > 0 && actp[gi - gw]) || (gy < gh - 1 && actp[gi + gw]) || (gx > 0 && actp[gi - 1]) || (gx < gw - 1 && actp[gi + 1])) a2[gi] = 1;
            }
            actp = a2;
        }
        std::vector<int32_t> alist; alist.reserve(n);
        std::fill(act_.begin(), act_.end(), 0); size_t nact = 0;
        for (int gy = 0; gy < gh; ++gy) for (int gx = 0; gx < gw; ++gx) if (actp[size_t(gy) * gw + gx]) {
            alist.push_back(int32_t(size_t(gy) * gw + gx));
            for (int y = 0; y < P; ++y) for (int x = 0; x < P; ++x) { act_[size_t(gy * P + y) * W + gx * P + x] = 1; ++nact; }
        }
        stats_.active_frac = double(nact) / N;
        lap();  // 3: active set
        // ---- 3. evidence on active patches only (box means with a 3-px margin so edges are right)
        const int M = 3;
        // four-colour partition of the active list: patches of one colour are never adjacent, so their
        // margins never overlap and each colour class can be processed in parallel without races
        std::vector<int32_t> colour[4];
        for (int gi : alist) colour[((gi / gw) % 2) * 2 + (gi % gw) % 2].push_back(gi);
        auto par_patches = [&](auto&& body) {
            for (int c = 0; c < 4; ++c) { auto& lst = colour[c]; parallel_for(lst.size(), [&](size_t a, size_t b) { for (size_t k = a; k < b; ++k) body(lst[k]); }); }
        };
        auto for_region = [&](int gi, int margin, auto&& fn) {
            const int gy = gi / gw, gx = gi % gw;
            const int y0 = std::max(gy * P - margin, 0), y1 = std::min(gy * P + P + margin, H), x0 = std::max(gx * P - margin, 0), x1 = std::min(gx * P + P + margin, W);
            for (int y = y0; y < y1; ++y) for (int x = x0; x < x1; ++x) fn(y, x);
        };
        auto boxv = [&](const std::vector<double>& src, int y, int x, int r) {
            double s = 0; for (int yy = y - r; yy <= y + r; ++yy) { const int cy = std::min(std::max(yy, 0), H - 1); for (int xx = x - r; xx <= x + r; ++xx) s += src[size_t(cy) * W + std::min(std::max(xx, 0), W - 1)]; }
            return s / (double(2 * r + 1) * (2 * r + 1));
        };
        par_patches([&](int gi) { for_region(gi, M, [&](int y, int x) { const size_t i = size_t(y) * W + x; t1_[i] = std::fabs(sgn_[i]); t2_[i] = std::fabs(A_[i]); }); });
        const double accn = std::sqrt(1.0 - prm_.rho * prm_.rho);
        par_patches([&](int gi) { for_region(gi, M - 1, [&](int y, int x) { const size_t i = size_t(y) * W + x; zs_[i] = std::max(boxv(t1_, y, x, 1), boxv(t2_, y, x, 1) * accn); }); });
        if (prm_.shadows) {
            par_patches([&](int gi) { for_region(gi, M - 1, [&](int y, int x) {
                const size_t i = size_t(y) * W + x;
                double mF = 0, mB = 0, sFF = 0, sBB = 0, sFB = 0, sR = 0, sRR = 0; int c = 0;
                for (int yy = y - 2; yy <= y + 2; ++yy) { const int cy = std::min(std::max(yy, 0), H - 1); for (int xx = x - 2; xx <= x + 2; ++xx) {
                    const size_t j = size_t(cy) * W + std::min(std::max(xx, 0), W - 1); const double a = F_[j], b = Bw_[j], r = a / std::max(b, 1.0);
                    mF += a; mB += b; sFF += a * a; sBB += b * b; sFB += a * b; sR += r; sRR += r * r; ++c; } }
                mF /= c; mB /= c; const double vF = sFF / c - mF * mF, vB = sBB / c - mB * mB, cov = sFB / c - mF * mB, rs = sR / c, rvar = sRR / c - rs * rs;
                const double ncc = cov / std::sqrt(std::max(vF, 1e-3) * std::max(vB, 1e-3)), ra = F_[i] / std::max(Bw_[i], 1.0);
                const bool textured = ra > 0.45 && ra < 0.95 && ncc > 0.8, flat = vB < 16.0 && rs > 0.55 && rs < 0.95 && rvar < 0.004;
                if (textured || flat) zs_[i] = std::min(zs_[i], prm_.z0 - 0.5);
            }); });
        }
        par_patches([&](int gi) { for_region(gi, M - 1, [&](int y, int x) { const size_t i = size_t(y) * W + x; f_[i] = std::min(3.0, std::max(-3.0, prm_.gamma * (prm_.z0 - zs_[i]))); t2_[i] = boxv(e_, y, x, 1); t1_[i] = boxv(F_, y, x, 1); }); });
        par_patches([&](int gi) { for_region(gi, M - 2, [&](int y, int x) {
            const size_t i = size_t(y) * W + x;
            const double gy = y < H - 1 ? t2_[i + W] - t2_[i] : 0.0, gx = x < W - 1 ? t2_[i + 1] - t2_[i] : 0.0;
            const double hy = y < H - 1 ? t1_[i + W] - t1_[i] : 0.0, hx = x < W - 1 ? t1_[i + 1] - t1_[i] : 0.0;
            const double wr = std::exp(-(gy * gy + gx * gx) / (2 * prm_.tau_edge * prm_.tau_edge)), wi = std::exp(-(hy * hy + hx * hx) / (2 * prm_.tau_img * prm_.tau_img));
            w_[i] = std::min(wr, wi) + 0.05;
        }); });
        lap();  // 4: evidence
        // ---- correction: K Chambolle-Pock steps, primal and dual on active pixels (+1 px ring for the dual)
        std::vector<double>& u = u_; std::vector<double>& ubar = t3_; std::vector<double>& uold = t4_; std::vector<double>& div = t5_;
        par_patches([&](int gi) { for_region(gi, 1, [&](int y, int x) { const size_t i = size_t(y) * W + x; ubar[i] = uhat_[i]; }); });
        par_patches([&](int gi) { for_region(gi, 0, [&](int y, int x) { const size_t i = size_t(y) * W + x; u[i] = uhat_[i]; }); });
        const double tau = 1.0 / std::sqrt(8.0), sig = tau, lam = prm_.lam, mu = prm_.mu;
        auto div_at = [&](int y, int x) {
            const size_t i = size_t(y) * W + x; double v = 0;
            if (y < H - 1) v += w_[i] * py_[i]; if (y > 0) v -= w_[i - W] * py_[i - W];
            if (x < W - 1) v += w_[i] * px_[i]; if (x > 0) v -= w_[i - 1] * px_[i - 1];
            return v;
        };
        for (int k = 0; k < prm_.steps; ++k) {
            par_patches([&](int gi) { for_region(gi, 1, [&](int y, int x) {
                const size_t i = size_t(y) * W + x;
                const double gy = y < H - 1 ? ubar[i + W] - ubar[i] : 0.0, gx = x < W - 1 ? ubar[i + 1] - ubar[i] : 0.0;
                double a = py_[i] + sig * w_[i] * gy, b = px_[i] + sig * w_[i] * gx;
                const double nr = std::max(1.0, std::sqrt(a * a + b * b) / lam); py_[i] = a / nr; px_[i] = b / nr;
            }); });
            par_patches([&](int gi) { for_region(gi, 0, [&](int y, int x) { const size_t i = size_t(y) * W + x; div[i] = div_at(y, x); }); });
            par_patches([&](int gi) { for_region(gi, 0, [&](int y, int x) {
                const size_t i = size_t(y) * W + x; uold[i] = u[i];
                const double un = (u[i] - tau * (-div[i] + f_[i]) + tau * mu * uhat_[i]) / (1.0 + tau * mu); u[i] = std::min(1.0, std::max(0.0, un));
                ubar[i] = 2 * u[i] - uold[i];
            }); });
        }
        lap();  // 5: correction
        // ---- KKT residual on active patches (frozen patches keep their last certified value)
        double ka = 0, kf = 0;
        par_patches([&](int gi) {
            double rpmax = 0;
            for_region(gi, 0, [&](int y, int x) {
                const size_t i = size_t(y) * W + x; const double g = -div_at(y, x) + f_[i] + mu * (u[i] - uhat_[i]);
                const double r = u[i] <= 0.0 ? std::max(0.0, -g) : (u[i] >= 1.0 ? std::max(0.0, g) : std::fabs(g)); rpmax = std::max(rpmax, r);
            });
            rp_[gi] = rpmax;
        });
        for (int gi : alist) ka = std::max(ka, rp_[gi]);
        for (size_t gi = 0; gi < n; ++gi) { kkt_active_[gi] = rp_[gi] > prm_.kkt_tol ? 1 : 0; if (!actp[gi]) kf = std::max(kf, rp_[gi]); }
        stats_.kkt_frozen_max = kf; stats_.kkt_active_max = ka;
        lap();  // 6: kkt
        // ---- threshold on active patches, previous mask elsewhere; certified boxes; clean-up
        std::copy(mask_.begin(), mask_.end(), m8b_.begin());
        for (int gi : alist) for_region(gi, 0, [&](int y, int x) { const size_t i = size_t(y) * W + x; m8b_[i] = u[i] > 0.5 ? 1 : 0; });
        if (prm_.certify) {
            for (int gi : alist) for (int r = 1; r <= 3; ++r) {
                const double D = 80.0 * std::pow(2.0 * r + 1, -0.5);
                for_region(gi, 0, [&](int y, int x) {
                    if (y < r || y >= H - r || x < r || x >= W - r) return;
                    if (std::fabs(boxv(e_, y, x, r)) >= 0.5 * D) for (int yy = y - r; yy <= y + r; ++yy) for (int xx = x - r; xx <= x + r; ++xx) m8b_[size_t(yy) * W + xx] = 1;
                });
            }
        }
        if (prm_.close_r > 0) { dilate4(m8b_.data(), H, W, prm_.close_r, m8a_.data(), m8c_); erode4(m8a_.data(), H, W, prm_.close_r, m8b_.data(), m8c_); }
        std::vector<std::array<int, 4>> boxes;   // kept components' bounding boxes (y0, y1, x0, x1)
        std::fill(m8c2_.begin(), m8c2_.end(), 0);  // ghost pixels (absorbed fast, not reported)
        {   // one labeling: area filter and line filter together, then hole filling per bounding box
            std::vector<int32_t> lb(N); std::vector<int32_t> ar; const int k = label_components(m8b_.data(), H, W, lb.data(), &ar);
            std::vector<int> y0(k + 1, H), y1(k + 1, -1), x0(k + 1, W), x1(k + 1, -1);
            for (size_t i = 0; i < N; ++i) if (lb[i]) { const int c = lb[i], y = int(i / W), x = int(i % W); y0[c] = std::min(y0[c], y); y1[c] = std::max(y1[c], y); x0[c] = std::min(x0[c], x); x1[c] = std::max(x1[c], x); }
            std::vector<char> drop(k + 1, 0);
            // contour contrast per component: for each boundary pixel, the largest studentised intensity step
            // within a 7x7 window (so a contour that overshoots the object by a pixel or two still finds its edge)
            // Ghost test: a ghost's outline is in the background model, not in the frame; a real object's is in
            // the frame, not in the model. Compare the contour contrast of F with that of B (windowed max step).
            std::vector<double> cF(k + 1, 0.0), cB(k + 1, 0.0); std::vector<int> ccnt(k + 1, 0);
            for (size_t i = 0; i < N; ++i) if (lb[i]) {
                const int y = int(i / W), x = int(i % W);
                const bool bnd = (y > 0 && !m8b_[i - W]) || (y < H - 1 && !m8b_[i + W]) || (x > 0 && !m8b_[i - 1]) || (x < W - 1 && !m8b_[i + 1]);
                if (!bnd) continue;
                double bf = 0, bb = 0; const double fi = F_[i], bi = Bw_[i], si = std::sqrt(var_[i]);
                for (int yy = std::max(y - 3, 0); yy <= std::min(y + 3, H - 1); ++yy) for (int xx = std::max(x - 3, 0); xx <= std::min(x + 3, W - 1); ++xx) { const size_t j = size_t(yy) * W + xx; bf = std::max(bf, std::fabs(F_[j] - fi)); bb = std::max(bb, std::fabs(Bw_[j] - bi)); }
                cF[lb[i]] += bf / si; cB[lb[i]] += bb / si; ccnt[lb[i]]++;
            }
            std::vector<char> ghost(k + 1, 0);
            for (int c = 1; c <= k; ++c) {
                const int hh = y1[c] - y0[c] + 1, ww = x1[c] - x0[c] + 1, mn = std::min(hh, ww), mx = std::max(hh, ww);
                if (ar[c] < prm_.min_area || (mn < prm_.min_side && mx > prm_.max_aspect * mn)) { drop[c] = 1; continue; }
                if (ccnt[c] && (cF[c] < prm_.ghost_contrast * ccnt[c] || cF[c] < 0.6 * cB[c])) { drop[c] = 1; ghost[c] = 1; continue; }
                boxes.push_back({y0[c], y1[c], x0[c], x1[c]});
            }
            parallel_for(N, [&](size_t a, size_t b) { for (size_t i = a; i < b; ++i) if (lb[i] && drop[lb[i]]) { if (ghost[lb[i]]) m8c2_[i] = 1; m8b_[i] = 0; } });
            std::vector<uint8_t> reach; std::vector<int32_t> q;
            for (auto& bx : boxes) fill_holes_box(m8b_.data(), H, W, bx[0], bx[1], bx[2], bx[3], reach, q);
        }
        lap();  // 7: clean-up (closing, components, holes, thin)
        if (prm_.snap_r > 0) {   // guided-filter boundary snap in a band around the contour (He, Sun & Tang 2010)
            const int r = prm_.snap_r; const double eps = prm_.snap_eps;
            std::vector<int32_t> band; { std::vector<uint8_t> sa, sb; for (auto& bx : boxes) band_box(m8b_.data(), H, W, bx[0], bx[1], bx[2], bx[3], r + 1, band, sa, sb); }
            std::sort(band.begin(), band.end()); band.erase(std::unique(band.begin(), band.end()), band.end());
            if (!band.empty()) {
                // a, b over the band plus r margin (needed for the final box of a, b)
                std::vector<int32_t> need; need.reserve(band.size() * 4);
                for (int32_t i : band) { const int y = i / W, x = i % W; for (int yy = std::max(y - r, 0); yy <= std::min(y + r, H - 1); ++yy) for (int xx = std::max(x - r, 0); xx <= std::min(x + r, W - 1); ++xx) need.push_back(int32_t(size_t(yy) * W + xx)); }
                std::sort(need.begin(), need.end()); need.erase(std::unique(need.begin(), need.end()), need.end());
                std::vector<double>& a = t3_; std::vector<double>& b = t4_;
                const double cnt = double(2 * r + 1) * (2 * r + 1);
                parallel_for(need.size(), [&](size_t k0, size_t k1) { for (size_t kk = k0; kk < k1; ++kk) { const size_t i = size_t(need[kk]);
                    const int y = int(i / W), x = int(i % W); double sI = 0, sP = 0, sII = 0, sIP = 0;
                    for (int yy = y - r; yy <= y + r; ++yy) { const int cy = std::min(std::max(yy, 0), H - 1); for (int xx = x - r; xx <= x + r; ++xx) { const size_t j = size_t(cy) * W + std::min(std::max(xx, 0), W - 1); const double I = F_[j], P = double(m8b_[j]); sI += I; sP += P; sII += I * I; sIP += I * P; } }
                    const double mI = sI / cnt, mP = sP / cnt, vI = sII / cnt - mI * mI, cIP = sIP / cnt - mI * mP;
                    a[i] = cIP / (vI + eps); b[i] = mP - a[i] * mI;
                } });
                std::vector<uint8_t> newv(band.size());
                parallel_for(band.size(), [&](size_t k0, size_t k1) { for (size_t k = k0; k < k1; ++k) {
                    const int32_t i = band[k]; const int y = i / W, x = i % W; double sa = 0, sb = 0;
                    for (int yy = y - r; yy <= y + r; ++yy) { const int cy = std::min(std::max(yy, 0), H - 1); for (int xx = x - r; xx <= x + r; ++xx) { const size_t j = size_t(cy) * W + std::min(std::max(xx, 0), W - 1); sa += a[j]; sb += b[j]; } }
                    newv[k] = (sa / cnt * F_[i] + sb / cnt) > 0.5 ? 1 : 0;
                } });
                for (size_t k = 0; k < band.size(); ++k) m8b_[band[k]] = newv[k];
                std::vector<uint8_t> reach; std::vector<int32_t> q;
                for (auto& bx : boxes) fill_holes_box(m8b_.data(), H, W, bx[0], bx[1], bx[2], bx[3], reach, q);
            }
        }
        lap();  // 8: snap
        // ---- background / noise update (full frame)
        dilate4(m8b_.data(), H, W, 3, m8a_.data(), m8c_);
        const double fl2 = prm_.sigma_floor * prm_.sigma_floor;
        parallel_for(N, [&](size_t a0, size_t b0) { for (size_t i = a0; i < b0; ++i) {
            const double a = m8c2_[i] ? prm_.alpha_ghost : (m8a_[i] ? prm_.alpha_fg : prm_.alpha_bg);
            B_[i] = (1 - a) * B_[i] + a * F_[i];
            if (!m8a_[i]) var_[i] = std::max((1 - prm_.alpha_bg) * var_[i] + prm_.alpha_bg * e_[i] * e_[i], fl2);
        } });
        parallel_for(size_t(H), [&](size_t y0, size_t y1) { for (size_t y = y0; y < y1; ++y) for (int x = 0; x < W; ++x) {
            const size_t i = y * W + x; const double r = B_[i];
            gxB_[i] = x == 0 ? double(B_[i + 1]) - r : (x == W - 1 ? r - double(B_[i - 1]) : 0.5 * (double(B_[i + 1]) - double(B_[i - 1])));
            gyB_[i] = y == 0 ? double(B_[i + W]) - r : (y + 1 == size_t(H) ? r - double(B_[i - W]) : 0.5 * (double(B_[i + W]) - double(B_[i - W])));
        } });
        lap();  // 9: background update + gradients
        // ---- tracking: match components to PREDICTED ids by overlap; split merged components by nearest predicted object
        std::vector<int32_t> lab(N); std::vector<int32_t> areas; const int nc = label_components(m8b_.data(), H, W, lab.data(), &areas);
        std::unordered_map<int, int> ref_area; for (size_t i = 0; i < N; ++i) if (plab_[i]) ref_area[plab_[i]]++;
        std::vector<std::unordered_map<int, int>> ov(nc + 1);
        for (size_t i = 0; i < N; ++i) if (lab[i] && plab_[i]) ov[lab[i]][plab_[i]]++;
        std::vector<char> used; used.assign(next_id_ + 1, 0);
        std::vector<int32_t> out(N, 0);
        std::vector<std::vector<int32_t>> cpix(nc + 1);
        for (size_t i = 0; i < N; ++i) if (lab[i]) cpix[lab[i]].push_back(int32_t(i));
        std::vector<int32_t> queue;
        for (int c = 1; c <= nc; ++c) {
            std::vector<std::pair<int, int>> cand;
            for (auto& kv : ov[c]) if (kv.first < (int)used.size() && !used[kv.first] && kv.second >= 0.2 * std::min(areas[c], ref_area[kv.first])) cand.push_back({kv.second, kv.first});
            std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
            if (cand.size() >= 2) {   // split only between substantial predicted objects
                const int top = ref_area[cand[0].second]; std::vector<std::pair<int, int>> big;
                for (auto& pc : cand) if (ref_area[pc.second] >= 0.3 * top && ref_area[pc.second] >= 0.2 * areas[c]) big.push_back(pc);
                if (big.size() >= 2) cand = big; else cand.resize(1);
            }
            if (cand.size() >= 2) {
                // seeds: component pixels carrying a candidate predicted label; BFS partition of the rest
                std::vector<char> isc; isc.assign(next_id_ + 1, 0); for (auto& pc : cand) isc[pc.second] = 1;
                queue.clear();
                for (int32_t i : cpix[c]) if (plab_[i] && isc[plab_[i]]) { out[i] = plab_[i]; queue.push_back(i); }
                for (size_t qh = 0; qh < queue.size(); ++qh) {
                    const int32_t i = queue[qh]; const int y = i / W, x = i % W; const int nb[4] = {y > 0 ? i - W : -1, y < H - 1 ? i + W : -1, x > 0 ? i - 1 : -1, x < W - 1 ? i + 1 : -1};
                    for (int j : nb) if (j >= 0 && lab[j] == c && !out[j]) { out[j] = out[i]; queue.push_back(j); }
                }
                for (int32_t i : cpix[c]) if (!out[i]) out[i] = cand[0].second;
                for (auto& pc : cand) used[pc.second] = 1;
                continue;
            }
            int best = cand.empty() ? 0 : cand[0].second;
            if (!best) { best = next_id_++; used.resize(next_id_ + 1, 0); }
            used[best] = 1;
            for (int32_t i : cpix[c]) out[i] = best;
        }
        std::unordered_map<int, std::pair<double, double>> cents; std::unordered_map<int, int> cnt;
        for (size_t i = 0; i < N; ++i) { labels_[i] = out[i]; if (out[i]) { auto& cc = cents[out[i]]; cc.first += double(i / W); cc.second += double(i % W); cnt[out[i]]++; } }
        for (auto& kv : cents) { kv.second.first /= cnt[kv.first]; kv.second.second /= cnt[kv.first]; }
        vel_.clear();
        for (auto& kv : cents) { auto it = cents_.find(kv.first); if (it != cents_.end()) vel_[kv.first] = {kv.second.first - it->second.first, kv.second.second - it->second.second}; }
        cents_ = cents; stats_.n_objects = int(cents.size());
        std::copy(m8b_.begin(), m8b_.end(), mask_.begin());
        lap();  // 10: tracking
    }

private:
    void gradients() { detail::gradients(B_, H_, W_, gxB_, gyB_); }
    void divergence(std::vector<double>& d) {
        const int H = H_, W = W_;
        for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
            const size_t i = size_t(y) * W + x; double v = 0;
            if (y < H - 1) v += w_[i] * py_[i];
            if (y > 0) v -= w_[i - W] * py_[i - W];
            if (x < W - 1) v += w_[i] * px_[i];
            if (x > 0) v -= w_[i - 1] * px_[i - 1];
            d[i] = v;
        }
    }
    void warp_full(double dy, double dx) {
        const int H = H_, W = W_;
        for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
            const double yy = detail::clampd(y + dy, 0, H - 1), xx = detail::clampd(x + dx, 0, W - 1);
            const int iy = int(std::floor(yy)), ix = int(std::floor(xx)), iy1 = std::min(iy + 1, H - 1), ix1 = std::min(ix + 1, W - 1);
            const double fy = yy - iy, fx = xx - ix;
            Bw_[size_t(y) * W + x] = (1 - fy) * (1 - fx) * B_[size_t(iy) * W + ix] + (1 - fy) * fx * B_[size_t(iy) * W + ix1]
                                   + fy * (1 - fx) * B_[size_t(iy1) * W + ix] + fy * fx * B_[size_t(iy1) * W + ix1];
        }
    }
    void global_shift(double& gdy, double& gdx) {
        const int P = P_, gh = gh_, gw = gw_, W = W_; const size_t n = size_t(gh) * gw;
        std::vector<double> energy(n); std::vector<uint8_t> occl(n);
        parallel_for(n, [&](size_t g0, size_t g1) { for (size_t gi = g0; gi < g1; ++gi) {
            const int gy = int(gi / gw), gx = int(gi % gw); double s = 0; int m = 0;
            for (int y = 0; y < P; ++y) for (int x = 0; x < P; ++x) { const size_t i = size_t(gy * P + y) * W + gx * P + x; s += gxB_[i] * gxB_[i] + gyB_[i] * gyB_[i]; m += mask_[i]; }
            energy[gi] = s; occl[gi] = m > 0.1 * P * P;
        } });
        std::vector<double> es = energy; std::nth_element(es.begin(), es.begin() + n / 2, es.end()); const double med = es[n / 2];
        std::vector<double> dyv(n, 0.0), dxv(n, 0.0); std::vector<uint8_t> okv(n, 0);
        size_t n_elig = 0; for (size_t gi = 0; gi < n; ++gi) if (energy[gi] >= med && !occl[gi]) ++n_elig;
        const size_t stride = std::max<size_t>(1, n_elig / 600);          // at most ~600 fits per frame
        std::vector<uint8_t> pick(n, 0); { size_t c = 0; for (size_t gi = 0; gi < n; ++gi) if (energy[gi] >= med && !occl[gi]) { if (c % stride == 0) pick[gi] = 1; ++c; } }
        parallel_for(n, [&](size_t g0, size_t g1) { std::vector<double> wp(size_t(P) * P); for (size_t gi = g0; gi < g1; ++gi) {
            if (!pick[gi]) continue;
            const int gy = int(gi / gw), gx = int(gi % gw);
            detail::fit_translation_patch<double>(B_, gxB_, gyB_, F_.data(), H_, W_, P, gy * P, gx * P, prm_.delta_max, prm_.fit_iters, wp.data(), dyv[gi], dxv[gi]); okv[gi] = 1;
        } });
        std::vector<double> dys, dxs; for (size_t gi = 0; gi < n; ++gi) if (okv[gi]) { dys.push_back(dyv[gi]); dxs.push_back(dxv[gi]); }
        if (dys.size() < 4) { gdy = gdx = 0; return; }
        std::nth_element(dys.begin(), dys.begin() + dys.size() / 2, dys.end()); std::nth_element(dxs.begin(), dxs.begin() + dxs.size() / 2, dxs.end());
        gdy = dys[dys.size() / 2]; gdx = dxs[dxs.size() / 2];
    }

    int H_, W_, P_; RtParams prm_; int gh_, gw_;
    std::vector<double> B_, var_, u_, py_, px_, gxB_, gyB_, F_, Bw_, e0_, e_, zs_, f_, w_, uhat_, t1_, t2_, t3_, t4_, t5_, act_d_, I_, rp_, A_, sgn_;
    std::vector<uint8_t> mask_, act_, m8a_, m8b_, m8c_, m8c2_, kkt_active_;
    std::vector<int32_t> labels_, plab_;
    std::unordered_map<int, std::pair<double, double>> cents_, vel_;
    int next_id_ = 1; bool have_bg_ = false; long t_ = -1; RtStats stats_;
};

} // namespace rt
} // namespace certskip
