// certskip.hpp — certified patch-skip rule for fixed-camera video.
//
// Rule: a patch is DROPPED while the spread (max - min) of its pixel change
// relative to the LAST KEPT copy stays strictly below a contrast level delta.
// Otherwise it is KEPT and becomes the new reference.
//
// Guarantee (persistence stability, Cohen-Steiner/Edelsbrunner/Harer 2007):
// if the change d = F - R has spread s < delta, then after removing the
// brightness shift c = (max d + min d)/2 we have ||F - (R + c)||_inf <= s/2,
// so the sublevel/superlevel persistence diagrams of F and R+c are within
// bottleneck distance s/2. No feature of contrast >= delta can be born from or
// die into the diagonal, and every feature's contrast changes by at most s.
//
// This header is dependency-free C++17 and is the single source of truth for
// the algorithm. The Python package has a numpy reference that must agree
// with it bit-for-bit (see tests/test_core.py).
#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>

namespace certskip {

// Accumulator wide enough that F - R cannot wrap.
template <typename T> struct Wide;
template <> struct Wide<uint8_t>  { using type = int32_t; };
template <> struct Wide<uint16_t> { using type = int32_t; };
template <> struct Wide<float>    { using type = float; };

struct Params {
    int patch = 16;          // patch side in pixels
    double delta = 32.0;     // contrast level (grey levels)
    double max_shift = -1;   // optional cap on |brightness shift|; <0 disables
};

// Per-frame output for one patch grid.
struct FrameResult {
    std::vector<uint8_t> keep;    // 1 = kept (re-encoded), 0 = dropped
    std::vector<float>   spread;  // max(d)-min(d) per patch
    std::vector<float>   shift;   // (max(d)+min(d))/2 per patch
    int gh = 0, gw = 0;           // grid size (patches)
    size_t kept = 0;
};

// Running state: reference image (last kept copy of every patch).
template <typename T>
class Pruner {
public:
    Pruner(int height, int width, Params p)
        : H_(height), W_(width), p_(p),
          gh_(height / p.patch), gw_(width / p.patch),
          ref_(static_cast<size_t>(height) * width, T{}),
          have_ref_(false) {}

    int grid_h() const { return gh_; }
    int grid_w() const { return gw_; }
    bool initialised() const { return have_ref_; }
    const std::vector<T>& reference() const { return ref_; }
    void reset() { have_ref_ = false; }

    // Process one frame (row-major, H*W, stride = W). Writes keep/spread/shift
    // for gh*gw patches. Pixels outside the patch grid (H%patch, W%patch) are
    // ignored by the rule but still copied into the reference on the first
    // frame so reference() is a full image.
    void step(const T* frame, uint8_t* keep, float* spread, float* shift) {
        const int P = p_.patch;
        const size_t n = static_cast<size_t>(gh_) * gw_;
        if (!have_ref_) {
            std::copy(frame, frame + static_cast<size_t>(H_) * W_, ref_.begin());
            std::fill(keep, keep + n, uint8_t{1});
            std::fill(spread, spread + n, 0.f);
            std::fill(shift, shift + n, 0.f);
            have_ref_ = true;
            return;
        }
        for (int gy = 0; gy < gh_; ++gy) {
            for (int gx = 0; gx < gw_; ++gx) {
                const size_t gi = static_cast<size_t>(gy) * gw_ + gx;
                // Reduce max/min of the difference over the P×P block.
                // Use a wide accumulator so uint8 differences do not wrap.
                using Acc = typename Wide<T>::type;
                Acc dmax = std::numeric_limits<Acc>::lowest();
                Acc dmin = std::numeric_limits<Acc>::max();
                const size_t base = static_cast<size_t>(gy) * P * W_ + static_cast<size_t>(gx) * P;
                for (int y = 0; y < P; ++y) {
                    const T* f = frame + base + static_cast<size_t>(y) * W_;
                    const T* r = ref_.data() + base + static_cast<size_t>(y) * W_;
                    for (int x = 0; x < P; ++x) {
                        const Acc d = static_cast<Acc>(f[x]) - static_cast<Acc>(r[x]);
                        dmax = d > dmax ? d : dmax;
                        dmin = d < dmin ? d : dmin;
                    }
                }
                const double s = static_cast<double>(dmax) - static_cast<double>(dmin);
                const double c = 0.5 * (static_cast<double>(dmax) + static_cast<double>(dmin));
                bool drop = s < p_.delta;
                if (p_.max_shift >= 0 && std::fabs(c) > p_.max_shift) drop = false;
                spread[gi] = static_cast<float>(s);
                shift[gi]  = static_cast<float>(c);
                keep[gi]   = drop ? 0 : 1;
                if (!drop) {
                    for (int y = 0; y < P; ++y) {
                        const T* f = frame + base + static_cast<size_t>(y) * W_;
                        T* r = ref_.data() + base + static_cast<size_t>(y) * W_;
                        std::copy(f, f + P, r);
                    }
                }
            }
        }
    }

    FrameResult step(const T* frame) {
        FrameResult out;
        out.gh = gh_; out.gw = gw_;
        const size_t n = static_cast<size_t>(gh_) * gw_;
        out.keep.resize(n); out.spread.resize(n); out.shift.resize(n);
        step(frame, out.keep.data(), out.spread.data(), out.shift.data());
        out.kept = static_cast<size_t>(std::count(out.keep.begin(), out.keep.end(), uint8_t{1}));
        return out;
    }

private:
    int H_, W_;
    Params p_;
    int gh_, gw_;
    std::vector<T> ref_;
    bool have_ref_;
};

// Baseline used only for comparison: mean |F_t - F_{t-1}| per patch < tau
// (the EVS / TimeChat-Online / run-length-tokenisation family). Stateless
// beyond the previous frame; has no guarantee. Returned so the CLI can print
// both rules side by side.
template <typename T>
inline void consecutive_mean(const T* prev, const T* cur, int H, int W, int P,
                             float* mean_abs_out) {
    const int gh = H / P, gw = W / P;
    for (int gy = 0; gy < gh; ++gy)
        for (int gx = 0; gx < gw; ++gx) {
            double acc = 0;
            const size_t base = static_cast<size_t>(gy) * P * W + static_cast<size_t>(gx) * P;
            for (int y = 0; y < P; ++y) {
                const T* a = prev + base + static_cast<size_t>(y) * W;
                const T* b = cur  + base + static_cast<size_t>(y) * W;
                for (int x = 0; x < P; ++x)
                    acc += std::fabs(static_cast<double>(b[x]) - static_cast<double>(a[x]));
            }
            mean_abs_out[static_cast<size_t>(gy) * gw + gx] = static_cast<float>(acc / (double(P) * P));
        }
}

} // namespace certskip

// ---------------------------------------------------------------------------
// Quotient rule: certify modulo a sub-pixel translation per patch (THEORY.md §3),
// and the sequential rule on top of it (THEORY.md §6).
// ---------------------------------------------------------------------------
namespace certskip {
namespace detail {

inline double clampd(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }

inline void solve3(double a00, double a01, double a02, double a11, double a12, double a22,
                   double b0, double b1, double b2, double& x0, double& x1, double& x2) {
    const double det = a00 * (a11 * a22 - a12 * a12) - a01 * (a01 * a22 - a12 * a02) + a02 * (a01 * a12 - a11 * a02);
    if (std::fabs(det) < 1e-12) { x0 = x1 = 0; x2 = b2 / a22; return; }
    x0 = (b0 * (a11 * a22 - a12 * a12) - a01 * (b1 * a22 - a12 * b2) + a02 * (b1 * a12 - a11 * b2)) / det;
    x1 = (a00 * (b1 * a22 - a12 * b2) - b0 * (a01 * a22 - a12 * a02) + a02 * (a01 * b2 - b1 * a02)) / det;
    x2 = (a00 * (a11 * b2 - b1 * a12) - a01 * (a01 * b2 - b1 * a02) + b0 * (a01 * a12 - a11 * a02)) / det;
}

// Central-difference gradients of a full frame (one-sided at the border).
template <typename T>
inline void gradients(const std::vector<T>& ref, int H, int W, std::vector<double>& gx, std::vector<double>& gy) {
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
        const size_t i = static_cast<size_t>(y) * W + x; const double r = ref[i];
        gx[i] = x == 0 ? double(ref[i + 1]) - r : (x == W - 1 ? r - double(ref[i - 1]) : 0.5 * (double(ref[i + 1]) - double(ref[i - 1])));
        gy[i] = y == 0 ? double(ref[i + W]) - r : (y == H - 1 ? r - double(ref[i - W]) : 0.5 * (double(ref[i + W]) - double(ref[i - W])));
    }
}

// Bilinear sample of the reference, shifted by (dy, dx), over one patch.
template <typename T>
inline void warp_patch(const std::vector<T>& ref, int H, int W, int P, int y0, int x0, double dy, double dx, double* out) {
    for (int y = 0; y < P; ++y) for (int x = 0; x < P; ++x) {
        const double yy = clampd(y0 + y + dy, 0, H - 1), xx = clampd(x0 + x + dx, 0, W - 1);
        const int iy = int(std::floor(yy)), ix = int(std::floor(xx));
        const int iy1 = std::min(iy + 1, H - 1), ix1 = std::min(ix + 1, W - 1);
        const double fy = yy - iy, fx = xx - ix;
        out[size_t(y) * P + x] = (1 - fy) * (1 - fx) * ref[size_t(iy) * W + ix] + (1 - fy) * fx * ref[size_t(iy) * W + ix1]
                               + fy * (1 - fx) * ref[size_t(iy1) * W + ix] + fy * fx * ref[size_t(iy1) * W + ix1];
    }
}

// Gauss-Newton fit of (dy, dx) on brightness constancy with a Tikhonov prior; writes the warped patch.
template <typename T>
inline void fit_translation_patch(const std::vector<T>& ref, const std::vector<double>& gx, const std::vector<double>& gy,
                                  const T* frame, int H, int W, int P, int y0, int x0, double dmax, int iters,
                                  double* wp, double& dy, double& dx) {
    double a00 = double(P) * P, a01 = 0, a02 = 0, a11 = double(P) * P, a12 = 0, a22 = double(P) * P;
    for (int y = 0; y < P; ++y) for (int x = 0; x < P; ++x) {
        const size_t i = size_t(y0 + y) * W + (x0 + x); const double rx = gx[i], ry = gy[i];
        a00 += rx * rx; a01 += rx * ry; a02 += rx; a11 += ry * ry; a12 += ry;
    }
    dx = 0; dy = 0;
    for (int it = 0; it < iters; ++it) {
        warp_patch(ref, H, W, P, y0, x0, dy, dx, wp);
        double b0 = 0, b1 = 0, b2 = 0;
        for (int y = 0; y < P; ++y) for (int x = 0; x < P; ++x) {
            const size_t i = size_t(y0 + y) * W + (x0 + x);
            const double d = double(frame[i]) - wp[size_t(y) * P + x];
            b0 += gx[i] * d; b1 += gy[i] * d; b2 += d;
        }
        double sx, sy, sc; solve3(a00, a01, a02, a11, a12, a22, b0, b1, b2, sx, sy, sc);
        dx = clampd(dx + sx, -dmax, dmax); dy = clampd(dy + sy, -dmax, dmax);
    }
    warp_patch(ref, H, W, P, y0, x0, dy, dx, wp);
}

// max over (2r+1)-boxes inside a P×P patch of |box mean|, via a 2-D prefix sum (scratch (P+1)^2).
inline double box_max_abs_mean(const double* e, int P, int r, double* S) {
    if (r == 0) { double m = 0; for (int i = 0; i < P * P; ++i) m = std::max(m, std::fabs(e[i])); return m; }
    const int k = 2 * r + 1; if (k > P) return 0.0;
    const int Q = P + 1;
    for (int i = 0; i < Q; ++i) { S[i] = 0; S[size_t(i) * Q] = 0; }
    for (int y = 1; y < Q; ++y) { double row = 0; for (int x = 1; x < Q; ++x) { row += e[size_t(y - 1) * P + (x - 1)]; S[size_t(y) * Q + x] = S[size_t(y - 1) * Q + x] + row; } }
    double m = 0;
    for (int y = k; y < Q; ++y) for (int x = k; x < Q; ++x) {
        const double b = S[size_t(y) * Q + x] - S[size_t(y - k) * Q + x] - S[size_t(y) * Q + x - k] + S[size_t(y - k) * Q + x - k];
        m = std::max(m, std::fabs(b));
    }
    return m / (double(k) * k);
}

} // namespace detail

struct Scale { int r; double delta; };   // multi-scale schedule entry: box half-width and object contrast threshold

template <typename T>
class QuotientPruner {
public:
    QuotientPruner(int height, int width, Params p, double delta_max = 1.0, int iters = 2, std::vector<Scale> ms = {})
        : H_(height), W_(width), p_(p), dmax_(delta_max), iters_(iters), ms_(std::move(ms)),
          gh_(height / p.patch), gw_(width / p.patch),
          ref_(size_t(height) * width, T{}), view_(size_t(height) * width, 0.f),
          gx_(size_t(height) * width, 0.0), gy_(size_t(height) * width, 0.0), have_ref_(false) {}

    int grid_h() const { return gh_; }
    int grid_w() const { return gw_; }
    bool initialised() const { return have_ref_; }
    const std::vector<T>& reference() const { return ref_; }
    const std::vector<float>& view() const { return view_; }
    void reset() { have_ref_ = false; }
    // true (default): remove a brightness offset per patch before the test, c = midrange of the change (THEORY.md §3).
    // false: no offset, the patch is certified against the held copy itself, brightness included (THEORY.md §9).
    void set_offset(bool per_patch) { offset_ = per_patch; }

    // keep/spread(or score)/shift per patch, motion (dy, dx) per patch.
    void step(const T* frame, uint8_t* keep, float* spread, float* shift, float* mdy, float* mdx) {
        const int P = p_.patch; const size_t n = size_t(gh_) * gw_, N = size_t(H_) * W_;
        if (!have_ref_) {
            std::copy(frame, frame + N, ref_.begin());
            for (size_t i = 0; i < N; ++i) view_[i] = float(frame[i]);
            std::fill(keep, keep + n, uint8_t{1}); std::fill(spread, spread + n, 0.f); std::fill(shift, shift + n, 0.f);
            std::fill(mdy, mdy + n, 0.f); std::fill(mdx, mdx + n, 0.f);
            have_ref_ = true; detail::gradients(ref_, H_, W_, gx_, gy_); return;
        }
        std::vector<double> wp(size_t(P) * P), e(size_t(P) * P), S(size_t(P + 1) * (P + 1));
        for (int gy = 0; gy < gh_; ++gy) for (int gx = 0; gx < gw_; ++gx) {
            const size_t gi = size_t(gy) * gw_ + gx; const int y0 = gy * P, x0 = gx * P;
            double dy, dx;
            detail::fit_translation_patch(ref_, gx_, gy_, frame, H_, W_, P, y0, x0, dmax_, iters_, wp.data(), dy, dx);
            double dmaxv = -1e300, dminv = 1e300;
            for (int y = 0; y < P; ++y) for (int x = 0; x < P; ++x) {
                const double d = double(frame[size_t(y0 + y) * W_ + (x0 + x)]) - wp[size_t(y) * P + x];
                e[size_t(y) * P + x] = d; dmaxv = std::max(dmaxv, d); dminv = std::min(dminv, d);
            }
            const double c = offset_ ? 0.5 * (dmaxv + dminv) : 0.0;
            const double s = offset_ ? dmaxv - dminv : 2.0 * std::max(std::fabs(dmaxv), std::fabs(dminv));   // 2 max|d - c|
            double score;
            if (ms_.empty()) score = s / p_.delta;
            else {
                for (auto& v : e) v -= c;
                score = 0;
                for (const auto& sc : ms_) score = std::max(score, detail::box_max_abs_mean(e.data(), P, sc.r, S.data()) / (0.5 * sc.delta));
            }
            const bool drop = score < 1.0;
            spread[gi] = float(ms_.empty() ? s : score); shift[gi] = float(c); keep[gi] = drop ? 0 : 1;
            mdy[gi] = drop ? float(dy) : 0.f; mdx[gi] = drop ? float(dx) : 0.f;
            for (int y = 0; y < P; ++y) for (int x = 0; x < P; ++x) {
                const size_t i = size_t(y0 + y) * W_ + (x0 + x);
                view_[i] = drop ? float(wp[size_t(y) * P + x] + c) : float(frame[i]);
            }
        }
        commit(frame, keep);
    }

protected:
    void commit(const T* frame, const uint8_t* keep) {
        const int P = p_.patch;
        for (int gy = 0; gy < gh_; ++gy) for (int gx = 0; gx < gw_; ++gx) {
            if (!keep[size_t(gy) * gw_ + gx]) continue;
            for (int y = 0; y < P; ++y) { const size_t b = size_t(gy * P + y) * W_ + size_t(gx) * P; std::copy(frame + b, frame + b + P, ref_.begin() + b); }
        }
        detail::gradients(ref_, H_, W_, gx_, gy_);
    }
    int H_, W_; Params p_; double dmax_; int iters_; std::vector<Scale> ms_; int gh_, gw_;
    std::vector<T> ref_; std::vector<float> view_; std::vector<double> gx_, gy_; bool have_ref_; bool offset_ = true;
};

// Sequential rule (THEORY.md §6): quotient + multi-scale memoryless certificate, plus a
// self-normalised space-time scan of the planar-projected quotient residual.
template <typename T>
class SequentialPruner : public QuotientPruner<T> {
    using Base = QuotientPruner<T>;
public:
    SequentialPruner(int height, int width, Params p, std::vector<Scale> ms, double z,
                     std::vector<int> windows = {1, 2, 4, 8, 16}, std::vector<int> radii = {1, 2, 3},
                     double delta_max = 1.0, int iters = 2, double alpha = 0.1, int min_calib = 8,
                     double scale_floor = 0.5, double veto_frac = 0.1)
        : Base(height, width, p, delta_max, iters, std::move(ms)), z_(z), wins_(std::move(windows)), rads_(std::move(radii)),
          alpha_(alpha), min_calib_(min_calib), scale_floor_(scale_floor), veto_frac_(veto_frac) {
        wmax_ = 0; for (int w : wins_) wmax_ = std::max(wmax_, w);
        L_ = 2 * wmax_ + 1;
        const int P = p.patch; Hg_ = this->gh_ * P; Wg_ = this->gw_ * P;
        csum_.assign(size_t(L_) * Hg_ * Wg_, 0.f);
        const size_t n = size_t(this->gh_) * this->gw_, K = wins_.size() * rads_.size();
        loc_.assign(K * n, std::numeric_limits<double>::quiet_NaN()); scale_.assign(K * n, 0.0); nscale_.assign(K * n, 0);
        t0_.assign(n, 0); t_ = -1;
        // orthonormal basis of (1, x, y) over the patch (Gram-Schmidt)
        Q_.assign(3 * size_t(P) * P, 0.0);
        std::vector<double> c1(size_t(P) * P, 1.0), cx(size_t(P) * P), cy(size_t(P) * P);
        for (int y = 0; y < P; ++y) for (int x = 0; x < P; ++x) { cx[size_t(y) * P + x] = x; cy[size_t(y) * P + x] = y; }
        auto normalise = [&](std::vector<double>& v) { double s = 0; for (double a : v) s += a * a; s = std::sqrt(s); for (double& a : v) a /= s; };
        auto project_out = [&](std::vector<double>& v, const std::vector<double>& q) { double d = 0; for (size_t i = 0; i < v.size(); ++i) d += v[i] * q[i]; for (size_t i = 0; i < v.size(); ++i) v[i] -= d * q[i]; };
        normalise(c1); project_out(cx, c1); normalise(cx); project_out(cy, c1); project_out(cy, cx); normalise(cy);
        std::copy(c1.begin(), c1.end(), Q_.begin()); std::copy(cx.begin(), cx.end(), Q_.begin() + size_t(P) * P); std::copy(cy.begin(), cy.end(), Q_.begin() + 2 * size_t(P) * P);
    }

    void reset() { Base::reset(); std::fill(csum_.begin(), csum_.end(), 0.f); t_ = -1; std::fill(t0_.begin(), t0_.end(), 0);
                   std::fill(loc_.begin(), loc_.end(), std::numeric_limits<double>::quiet_NaN()); std::fill(scale_.begin(), scale_.end(), 0.0); std::fill(nscale_.begin(), nscale_.end(), 0); }

    // keep / total score (max of memoryless score and z'/z) / shift; seq_z = sequential z' per patch.
    void step(const T* frame, uint8_t* keep, float* score_out, float* shift, float* seq_z) {
        const int P = this->p_.patch; const size_t n = size_t(this->gh_) * this->gw_, N = size_t(this->H_) * this->W_;
        ++t_;
        float* C = csum_.data() + size_t(t_ % L_) * Hg_ * Wg_;
        if (!this->have_ref_) {
            std::copy(frame, frame + N, this->ref_.begin());
            for (size_t i = 0; i < N; ++i) this->view_[i] = float(frame[i]);
            std::fill(keep, keep + n, uint8_t{1}); std::fill(score_out, score_out + n, 0.f); std::fill(shift, shift + n, 0.f); std::fill(seq_z, seq_z + n, 0.f);
            std::fill(C, C + size_t(Hg_) * Wg_, 0.f); std::fill(t0_.begin(), t0_.end(), 0);
            this->have_ref_ = true; detail::gradients(this->ref_, this->H_, this->W_, this->gx_, this->gy_); return;
        }
        const float* Cprev = csum_.data() + size_t((t_ - 1 + L_) % L_) * Hg_ * Wg_;
        std::vector<double> wp(size_t(P) * P), e(size_t(P) * P), em(size_t(P) * P), S(size_t(P + 1) * (P + 1));
        std::vector<double> mscore(n), cmid(n), dyv(n), dxv(n);
        // ---- memoryless part + residual prefix sums
        for (int gy = 0; gy < this->gh_; ++gy) for (int gx = 0; gx < this->gw_; ++gx) {
            const size_t gi = size_t(gy) * this->gw_ + gx; const int y0 = gy * P, x0 = gx * P;
            double dy, dx;
            detail::fit_translation_patch(this->ref_, this->gx_, this->gy_, frame, this->H_, this->W_, P, y0, x0, this->dmax_, this->iters_, wp.data(), dy, dx);
            double dmaxv = -1e300, dminv = 1e300;
            for (int y = 0; y < P; ++y) for (int x = 0; x < P; ++x) {
                const double d = double(frame[size_t(y0 + y) * this->W_ + (x0 + x)]) - wp[size_t(y) * P + x];
                e[size_t(y) * P + x] = d; dmaxv = std::max(dmaxv, d); dminv = std::min(dminv, d);
            }
            const double c = this->offset_ ? 0.5 * (dmaxv + dminv) : 0.0;
            double sc = 0;
            if (this->ms_.empty()) sc = (this->offset_ ? dmaxv - dminv : 2.0 * std::max(std::fabs(dmaxv), std::fabs(dminv))) / this->p_.delta;
            else { for (int i = 0; i < P * P; ++i) em[i] = e[i] - c; for (const auto& s : this->ms_) sc = std::max(sc, detail::box_max_abs_mean(em.data(), P, s.r, S.data()) / (0.5 * s.delta)); }
            mscore[gi] = sc; cmid[gi] = c; dyv[gi] = dy; dxv[gi] = dx;
            // planar projection and prefix-sum update
            double q0 = 0, q1 = 0, q2 = 0;
            const double* Q0 = Q_.data(); const double* Q1 = Q0 + size_t(P) * P; const double* Q2 = Q1 + size_t(P) * P;
            for (int i = 0; i < P * P; ++i) { q0 += Q0[i] * e[i]; q1 += Q1[i] * e[i]; q2 += Q2[i] * e[i]; }
            for (int y = 0; y < P; ++y) for (int x = 0; x < P; ++x) {
                const int i = y * P + x; const double ep = e[i] - q0 * Q0[i] - q1 * Q1[i] - q2 * Q2[i];
                const size_t pix = size_t(y0 + y) * Wg_ + (x0 + x);
                C[pix] = Cprev[pix] + float(ep);
            }
            // view for a dropped patch (overwritten below if kept)
            for (int y = 0; y < P; ++y) for (int x = 0; x < P; ++x) this->view_[size_t(y0 + y) * this->W_ + (x0 + x)] = float(wp[size_t(y) * P + x] + c);
        }
        // ---- sequential part
        std::vector<double> zbest(n, 0.0);
        std::vector<double> Mstore(wins_.size() * rads_.size() * n, 0.0);
        std::vector<uint8_t> validw(wins_.size() * n, 0);
        std::vector<double> Dw(size_t(P) * P);
        for (size_t wi = 0; wi < wins_.size(); ++wi) {
            const int w = wins_[wi]; if (t_ - 2 * w < 0) continue;
            const float* Ct = C; const float* Cw = csum_.data() + size_t((t_ - w + L_ * 4) % L_) * Hg_ * Wg_; const float* C2w = csum_.data() + size_t((t_ - 2 * w + L_ * 4) % L_) * Hg_ * Wg_;
            for (int gy = 0; gy < this->gh_; ++gy) for (int gx = 0; gx < this->gw_; ++gx) {
                const size_t gi = size_t(gy) * this->gw_ + gx;
                const bool valid = (t_ - 2 * w) >= t0_[gi]; validw[wi * n + gi] = valid;
                if (!valid) continue;
                for (int y = 0; y < P; ++y) for (int x = 0; x < P; ++x) {
                    const size_t pix = size_t(gy * P + y) * Wg_ + (gx * P + x);
                    Dw[size_t(y) * P + x] = (double(Ct[pix]) - 2.0 * double(Cw[pix]) + double(C2w[pix])) / w;
                }
                for (size_t ri = 0; ri < rads_.size(); ++ri) {
                    const double M = detail::box_max_abs_mean(Dw.data(), P, rads_[ri], S.data());
                    const size_t k = (wi * rads_.size() + ri) * n + gi; Mstore[k] = M;
                    if (nscale_[k] >= min_calib_) zbest[gi] = std::max(zbest[gi], (M - loc_[k]) / std::max(scale_[k], scale_floor_));
                }
            }
        }
        // ---- global-coincidence veto: a scene-wide firing is not objects (illumination, keyframe); train on it instead
        size_t nfire = 0; for (size_t gi = 0; gi < n; ++gi) if (zbest[gi] >= z_) ++nfire;
        const bool vetoed = double(nfire) >= veto_frac_ * double(n);
        vetoed_ = vetoed;
        // ---- decisions, learning, commit
        for (size_t gi = 0; gi < n; ++gi) {
            const bool seq = !vetoed && zbest[gi] >= z_, mem = mscore[gi] >= 1.0, k = seq || mem;
            keep[gi] = k ? 1 : 0; score_out[gi] = float(std::max(mscore[gi], zbest[gi] / z_)); shift[gi] = float(cmid[gi]); seq_z[gi] = float(zbest[gi]);
            const bool quiet = !seq && !mem;
            for (size_t wi = 0; wi < wins_.size(); ++wi) {
                if (!validw[wi * n + gi] || !quiet) continue;
                for (size_t ri = 0; ri < rads_.size(); ++ri) {
                    const size_t kk = (wi * rads_.size() + ri) * n + gi; const double M = Mstore[kk];
                    if (std::isnan(loc_[kk])) { loc_[kk] = M; scale_[kk] = std::max(0.25 * M, 1e-3); }
                    else { const double dev = std::fabs(M - loc_[kk]); loc_[kk] = (1 - alpha_) * loc_[kk] + alpha_ * M; scale_[kk] = std::max((1 - alpha_) * scale_[kk] + alpha_ * dev, 1e-3); }
                    nscale_[kk] += 1;
                }
            }
            if (k) {
                t0_[gi] = t_;
                const int gy = int(gi / this->gw_), gx = int(gi % this->gw_);
                for (int y = 0; y < P; ++y) for (int x = 0; x < P; ++x) { const size_t i = size_t(gy * P + y) * this->W_ + (gx * P + x); this->view_[i] = float(frame[i]); }
            }
        }
        this->commit(frame, keep);
    }

private:
    double z_; std::vector<int> wins_, rads_; double alpha_; int min_calib_; double scale_floor_, veto_frac_; bool vetoed_ = false; int wmax_, L_, Hg_, Wg_;
    std::vector<float> csum_; std::vector<double> loc_, scale_; std::vector<long> nscale_; std::vector<long> t0_; long t_;
    std::vector<double> Q_;
};

} // namespace certskip

// ---------------------------------------------------------------------------
// Binary mask utilities for the segmentation layer (python/certskip/segment.py):
// connected components (8-connectivity), hole filling, small-component removal.
// ---------------------------------------------------------------------------
namespace certskip {

// Labels 8-connected components of mask (nonzero = foreground). labels: 0 = background,
// 1..n = component id. Returns n. areas (optional, size >= n+1) receives pixel counts.
inline int label_components(const uint8_t* mask, int H, int W, int32_t* labels, std::vector<int32_t>* areas = nullptr) {
    const size_t N = size_t(H) * W;
    std::fill(labels, labels + N, 0);
    std::vector<int32_t> stack; stack.reserve(1024);
    int n = 0;
    if (areas) { areas->clear(); areas->push_back(0); }
    for (size_t s = 0; s < N; ++s) {
        if (!mask[s] || labels[s]) continue;
        ++n; labels[s] = n; stack.push_back(int32_t(s)); int32_t area = 0;
        while (!stack.empty()) {
            const int32_t i = stack.back(); stack.pop_back(); ++area;
            const int y = i / W, x = i % W;
            for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
                if (!dy && !dx) continue;
                const int ny = y + dy, nx = x + dx;
                if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
                const size_t j = size_t(ny) * W + nx;
                if (mask[j] && !labels[j]) { labels[j] = n; stack.push_back(int32_t(j)); }
            }
        }
        if (areas) areas->push_back(area);
    }
    return n;
}

// Fills holes: background pixels not 4-connected to the image border become foreground.
inline void fill_holes(uint8_t* mask, int H, int W) {
    const size_t N = size_t(H) * W;
    std::vector<uint8_t> reach(N, 0);
    std::vector<int32_t> stack;
    auto push = [&](int y, int x) { const size_t i = size_t(y) * W + x; if (!mask[i] && !reach[i]) { reach[i] = 1; stack.push_back(int32_t(i)); } };
    for (int x = 0; x < W; ++x) { push(0, x); push(H - 1, x); }
    for (int y = 0; y < H; ++y) { push(y, 0); push(y, W - 1); }
    while (!stack.empty()) {
        const int32_t i = stack.back(); stack.pop_back(); const int y = i / W, x = i % W;
        if (y > 0) push(y - 1, x); if (y < H - 1) push(y + 1, x); if (x > 0) push(y, x - 1); if (x < W - 1) push(y, x + 1);
    }
    for (size_t i = 0; i < N; ++i) if (!mask[i] && !reach[i]) mask[i] = 1;
}

// Removes 8-connected components with fewer than min_area pixels.
inline void remove_small(uint8_t* mask, int H, int W, int min_area) {
    const size_t N = size_t(H) * W;
    std::vector<int32_t> labels(N); std::vector<int32_t> areas;
    label_components(mask, H, W, labels.data(), &areas);
    for (size_t i = 0; i < N; ++i) if (mask[i] && areas[labels[i]] < min_area) mask[i] = 0;
}

} // namespace certskip
