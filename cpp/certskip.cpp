// certskip.cpp — C ABI over certskip.hpp for ctypes / FFI consumers.
#include "certskip.hpp"
#include <new>

using namespace certskip;

extern "C" {

// Opaque handle. dtype: 0 = uint8, 1 = uint16, 2 = float32.
struct cs_handle {
    int dtype;
    void* impl;
};

cs_handle* cs_create(int height, int width, int patch, double delta, double max_shift, int dtype) {
    if (height <= 0 || width <= 0 || patch <= 0 || patch > height || patch > width) return nullptr;
    Params p; p.patch = patch; p.delta = delta; p.max_shift = max_shift;
    cs_handle* h = new (std::nothrow) cs_handle{dtype, nullptr};
    if (!h) return nullptr;
    switch (dtype) {
        case 0: h->impl = new (std::nothrow) Pruner<uint8_t>(height, width, p); break;
        case 1: h->impl = new (std::nothrow) Pruner<uint16_t>(height, width, p); break;
        case 2: h->impl = new (std::nothrow) Pruner<float>(height, width, p); break;
        default: delete h; return nullptr;
    }
    if (!h->impl) { delete h; return nullptr; }
    return h;
}

void cs_destroy(cs_handle* h) {
    if (!h) return;
    switch (h->dtype) {
        case 0: delete static_cast<Pruner<uint8_t>*>(h->impl); break;
        case 1: delete static_cast<Pruner<uint16_t>*>(h->impl); break;
        case 2: delete static_cast<Pruner<float>*>(h->impl); break;
    }
    delete h;
}

int cs_grid_h(const cs_handle* h) {
    switch (h->dtype) {
        case 0: return static_cast<const Pruner<uint8_t>*>(h->impl)->grid_h();
        case 1: return static_cast<const Pruner<uint16_t>*>(h->impl)->grid_h();
        default: return static_cast<const Pruner<float>*>(h->impl)->grid_h();
    }
}
int cs_grid_w(const cs_handle* h) {
    switch (h->dtype) {
        case 0: return static_cast<const Pruner<uint8_t>*>(h->impl)->grid_w();
        case 1: return static_cast<const Pruner<uint16_t>*>(h->impl)->grid_w();
        default: return static_cast<const Pruner<float>*>(h->impl)->grid_w();
    }
}

void cs_reset(cs_handle* h) {
    switch (h->dtype) {
        case 0: static_cast<Pruner<uint8_t>*>(h->impl)->reset(); break;
        case 1: static_cast<Pruner<uint16_t>*>(h->impl)->reset(); break;
        default: static_cast<Pruner<float>*>(h->impl)->reset(); break;
    }
}

// Process one frame. `frame` must point to H*W contiguous elements of the
// handle's dtype. Outputs are gh*gw arrays. Returns number of kept patches.
long cs_step(cs_handle* h, const void* frame, uint8_t* keep, float* spread, float* shift) {
    long gh = cs_grid_h(h), gw = cs_grid_w(h);
    switch (h->dtype) {
        case 0: static_cast<Pruner<uint8_t>*>(h->impl)->step(static_cast<const uint8_t*>(frame), keep, spread, shift); break;
        case 1: static_cast<Pruner<uint16_t>*>(h->impl)->step(static_cast<const uint16_t*>(frame), keep, spread, shift); break;
        default: static_cast<Pruner<float>*>(h->impl)->step(static_cast<const float*>(frame), keep, spread, shift); break;
    }
    long kept = 0;
    for (long i = 0; i < gh * gw; ++i) kept += keep[i];
    return kept;
}

// Process T frames stored contiguously (T*H*W). Outputs are T*gh*gw.
long cs_run(cs_handle* h, const void* frames, long nframes, long frame_elems,
            uint8_t* keep, float* spread, float* shift) {
    long gh = cs_grid_h(h), gw = cs_grid_w(h), n = gh * gw, kept = 0;
    size_t esz = h->dtype == 0 ? 1 : (h->dtype == 1 ? 2 : 4);
    const char* base = static_cast<const char*>(frames);
    for (long t = 0; t < nframes; ++t)
        kept += cs_step(h, base + static_cast<size_t>(t) * frame_elems * esz,
                        keep + t * n, spread + t * n, shift + t * n);
    return kept;
}

// Copy the current reference image (what the model sees) into out (H*W).
void cs_reference(const cs_handle* h, void* out, long elems) {
    switch (h->dtype) {
        case 0: { auto& r = static_cast<const Pruner<uint8_t>*>(h->impl)->reference();
                  std::copy(r.begin(), r.begin() + std::min<long>(elems, r.size()), static_cast<uint8_t*>(out)); break; }
        case 1: { auto& r = static_cast<const Pruner<uint16_t>*>(h->impl)->reference();
                  std::copy(r.begin(), r.begin() + std::min<long>(elems, r.size()), static_cast<uint16_t*>(out)); break; }
        default:{ auto& r = static_cast<const Pruner<float>*>(h->impl)->reference();
                  std::copy(r.begin(), r.begin() + std::min<long>(elems, r.size()), static_cast<float*>(out)); break; }
    }
}

// Baseline helper (uint8 only): mean |cur - prev| per patch.
void cs_consecutive_mean_u8(const uint8_t* prev, const uint8_t* cur, int H, int W, int P, float* out) {
    consecutive_mean<uint8_t>(prev, cur, H, W, P, out);
}

const char* cs_version(void) { return "certskip 0.1.0"; }

} // extern "C"

// ------------------------------------------------------------ quotient rule
extern "C" {

struct csq_handle { QuotientPruner<uint8_t>* impl; };

csq_handle* csq_create(int height, int width, int patch, double delta, double delta_max, int iters) {
    if (height <= 0 || width <= 0 || patch <= 0 || patch > height || patch > width) return nullptr;
    Params p; p.patch = patch; p.delta = delta;
    auto* impl = new (std::nothrow) QuotientPruner<uint8_t>(height, width, p, delta_max, iters);
    if (!impl) return nullptr;
    return new (std::nothrow) csq_handle{impl};
}
void csq_destroy(csq_handle* h) { if (h) { delete h->impl; delete h; } }
int csq_grid_h(const csq_handle* h) { return h->impl->grid_h(); }
int csq_grid_w(const csq_handle* h) { return h->impl->grid_w(); }
void csq_reset(csq_handle* h) { h->impl->reset(); }
long csq_step(csq_handle* h, const uint8_t* frame, uint8_t* keep, float* spread, float* shift, float* mdy, float* mdx) {
    h->impl->step(frame, keep, spread, shift, mdy, mdx);
    long n = (long)h->impl->grid_h() * h->impl->grid_w(), kept = 0;
    for (long i = 0; i < n; ++i) kept += keep[i];
    return kept;
}
void csq_reference(const csq_handle* h, uint8_t* out, long elems) { auto& r = h->impl->reference(); std::copy(r.begin(), r.begin() + std::min<long>(elems, r.size()), out); }
void csq_view(const csq_handle* h, float* out, long elems) { auto& v = h->impl->view(); std::copy(v.begin(), v.begin() + std::min<long>(elems, v.size()), out); }

// ------------------------------------------------------------ sequential rule
struct css_handle { SequentialPruner<uint8_t>* impl; };

css_handle* css_create(int height, int width, int patch, double delta_max, int iters, double z,
                       int n_ms, const int* ms_r, const double* ms_delta,
                       int n_w, const int* windows, int n_r, const int* radii, double alpha, int min_calib,
                       double scale_floor, double veto_frac) {
    if (height <= 0 || width <= 0 || patch <= 0 || patch > height || patch > width) return nullptr;
    Params p; p.patch = patch; p.delta = 32.0;
    std::vector<Scale> ms; for (int i = 0; i < n_ms; ++i) ms.push_back({ms_r[i], ms_delta[i]});
    std::vector<int> W(windows, windows + n_w), R(radii, radii + n_r);
    auto* impl = new (std::nothrow) SequentialPruner<uint8_t>(height, width, p, ms, z, W, R, delta_max, iters, alpha, min_calib, scale_floor, veto_frac);
    if (!impl) return nullptr;
    return new (std::nothrow) css_handle{impl};
}
void css_destroy(css_handle* h) { if (h) { delete h->impl; delete h; } }
int css_grid_h(const css_handle* h) { return h->impl->grid_h(); }
int css_grid_w(const css_handle* h) { return h->impl->grid_w(); }
void css_reset(css_handle* h) { h->impl->reset(); }
long css_step(css_handle* h, const uint8_t* frame, uint8_t* keep, float* score, float* shift, float* seq_z) {
    h->impl->step(frame, keep, score, shift, seq_z);
    long n = (long)h->impl->grid_h() * h->impl->grid_w(), kept = 0;
    for (long i = 0; i < n; ++i) kept += keep[i];
    return kept;
}
void css_reference(const css_handle* h, uint8_t* out, long elems) { auto& r = h->impl->reference(); std::copy(r.begin(), r.begin() + std::min<long>(elems, r.size()), out); }
void css_view(const css_handle* h, float* out, long elems) { auto& v = h->impl->view(); std::copy(v.begin(), v.begin() + std::min<long>(elems, v.size()), out); }

} // extern "C"

// ------------------------------------------------------------ mask utilities
extern "C" {
int cs_label_components(const uint8_t* mask, int H, int W, int32_t* labels) { return label_components(mask, H, W, labels); }
void cs_fill_holes(uint8_t* mask, int H, int W) { fill_holes(mask, H, W); }
void cs_remove_small(uint8_t* mask, int H, int W, int min_area) { remove_small(mask, H, W, min_area); }
}

// ------------------------------------------------------------ real-time segmenter
#include "rtseg.hpp"
extern "C" {
struct csg_handle { certskip::rt::RtSegmenter* impl; };
csg_handle* csg_create(int height, int width, int patch, const double* p /* z0,gamma,lam,tau_edge,mu,alpha_bg,alpha_fg,sigma_floor,kkt_tol,act_delta0,delta_max */,
                       int steps, int min_area, int halo, int fit_iters, int shadows, int certify) {
    if (height <= 0 || width <= 0 || patch <= 0 || patch > height || patch > width) return nullptr;
    certskip::rt::RtParams q; q.patch = patch; q.z0 = p[0]; q.gamma = p[1]; q.lam = p[2]; q.tau_edge = p[3]; q.mu = p[4];
    q.alpha_bg = p[5]; q.alpha_fg = p[6]; q.sigma_floor = p[7]; q.kkt_tol = p[8]; q.act_delta0 = p[9]; q.delta_max = p[10];
    q.steps = steps; q.min_area = min_area; q.halo = halo; q.fit_iters = fit_iters; q.shadows = shadows != 0; q.certify = certify != 0;
    q.close_r = int(p[11]); q.min_side = int(p[12]); q.max_aspect = p[13];
    q.rho = p[14]; q.tau_img = p[15]; q.snap_r = int(p[16]); q.snap_eps = p[17]; q.act_z = p[18]; q.threads = int(p[19]); q.ghost_contrast = p[20]; q.alpha_ghost = p[21];
    auto* impl = new (std::nothrow) certskip::rt::RtSegmenter(height, width, q);
    return impl ? new (std::nothrow) csg_handle{impl} : nullptr;
}
void csg_destroy(csg_handle* h) { if (h) { delete h->impl; delete h; } }
void csg_stage_ms(const csg_handle* h, double* out) { const auto& s = h->impl->stats(); for (int i = 0; i < 12; ++i) out[i] = s.stage_ms[i]; }
void csg_active_why(const csg_handle* h, int* out) { const auto& s = h->impl->stats(); out[0] = s.n_score; out[1] = s.n_near; out[2] = s.n_kkt; out[3] = s.n_patches; }
void csg_init_background(csg_handle* h, const uint8_t* frames, int n) { h->impl->init_background(frames, n); }
void csg_step(csg_handle* h, const uint8_t* frame, uint8_t* mask_out, int32_t* labels_out, double* stats_out) {
    h->impl->step(frame);
    const auto& m = h->impl->mask(); const auto& l = h->impl->labels(); std::copy(m.begin(), m.end(), mask_out); std::copy(l.begin(), l.end(), labels_out);
    const auto& s = h->impl->stats(); stats_out[0] = s.active_frac; stats_out[1] = s.kkt_frozen_max; stats_out[2] = s.kkt_active_max; stats_out[3] = s.dy; stats_out[4] = s.dx; stats_out[5] = s.n_objects; stats_out[6] = s.pred_shift;
}
}
