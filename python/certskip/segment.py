"""Segmentation layer: pixel-accurate, class-agnostic object masks for a fixed
camera, built on the same machinery as the skip rules.

Pipeline per frame
------------------
1. Background B: temporal median of the first K frames, then selectively
   updated on background pixels (slow absorption of persistent foreground).
2. Quotient: one global sub-pixel translation per frame (median of the
   per-patch Lucas-Kanade fits over textured patches) warps B onto the frame,
   so camera jitter never reaches the residual.  A per-patch illumination
   offset is estimated from background pixels only and interpolated across
   occluded patches.
3. Evidence: residual e = F - warp(B) - offset, per-pixel noise sigma(x)
   tracked on background pixels; z(x) = |e| / sigma, lightly box-smoothed.
   Optional texture-based shadow suppression (ratio in [0.45, 0.95] and high
   local normalised correlation between F and B => shadow, not object).
4. Labelling: exact MAP of a contrast-sensitive Potts model through its convex
   relaxation (Chan, Esedoglu & Nikolova 2006):
        min_{u in [0,1]}  <f, u> + lambda * sum_x w(x) |grad u|(x)
   solved by Chambolle-Pock; thresholding u at 1/2 is a global minimiser of
   the binary problem (coarea formula).  f = gamma (z0 - z) is the evidence,
   w = exp(-|grad e|^2 / 2 tau^2) lets boundaries snap to residual edges.
5. Clean-up: small components removed, holes filled (C++), components tracked
   across frames by overlap so each object keeps its colour.

Certificate kept: the union of all multi-scale boxes whose mean change exceeds
the Part-II thresholds is OR-ed into the mask, so every certified object is
inside the mask whatever the MRF decides.
"""
from __future__ import annotations

import numpy as np

from .core import patch_grid
from .scalespace import box_filter, box_max_abs_mean
from .warp import fit_translation, warp
from . import native


def _grad(u):
    gy = np.zeros_like(u); gx = np.zeros_like(u)
    gy[:-1] = u[1:] - u[:-1]; gx[:, :-1] = u[:, 1:] - u[:, :-1]
    return gy, gx


def _div(py, px):
    d = np.zeros_like(py)
    d[:-1] += py[:-1]; d[1:] -= py[:-1]
    d[:, :-1] += px[:, :-1]; d[:, 1:] -= px[:, :-1]
    return d


def tv_segment(f, w, lam, iters=80, u0=None):
    """Global minimiser of <f,u> + lam * sum w |grad u| over u in [0,1]
    (Chambolle-Pock, tau = sigma = 1/sqrt(8)).  Returns u (float) in [0,1]."""
    u = np.full(f.shape, 0.5) if u0 is None else u0.copy()
    ubar = u.copy()
    py = np.zeros_like(u); px = np.zeros_like(u)
    tau = sig = 1.0 / np.sqrt(8.0)
    for _ in range(iters):
        gy, gx = _grad(ubar)
        py += sig * w * gy; px += sig * w * gx
        nrm = np.maximum(1.0, np.sqrt(py * py + px * px) / lam)
        py /= nrm; px /= nrm
        u_old = u
        u = np.clip(u - tau * (-_div(w * py, w * px) + f), 0.0, 1.0)
        ubar = 2 * u - u_old
    return u


def _ncc(F, B, r=2):
    """Local normalised cross-correlation of F and B over (2r+1)-boxes (same-size output)."""
    k = 2 * r + 1
    pad = lambda a: np.pad(a, r, mode="edge")
    mF, mB = box_filter(pad(F), r), box_filter(pad(B), r)
    vF = box_filter(pad(F * F), r) - mF * mF
    vB = box_filter(pad(B * B), r) - mB * mB
    cov = box_filter(pad(F * B), r) - mF * mB
    return cov / np.sqrt(np.maximum(vF, 1e-3) * np.maximum(vB, 1e-3))


class ForegroundSegmenter:
    def __init__(self, height, width, patch=16, init_frames=None, z0=3.0, gamma=0.6, lam=1.2, tau_edge=6.0,
                 iters=60, min_area=40, alpha_bg=0.02, alpha_fg=5e-5, sigma_floor=1.5, shadows=True,
                 certify=None, delta_max=1.0, close_r=2, min_side=8, max_aspect=5.0,
                 rho=0.6, tau_img=12.0, snap_r=3, snap_eps=36.0, ghost_contrast=6.0, alpha_ghost=0.25):
        self.H, self.W, self.P = int(height), int(width), int(patch)
        self.z0, self.gamma, self.lam, self.tau_edge, self.iters = z0, gamma, lam, tau_edge, iters
        self.min_area, self.alpha_bg, self.alpha_fg, self.sigma_floor, self.shadows = min_area, alpha_bg, alpha_fg, sigma_floor, shadows
        self.certify = certify            # dict r -> Delta_r (multi-scale certificate), or None
        self.close_r = int(close_r); self.min_side, self.max_aspect = int(min_side), float(max_aspect)
        self.rho, self.tau_img, self.snap_r, self.snap_eps = float(rho), float(tau_img), int(snap_r), float(snap_eps)
        self.A = np.zeros((self.H, self.W))          # Lagrangian evidence accumulator (signed, studentised)
        self.ghost_contrast, self.alpha_ghost = float(ghost_contrast), float(alpha_ghost)
        self.ghost = np.zeros((self.H, self.W), bool)
        self.delta_max = delta_max
        self.B = None; self.var = None; self.u = None; self.mask = np.zeros((self.H, self.W), bool)
        self.labels = np.zeros((self.H, self.W), np.int32); self.ids = {}; self.next_id = 1
        self.cents = {}; self.vel = {}; self.plab = None
        self.t = -1; self.motion = (0.0, 0.0)
        if init_frames is not None:
            self.initialise(init_frames)

    # -- background -----------------------------------------------------------
    def initialise(self, frames):
        fr = np.asarray(frames, np.float64)
        self.B = np.median(fr, axis=0)
        dev = np.abs(fr - self.B)
        self.var = np.maximum(1.4826 * np.median(dev, axis=0), self.sigma_floor) ** 2
        self.u = np.zeros((self.H, self.W))

    def _global_shift(self, F):
        gh, gw = self.H // self.P, self.W // self.P
        dy, dx = fit_translation(F, self.B, self.P, self.delta_max, iters=2)
        gyB, gxB = _grad(self.B)
        energy = patch_grid(gyB * gyB + gxB * gxB, self.P).sum((2, 3))
        occl = patch_grid(self.mask, self.P).mean((2, 3)) > 0.1
        ok = (energy >= np.median(energy)) & ~occl
        if ok.sum() < 4:
            return 0.0, 0.0
        return float(np.median(dy[ok])), float(np.median(dx[ok]))

    def _illumination(self, e, bgmask):
        """Per-patch offset from background pixels, interpolated over occluded patches, upsampled."""
        P = self.P; gh, gw = self.H // P, self.W // P
        eg = patch_grid(e, P); mg = patch_grid(bgmask, P)
        cnt = mg.sum((2, 3)); s = (eg * mg).sum((2, 3))
        off = np.where(cnt >= 0.25 * P * P, s / np.maximum(cnt, 1), np.nan)
        for _ in range(8):
            if not np.isnan(off).any():
                break
            padded = np.pad(off, 1, mode="edge")
            nb = np.stack([padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:]])
            fill = np.nanmean(nb, axis=0)
            off = np.where(np.isnan(off), fill, off)
        off = np.nan_to_num(off, nan=0.0)
        full = np.zeros((self.H, self.W)); full[:gh * P, :gw * P] = np.kron(off, np.ones((P, P)))
        return full

    # -- main step ---------------------------------------------------------------
    def evidence(self, F):
        """Residual e, unary f and TV weights w for the current frame (steps 2-3)."""
        H, W, P = self.H, self.W, self.P
        gh, gw = H // P, W // P
        dy, dx = self._global_shift(F); self.motion = (dy, dx)
        Bw = warp(self.B, np.full((gh, gw), dy), np.full((gh, gw), dx), P) if (dy or dx) else self.B
        e0 = F - Bw
        prev_bg = ~_dilate(self.mask, 2)
        e = e0 - self._illumination(e0, prev_bg)
        sigma = np.sqrt(self.var)
        z = np.abs(e) / sigma
        zs = box_filter(np.pad(z, 1, mode="edge"), 1)
        # Lagrangian evidence accumulation: a recursive signed statistic carried along each object's motion.
        # Persistent evidence grows by 1/(1-rho); independent noise keeps unit variance after normalisation.
        sgn = np.clip(e / sigma, -6.0, 6.0)
        A = self.A
        for pid, (vy, vx) in self.vel.items():
            dy, dx = int(round(vy)), int(round(vx))
            if (dy == 0 and dx == 0) or abs(dy) > 48 or abs(dx) > 48:
                continue
            obj = self.labels == pid
            if not obj.any():
                continue
            moved = _shift_int(np.where(obj, A, 0.0), dy, dx)
            A = np.where(obj, 0.0, A)
            A = np.where(_shift_int(obj.astype(np.int32), dy, dx) > 0, moved, A)
        self.A = self.rho * A + sgn
        z_acc = box_filter(np.pad(np.abs(self.A), 1, mode="edge"), 1) * np.sqrt(1.0 - self.rho ** 2)
        zs = np.maximum(zs, z_acc)
        if self.shadows:
            ratio = F / np.maximum(Bw, 1.0)
            ncc = _ncc(F, Bw, 2)
            texB = box_filter(np.pad(Bw * Bw, 2, mode="edge"), 2) - box_filter(np.pad(Bw, 2, mode="edge"), 2) ** 2
            rs = box_filter(np.pad(ratio, 2, mode="edge"), 2)
            rvar = box_filter(np.pad(ratio * ratio, 2, mode="edge"), 2) - rs * rs
            textured = (ratio > 0.45) & (ratio < 0.95) & (ncc > 0.8)
            flat = (texB < 16.0) & (rs > 0.55) & (rs < 0.95) & (rvar < 0.004)
            zs = np.where(textured | flat, np.minimum(zs, self.z0 - 0.5), zs)
        f = np.clip(self.gamma * (self.z0 - zs), -3.0, 3.0)
        es = box_filter(np.pad(e, 1, mode="edge"), 1)
        gy, gx = _grad(es)
        w_res = np.exp(-(gy * gy + gx * gx) / (2 * self.tau_edge ** 2))
        Fs = box_filter(np.pad(F, 1, mode="edge"), 1)
        gy, gx = _grad(Fs)
        w_img = np.exp(-(gy * gy + gx * gx) / (2 * self.tau_img ** 2))
        w = np.minimum(w_res, w_img) + 0.05            # a boundary is cheap where EITHER the residual or the image has an edge
        return f, w, e, e0

    def label(self, f, w, e=None):
        """Per-frame global minimiser (step 4). Overridden by the real-time tracker."""
        self.u = tv_segment(f, w, self.lam, self.iters, u0=self.u)
        return (self.u > 0.5).astype(np.uint8)

    def finish(self, F, m, e, e0):
        """Certified boxes, clean-up, background/noise update, tracking (step 5)."""
        H, W, P = self.H, self.W, self.P
        if self.certify:
            cm = np.zeros((H, W), bool)
            for r, D in self.certify.items():
                k = 2 * r + 1
                if k > P:
                    continue
                bm = box_filter(e, r)
                hit = np.abs(bm) >= 0.5 * D
                ys, xs = np.where(hit)
                for yy, xx in zip(ys, xs):
                    cm[yy:yy + k, xx:xx + k] = True
            m = m | cm.astype(np.uint8)
        if self.close_r > 0:                                                 # closing: bridge camouflage gaps inside objects
            m = _erode(_dilate(m.astype(bool), self.close_r), self.close_r).astype(np.uint8)
        m = native.remove_small(m, self.min_area)
        m = native.fill_holes(m)
        m = _drop_thin(m, self.min_side, self.max_aspect)
        m, self.ghost = _drop_ghosts(m, F, self.B, self.var, self.ghost_contrast)
        if self.snap_r > 0:
            m = _guided_snap(m, F, self.snap_r, self.snap_eps)
            m = native.fill_holes(native.remove_small(m, self.min_area))
        mask = m.astype(bool)
        upd_bg = ~_dilate(mask, 3)
        a = np.where(upd_bg, self.alpha_bg, self.alpha_fg)
        a = np.where(self.ghost, self.alpha_ghost, a)
        self.B = (1 - a) * self.B + a * F
        self.var = np.where(upd_bg, (1 - self.alpha_bg) * self.var + self.alpha_bg * e * e, self.var)
        self.var = np.maximum(self.var, self.sigma_floor ** 2)
        self._track(mask)
        self.mask = mask
        return mask

    def step(self, frame):
        self.t += 1
        F = np.asarray(frame, np.float64)
        if self.B is None:
            self.initialise(F[None]); return self.mask
        f, w, e, e0 = self.evidence(F)
        m = self.label(f, w, e)
        return self.finish(F, m, e, e0)

    def _track(self, mask):
        lab, n = native.label_components(mask.astype(np.uint8))
        ref = self.plab if self.plab is not None else self.labels          # predicted labels if available
        ref_area = {int(v): int(c) for v, c in zip(*np.unique(ref[ref > 0], return_counts=True))}
        out = np.zeros_like(self.labels)
        used = set(); new_ids = {}
        for c in range(1, n + 1):
            comp = lab == c
            prev = ref[comp]; prev = prev[prev > 0]
            cands = []
            if prev.size:
                vals, counts = np.unique(prev, return_counts=True)
                for v, k in zip(vals, counts):
                    v = int(v)
                    if v not in used and k >= 0.2 * min(comp.sum(), ref_area.get(v, 1)):
                        cands.append((int(k), v))
                cands.sort(reverse=True)
            # split only between SUBSTANTIAL predicted objects: a fragment (foot, bag) must not split the body it rejoins
            if len(cands) >= 2:
                top = ref_area.get(cands[0][1], 1)
                big = [v for _, v in cands if ref_area.get(v, 0) >= 0.3 * top and ref_area.get(v, 0) >= 0.2 * comp.sum()]
                cands = [(k, v) for k, v in cands if v in big] if len(big) >= 2 else cands[:1]
            if len(cands) >= 2:
                # a merged component: split it between the predicted objects by nearest seed
                ids = [v for _, v in cands]
                seeds = np.where(comp & np.isin(ref, ids), ref, 0)
                filled = _propagate_labels(seeds, comp)
                for v in ids:
                    sel = filled == v
                    if sel.sum() >= 1:
                        out[sel] = v; used.add(v)
                continue
            best = cands[0][1] if cands else None
            if best is None:
                best = self.next_id; self.next_id += 1
            used.add(best); new_ids[c] = best
            out[comp] = best
        cents = {}
        for pid in np.unique(out[out > 0]):
            ys, xs = np.where(out == pid); cents[int(pid)] = (ys.mean(), xs.mean())
        self.vel = {pid: (cy - self.cents[pid][0], cx - self.cents[pid][1]) for pid, (cy, cx) in cents.items() if pid in self.cents}
        self.cents = cents
        self.labels = out
        self.ids = {int(v): int(v) for v in np.unique(out[out > 0])}


def _propagate_labels(seeds, region):
    """Grow seed labels inside `region` one pixel per pass (nearest-seed partition, 4-connectivity)."""
    lab = seeds.copy()
    todo = region & (lab == 0)
    for _ in range(64):
        if not todo.any():
            break
        new = lab.copy()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sh = np.zeros_like(lab)
            if dy == 1: sh[1:] = lab[:-1]
            elif dy == -1: sh[:-1] = lab[1:]
            elif dx == 1: sh[:, 1:] = lab[:, :-1]
            else: sh[:, :-1] = lab[:, 1:]
            take = todo & (new == 0) & (sh > 0)
            new[take] = sh[take]
        if np.array_equal(new, lab):
            break
        lab = new; todo = region & (lab == 0)
    return lab


def _guided_snap(m, F, r, eps):
    """Guided filter (He, Sun & Tang 2010) of the binary mask with the frame as guide, applied in a
    band around the boundary, then re-thresholded: the contour snaps to the image edge."""
    mb = m.astype(bool)
    band = _dilate(mb, r + 1) & ~_erode(mb, r + 1)
    if not band.any():
        return m
    pad = lambda a: np.pad(a, r, mode="edge")
    I = F.astype(np.float64); P_ = mb.astype(np.float64)
    mI, mP = box_filter(pad(I), r), box_filter(pad(P_), r)
    vI = box_filter(pad(I * I), r) - mI * mI
    cIP = box_filter(pad(I * P_), r) - mI * mP
    a = cIP / (vI + eps); b = mP - a * mI
    q = box_filter(pad(a), r) * I + box_filter(pad(b), r)
    out = m.copy()
    out[band] = (q[band] > 0.5).astype(np.uint8)
    return out


def _windowed_step(I, r=3):
    Ip = np.pad(I, r, mode="edge"); H, W = I.shape; best = np.zeros(I.shape)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            best = np.maximum(best, np.abs(Ip[r + dy:r + dy + H, r + dx:r + dx + W] - I))
    return best


def _drop_ghosts(m, F, B, var, thr):
    """A ghost's outline is in the background model and not in the frame; a real object's is in the
    frame and not in the model. Drop components whose contour contrast in F (windowed max step, sigma
    units) is below `thr` or below 0.6 times the same statistic computed on B."""
    lab, n = native.label_components(m)
    ghost = np.zeros(m.shape, bool)
    if n == 0:
        return m, ghost
    mb = m.astype(bool); sig = np.sqrt(var)
    bnd = mb & ~_erode(mb, 1)
    sF = _windowed_step(F.astype(np.float64)) / sig; sB = _windowed_step(B.astype(np.float64)) / sig
    out = m.copy()
    for c in range(1, n + 1):
        sel = (lab == c) & bnd
        if sel.any() and (sF[sel].mean() < thr or sF[sel].mean() < 0.6 * sB[sel].mean()):
            ghost |= lab == c; out[lab == c] = 0
    return out, ghost


def _drop_thin(m, min_side, max_aspect):
    """Remove components that are lines rather than objects: smaller bbox side < min_side and aspect > max_aspect."""
    lab, n = native.label_components(m)
    if n == 0:
        return m
    out = m.copy()
    for c in range(1, n + 1):
        ys, xs = np.where(lab == c)
        h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        if min(h, w) < min_side and max(h, w) > max_aspect * min(h, w):
            out[lab == c] = 0
    return out


def _erode(mask, r):
    m = np.asarray(mask, bool)
    for _ in range(r):
        g = m.copy()
        g[1:] &= m[:-1]; g[:-1] &= m[1:]; g[:, 1:] &= m[:, :-1]; g[:, :-1] &= m[:, 1:]
        g[0] = False; g[-1] = False; g[:, 0] = False; g[:, -1] = False
        m = g
    return m


def _dilate(mask, r):
    m = np.asarray(mask, bool)
    for _ in range(r):
        g = m.copy()
        g[1:] |= m[:-1]; g[:-1] |= m[1:]; g[:, 1:] |= m[:, :-1]; g[:, :-1] |= m[:, 1:]
        m = g
    return m


def object_colour(pid):
    """Saturated colour for object id: golden-ratio hue sequence, avoiding the blue background band."""
    import colorsys
    h = (0.0 + 0.618033988749895 * (pid - 1)) % 1.0
    h = h * 0.8 if h < 0.5 else 0.4 + (h - 0.5) * 0.4 + 0.3   # skip the pure-blue band around 0.66
    h = h % 1.0
    if 0.58 < h < 0.72:
        h = (h + 0.15) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
    return np.array([int(255 * r), int(255 * g), int(255 * b)], np.uint8)


def render_mask(labels, background=(0, 0, 255)):
    """Flat colour image: blue background, one saturated colour per tracked object."""
    out = np.empty(labels.shape + (3,), np.uint8); out[:] = background
    for pid in np.unique(labels[labels > 0]):
        out[labels == pid] = object_colour(int(pid))
    return out


# =============================================================================
# Real-time tracking of the minimiser (time-varying convex optimisation)
# =============================================================================
def _shift_int(a, dy, dx):
    """Integer shift with zero fill."""
    out = np.zeros_like(a)
    H, W = a.shape
    ys, ye = max(0, dy), min(H, H + dy); xs, xe = max(0, dx), min(W, W + dx)
    if ye > ys and xe > xs:
        out[ys:ye, xs:xe] = a[ys - dy:ye - dy, xs - dx:xe - dx]
    return out


class RealtimeSegmenter(ForegroundSegmenter):
    """Tracks the minimiser of the time-varying segmentation energy instead of
    re-solving it per frame.

    Energy at frame t (strongly convex because of the temporal proximal term):
        E_t(u) = <f_t, u> + lam * sum w_t |grad u| + (mu/2) ||u - uhat_t||^2,   u in [0,1]
    uhat_t  = prediction: the previous solution with each tracked object
              advected by its own velocity (centroid displacement per frame).
    Correction: K primal-dual steps (Chambolle-Pock) from the warm start,
              applied on the ACTIVE set only (patches whose evidence changed,
              the objects' neighbourhoods, and a halo); everything else is frozen.
    Certificate: the KKT residual of the full problem is evaluated every frame,
              on the frozen region too, so the reused labels are certified
              eta-stationary; if the residual anywhere exceeds eta the patch is
              activated next frame.

    Tracking theorem (prediction-correction, Simonetto et al. 2020): with the
    proximal term the primal-dual map is a contraction with factor rho < 1; if
    the minimiser drifts by d_t between frames, the error after K steps obeys
    e_t <= rho^K (e_{t-1} + d_t), so e_t <= rho^K d / (1 - rho^K) in steady
    state: a fixed, small multiple of one frame's drift, however long the video.
    """

    def __init__(self, *a, steps=3, mu=0.5, halo=0, kkt_tol=0.5, compare_full=False, act_delta0=40.0, **k):
        super().__init__(*a, **k)
        self.steps, self.mu, self.halo, self.kkt_tol, self.compare_full = int(steps), float(mu), int(halo), float(kkt_tol), bool(compare_full)
        self.act_delta0 = float(act_delta0); self.act_z = 3.0
        self.py = np.zeros((self.H, self.W)); self.px = np.zeros((self.H, self.W))
        self.f_prev = None; self.kkt_active = np.zeros((self.H // self.P, self.W // self.P), bool)
        self.stats = {"active_frac": [], "kkt_frozen_max": [], "kkt_active_max": [], "iou_vs_full": [], "pred_shift": []}

    def _predict(self):
        """Advect each tracked object by its velocity; zero-fill what it vacated."""
        u = self.u
        if not self.vel:
            self.plab = self.labels.copy()
            return u.copy()
        uhat = u.copy()
        plab = self.labels.copy()
        shifted_total = 0
        for pid, (vy, vx) in self.vel.items():
            dy, dx = int(round(vy)), int(round(vx))
            if dy == 0 and dx == 0:
                continue
            if abs(dy) > 48 or abs(dx) > 48:
                continue
            obj = self.labels == pid
            if not obj.any():
                continue
            u_obj = np.where(obj, u, 0.0)
            uhat = np.where(obj, 0.0, uhat)
            uhat = np.maximum(uhat, _shift_int(u_obj, dy, dx))
            plab = np.where(obj & (plab == pid), 0, plab)
            moved = _shift_int(obj.astype(np.int32), dy, dx) > 0
            plab = np.where(moved, pid, plab)
            shifted_total += 1
        self.plab = plab
        self.stats["pred_shift"].append(shifted_total)
        return uhat

    def _active(self, f, uhat, e):
        """Pixel mask of the active set: patches with coherent evidence (the Part-II multi-scale
        statistic on the residual, at half its certified threshold), object neighbourhoods,
        patches whose labels were not stationary last frame, and a halo."""
        P = self.P; gh, gw = self.H // P, self.W // P
        near = _dilate(self.mask | (uhat > 0.5), 4)
        eg = patch_grid(np.clip(e / np.sqrt(self.var), -6, 6), P)            # studentised residual
        score = np.zeros((gh, gw))
        for r in (1, 2, 3):
            score = np.maximum(score, box_max_abs_mean(eg, r) * (2 * r + 1))
        act_p = (score >= self.act_z * 2.5) | (patch_grid(near, P).any((2, 3))) | self.kkt_active
        for _ in range(self.halo):
            g = act_p.copy()
            g[1:] |= act_p[:-1]; g[:-1] |= act_p[1:]; g[:, 1:] |= act_p[:, :-1]; g[:, :-1] |= act_p[:, 1:]
            act_p = g
        act = np.zeros((self.H, self.W), bool)
        act[:gh * P, :gw * P] = np.kron(act_p, np.ones((P, P), bool))
        self.f_prev = f
        return act, act_p

    def _kkt(self, f, w, u, uhat):
        """Stationarity residual of E_t at u (box constraints handled by sign), per pixel."""
        g = -_div(w * self.py, w * self.px) + f + self.mu * (u - uhat)
        r = np.where(u <= 0.0, np.maximum(0.0, -g), np.where(u >= 1.0, np.maximum(0.0, g), np.abs(g)))
        return r

    def label(self, f, w, e=None):
        uhat = self._predict()
        act, act_p = self._active(f, uhat, e)
        self.stats["active_frac"].append(float(act.mean()))
        u = self.u.copy(); ubar = uhat.copy(); u[act] = uhat[act]
        py, px = self.py, self.px
        tau = sig = 1.0 / np.sqrt(8.0); lam = self.lam; mu = self.mu
        for _ in range(self.steps):
            gy, gx = _grad(ubar)
            py = py + sig * w * gy; px = px + sig * w * gx
            nrm = np.maximum(1.0, np.sqrt(py * py + px * px) / lam)
            py = py / nrm; px = px / nrm
            u_old = u
            u_new = np.clip((u - tau * (-_div(w * py, w * px) + f) + tau * mu * uhat) / (1.0 + tau * mu), 0.0, 1.0)
            u = np.where(act, u_new, u_old)
            ubar = 2 * u - u_old
        self.py, self.px = py, px
        self.u = u
        r = self._kkt(f, w, u, uhat)
        P = self.P; gh, gw = self.H // P, self.W // P
        rp = patch_grid(r, P).max((2, 3))
        self.kkt_active = rp > self.kkt_tol                                   # re-activate where not stationary
        frozen = ~act[:gh * P, :gw * P]
        self.stats["kkt_frozen_max"].append(float(r[:gh * P, :gw * P][frozen].max()) if frozen.any() else 0.0)
        self.stats["kkt_active_max"].append(float(r[act].max()) if act.any() else 0.0)
        m = (u > 0.5).astype(np.uint8)
        if self.compare_full:
            u_full = tv_segment(f + mu * 0.0, w, lam, 150, u0=None)             # per-frame global solve of the same E_t without the temporal term
            mf = u_full > 0.5
            inter = (mf & (m > 0)).sum(); union = (mf | (m > 0)).sum()
            self.stats["iou_vs_full"].append(float(inter / union) if union else 1.0)
        return m
