"""Quotient certificate over a group of warps: the certificate for moving cameras.

THEORY.md §3 certifies a patch modulo a sub-pixel translation.  This module
replaces the translation by a per-frame homography (camera motion) composed
with a per-patch sub-pixel translation (local refinement), keeps the reference
in a world-anchored canvas, and states exactly where interpolation enters.

Mechanics
---------
A canvas C holds the last kept copy of every patch in the coordinates of the
first frame (plus a margin).  G_t maps frame-t pixels to canvas coordinates.
For a new frame F:

  1. Estimate G_t by Gauss-Newton on brightness constancy F(x) ~ C(G x) + c,
     coarse to fine, with Huber weights so that moving objects do not steer
     the fit.  The estimate is a heuristic and does not affect soundness.
  2. Per patch, optionally refine with a sub-pixel translation delta
     (|delta|_inf <= delta_max) in canvas coordinates: W(x) = C(G x + delta).
  3. Certify the residual F - W with the range rule or the multi-scale rule
     (THEORY.md §2-3), exactly as before.  A patch whose sampling region is
     not yet in the canvas has no reference and is kept.
  4. Kept patches are written into the canvas by inverse warping (one bilinear
     resample).  If more than `cut_frac` of the patches are kept the frame is
     treated as a shot cut: the canvas is reset to this frame, all patches are
     kept, and the cost is counted.

For a fixed camera G_t is the identity (plus the integer margin), every sample
falls on an integer canvas position, the interpolation term below is zero and
the rule is bit for bit the range rule of core.py (tests/test_warpcert.py).

Theorem A (warp-quotient certificate; exact).  For any warp g and any
construction of W from previously kept pixels, if s = range(F - W) < Delta on
the patch then, with c the midrange of F - W,
    ||F - (W + c)||_inf < Delta/2   and   d_B(Dgm F, Dgm(W + c)) < Delta/2
for dark and for bright blobs: no feature of contrast >= Delta appears,
vanishes, splits or merges in F relative to W.  (Proof: THEORY.md Theorem 4,
which never used that the warp is a translation.)  With the multi-scale rule
the statement is THEORY.md Theorem 3 applied to F against W + c.

Proposition B (where interpolation enters; observable).  W(x) is a convex
combination of the four canvas pixels around G x + delta, and each canvas
pixel is a convex combination of the four frame pixels around its own source
position when it was written.  Let W_NN(x) be the actual kept pixel nearest
to the sample point (one of those corners, at one of the write corners).
Then |W(x) - W_NN(x)| <= iota(x), where iota(x) is the oscillation (max -
min) of the four read corners plus the largest write-time oscillation stored
under those corners.  Hence
    d_B(Dgm F, Dgm(W_NN + c)) < Delta/2 + iota(patch),  iota(patch) = max_x iota(x).
iota is zero wherever the sample lands on an integer position of an
integer-written canvas (every fixed-camera patch), and it is large only where
the reference has an edge sharper than one pixel at the sample point.  The
pruner reports iota per patch and per frame.  Whether Dgm(W_NN) equals the
diagram of the kept pixels re-indexed by the warp is exact for integer
translations and holds up to adjacency changes of the re-indexing otherwise;
that part is measured (scripts/warp_audit.py), not proved.

Proposition C (tightness).  For any g in the group and any residual pattern
with spread Delta there is a canvas, a flat one, on which that residual is a
feature of contrast exactly Delta.  No rule that reads only the residual can
drop at spread >= Delta and keep Theorem A.  (THEORY.md Proposition 2 with W
flat; a warp of a flat image is flat.)
"""
from __future__ import annotations

import numpy as np

from .core import patch_grid
from .scalespace import box_max_abs_mean

_SNAP = 1e-6


# ---------------------------------------------------------------- geometry
def apply_h(G: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Apply a 3x3 homography to pixel coordinates. Returns (u, v)."""
    u = G[0, 0] * x + G[0, 1] * y + G[0, 2]
    v = G[1, 0] * x + G[1, 1] * y + G[1, 2]
    w = G[2, 0] * x + G[2, 1] * y + G[2, 2]
    return u / w, v / w


def translation(tx: float, ty: float) -> np.ndarray:
    G = np.eye(3)
    G[0, 2] = tx; G[1, 2] = ty
    return G


def _snap(a: np.ndarray) -> np.ndarray:
    """Snap values within 1e-6 of an integer to that integer, so a fixed camera samples exactly."""
    r = np.rint(a)
    return np.where(np.abs(a - r) < _SNAP, r, a)


def sample_bilinear(img: np.ndarray, valid: np.ndarray, px: np.ndarray, py: np.ndarray, extra: np.ndarray | None = None):
    """Bilinear sample of `img` at float positions (px, py).

    Returns (value, ok, osc, nn) where ok marks samples whose four corners are inside the image and
    valid; osc is the oscillation of the four corner values plus the largest `extra` (write-time
    oscillation) under them, i.e. the per-pixel interpolation term of Proposition B; nn is the
    nearest-corner value W_NN.  Samples exactly on an integer position have osc = 0 (plus extra)."""
    H, W = img.shape
    px = _snap(np.asarray(px, np.float64)); py = _snap(np.asarray(py, np.float64))
    x0 = np.floor(px); y0 = np.floor(py)
    fx = px - x0; fy = py - y0
    x0 = x0.astype(np.int64); y0 = y0.astype(np.int64)
    exact_x = fx == 0; exact_y = fy == 0
    # a sample exactly on a column needs only that column; treat x1 = x0 there
    x1 = np.where(exact_x, x0, x0 + 1); y1 = np.where(exact_y, y0, y0 + 1)
    inside = (x0 >= 0) & (y0 >= 0) & (x1 <= W - 1) & (y1 <= H - 1)
    x0c = np.clip(x0, 0, W - 1); x1c = np.clip(x1, 0, W - 1); y0c = np.clip(y0, 0, H - 1); y1c = np.clip(y1, 0, H - 1)
    v00 = img[y0c, x0c]; v01 = img[y0c, x1c]; v10 = img[y1c, x0c]; v11 = img[y1c, x1c]
    ok = inside & valid[y0c, x0c] & valid[y0c, x1c] & valid[y1c, x0c] & valid[y1c, x1c]
    val = (1 - fy) * ((1 - fx) * v00 + fx * v01) + fy * ((1 - fx) * v10 + fx * v11)
    stack = np.stack([v00, v01, v10, v11])
    osc = stack.max(0) - stack.min(0)
    if extra is not None:
        e = np.stack([extra[y0c, x0c], extra[y0c, x1c], extra[y1c, x0c], extra[y1c, x1c]]).max(0)
        osc = osc + e
    nx = np.where(fx < 0.5, x0c, x1c); ny = np.where(fy < 0.5, y0c, y1c)
    nn = img[ny, nx]
    return val, ok, osc, nn


def _gradients(img: np.ndarray, valid: np.ndarray | None = None):
    g = np.asarray(img, np.float64)
    gy = np.zeros_like(g); gx = np.zeros_like(g)
    gy[1:-1] = 0.5 * (g[2:] - g[:-2]); gx[:, 1:-1] = 0.5 * (g[:, 2:] - g[:, :-2])
    if valid is not None:
        ok = valid.copy()
        ok[1:-1] &= valid[2:] & valid[:-2]
        gy = np.where(ok, gy, 0.0)
        ok = valid.copy()
        ok[:, 1:-1] &= valid[:, 2:] & valid[:, :-2]
        gx = np.where(ok, gx, 0.0)
    return gy, gx


def _downsample(img: np.ndarray, valid: np.ndarray, s: int):
    """Box downsample by s with a validity mask (coarse pixel valid if at least half its fine pixels are)."""
    if s == 1:
        return np.asarray(img, np.float64), valid.copy()
    H, W = img.shape
    h, w = H // s, W // s
    a = np.asarray(img, np.float64)[: h * s, : w * s].reshape(h, s, w, s)
    v = valid[: h * s, : w * s].reshape(h, s, w, s)
    cnt = v.sum((1, 3))
    out = np.where(cnt > 0, (a * v).sum((1, 3)) / np.maximum(cnt, 1), 0.0)
    return out, cnt >= (s * s + 1) // 2


# ---------------------------------------------------------------- global registration
def fit_global(F: np.ndarray, C: np.ndarray, valid: np.ndarray, G0: np.ndarray, model: str = "homography",
               levels=(4, 2, 1), iters=(10, 8, 4), huber: bool = True, tol: float = 0.01):
    """Gauss-Newton fit of G (frame -> canvas) and a brightness offset c on F(x) ~ C(G x) + c.

    Coarse to fine over `levels` (downsampling factors), with Huber weights on the residual so that
    the fit follows the majority (the background) and not the moving objects.  `model` is
    'homography' (8 parameters), 'affine' (6) or 'translation' (2).  Returns (G, c, info) where
    info has the robust residual scale and the fraction of frame pixels with a valid reference.
    A heuristic: the certificate never depends on how G was chosen."""
    G = np.asarray(G0, np.float64).copy(); G /= G[2, 2]
    F = np.asarray(F, np.float64)
    nparam = {"homography": 8, "affine": 6, "translation": 2}[model]
    c = 0.0
    info = {"scale": np.nan, "coverage": 0.0}
    for s, it in zip(levels, iters):
        Fs, _ = _downsample(F, np.ones(F.shape, bool), s)
        Cs, Vs = _downsample(C, valid, s)
        if Vs.sum() < 64:
            continue
        Cy, Cx = _gradients(Cs, Vs)
        S = np.diag([1.0 / s, 1.0 / s, 1.0]); Si = np.diag([float(s), float(s), 1.0])
        Gs = S @ G @ Si; Gs /= Gs[2, 2]
        h, w = Fs.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        x = xx.ravel(); y = yy.ravel(); f = Fs.ravel()
        for _ in range(it):
            u, v = apply_h(Gs, x, y)
            wv, ok, _, _ = sample_bilinear(Cs, Vs, u, v)
            gx, _, _, _ = sample_bilinear(Cx, Vs, u, v)
            gy, _, _, _ = sample_bilinear(Cy, Vs, u, v)
            r = f - wv - c
            if ok.sum() < 32:
                break
            r = r[ok]; gx = gx[ok]; gy = gy[ok]; xo = x[ok]; yo = y[ok]; uo = u[ok]; vo = v[ok]
            wdn = Gs[2, 0] * xo + Gs[2, 1] * yo + 1.0
            if huber:
                med = np.median(r); sc = 1.4826 * np.median(np.abs(r - med)) + 1e-3
                wt = np.minimum(1.0, 1.345 * sc / np.maximum(np.abs(r - med), 1e-9))
            else:
                wt = np.ones_like(r)
            cols = []
            if model == "translation":
                cols = [gx / wdn, gy / wdn]
            else:
                cols = [gx * xo / wdn, gx * yo / wdn, gx / wdn, gy * xo / wdn, gy * yo / wdn, gy / wdn]
                if model == "homography":
                    cols += [-(gx * uo + gy * vo) * xo / wdn, -(gx * uo + gy * vo) * yo / wdn]
            cols.append(np.ones_like(r))
            J = np.stack(cols, 1)
            A = (J * wt[:, None]).T @ J + 1e-6 * np.eye(J.shape[1])
            b = (J * wt[:, None]).T @ r
            try:
                step = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                break
            cost0 = float((wt * r * r).sum())
            # damped acceptance
            accepted = False
            for damp in (1.0, 0.5, 0.25, 0.125):
                Gn = Gs.copy(); cn = c + damp * step[-1]
                st = damp * step[:-1]
                if model == "translation":
                    Gn[0, 2] += st[0]; Gn[1, 2] += st[1]
                else:
                    Gn[0, 0] += st[0]; Gn[0, 1] += st[1]; Gn[0, 2] += st[2]
                    Gn[1, 0] += st[3]; Gn[1, 1] += st[4]; Gn[1, 2] += st[5]
                    if model == "homography":
                        Gn[2, 0] += st[6]; Gn[2, 1] += st[7]
                un, vn = apply_h(Gn, xo, yo)
                wvn, okn, _, _ = sample_bilinear(Cs, Vs, un, vn)
                rn = np.where(okn, f[ok] - wvn - cn, 0.0)
                cost1 = float((wt * rn * rn).sum())
                if cost1 <= cost0:
                    Gs, c, accepted = Gn, cn, True
                    break
            if not accepted:
                break
            # convergence: largest displacement change over the frame
            u2, v2 = apply_h(Gs, xo, yo)
            if max(np.abs(u2 - uo).max(), np.abs(v2 - vo).max()) < tol / s:
                break
        G = Si @ Gs @ S; G /= G[2, 2]
    # final residual at full resolution; also try the nearest integer translation, so that a camera
    # that did not move lands on exact pixels (interpolation term zero) when that is at least as good
    h, w = F.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    x = xx.ravel(); y = yy.ravel(); f = F.ravel()

    def robust_cost(Gc):
        u, v = apply_h(Gc, x, y)
        wv, ok, _, _ = sample_bilinear(C, valid, u, v)
        if ok.sum() < 32:
            return np.inf, np.nan, 0.0, c
        r = (f - wv)[ok]
        cc = float(np.median(r)); r = r - cc
        sc = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-3
        k = 1.345 * sc
        hub = np.where(np.abs(r) <= k, 0.5 * r * r, k * (np.abs(r) - 0.5 * k))
        return float(hub.mean()), float(sc), float(ok.mean()), cc

    cost, scale, cov, c = robust_cost(G)
    lin = G[:2, :2] - np.eye(2)
    if np.abs(lin).max() < 1e-3 and np.abs(G[2, :2]).max() < 1e-6:
        Gs = G.copy(); Gs[:2, :2] = np.eye(2); Gs[2, :2] = 0; Gs[0, 2] = np.rint(G[0, 2]); Gs[1, 2] = np.rint(G[1, 2])
        if np.abs(Gs[:2, 2] - G[:2, 2]).max() < 0.5:
            cost_s, scale_s, cov_s, c_s = robust_cost(Gs)
            if cost_s <= cost:
                G, cost, scale, cov, c = Gs, cost_s, scale_s, cov_s, c_s
                info["snapped"] = True
    info["scale"] = scale; info["coverage"] = cov; info["cost"] = cost
    return G, float(c), info


# ---------------------------------------------------------------- the pruner
class WarpGroupPruner:
    """Quotient certificate modulo a per-frame homography (camera) and a per-patch sub-pixel
    translation (refinement), with the reference held in a world canvas.

    Parameters
    ----------
    height, width, patch : as core.Pruner
    delta        : contrast threshold for the range certificate (ignored if `multiscale` is given)
    multiscale   : dict r -> Delta_r for the multi-scale certificate on the residual
    model        : 'homography' | 'affine' | 'translation' | 'none' (fixed camera, no global fit)
    delta_max    : per-patch refinement bound in pixels (0 disables)
    margin       : canvas margin on each side, in pixels
    cut_scale    : a frame is a shot cut when the robust residual scale of the global registration
                   (grey levels, after convergence) exceeds this, or when more than `cut_frac` of the
                   patches are kept anyway; the canvas is then reset to the frame and every patch is kept
    max_shift    : optional cap on |c|: a patch whose brightness shift exceeds it is kept
    candidates   : also evaluate the residual against the previous registration G_{t-1} (no refinement)
                   and let each patch be certified by whichever warp fits it better.  Both are elements
                   of the group, so the certificate is unchanged (F equals the reference moved by *some*
                   g in G); on a camera that has not moved, G_{t-1} lands on integer pixels and the
                   fixed-camera rule is recovered exactly
    """

    def __init__(self, height, width, patch=16, delta=32.0, multiscale: dict | None = None,
                 model: str = "homography", delta_max: float = 1.0, iters: int = 2, margin: int | None = None,
                 cut_frac: float = 0.9, cut_scale: float = 24.0, max_shift: float | None = None,
                 levels=(8, 4, 2), gn_iters=(8, 6, 4), candidates: bool = True):
        self.H, self.W, self.P = int(height), int(width), int(patch)
        self.delta, self.multiscale = float(delta), (dict(multiscale) if multiscale else None)
        self.model, self.delta_max, self.iters = model, float(delta_max), int(iters)
        self.margin = int(margin) if margin is not None else int(max(self.H, self.W) // 2)
        self.cut_frac, self.cut_scale, self.max_shift = float(cut_frac), float(cut_scale), max_shift
        self.candidates = bool(candidates)
        self.levels, self.gn_iters = levels, gn_iters
        self.gh, self.gw = self.H // self.P, self.W // self.P
        self.CH, self.CW = self.H + 2 * self.margin, self.W + 2 * self.margin
        self.C = np.zeros((self.CH, self.CW), np.float64)
        self.valid = np.zeros((self.CH, self.CW), bool)
        self.iota_w = np.zeros((self.CH, self.CW), np.float64)    # write-time oscillation per canvas pixel
        self.G = None; self.G_prev = None
        self.t = -1
        self.n_cuts = 0; self.n_recentre = 0
        self.last = {}
        self._view = None
        self._yy, self._xx = np.mgrid[0:self.H, 0:self.W].astype(np.float64)

    # -- state ---------------------------------------------------------------
    @property
    def initialised(self):
        return self.G is not None

    def reset(self):
        self.valid[:] = False; self.C[:] = 0; self.iota_w[:] = 0
        self.G = None; self.G_prev = None; self.t = -1

    def _reset_to(self, frame):
        self.valid[:] = False; self.C[:] = 0; self.iota_w[:] = 0
        self.G = translation(self.margin, self.margin); self.G_prev = None
        self._write(frame, np.ones((self.gh, self.gw), bool))

    def _write(self, frame, keep):
        """Write the kept patches of `frame` into the canvas by inverse warping (one bilinear resample)."""
        P = self.P
        Gi = np.linalg.inv(self.G)
        # canvas bounding box of the frame
        cx, cy = apply_h(self.G, np.array([0, self.W - 1, 0, self.W - 1], float), np.array([0, 0, self.H - 1, self.H - 1], float))
        x0 = max(0, int(np.floor(cx.min()))); x1 = min(self.CW - 1, int(np.ceil(cx.max())))
        y0 = max(0, int(np.floor(cy.min()))); y1 = min(self.CH - 1, int(np.ceil(cy.max())))
        if x1 <= x0 or y1 <= y0:
            return
        py, px = np.mgrid[y0:y1 + 1, x0:x1 + 1].astype(np.float64)
        sx, sy = apply_h(Gi, px, py)
        sx = _snap(sx); sy = _snap(sy)
        inside = (sx >= 0) & (sy >= 0) & (sx <= self.gw * P - 1) & (sy <= self.gh * P - 1)
        gx = np.clip(np.floor(sx / P).astype(np.int64), 0, self.gw - 1)
        gy = np.clip(np.floor(sy / P).astype(np.int64), 0, self.gh - 1)
        sel = inside & keep[gy, gx]
        if not sel.any():
            return
        fr = np.asarray(frame, np.float64)
        val, ok, osc, _ = sample_bilinear(fr, np.ones(fr.shape, bool), sx[sel], sy[sel])
        rows = py[sel].astype(np.int64); cols = px[sel].astype(np.int64)
        okr = ok
        self.C[rows[okr], cols[okr]] = val[okr]
        self.valid[rows[okr], cols[okr]] = True
        self.iota_w[rows[okr], cols[okr]] = osc[okr]

    def _recentre_if_needed(self):
        """If the frame's canvas footprint leaves the canvas, shift the canvas by an integer offset (lossless)."""
        cx, cy = apply_h(self.G, np.array([0, self.W - 1, 0, self.W - 1], float), np.array([0, 0, self.H - 1, self.H - 1], float))
        dx = dy = 0
        if cx.min() < 1: dx = int(np.ceil(1 - cx.min())) + self.margin // 2
        if cx.max() > self.CW - 2: dx = -(int(np.ceil(cx.max() - (self.CW - 2))) + self.margin // 2)
        if cy.min() < 1: dy = int(np.ceil(1 - cy.min())) + self.margin // 2
        if cy.max() > self.CH - 2: dy = -(int(np.ceil(cy.max() - (self.CH - 2))) + self.margin // 2)
        if dx == 0 and dy == 0:
            return
        for arr, fill in ((self.C, 0.0), (self.iota_w, 0.0)):
            moved = np.full_like(arr, fill)
            # A shift as large as the canvas leaves nothing to carry: clamp so both slices are empty (the canvas is
            # cleared and every patch is kept for lack of a reference) instead of a negative stop selecting rows.
            ys = slice(max(0, dy), max(0, min(self.CH, self.CH + dy))); yd = slice(max(0, -dy), max(0, min(self.CH, self.CH - dy)))
            xs = slice(max(0, dx), max(0, min(self.CW, self.CW + dx))); xd = slice(max(0, -dx), max(0, min(self.CW, self.CW - dx)))
            moved[ys, xs] = arr[yd, xd]
            arr[:] = moved
        movedv = np.zeros_like(self.valid)
        movedv[ys, xs] = self.valid[yd, xd]
        self.valid[:] = movedv
        T = translation(dx, dy)
        self.G = T @ self.G
        if self.G_prev is not None:
            self.G_prev = T @ self.G_prev
        self.n_recentre += 1

    # -- rule ----------------------------------------------------------------
    def step(self, frame):
        """Process one frame.  Returns (keep[gh,gw], score[gh,gw], shift[gh,gw]); details in self.last:
        iota (per patch), G, cut, noref (patches kept for lack of a reference), reg (registration info)."""
        frame = np.ascontiguousarray(frame)
        if frame.shape != (self.H, self.W):
            raise ValueError(f"frame shape {frame.shape} != {(self.H, self.W)}")
        self.t += 1
        P, gh, gw = self.P, self.gh, self.gw
        z = np.zeros((gh, gw), np.float32)
        if self.G is None:
            self._reset_to(frame)
            self._view = frame.astype(np.float64)
            self.last = {"iota": z.copy(), "G": self.G.copy(), "cut": False, "noref": np.ones((gh, gw), bool), "reg": {}, "c": 0.0,
                         "revealed": np.ones((gh, gw), bool), "chosen": np.zeros((gh, gw), np.int8)}
            return np.ones((gh, gw), bool), z, z.copy()
        fr = frame.astype(np.float64)
        # 1. global registration (constant-velocity prediction, then Gauss-Newton against the canvas)
        reg = {}
        if self.model != "none":
            G0 = self.G if self.G_prev is None else self.G @ np.linalg.inv(self.G_prev) @ self.G
            G0 = G0 / G0[2, 2]
            Gn, cglob, reg = fit_global(fr, self.C, self.valid, G0, self.model, self.levels, self.gn_iters)
            if np.all(np.isfinite(Gn)) and reg.get("coverage", 0) > 0.05:
                # a translation-like registration is rounded to whole pixels when per-patch refinement is on:
                # the canvas is then written without resampling and the sub-pixel part is the refinement's job,
                # which is exactly the fixed-camera quotient rule once the integer part of the pan is removed
                if self.delta_max >= 0.5:
                    cx, cy = apply_h(Gn, np.array([0, self.W - 1, 0, self.W - 1, 0.5 * (self.W - 1)]), np.array([0, 0, self.H - 1, self.H - 1, 0.5 * (self.H - 1)]))
                    tx, ty = np.rint(cx[4] - 0.5 * (self.W - 1)), np.rint(cy[4] - 0.5 * (self.H - 1))
                    disp = max(np.abs(cx[:4] - (np.array([0, self.W - 1, 0, self.W - 1]) + tx)).max(),
                               np.abs(cy[:4] - (np.array([0, 0, self.H - 1, self.H - 1]) + ty)).max())
                    if disp < 0.75 * self.delta_max:
                        Gn = translation(tx, ty)
                        reg["snapped"] = True
                self.G_prev, self.G = self.G, Gn
            else:
                self.G_prev = self.G
        self._recentre_if_needed()
        # 2. warped reference at G x (+ per-patch delta), with the interpolation term
        u, v = apply_h(self.G, self._xx, self._yy)
        W0, ok0, osc0, _ = sample_bilinear(self.C, self.valid, u, v, self.iota_w)
        dy = np.zeros((gh, gw)); dx = np.zeros((gh, gw))
        if self.delta_max > 0:
            dy, dx = self._refine(fr, W0, ok0, u, v)
            uu = u + np.kron(dx, np.ones((P, P)))[: self.H, : self.W]
            vv = v + np.kron(dy, np.ones((P, P)))[: self.H, : self.W]
            Wimg, ok, osc, nn = sample_bilinear(self.C, self.valid, uu, vv, self.iota_w)
        else:
            Wimg, ok, osc, nn = W0, ok0, osc0, None
        Hg, Wg = gh * P, gw * P

        def certify(Wc, okc):
            okg_ = patch_grid(okc[:Hg, :Wg], P).all((2, 3))
            d_ = patch_grid(fr[:Hg, :Wg] - Wc[:Hg, :Wg], P)
            c_ = 0.5 * (d_.max((2, 3)) + d_.min((2, 3)))
            e_ = d_ - c_[..., None, None]
            if self.multiscale is None:
                sc_ = 2.0 * np.abs(e_).max((2, 3)) / self.delta
            else:
                sc_ = np.zeros((gh, gw))
                for r, D in self.multiscale.items():
                    sc_ = np.maximum(sc_, box_max_abs_mean(e_, r) / (0.5 * D))
            return okg_, np.where(okg_, sc_, np.inf), c_

        # 3. certificate on the residual, against the best of the candidate warps per patch
        okg, score, c = certify(Wimg, ok)
        revealed = ~okg                                   # no reference under the current registration
        chosen = np.zeros((gh, gw), np.int8)
        if self.candidates and self.G_prev is not None and self.model != "none":
            u2, v2 = apply_h(self.G_prev, self._xx, self._yy)
            W2, ok2, osc2, _ = sample_bilinear(self.C, self.valid, u2, v2, self.iota_w)
            okg2, score2, c2 = certify(W2, ok2)
            better = score2 < score
            if better.any():
                chosen[better] = 1
                score = np.where(better, score2, score); c = np.where(better, c2, c); okg = okg | (better & okg2)
                m = np.kron(better, np.ones((P, P), bool))
                Wimg = Wimg.copy(); osc = osc.copy(); ok = ok.copy()
                Wimg[:Hg, :Wg][m] = W2[:Hg, :Wg][m]; osc[:Hg, :Wg][m] = osc2[:Hg, :Wg][m]; ok[:Hg, :Wg][m] = ok2[:Hg, :Wg][m]
        noref = ~okg
        keep = score >= 1.0
        if self.max_shift is not None:
            keep |= np.abs(c) > self.max_shift
        reg_fail = (self.model != "none") and (not np.isfinite(reg.get("scale", np.nan)) or reg.get("scale", 0.0) > self.cut_scale)
        cut = bool(keep.mean() > self.cut_frac) or bool(reg_fail)
        iota = patch_grid(osc[:Hg, :Wg], P).max((2, 3))
        iota = np.where(okg, iota, 0.0)
        if cut:
            self.n_cuts += 1
            self._reset_to(frame)
            keep = np.ones((gh, gw), bool)
            self._view = fr.copy()
        else:
            view = Wimg.copy()
            view[:Hg, :Wg] += np.kron(np.where(keep, 0.0, c), np.ones((P, P)))
            fg = patch_grid(fr, P); vg = patch_grid(view, P); vg[keep] = fg[keep]
            view[Hg:, :] = fr[Hg:, :]; view[:, Wg:] = fr[:, Wg:]
            self._view = view
            self._write(frame, keep)
        self.last = {"iota": iota.astype(np.float32), "G": self.G.copy(), "cut": cut, "noref": noref, "reg": reg,
                     "c": c.astype(np.float32), "dy": dy.astype(np.float32), "dx": dx.astype(np.float32), "ok": okg,
                     "iota_pix": osc[:Hg, :Wg].astype(np.float32), "chosen": chosen, "revealed": revealed}
        return keep, score.astype(np.float32), c.astype(np.float32)

    def _refine(self, fr, W0, ok0, u, v):
        """Per-patch sub-pixel translation in canvas coordinates (Gauss-Newton, Tikhonov prior, clamped)."""
        P, gh, gw = self.P, self.gh, self.gw
        Hg, Wg = gh * P, gw * P
        Wy, Wx = _gradients(W0, ok0)
        Gy, Gx = patch_grid(Wy[:Hg, :Wg], P), patch_grid(Wx[:Hg, :Wg], P)
        n = P * P
        A = np.empty((gh, gw, 3, 3))
        A[..., 0, 0] = (Gx * Gx).sum((2, 3)) + n; A[..., 0, 1] = A[..., 1, 0] = (Gx * Gy).sum((2, 3))
        A[..., 1, 1] = (Gy * Gy).sum((2, 3)) + n; A[..., 0, 2] = A[..., 2, 0] = Gx.sum((2, 3))
        A[..., 1, 2] = A[..., 2, 1] = Gy.sum((2, 3)); A[..., 2, 2] = n
        dy = np.zeros((gh, gw)); dx = np.zeros((gh, gw))
        for _ in range(self.iters):
            uu = u + np.kron(dx, np.ones((P, P)))[: self.H, : self.W]
            vv = v + np.kron(dy, np.ones((P, P)))[: self.H, : self.W]
            Wimg, ok, _, _ = sample_bilinear(self.C, self.valid, uu, vv)
            D = patch_grid(np.where(ok, fr - Wimg, 0.0)[:Hg, :Wg], P)
            b = np.stack([(Gx * D).sum((2, 3)), (Gy * D).sum((2, 3)), D.sum((2, 3))], -1)
            sol = np.linalg.solve(A, b[..., None])[..., 0]
            dx = np.clip(dx + sol[..., 0], -self.delta_max, self.delta_max)
            dy = np.clip(dy + sol[..., 1], -self.delta_max, self.delta_max)
        return dy, dx

    def run(self, frames):
        ks, ss, cs, io, cuts, nr, rv = [], [], [], [], [], [], []
        for f in frames:
            k, s, c = self.step(f)
            ks.append(k); ss.append(s); cs.append(c); io.append(self.last["iota"]); cuts.append(self.last["cut"])
            nr.append(self.last["noref"]); rv.append(self.last["revealed"])
        return {"keep": np.stack(ks), "spread": np.stack(ss), "shift": np.stack(cs), "iota": np.stack(io),
                "cut": np.array(cuts), "noref": np.stack(nr), "revealed": np.stack(rv)}

    def view(self, compensate_shift: bool = True):
        v = self._view
        return np.clip(np.rint(v), 0, 255).astype(np.uint8)

    def warped_reference(self):
        """The reference as the current frame sees it: W = C(G x) over the frame (nan where no reference)."""
        u, v = apply_h(self.G, self._xx, self._yy)
        Wimg, ok, _, _ = sample_bilinear(self.C, self.valid, u, v)
        return np.where(ok, Wimg, np.nan)
