"""Quotient certificate: certify modulo a small motion.

The range rule pays full price for a sub-pixel jiggle of a high-contrast
edge although nothing appeared.  Translations do not change what is in a
patch, so quotient them out: explain the change as a sub-pixel translation
delta (|delta| <= delta_max) plus a brightness shift c of the reference, and
certify only the residual.

Per patch (vectorised over the frame), one Gauss-Newton step of brightness
constancy  F(x) ~ R(x + delta) + c  gives

    [sum Rx^2  sum RxRy  sum Rx] [dx]   [sum Rx d]
    [sum RxRy  sum Ry^2  sum Ry] [dy] = [sum Ry d]      d = F - R
    [sum Rx    sum Ry    n     ] [c ]   [sum d  ]

The fitted delta is clamped to |delta|_inf <= delta_max; the warped
reference W = R(x + delta) is sampled bilinearly from the FULL reference
frame (so patch borders are fine), and the residual e = F - W - c' with
c' = midrange(F - W) is certified with the range rule (or the multi-scale
rule).  The fit is only a heuristic for choosing delta: the certificate is
computed exactly against W + c', so it is sound whatever the fit does.

Guarantee while dropped: F is within s/2 (s = range of F - W) in L_inf of a
sub-pixel translate (by at most delta_max) of the reference, up to a
brightness shift.  By stability no feature of contrast >= Delta appears,
vanishes, splits or merges relative to that translate; and a translate has
the same features as the reference at sub-pixel-shifted positions.  The
model's token encodes R; the certificate says the scene in F is R, moved
by less than delta_max pixels, up to Delta-contrast topology.

This also happens to be the first step towards moving cameras: replace the
per-patch translation by a per-frame or per-patch affine motion and the
same certificate applies to the registered residual.
"""
from __future__ import annotations

import numpy as np

from .core import patch_grid
from .scalespace import box_max_abs_mean


def _grad(img: np.ndarray):
    g = np.asarray(img, np.float64)
    gy = np.zeros_like(g); gx = np.zeros_like(g)
    gy[1:-1] = 0.5 * (g[2:] - g[:-2]); gy[0] = g[1] - g[0]; gy[-1] = g[-1] - g[-2]
    gx[:, 1:-1] = 0.5 * (g[:, 2:] - g[:, :-2]); gx[:, 0] = g[:, 1] - g[:, 0]; gx[:, -1] = g[:, -1] - g[:, -2]
    return gy, gx


def fit_translation(frame, ref, patch, delta_max=1.0, iters=2):
    """Per-patch (dy, dx) with |.| <= delta_max minimising the brightness
    constancy residual (Gauss-Newton, `iters` steps). Returns (dy, dx)[gh, gw]."""
    F = np.asarray(frame, np.float64); R = np.asarray(ref, np.float64)
    H, W = F.shape; gh, gw = H // patch, W // patch
    gy, gx = _grad(R)
    Gy, Gx = patch_grid(gy, patch), patch_grid(gx, patch)
    n = patch * patch
    A = np.empty((gh, gw, 3, 3)); A[..., 0, 0] = (Gx * Gx).sum((2, 3)); A[..., 0, 1] = A[..., 1, 0] = (Gx * Gy).sum((2, 3))
    A[..., 1, 1] = (Gy * Gy).sum((2, 3)); A[..., 0, 2] = A[..., 2, 0] = Gx.sum((2, 3)); A[..., 1, 2] = A[..., 2, 1] = Gy.sum((2, 3)); A[..., 2, 2] = n
    A[..., 0, 0] += n; A[..., 1, 1] += n          # Tikhonov prior: motion is 0 unless gradient energy >> n grey^2
    dy = np.zeros((gh, gw)); dx = np.zeros((gh, gw))
    for _ in range(iters):
        Wimg = warp(R, dy, dx, patch)
        D = patch_grid(F - Wimg, patch)
        b = np.stack([(Gx * D).sum((2, 3)), (Gy * D).sum((2, 3)), D.sum((2, 3))], -1)
        sol = np.linalg.solve(A, b[..., None])[..., 0]
        dx = np.clip(dx + sol[..., 0], -delta_max, delta_max)
        dy = np.clip(dy + sol[..., 1], -delta_max, delta_max)
    return dy, dx


def warp(ref, dy, dx, patch):
    """Bilinear sample R(y + dy, x + dx) with a per-patch (dy, dx), from the
    full frame, clamped at the frame border. Returns float64 (H, W) over the grid area."""
    R = np.asarray(ref, np.float64)
    H, W = R.shape; gh, gw = H // patch, W // patch
    Hg, Wg = gh * patch, gw * patch
    yy, xx = np.mgrid[0:Hg, 0:Wg].astype(np.float64)
    yy += np.kron(dy, np.ones((patch, patch))); xx += np.kron(dx, np.ones((patch, patch)))
    yy = np.clip(yy, 0, H - 1); xx = np.clip(xx, 0, W - 1)
    y0 = np.floor(yy).astype(int); x0 = np.floor(xx).astype(int)
    y1 = np.minimum(y0 + 1, H - 1); x1 = np.minimum(x0 + 1, W - 1)
    fy = yy - y0; fx = xx - x0
    out = np.empty_like(R)
    out[:Hg, :Wg] = ((1 - fy) * (1 - fx) * R[y0, x0] + (1 - fy) * fx * R[y0, x1]
                     + fy * (1 - fx) * R[y1, x0] + fy * fx * R[y1, x1])
    out[Hg:, :] = R[Hg:, :]; out[:, Wg:] = R[:, Wg:]
    return out


class WarpPruner:
    """Same interface as core.Pruner.  certify = 'range' (threshold delta) or
    'multiscale' (dict r -> Delta_r, as in scalespace)."""

    def __init__(self, height, width, patch=16, delta=32.0, delta_max=1.0, iters=2,
                 multiscale: dict | None = None):
        self.H, self.W, self.patch = int(height), int(width), int(patch)
        self.delta, self.delta_max, self.iters = float(delta), float(delta_max), int(iters)
        self.multiscale = multiscale
        self.gh, self.gw = self.H // patch, self.W // patch
        self.ref = None
        self._view = None
        self.last_motion = None

    @property
    def initialised(self):
        return self.ref is not None

    def reset(self):
        self.ref = None; self._view = None

    def step(self, frame):
        frame = np.ascontiguousarray(frame)
        if self.ref is None:
            self.ref = frame.copy(); self._view = frame.astype(np.float64)
            z = np.zeros((self.gh, self.gw), np.float32)
            self.last_motion = (z, z)
            return np.ones((self.gh, self.gw), bool), z, z.copy()
        P = self.patch
        dy, dx = fit_translation(frame, self.ref, P, self.delta_max, self.iters)
        Wimg = warp(self.ref, dy, dx, P)
        d = patch_grid(frame.astype(np.float64) - Wimg, P)
        c = 0.5 * (d.max((2, 3)) + d.min((2, 3)))
        e = d - c[..., None, None]
        if self.multiscale is None:
            score = 2.0 * np.abs(e).max((2, 3)) / self.delta           # = spread / delta
            spread = 2.0 * np.abs(e).max((2, 3))
        else:
            score = np.zeros((self.gh, self.gw))
            for r, D in self.multiscale.items():
                score = np.maximum(score, box_max_abs_mean(e, r) / (0.5 * D))
            spread = score
        keep = score >= 1.0
        # the model's view: warped reference + shift on dropped patches, truth on kept
        view = Wimg + np.kron(np.where(keep, 0.0, c), np.ones((P, P)))
        fg = patch_grid(frame, P)
        vg = patch_grid(view, P)
        vg[keep] = fg[keep]
        self._view = view
        rg = patch_grid(self.ref, P)
        rg[keep] = fg[keep]
        self.last_motion = (np.where(keep, 0, dy).astype(np.float32), np.where(keep, 0, dx).astype(np.float32))
        return keep, spread.astype(np.float32), c.astype(np.float32)

    def run(self, frames):
        ks, ss, cs = [], [], []
        for f in frames:
            k, s, c = self.step(f)
            ks.append(k); ss.append(s); cs.append(c)
        return {"keep": np.stack(ks), "spread": np.stack(ss), "shift": np.stack(cs)}

    def reference(self):
        return self.ref.copy()

    def view(self, compensate_shift=True):
        v = self._view if compensate_shift else self.ref.astype(np.float64)
        if self.ref.dtype.kind in "ui":
            info = np.iinfo(self.ref.dtype)
            v = np.clip(np.rint(v), info.min, info.max)
        return v.astype(self.ref.dtype)
