"""Sequential (quickest-detection) rule: evidence accumulates over time.

Every skip rule so far is memoryless: it compares one frame with the reference
and can never see a change fainter than one frame's noise.  Sequential
analysis (Page 1954; Lorden 1971; Moustakides 1986; Lai 1998) is the theory of
detecting a persistent change as early as possible for a given false-alarm
rate, and it says how: accumulate the log-likelihood of "changed" against
"unchanged" and stop when it crosses a threshold.  With the change's time,
place, size and magnitude unknown, the optimal procedure is the window-limited
generalised likelihood ratio (Lai 1998), which for a mean shift in Gaussian
noise is a *scan statistic*: the largest normalised two-sample difference over
a family of time windows and spatial boxes.  Arias-Castro, Donoho and Huo
(2005) prove such multiscale scans are minimax-optimal for detecting
geometric objects in noise.

Statistic.  Let e_t be the quotient residual image of frame t (F_t minus the
sub-pixel-warped, shift-compensated reference).  For a window length w and
box half-width r, with A = mean of e over the last w frames and B = mean over
the w frames before those,

    Z_{w,r}(patch) = max over (2r+1)-boxes |mean_box (A - B)| * (2r+1) * sqrt(w/2) / sigma

is the z-score of a Haar-in-time, box-in-space test.  Fire (keep) iff
max_{w,r} Z_{w,r} >= z.  Reference noise and any static bias cancel in A - B;
independent noise averages down by sqrt(w); a persistent object does not.

Guarantees (under independent noise of per-pixel std sigma, estimated per
patch from temporal differences):
  * false fires: by the union bound over the N tests per patch and frame,
    P(fire | no change) <= N * (1 - Phi(z)); z = 4.5 gives about 1e-2 per
    patch-frame for N ~ 2000 (the a-contrario "number of false alarms");
  * delay: an object of contrast Delta containing a (2r+1)-box that appears
    at frame nu and persists is kept by frame nu + w - 1 for the smallest
    w in W with Delta (2r+1) sqrt(w/2) >= (z + z') sigma, where z' sets the
    miss probability; with dyadic W this is within a factor 2 of the
    minimax-optimal delay (no procedure can detect reliably while
    Delta (2r+1) sqrt(w) is below a constant times sigma sqrt(log N)).

Self-normalisation.  Real camera noise is neither white nor Gaussian: codec
noise is spatially blocky and drifts in time, analog noise is correlated
along scan lines.  Measured on three cameras, the per-pixel z-scores were
fine but box averages at long windows were 2-3x wider than independence
predicts.  So each (w, r) statistic is studentised by its own running robust
scale, estimated per patch from quiet frames (the classical self-normalised
sequential test); the threshold z then means "z robust standard deviations of
this patch's own fluctuation at this time scale", whatever the noise
spectrum.  A planar illumination term (1, x, y) is projected out per patch
before accumulation so smooth lighting drift is not mistaken for an object.

The sequential test only ADDS keeps to a memoryless certified rule, so every
dropped patch still carries that rule's worst-case certificate.  What the
sequential part adds is statistical: faint persistent changes are caught
within a provable delay.
"""
from __future__ import annotations

import numpy as np

from .core import patch_grid
from .scalespace import box_max_abs_mean
from .warp import WarpPruner, fit_translation, warp


class SequentialPruner(WarpPruner):
    """Quotient + multi-scale memoryless certificate, plus the sequential
    space-time scan.  Interface as core.Pruner; step() returns (keep, score,
    shift) where score = max(memoryless score, sequential z / z_thr)."""

    def __init__(self, height, width, patch=16, multiscale: dict | None = None,
                 delta=32.0, delta_max=1.0, iters=2,
                 windows=(1, 2, 4, 8, 16), radii=(1, 2, 3), z=4.5,
                 sigma_floor=0.5, sigma_alpha=0.1, scale_floor=0.5, veto_frac=0.1, offset: str = "patch"):
        super().__init__(height, width, patch, delta, delta_max, iters, multiscale, offset)
        self.windows = tuple(int(w) for w in windows)
        self.radii = tuple(int(r) for r in radii)
        self.z = float(z)
        self.sigma_floor, self.sigma_alpha = float(sigma_floor), float(sigma_alpha)
        # scale_floor: the null spread of a box-mean statistic cannot be resolved below a fraction of a
        # quantisation step; without it, codec-clean blocks that repeat exactly give a zero spread and the
        # first periodic requantisation (GOP breathing) fires every patch at once.
        # veto_frac: if more than this fraction of patches fire sequentially in one frame, the event is
        # scene-wide (illumination, keyframe) and not an object: no keeps, and the frame trains the null.
        self.scale_floor, self.veto_frac = float(scale_floor), float(veto_frac)
        self.L = 2 * max(self.windows) + 1
        Hg, Wg = self.gh * patch, self.gw * patch
        self._csum = np.zeros((self.L, Hg, Wg), np.float64)   # ring buffer of temporal prefix sums
        self._t = -1
        self._t0 = np.zeros((self.gh, self.gw), np.int64)       # last keep time per patch
        self._prev_e = None
        self._sigma = np.full((self.gh, self.gw), np.nan)
        self.last_seq = None
        # self-normalisation of the extreme statistic: running location and spread of
        # M_{w,r} = max over boxes |mean_box(A - B)| per patch, learned on quiet frames.
        # Nonparametric in the noise's spatial correlation (which biases any
        # position-wise scale estimate) and in its temporal spectrum.
        self._loc = {(w, r): np.full((self.gh, self.gw), np.nan) for w in self.windows for r in self.radii}
        self._scale = {(w, r): np.full((self.gh, self.gw), np.nan) for w in self.windows for r in self.radii}
        self._nscale = {(w, r): np.zeros((self.gh, self.gw), np.int64) for w in self.windows for r in self.radii}
        self.min_calib = 8
        # planar illumination basis (1, x, y), orthonormal over the patch
        yy, xx = np.mgrid[0:patch, 0:patch].astype(np.float64)
        Q, _ = np.linalg.qr(np.stack([np.ones(patch * patch), xx.ravel(), yy.ravel()], 1))
        self._Q = Q

    def reset(self):
        super().reset()
        self._csum[:] = 0; self._t = -1; self._t0[:] = 0; self._prev_e = None; self._sigma[:] = np.nan
        for k in self._scale:
            self._scale[k][:] = np.nan; self._loc[k][:] = np.nan; self._nscale[k][:] = 0

    def _C(self, t):
        return self._csum[t % self.L]

    def step(self, frame):
        frame = np.ascontiguousarray(frame)
        P = self.patch
        Hg, Wg = self.gh * P, self.gw * P
        self._t += 1
        t = self._t
        if self.ref is None:
            self.ref = frame.copy(); self._view = frame.astype(np.float64)
            z = np.zeros((self.gh, self.gw), np.float32)
            self.last_motion = (z, z); self.last_seq = z.copy()
            self._C(t)[:] = 0
            self._t0[:] = 0
            return np.ones((self.gh, self.gw), bool), z, z.copy()

        # ---- memoryless part (quotient + multi-scale), as in WarpPruner.step
        dy, dx = fit_translation(frame, self.ref, P, self.delta_max, self.iters)
        Wimg = warp(self.ref, dy, dx, P)
        d = patch_grid(frame.astype(np.float64) - Wimg, P)
        c_mid = 0.5 * (d.max((2, 3)) + d.min((2, 3))) if self.offset == "patch" else np.zeros((self.gh, self.gw))
        e_mid = d - c_mid[..., None, None]
        if self.multiscale is None:
            score = 2.0 * np.abs(e_mid).max((2, 3)) / self.delta
        else:
            score = np.zeros((self.gh, self.gw))
            for r, D in self.multiscale.items():
                score = np.maximum(score, box_max_abs_mean(e_mid, r) / (0.5 * D))

        # ---- sequential part on the mean-shift-removed residual
        flat = d.reshape(self.gh, self.gw, P * P)
        e = (flat - (flat @ self._Q) @ self._Q.T).reshape(self.gh, self.gw, P, P)   # remove planar illumination
        e_img = e.swapaxes(1, 2).reshape(Hg, Wg)
        # per-patch noise from temporal differences of the residual (valid only if same reference)
        if self._prev_e is not None:
            same_ref = self._t0 < t - 1                                    # no keep at t-1
            tau = e - self._prev_e
            med = np.median(tau, axis=(2, 3), keepdims=True)
            s_now = 1.4826 * np.median(np.abs(tau - med), axis=(2, 3)) / np.sqrt(2.0)
            s_now = np.maximum(s_now, self.sigma_floor)
            upd = same_ref
            self._sigma = np.where(upd & np.isnan(self._sigma), s_now, self._sigma)
            self._sigma = np.where(upd & ~np.isnan(self._sigma),
                                   (1 - self.sigma_alpha) * self._sigma + self.sigma_alpha * s_now, self._sigma)
        self._prev_e = e
        self._C(t)[:] = self._C(t - 1) + e_img
        zbest = np.zeros((self.gh, self.gw))
        pending = []                                                        # (w, r, valid, mad) for scale updates
        for w in self.windows:
            if t - 2 * w < 0:
                continue
            valid = (t - 2 * w) >= self._t0                                 # both windows after the last keep
            if not valid.any():
                continue
            A = (self._C(t) - self._C(t - w)) / w
            B = (self._C(t - w) - self._C(t - 2 * w)) / w
            Dw = patch_grid(A - B, P)
            for r in self.radii:
                M = box_max_abs_mean(Dw, r)                                 # extreme statistic per patch
                loc = self._loc[(w, r)]; sc = self._scale[(w, r)]; n = self._nscale[(w, r)]
                ok = valid & (n >= self.min_calib)
                zz = np.where(ok, (M - loc) / np.maximum(sc, self.scale_floor), 0.0)
                zbest = np.maximum(zbest, zz)
                pending.append((w, r, valid, M))
        self.last_seq = zbest.astype(np.float32)
        seq_keep = zbest >= self.z
        self.vetoed = bool(seq_keep.mean() >= self.veto_frac)
        if self.vetoed:
            seq_keep[:] = False                      # scene-wide: not objects; the frame trains the null below
        # learn location/spread of the extreme statistic on quiet patches only
        quiet = ~seq_keep & (score < 1.0)
        a = self.sigma_alpha
        for w, r, valid, M in pending:
            loc = self._loc[(w, r)]; sc = self._scale[(w, r)]; n = self._nscale[(w, r)]
            upd = valid & quiet
            first = upd & np.isnan(loc)
            loc[first] = M[first]; sc[first] = np.maximum(0.25 * M[first], 1e-3)
            later = upd & ~first
            dev = np.abs(M - loc)
            loc[later] = (1 - a) * loc[later] + a * M[later]
            sc[later] = np.maximum((1 - a) * sc[later] + a * dev[later], 1e-3)      # the floor is applied at use
            n[upd] += 1

        keep = (score >= 1.0) | seq_keep
        total = np.maximum(score, zbest / self.z)

        # ---- bookkeeping: view, reference, keep times
        view = Wimg + np.kron(np.where(keep, 0.0, c_mid), np.ones((P, P)))
        fg = patch_grid(frame, P); vg = patch_grid(view, P); vg[keep] = fg[keep]
        self._view = view
        rg = patch_grid(self.ref, P); rg[keep] = fg[keep]
        self._t0 = np.where(keep, t, self._t0)
        # scales are properties of the camera noise, not of the reference: keep them across keeps
        self.last_motion = (np.where(keep, 0, dy).astype(np.float32), np.where(keep, 0, dx).astype(np.float32))
        return keep, total.astype(np.float32), c_mid.astype(np.float32)


def _box_means(e, r):
    """All (2r+1)-box means inside each patch: e[..., P, P] -> (..., n_pos)."""
    k = 2 * r + 1
    Pp = e.shape[-1]
    S = np.zeros(e.shape[:-2] + (Pp + 1, Pp + 1), np.float64)
    S[..., 1:, 1:] = e.cumsum(axis=-2).cumsum(axis=-1)
    box = S[..., k:, k:] - S[..., :-k, k:] - S[..., k:, :-k] + S[..., :-k, :-k]
    return (box / (k * k)).reshape(e.shape[:-2] + (-1,))


def delay_bound(delta, r, sigma, z, z_miss=2.0, windows=(1, 2, 4, 8, 16), scales=None):
    """Smallest window in `windows` guaranteeing detection of a persistent
    object of contrast `delta` containing a (2r+1)-box.  With white noise of
    per-pixel std `sigma` the scale of the (w, r) statistic is
    sigma*sqrt(2/w)/(2r+1); with `scales` = {w: measured scale} (the
    self-normalised case) those are used instead.  None if no window suffices."""
    for w in windows:
        sc = scales[w] if scales is not None else sigma * np.sqrt(2.0 / w) / (2 * r + 1)
        if delta >= (z + z_miss) * sc:
            return w
    return None
