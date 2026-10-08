"""Multi-scale box certificate.

Statistic.  For the shift-removed change e = F - R - c (c = midrange of d so
that scale 0 reproduces the range rule) and box half-widths r in R:

    M_r(e) = max over (2r+1)x(2r+1) boxes B inside the patch of | mean_B e |

Rule.  Drop iff M_r(e) < eps_r for every r in R.  With eps_0 = Delta/2 and
R = {0} this is exactly the range rule.

Why.  Noise of per-pixel std sigma_d has box-mean std sigma_d/sqrt(n_r); a
compact object of contrast Delta containing a (2r+1)-box has box mean Delta.
A one-pixel-wide jittering edge in a (2r+1)-box contributes only 1/(2r+1) of
its contrast.  So larger boxes separate objects from noise and jitter far
better than single pixels, at the price of only certifying objects that
contain a box of that size.

Guarantees while a patch is dropped (all deterministic, about the actual
frames):
  * every (2r+1)-box average of the compensated view R + c is within eps_r of
    the truth;
  * no object of contrast >= 2 eps_r that contains a (2r+1)-box can appear,
    vanish, split or merge, in the following precise sense: the persistence
    diagrams of the (2r+1)-box-filtered frames of F and R + c are within
    bottleneck distance eps_r (stability applied to the filtered images),
    so no feature of the filtered image with persistence >= 2 eps_r is born
    or dies.
Tightness.  A change equal to 2 eps_r on exactly one (2r+1)-box (midrange
removed it is +-eps_r) has M_r = eps_r and creates a 2 eps_r object, so no
rule that looks only at the M-statistics can drop at M_r >= eps_r.

Default schedule.  Delta_r = Delta_0 / sqrt(n_r), i.e. thresholds fall like
the noise does; a single knob Delta_0 moves the whole curve.
"""
from __future__ import annotations

import numpy as np

from .core import patch_grid


def box_max_abs_mean(e: np.ndarray, r: int) -> np.ndarray:
    """max over all (2r+1)-boxes fully inside the patch of |mean e|, for a stack e[..., P, P]."""
    if r == 0:
        return np.abs(e).max(axis=(-2, -1))
    k = 2 * r + 1
    P = e.shape[-1]
    if k > P:
        return np.zeros(e.shape[:-2])
    S = np.zeros(e.shape[:-2] + (P + 1, P + 1), np.float64)
    S[..., 1:, 1:] = e.cumsum(axis=-2).cumsum(axis=-1)
    box = S[..., k:, k:] - S[..., :-k, k:] - S[..., k:, :-k] + S[..., :-k, :-k]
    return np.abs(box).max(axis=(-2, -1)) / (k * k)


def box_filter(img: np.ndarray, r: int) -> np.ndarray:
    """(2r+1)-box mean filter of a 2-D image, valid region only."""
    if r == 0:
        return np.asarray(img, np.float64)
    k = 2 * r + 1
    S = np.zeros((img.shape[0] + 1, img.shape[1] + 1), np.float64)
    S[1:, 1:] = np.asarray(img, np.float64).cumsum(0).cumsum(1)
    return (S[k:, k:] - S[:-k, k:] - S[k:, :-k] + S[:-k, :-k]) / (k * k)


def default_schedule(delta0: float, radii=(0, 1, 2, 3)) -> dict:
    """Delta_r = Delta_0 / sqrt(n_r) with n_r = (2r+1)^2."""
    return {r: delta0 / (2 * r + 1) for r in radii}


class MultiScalePruner:
    """Same interface as core.Pruner.  `deltas` maps box half-width r -> Delta_r
    (object-contrast threshold; eps_r = Delta_r / 2)."""

    def __init__(self, height, width, patch=16, deltas=None, delta0: float = 64.0, radii=(0, 1, 2, 3)):
        self.H, self.W, self.patch = int(height), int(width), int(patch)
        self.deltas = dict(deltas) if deltas is not None else default_schedule(delta0, radii)
        self.gh, self.gw = self.H // patch, self.W // patch
        self.ref = None
        self.last_shift = np.zeros((self.gh, self.gw), np.float32)

    @property
    def initialised(self):
        return self.ref is not None

    def reset(self):
        self.ref = None
        self.last_shift[:] = 0

    def scores(self, frame):
        acc = np.float32 if frame.dtype.kind == "f" else np.int32
        d = patch_grid(frame.astype(acc) - self.ref.astype(acc), self.patch).astype(np.float64)
        c = 0.5 * (d.max(axis=(2, 3)) + d.min(axis=(2, 3)))
        e = d - c[..., None, None]
        # normalised score: max_r M_r / eps_r  (>= 1 means keep)
        score = np.zeros((self.gh, self.gw))
        for r, D in self.deltas.items():
            score = np.maximum(score, box_max_abs_mean(e, r) / (0.5 * D))
        return score.astype(np.float32), c.astype(np.float32)

    def step(self, frame):
        frame = np.ascontiguousarray(frame)
        if self.ref is None:
            self.ref = frame.copy()
            z = np.zeros((self.gh, self.gw), np.float32)
            return np.ones((self.gh, self.gw), bool), z, z.copy()
        score, c = self.scores(frame)
        keep = score >= 1.0
        rg = patch_grid(self.ref, self.patch)
        rg[keep] = patch_grid(frame, self.patch)[keep]
        self.last_shift = np.where(keep, 0.0, c).astype(np.float32)
        return keep, score, c

    def run(self, frames):
        ks, ss, cs = [], [], []
        for f in frames:
            k, s, c = self.step(f)
            ks.append(k); ss.append(s); cs.append(c)
        return {"keep": np.stack(ks), "spread": np.stack(ss), "shift": np.stack(cs)}

    def reference(self):
        return self.ref.copy()

    def view(self, compensate_shift=True):
        v = self.ref.astype(np.float32)
        if compensate_shift:
            P = self.patch
            v[: self.gh * P, : self.gw * P] += np.kron(self.last_shift, np.ones((P, P), np.float32))
        if self.ref.dtype.kind in "ui":
            info = np.iinfo(self.ref.dtype)
            v = np.clip(np.rint(v), info.min, info.max)
        return v.astype(self.ref.dtype)
