"""What to hold: the fewest sends a certificate allows, and a rule that comes close.

Every rule in this package holds a frame: when the certificate breaks, the current frame is sent and held until
it breaks again. THEORY.md section 11 asks what the fewest sends are for a given certificate, whatever is held,
and answers it with the geometry of the certificate's norm.

Setting, per patch: frames x_0, x_1, ... and a held copy h_t that changes only when something is sent, with the
certificate ||x_t - h_t|| < 1 at every frame. The norm is the multi-scale norm without offset,

    ||e|| = max over r in the schedule, over (2r+1)-boxes B inside the patch, of |mean_B e| / eps_r,   eps_r = Delta_r / 2,

which for the schedule {0: Delta} is the sup norm divided by Delta / 2. `features` is the linear map e -> (every
box mean) and `thresholds` the eps_r of each; the norm of e is the largest |feature| / threshold, so the patch
space sits isometrically inside a (weighted) sup-norm space, and distances between frames are read off the
features coordinate by coordinate. Features are kept in grey levels so that a tie with a threshold is exact.

  fewest_sends   A run of frames can share one held copy only if its diameter is below 2 (Theorem 13). Splitting
                 the clip greedily into such runs counts a number of sends that no policy can beat, with any
                 latency, holding anything. For the sup norm the bound is attained: hold each run's midrange.
  CentrePruner   On a break at frame s, look at frames s..s+L and hold the centre (per-pixel midrange) of the
                 longest run s..s+j that the centre certifies. L = 0 is send-on-delta: the current frame is held.
                 The model's picture lags L frames; the certificate is the same at every L (Theorem 16).

Send-on-delta is never better than `fewest_sends` at tolerance 1 and never worse than it at tolerance 1/2
(Theorem 14); `tests/test_centre.py` checks all of it on data.
"""
from __future__ import annotations

import numpy as np

from .core import patch_grid


def thresholds(patch: int, schedule: dict) -> np.ndarray:
    """eps_r of every feature, in the order `features` lists them."""
    return np.concatenate([np.full((patch - 2 * r) ** 2, 0.5 * D) for r, D in schedule.items() if 2 * r + 1 <= patch])


def features(img, patch: int, schedule: dict) -> np.ndarray:
    """(Hg, Wg) image -> (gh, gw, n): per patch, every box mean of the schedule, in grey levels."""
    g = patch_grid(np.asarray(img, np.float64), patch)
    gh, gw, P = g.shape[0], g.shape[1], patch
    out = []
    for r, D in schedule.items():
        k = 2 * r + 1
        if k == 1:
            b = g.reshape(gh, gw, -1)
        elif k > P:
            continue
        else:
            S = np.zeros((gh, gw, P + 1, P + 1)); S[..., 1:, 1:] = g.cumsum(-2).cumsum(-1)
            b = ((S[..., k:, k:] - S[..., :-k, k:] - S[..., k:, :-k] + S[..., :-k, :-k]) / (k * k)).reshape(gh, gw, -1)
        out.append(b)
    return np.concatenate(out, -1)


def _schedule(multiscale, delta):
    return dict(multiscale) if multiscale else {0: float(delta)}


def fewest_sends(frames, patch: int = 16, multiscale: dict | None = None, delta: float = 32.0, tolerance: float = 1.0) -> np.ndarray:
    """Per patch, the number of greedy runs of diameter < 2 * tolerance: a lower bound on the sends of any
    certified policy at that tolerance (exact for the sup norm), first frame included."""
    sched = _schedule(multiscale, delta); thr = thresholds(patch, sched)
    H, W = frames[0].shape; gh, gw = H // patch, W // patch
    f = features(frames[0][: gh * patch, : gw * patch], patch, sched); hi, lo = f.copy(), f.copy()
    n = np.ones((gh, gw), int)
    for t in range(1, len(frames)):
        f = features(frames[t][: gh * patch, : gw * patch], patch, sched)
        hi2, lo2 = np.maximum(hi, f), np.minimum(lo, f)
        brk = ((hi2 - lo2) >= 2.0 * tolerance * thr).any(-1)
        hi = np.where(brk[..., None], f, hi2); lo = np.where(brk[..., None], f, lo2); n += brk
    return n


class CentrePruner:
    """Hold the centre of the next frames. run() returns keep[T, gh, gw] (a patch is sent on frame s) and the
    largest certificate norm met on any frame, which the certificate says is below 1. The held copy is the
    midrange itself, so it can end in .5; an 8-bit copy of it is half a grey level further from the truth."""

    def __init__(self, height, width, patch=16, multiscale: dict | None = None, delta: float = 32.0, lookahead: int = 0):
        self.H, self.W, self.patch = int(height), int(width), int(patch)
        self.schedule = _schedule(multiscale, delta); self.lookahead = int(lookahead)
        self.thr = thresholds(self.patch, self.schedule)
        self.gh, self.gw = self.H // patch, self.W // patch

    def _spread(self, mask):
        return np.kron(mask, np.ones((self.patch, self.patch), bool))

    def run(self, frames, on_frame=None):
        P, L, gh, gw, thr = self.patch, self.lookahead, self.gh, self.gw, self.thr
        T = len(frames); Hg, Wg = gh * P, gw * P
        crop = lambda t: np.asarray(frames[t][:Hg, :Wg], np.float64)
        cache = {}

        def feat(t):
            if t not in cache:
                cache[t] = features(crop(t), P, self.schedule)
            return cache[t]

        held = fheld = None; keep = np.zeros((T, gh, gw), bool); worst = 0.0
        for s in range(T):
            cache.pop(s - 1, None)
            x, fx = crop(s), feat(s)
            brk = np.ones((gh, gw), bool) if held is None else (np.abs(fx - fheld) >= thr).any(-1)
            if brk.any():
                hi, lo, c, fc, fhi, flo = x, x, x.copy(), fx.copy(), fx, fx
                alive = brk.copy()
                for j in range(1, L + 1):
                    if s + j >= T or not alive.any():
                        break
                    g, fg = crop(s + j), feat(s + j)
                    hi2, lo2 = np.maximum(hi, g), np.minimum(lo, g)
                    c2 = 0.5 * (hi2 + lo2); fc2 = features(c2, P, self.schedule)
                    fhi2, flo2 = np.maximum(fhi, fg), np.minimum(flo, fg)
                    # the centre must certify every frame it stands for: every feature of every frame within its threshold
                    ok = alive & ((fhi2 - fc2 < thr) & (fc2 - flo2 < thr)).all(-1)
                    m, mf = self._spread(ok), ok[..., None]
                    hi, lo, c = np.where(m, hi2, hi), np.where(m, lo2, lo), np.where(m, c2, c)
                    fc, fhi, flo = np.where(mf, fc2, fc), np.where(mf, fhi2, fhi), np.where(mf, flo2, flo)
                    alive = ok
                m = self._spread(brk)
                held = c if held is None else np.where(m, c, held)
                fheld = fc if fheld is None else np.where(brk[..., None], fc, fheld)
            keep[s] = brk
            worst = max(worst, float((np.abs(fx - fheld) / thr).max()))
            if on_frame is not None:
                on_frame(s, brk, held)
        return {"keep": keep, "worst": worst}
