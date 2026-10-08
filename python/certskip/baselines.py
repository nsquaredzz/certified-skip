"""Simplified versions of published skip rules, for comparison only.

None of these carries a guarantee.  They are the rules the certified rule is
measured against, each reduced to its decision statistic:

consecutive_mean   mean |F_t - F_{t-1}| per patch < tau  -> drop
                   (Efficient Video Sampling, TimeChat-Online differential
                   token dropping, run-length tokenisation all threshold a
                   consecutive-frame average)
mean_vs_kept       mean |F_t - R| per patch < tau -> drop, R = last kept copy
                   (same statistic as above but with the certified rule's
                   reference, to separate "statistic" from "reference")
uniform            keep every k-th frame in full, drop the rest

All functions take frames as a (T, H, W) array and return keep as a
(T, gh, gw) bool array with frame 0 fully kept.
"""
from __future__ import annotations

import numpy as np

from .core import patch_grid, Pruner, drop_rate


def _grid_mean_abs(a: np.ndarray, b: np.ndarray, patch: int) -> np.ndarray:
    d = np.abs(a.astype(np.float32) - b.astype(np.float32))
    return patch_grid(d, patch).mean(axis=(2, 3))


def consecutive_mean_scores(frames: np.ndarray, patch: int) -> np.ndarray:
    """(T-1, gh, gw) mean absolute consecutive difference."""
    return np.stack([_grid_mean_abs(frames[t], frames[t - 1], patch) for t in range(1, len(frames))])


def consecutive_mean(frames: np.ndarray, patch: int, tau: float, scores=None) -> np.ndarray:
    if scores is None:
        scores = consecutive_mean_scores(frames, patch)
    T, gh, gw = len(frames), scores.shape[1], scores.shape[2]
    keep = np.ones((T, gh, gw), bool)
    keep[1:] = scores >= tau
    return keep


def mean_vs_kept(frames: np.ndarray, patch: int, tau: float) -> np.ndarray:
    T, H, W = frames.shape
    gh, gw = H // patch, W // patch
    keep = np.ones((T, gh, gw), bool)
    ref = frames[0].copy()
    rg = patch_grid(ref, patch)
    for t in range(1, T):
        k = _grid_mean_abs(frames[t], ref, patch) >= tau
        keep[t] = k
        rg[k] = patch_grid(frames[t], patch)[k]
    return keep


def uniform(T: int, gh: int, gw: int, stride: int) -> np.ndarray:
    keep = np.zeros((T, gh, gw), bool)
    keep[::stride] = True
    return keep


def certified(frames: np.ndarray, patch: int, delta: float, max_shift=None, pruner_cls=Pruner) -> np.ndarray:
    """The certified rule in the same (frames, patch, threshold) -> keep shape."""
    T, H, W = frames.shape
    pr = pruner_cls(H, W, patch, delta, max_shift) if pruner_cls is Pruner else pruner_cls(H, W, patch, delta, max_shift, dtype=frames.dtype)
    return pr.run(frames)["keep"]


def calibrate(rule, frames: np.ndarray, patch: int, target_drop: float,
              lo: float = 0.0, hi: float = 255.0, iters: int = 30):
    """Bisection on the threshold so that rule(frames, patch, thr) drops
    approximately `target_drop` of patches (frame 0 excluded).  Returns
    (threshold, keep, achieved_drop).  Drop rate is monotone in the threshold
    for every rule in this module."""
    best = None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        keep = rule(frames, patch, mid)
        d = drop_rate(keep)
        if best is None or abs(d - target_drop) < abs(best[2] - target_drop):
            best = (mid, keep, d)
        if d < target_drop:
            lo = mid
        else:
            hi = mid
    return best


def uniform_for_drop(T: int, gh: int, gw: int, target_drop: float) -> np.ndarray:
    """Uniform sampling with the stride closest to the requested drop rate."""
    stride = max(1, int(round(1.0 / max(1e-9, 1.0 - target_drop))))
    return uniform(T, gh, gw, stride)
