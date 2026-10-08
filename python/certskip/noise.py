"""Temporal-noise analysis: the gating experiment for real footage.

The certified rule thresholds the *range* of the patch difference, which is
the most noise-sensitive statistic possible.  For i.i.d. Gaussian pixel noise
with standard deviation sigma, the difference of two frames has standard
deviation sigma*sqrt(2) and the expected range over n = patch**2 samples is
roughly 5.3..6 times that, i.e. about 7.5*sigma for 14- to 16-pixel patches.
So delta must sit comfortably above ~8*sigma or nothing is dropped.

Functions here estimate sigma from footage, predict the spread distribution
and the resulting drop rate under the Gaussian model, and measure the actual
drop-rate-versus-delta curve, so the two can be compared.
"""
from __future__ import annotations

import numpy as np

from .core import drop_rate, certified_error
from .native import make_pruner


def estimate_temporal_noise(frames: np.ndarray, patch: int = 16, static_fraction: float = 0.5) -> dict:
    """Robust per-pixel temporal noise estimate.

    For each consecutive frame pair the difference is taken; on each patch a
    robust std (1.4826 * MAD) of the difference is computed; the most static
    `static_fraction` of patches (lowest robust std) are averaged so motion
    does not inflate the estimate.  Returns sigma_pixel = sigma_diff/sqrt(2)
    plus the distribution over patches."""
    from .core import patch_grid
    T = len(frames)
    if T < 2:
        raise ValueError("need at least two frames")
    acc = np.float32 if frames.dtype.kind == "f" else np.int32
    per_patch = []
    for t in range(1, T):
        d = frames[t].astype(acc) - frames[t - 1].astype(acc)
        g = patch_grid(d, patch).astype(np.float32)
        med = np.median(g, axis=(2, 3), keepdims=True)
        mad = np.median(np.abs(g - med), axis=(2, 3))
        per_patch.append(1.4826 * mad)
    pp = np.stack(per_patch)                              # (T-1, gh, gw) sigma of the difference
    flat = np.sort(pp.ravel())
    k = max(1, int(len(flat) * static_fraction))
    sigma_diff = float(flat[:k].mean())
    return {
        "sigma_pixel": sigma_diff / np.sqrt(2.0),
        "sigma_diff": sigma_diff,
        "sigma_diff_median_all_patches": float(np.median(flat)),
        "sigma_diff_p90_all_patches": float(np.quantile(flat, 0.9)),
        "per_patch_sigma_diff": pp,
    }


def simulate_spread(sigma_pixel: float, patch: int = 16, n: int = 20000, rng=None) -> np.ndarray:
    """Monte-Carlo samples of the spread of a patch difference under i.i.d.
    Gaussian noise (both frames noisy, scene static)."""
    rng = np.random.default_rng(0) if rng is None else rng
    d = rng.normal(0.0, sigma_pixel * np.sqrt(2.0), size=(n, patch * patch))
    return d.max(axis=1) - d.min(axis=1)


def predicted_drop_rate(sigma_pixel: float, delta: float, patch: int = 16, samples=None) -> float:
    """P(spread < delta) under the Gaussian model: the ceiling on the drop
    rate for a perfectly static scene with this noise level."""
    s = simulate_spread(sigma_pixel, patch) if samples is None else samples
    return float(np.mean(s < delta))


def delta_for_drop_rate(sigma_pixel: float, target: float, patch: int = 16) -> float:
    """Smallest delta whose Gaussian-model drop rate reaches `target`."""
    s = simulate_spread(sigma_pixel, patch)
    return float(np.quantile(s, target))


def sweep_delta(frames: np.ndarray, deltas, patch: int = 16, max_shift=None) -> list[dict]:
    """Measured drop rate and certified max view error for each delta."""
    T, H, W = frames.shape
    rows = []
    for d in deltas:
        pr = make_pruner(H, W, patch, float(d), max_shift, dtype=frames.dtype)
        out = pr.run(frames)
        rows.append({
            "delta": float(d),
            "drop_rate": drop_rate(out["keep"]),
            "certified_max_error": certified_error(out["keep"], out["spread"]),
            "per_frame_keep_fraction": out["keep"][1:].mean(axis=(1, 2)),
        })
    return rows


def keyframe_spikes(keep_fraction: np.ndarray, z: float = 6.0) -> dict:
    """Detect frames where the keep fraction jumps (typically codec I-frames,
    which re-quantise the whole picture).  Returns spike indices and an
    estimated period from the spike spacing."""
    kf = np.asarray(keep_fraction, float)
    if kf.size < 4:
        return {"spikes": np.array([], int), "period": None, "fraction_of_frames": 0.0}
    med = np.median(kf)
    mad = 1.4826 * np.median(np.abs(kf - med)) + 1e-9
    spikes = np.where(kf > med + z * mad)[0]
    period = None
    if len(spikes) >= 3:
        gaps = np.diff(spikes)
        period = float(np.median(gaps))
    return {"spikes": spikes, "period": period, "fraction_of_frames": len(spikes) / kf.size,
            "median_keep_fraction": float(med), "spike_mean_keep_fraction": float(kf[spikes].mean()) if len(spikes) else None}
