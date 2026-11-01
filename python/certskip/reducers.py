"""Decision rules of published video token reducers, reduced to what they compare.

Each function maps frames (T, H, W) and a threshold to a keep mask (T, gh, gw) with frame 0 kept,
the interface of baselines.py, so that baselines.calibrate can set every rule to the same budget.
None of these is the published system: the model-side parts (ViT features, attention scores, token
merging inside the LLM) are replaced by the pixel-space statistic that plays the same role, and the
docstring of each says what is kept and what is not.  What the audit tests is the *temporal
comparison structure* of each method: against which frame a token is judged redundant.

  consecutive          mean |F_t - F_{t-1}| per patch < tau            EVS, TimeChat-Online, and the
                                                                        pixel-L1 test of run-length
                                                                        tokenization (RLT) exactly
  cosine_consecutive   cosine similarity of patch vectors of F_t and    the merge test of DyCoke (TTM),
                       F_{t-1} > 1 - tau; vectors are mean-removed      FrameFusion (adjacent-frame
                       random projections, a proxy for a linear patch   merging) and PruneVid's static
                       embedding                                        tokens, which use ViT/LLM features
  dycoke               cosine_consecutive within groups of `group`      DyCoke keeps the first frame of
                       frames; the first frame of each group is kept    every group of 4 sampled frames
  segment_static       within fixed segments of `seg` frames, a patch   PruneVid: static tokens of a
                       is static iff mean |F_t - F_seg0| < tau for the   temporal segment are merged
                       segment's first frame                            into one
  frame_consecutive    the whole frame is dropped iff the frame-level   LongVU's temporal reduction by
                       mean |F_t - F_{t-1}| < tau                        adjacent-frame feature similarity
  consecutive_refresh  consecutive with a full refresh every `period`   the periodic-keyframe variant
                       frames                                           the write-up listed as missing
  mean_vs_kept         mean |F_t - R| < tau against the last kept copy  anchored, no certificate
  uniform              every k-th frame in full                         frame sampling at a fixed rate
  attention_stub       raises: FastV / VisionZip / SparseVLM prune by   needs the model; the harness
                       attention inside the model and have no temporal  accepts a precomputed score
                       statistic to re-implement in pixel space         file for them instead

Proposition E (consecutive comparison cannot see slow change).  Let a rule decide skip at (t, patch)
by sigma(F_t - F_{t-1}) < tau for a seminorm sigma, and let the patch carry a change whose per-frame
increment i_t satisfies sigma(i_t) <= tau - sigma(n_t - n_{t-1}) where n is the noise.  Then the rule
skips the patch on every frame of the change, whatever the accumulated change becomes: by the
triangle inequality sigma(F_t - F_{t-1}) <= sigma(i_t) + sigma(n_t - n_{t-1}) < tau.  With noise that
occasionally exceeds tau - sigma(i_t), the rule fires on the fading patch at that patch's own
static false-fire rate rho, which is independent of the change, so a fade of T frames goes unseen
with probability (1 - rho)^T, and a budget of drop rate D on static footage bounds the average rho
by 1 - D.  The anchored rule sends the patch as soon as the accumulated change reaches its
threshold.  Both statements are measured in scripts/audit_reducers.py.
"""
from __future__ import annotations

import numpy as np

from .core import patch_grid
from .baselines import consecutive_mean, consecutive_mean_scores, mean_vs_kept, uniform, uniform_for_drop  # noqa: F401

consecutive = consecutive_mean


# ---------------------------------------------------------------- patch vectors for cosine rules
_PROJ = {}


def _patch_vectors(frames: np.ndarray, patch: int, dim: int = 64, seed: int = 0) -> np.ndarray:
    """(T, gh, gw, dim) mean-removed random linear projection of every patch: a fixed, untrained
    stand-in for a linear patch embedding.  Cosine similarity between such vectors behaves like the
    cosine similarity of ViT patch embeddings in that it ignores brightness and compares texture."""
    key = (patch, dim, seed)
    if key not in _PROJ:
        rng = np.random.default_rng(seed)
        _PROJ[key] = rng.standard_normal((patch * patch, dim)) / np.sqrt(patch * patch)
    Wp = _PROJ[key]
    g = patch_grid(frames.astype(np.float32), patch) if frames.ndim == 2 else np.stack([patch_grid(f.astype(np.float32), patch) for f in frames])
    v = g.reshape(g.shape[:-2] + (patch * patch,))
    v = v - v.mean(-1, keepdims=True)
    return v @ Wp.astype(np.float32)


def cosine_consecutive_scores(frames: np.ndarray, patch: int) -> np.ndarray:
    """(T-1, gh, gw): 1 - cosine similarity between each patch and the same patch one frame earlier."""
    v = _patch_vectors(frames, patch)
    n = np.linalg.norm(v, axis=-1) + 1e-6
    cos = (v[1:] * v[:-1]).sum(-1) / (n[1:] * n[:-1])
    return 1.0 - cos


def cosine_consecutive(frames: np.ndarray, patch: int, tau: float, scores=None) -> np.ndarray:
    if scores is None:
        scores = cosine_consecutive_scores(frames, patch)
    T = len(frames)
    keep = np.ones((T,) + scores.shape[1:], bool)
    keep[1:] = scores >= tau
    return keep


def dycoke(frames: np.ndarray, patch: int, tau: float, group: int = 4, scores=None) -> np.ndarray:
    keep = cosine_consecutive(frames, patch, tau, scores)
    keep[::group] = True
    return keep


def segment_static(frames: np.ndarray, patch: int, tau: float, seg: int = 16) -> np.ndarray:
    T, H, W = frames.shape
    gh, gw = H // patch, W // patch
    keep = np.ones((T, gh, gw), bool)
    for s0 in range(0, T, seg):
        ref = frames[s0].astype(np.float32)
        for t in range(s0 + 1, min(T, s0 + seg)):
            d = np.abs(frames[t].astype(np.float32) - ref)
            keep[t] = patch_grid(d, patch).mean(axis=(2, 3)) >= tau
    return keep


def frame_consecutive(frames: np.ndarray, patch: int, tau: float) -> np.ndarray:
    T, H, W = frames.shape
    gh, gw = H // patch, W // patch
    keep = np.ones((T, gh, gw), bool)
    for t in range(1, T):
        m = np.abs(frames[t].astype(np.float32) - frames[t - 1].astype(np.float32)).mean()
        keep[t] = m >= tau
    return keep


def consecutive_refresh(frames: np.ndarray, patch: int, tau: float, period: int = 60, scores=None) -> np.ndarray:
    keep = consecutive_mean(frames, patch, tau, scores)
    keep[::period] = True
    return keep


def attention_stub(frames, patch, tau):
    raise NotImplementedError("FastV / VisionZip / SparseVLM prune by attention inside the model; "
                              "supply per-patch scores from the model through audit_reducers.py --scores")


# registry: name -> (callable(frames, patch, thr) -> keep, threshold grid, what it stands for)
REGISTRY = {
    "consecutive (EVS/RLT)":        (consecutive,            None, "mean |F_t - F_{t-1}| < tau"),
    "cosine consecutive (DyCoke/FrameFusion)": (cosine_consecutive, None, "1 - cos(v_t, v_{t-1}) < tau on projected patches"),
    "dycoke groups of 4":           (dycoke,                 None, "cosine consecutive, first of every 4 kept"),
    "segment static (PruneVid)":    (segment_static,         None, "mean |F_t - F_seg0| < tau in 16-frame segments"),
    "frame consecutive (LongVU)":   (frame_consecutive,      None, "frame-level mean |F_t - F_{t-1}| < tau"),
    "consecutive + refresh/60":     (consecutive_refresh,    None, "consecutive, full refresh every 60 frames"),
    "mean vs last kept":            (mean_vs_kept,           None, "mean |F_t - R| < tau, R = last kept"),
}
