"""numpy reference implementation of the certified skip rule.

This file is deliberately small and literal.  The C++ core must agree with it
exactly (tests/test_core.py checks keep masks and spreads bit-for-bit on uint8
input), so if you change the rule change it here first.
"""
from __future__ import annotations

import numpy as np


def patch_grid(frame: np.ndarray, patch: int) -> np.ndarray:
    """View an (H, W) frame as (gh, gw, patch, patch), dropping the remainder."""
    H, W = frame.shape
    gh, gw = H // patch, W // patch
    f = frame[: gh * patch, : gw * patch]
    return f.reshape(gh, patch, gw, patch).swapaxes(1, 2)


def spread_and_shift(frame: np.ndarray, ref: np.ndarray, patch: int):
    """Per-patch spread = max(d) - min(d) and shift = (max(d) + min(d)) / 2,
    where d = frame - ref, computed in a wide dtype so uint8 cannot wrap."""
    acc = np.float32 if frame.dtype.kind == "f" else np.int32
    d = frame.astype(acc) - ref.astype(acc)
    g = patch_grid(d, patch)
    dmax = g.max(axis=(2, 3)).astype(np.float64)
    dmin = g.min(axis=(2, 3)).astype(np.float64)
    return (dmax - dmin).astype(np.float32), (0.5 * (dmax + dmin)).astype(np.float32)


class Pruner:
    """Stateful certified pruner.

    Parameters
    ----------
    height, width : frame size in pixels
    patch         : patch side; frame is tiled into (H//patch, W//patch) patches
    delta         : contrast level in the frame's units (grey levels for uint8)
    max_shift     : optional cap on |brightness shift|.  None keeps the pure
                    topological rule (a uniform brightness change is dropped
                    and should be compensated at reconstruction time).
    """

    def __init__(self, height: int, width: int, patch: int = 16,
                 delta: float = 32.0, max_shift: float | None = None):
        if patch <= 0 or patch > height or patch > width:
            raise ValueError("patch must be in 1..min(height, width)")
        self.H, self.W, self.patch = int(height), int(width), int(patch)
        self.delta, self.max_shift = float(delta), max_shift
        self.gh, self.gw = self.H // self.patch, self.W // self.patch
        self.ref: np.ndarray | None = None
        self.last_shift = np.zeros((self.gh, self.gw), np.float32)

    # -- state --------------------------------------------------------------
    def reset(self) -> None:
        self.ref = None
        self.last_shift[:] = 0

    @property
    def initialised(self) -> bool:
        return self.ref is not None

    def reference(self) -> np.ndarray:
        """The last kept copy of every patch: what the model sees (uncompensated)."""
        if self.ref is None:
            raise RuntimeError("no frame processed yet")
        return self.ref.copy()

    def view(self, compensate_shift: bool = True) -> np.ndarray:
        """Model's view of the current frame.  With compensation the per-patch
        brightness shift of the latest frame is added back, which makes the
        pointwise error at most spread/2 on every dropped patch."""
        if self.ref is None:
            raise RuntimeError("no frame processed yet")
        if not compensate_shift:
            return self.ref.copy()
        v = self.ref.astype(np.float32)
        P = self.patch
        sh = np.kron(self.last_shift, np.ones((P, P), np.float32))
        v[: self.gh * P, : self.gw * P] += sh
        if self.ref.dtype.kind in "ui":
            info = np.iinfo(self.ref.dtype)
            v = np.clip(np.rint(v), info.min, info.max)
        return v.astype(self.ref.dtype)

    # -- rule ---------------------------------------------------------------
    def step(self, frame: np.ndarray):
        """Process one frame. Returns (keep[gh,gw] bool, spread[gh,gw], shift[gh,gw])."""
        frame = np.ascontiguousarray(frame)
        if frame.shape != (self.H, self.W):
            raise ValueError(f"frame shape {frame.shape} != {(self.H, self.W)}")
        if self.ref is None:
            self.ref = frame.copy()
            z = np.zeros((self.gh, self.gw), np.float32)
            return np.ones((self.gh, self.gw), bool), z, z.copy()
        spread, shift = spread_and_shift(frame, self.ref, self.patch)
        drop = spread < self.delta
        if self.max_shift is not None:
            drop &= np.abs(shift) <= self.max_shift
        keep = ~drop
        # Update the reference on kept patches only.
        P = self.patch
        rg = patch_grid(self.ref, P)       # view into self.ref
        fg = patch_grid(frame, P)
        rg[keep] = fg[keep]
        self.last_shift = np.where(keep, 0.0, shift).astype(np.float32)
        return keep, spread, shift

    def run(self, frames):
        """Process an iterable / (T,H,W) array of frames.
        Returns dict with keep (T,gh,gw) bool, spread, shift (T,gh,gw) float32."""
        keeps, spreads, shifts = [], [], []
        for f in frames:
            k, s, c = self.step(f)
            keeps.append(k); spreads.append(s); shifts.append(c)
        return {"keep": np.stack(keeps), "spread": np.stack(spreads), "shift": np.stack(shifts)}


def drop_rate(keep: np.ndarray, skip_first: bool = True) -> float:
    """Fraction of patches dropped. The first frame is always fully kept and
    is excluded by default so short clips are not biased."""
    k = keep[1:] if (skip_first and keep.shape[0] > 1) else keep
    return float(1.0 - k.mean())


def certified_error(keep: np.ndarray, spread: np.ndarray) -> float:
    """Largest guaranteed pointwise error of the compensated view over all
    dropped patches: max spread/2 where keep == 0."""
    dropped = ~keep
    return float(0.5 * spread[dropped].max()) if dropped.any() else 0.0
