"""A tiny world with a persistent background and a few moving, coloured discs: long-horizon coherence is measurable.

An episode has a fixed smooth background and k in {1, 2, 3} discs, each with its own colour from a palette, moving
in straight lines and bouncing off the walls. Frames are 3 x 32 x 32 in [-1, 1].
"""
from __future__ import annotations

import numpy as np

PALETTE = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], np.float32) * 2 - 1
S, R = 32, 3.5


def background(rng):
    low = rng.uniform(-0.5, 0.5, (3, 4, 4)).astype(np.float32)
    t = np.linspace(0, 3, S)
    i0 = np.clip(t.astype(int), 0, 2); f = (t - i0).astype(np.float32)
    rows = low[:, i0, :] * (1 - f)[None, :, None] + low[:, i0 + 1, :] * f[None, :, None]
    return rows[:, :, i0] * (1 - f)[None, None] + rows[:, :, i0 + 1] * f[None, None]


def episode(rng, T):
    bg = background(rng); fr, cols = _discs(rng, T, lambda t: bg)
    return fr, bg, cols


def background_periodic(rng, W):
    """A smooth background W pixels wide that wraps around horizontally (for a camera that pans forever)."""
    cells = W // 8; low = rng.uniform(-0.5, 0.5, (3, 4, cells)).astype(np.float32)
    t = np.linspace(0, 3, S); i0 = np.clip(t.astype(int), 0, 2); f = (t - i0).astype(np.float32)
    rows = low[:, i0, :] * (1 - f)[None, :, None] + low[:, i0 + 1, :] * f[None, :, None]          # (3, S, cells)
    u = np.arange(W) / 8.0; j0 = u.astype(int) % cells; j1 = (j0 + 1) % cells; g = (u - u.astype(int)).astype(np.float32)
    return rows[:, :, j0] * (1 - g) + rows[:, :, j1] * g


def episode_pan(rng, T, speeds=(-8, -4, 4, 8), W=256):
    """The camera pans horizontally at a constant speed (pixels per frame) over a wrapping background; the discs move
    on the screen as in `episode`. Returns frames, the background of every frame (T, 3, S, S), and the colours."""
    wide = background_periodic(rng, W); v = int(rng.choice(speeds)); x0 = int(rng.integers(W))
    bgs = np.stack([wide[:, :, (x0 + v * t + np.arange(S)) % W] for t in range(T)])
    fr, cols = _discs(rng, T, lambda t: bgs[t])
    return fr, bgs, cols


def batch_world(rng, B, T, world="static", speeds=(-8, -4, 4, 8)):
    """Frames (B, T, 3, S, S), the background of every frame (B, T, 3, S, S), and each episode's colours."""
    eps = [episode_pan(rng, T, speeds) if world == "pan" else episode(rng, T) for _ in range(B)]
    bgs = [e[1] if world == "pan" else np.repeat(e[1][None], T, 0) for e in eps]
    return np.stack([e[0] for e in eps]), np.stack(bgs), [e[2] for e in eps]


def _discs(rng, T, bg_of):
    k = rng.integers(1, 4); cols = PALETTE[rng.choice(len(PALETTE), k, replace=False)]
    p = rng.uniform(R, S - R, (k, 2)); v = rng.uniform(0.4, 1.2, (k, 2)) * rng.choice([-1, 1], (k, 2))
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32); out = np.empty((T, 3, S, S), np.float32)
    for t in range(T):
        fr = bg_of(t).copy()
        for j in range(k):
            m = (yy - p[j, 0]) ** 2 + (xx - p[j, 1]) ** 2 <= R * R
            fr[:, m] = cols[j][:, None]
        out[t] = fr; p = p + v
        for a in range(2):
            lo, hi = p[:, a] < R, p[:, a] > S - R
            v[lo | hi, a] *= -1; p[:, a] = np.clip(p[:, a], R, S - R)
    return out, cols


def batch(rng, B, T):
    eps = [episode(rng, T) for _ in range(B)]
    return np.stack([e[0] for e in eps]), np.stack([e[1] for e in eps]), [e[2] for e in eps]
