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
    bg = background(rng); k = rng.integers(1, 4); cols = PALETTE[rng.choice(len(PALETTE), k, replace=False)]
    p = rng.uniform(R, S - R, (k, 2)); v = rng.uniform(0.4, 1.2, (k, 2)) * rng.choice([-1, 1], (k, 2))
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32); out = np.empty((T, 3, S, S), np.float32)
    for t in range(T):
        fr = bg.copy()
        for j in range(k):
            m = (yy - p[j, 0]) ** 2 + (xx - p[j, 1]) ** 2 <= R * R
            fr[:, m] = cols[j][:, None]
        out[t] = fr; p = p + v
        for a in range(2):
            lo, hi = p[:, a] < R, p[:, a] > S - R
            v[lo | hi, a] *= -1; p[:, a] = np.clip(p[:, a], R, S - R)
    return out, bg, cols


def batch(rng, B, T):
    eps = [episode(rng, T) for _ in range(B)]
    return np.stack([e[0] for e in eps]), np.stack([e[1] for e in eps]), [e[2] for e in eps]
