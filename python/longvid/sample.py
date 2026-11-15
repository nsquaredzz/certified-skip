"""Sampling with a KV cache: prompt frames, then frame by frame, each denoised against the cache and then committed."""
from __future__ import annotations

import torch

from .model import KVCache


def schedule(steps, shift=3.0):
    s = torch.linspace(1, 0, steps + 1)
    return shift * s / (1 + (shift - 1) * s)


@torch.no_grad()
def rollout(model, prompt, n_frames, steps=8, sink=0, window=None, reindex=False, gen=None, on_frame=None):
    """prompt (B, P, C, H, W) clean frames. Returns (B, n_frames, C, H, W) generated after the prompt."""
    B, Pn = prompt.shape[:2]; cache = KVCache(model.depth, sink, window, reindex); sig = schedule(steps)
    for t in range(Pn):
        model.forward_chunk(model.patchify(prompt[:, t:t + 1])[:, 0], torch.zeros(B), t, cache, write=True)
    out = []
    for n in range(n_frames):
        t = Pn + n; x = torch.randn(B, model.N, model.pd, generator=gen)
        for i in range(steps):
            s0, s1 = sig[i], sig[i + 1]; v = model.forward_chunk(x, torch.full((B,), float(s0)), t, cache, write=False)
            x = x + (s1 - s0) * v
        x = x.clamp(-1, 1); model.forward_chunk(x, torch.zeros(B), t, cache, write=True)
        out.append(model.unpatchify(x[:, None])[:, 0])
        if on_frame: on_frame(n, len(cache))
    return torch.stack(out, 1)
