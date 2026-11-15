"""Sampling with a KV cache: prompt frames, then frame by frame, each denoised against the cache and then committed."""
from __future__ import annotations

import torch

from .model import KVCache, velocity


def schedule(steps, shift=3.0):
    s = torch.linspace(1, 0, steps + 1)
    return shift * s / (1 + (shift - 1) * s)


@torch.no_grad()
def rollout(model, prompt, n_frames, steps=8, sink=0, window=None, reindex=False, gen=None, on_frame=None, make_cache=None):
    """prompt (B, P, C, H, W) clean frames. Returns (B, n_frames, C, H, W) generated after the prompt.
    `make_cache(depth)` builds the cache; the default is a KVCache with (sink, window, reindex)."""
    B, Pn = prompt.shape[:2]; dev = prompt.device
    cache = make_cache(model.depth) if make_cache else KVCache(model.depth, sink, window, reindex); sig = schedule(steps)
    for t in range(Pn):
        model.forward_chunk(model.patchify(prompt[:, t:t + 1])[:, 0], torch.zeros(B, device=dev), t, cache, write=True)
    out = []
    for n in range(n_frames):
        t = Pn + n; x = torch.randn(B, model.N, model.pd, generator=gen, device=dev)
        for i in range(steps):
            s0, s1 = float(sig[i]), float(sig[i + 1]); sg = torch.full((B,), s0, device=dev)
            v = velocity(x, sg, model.forward_chunk(x, sg, t, cache, write=False)); x = x + (s1 - s0) * v
        x = x.clamp(-1, 1); model.forward_chunk(x, torch.zeros(B, device=dev), t, cache, write=True)
        out.append(model.unpatchify(x[:, None])[:, 0])
        if on_frame: on_frame(n, cache.entries())
    return torch.stack(out, 1)
