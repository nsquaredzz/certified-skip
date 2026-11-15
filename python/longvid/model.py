"""A small block-causal video diffusion transformer, with the ingredients current long-video models share.

  tokens        a frame is cut into patches; one token per patch, one block of tokens per frame
  attention     bidirectional inside a frame, causal across frames; 3D rotary positions (t, y, x); QK-norm
  conditioning  adaLN-zero on the noise level of each frame separately (diffusion forcing: every frame has its own)
  objective     rectified flow, x_sigma = (1 - sigma) x0 + sigma eps, the network predicts v = eps - x0
  clean context training lays the sequence out as [clean frames][noisy frames]; a noisy frame sees the clean frames
                before it and itself, so at inference the clean frames are exactly what a KV cache holds

The cache stores keys *before* the rotary rotation and rotates on read, so a frame's position can be re-indexed
when frames are evicted (`reindex`). Eviction is a policy (`KVCache.policy`): the place the experiments go.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rope_angles(pos, dt, dy, dx, base=10000.0):
    """pos (N, 3) = (t, y, x) -> angles (N, (dt + dy + dx) / 2)."""
    fs = [1.0 / (base ** (torch.arange(0, d, 2, device=pos.device).float() / d)) for d in (dt, dy, dx)]
    return torch.cat([pos[:, i:i + 1].float() * fs[i][None] for i in range(3)], -1)


def rotate(x, ang):
    """x (B, N, H, dh), ang (N, dh / 2): rotate consecutive pairs."""
    c, s = ang.cos()[None, :, None], ang.sin()[None, :, None]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, dim, heads, split, mlp=4, temp=10.0):
        super().__init__()
        self.H, self.dh, self.split = heads, dim // heads, split
        self.n1, self.n2 = nn.LayerNorm(dim, elementwise_affine=False), nn.LayerNorm(dim, elementwise_affine=False)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp * dim), nn.GELU(), nn.Linear(mlp * dim, dim))
        self.ada = nn.Linear(dim, 6 * dim); nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)
        self.temp = nn.Parameter(torch.full((heads, 1, 1), float(temp)))

    def forward(self, x, cond, pos, past=None, mask=None):
        """x (B, n, D); cond (B, n, D); pos (n, 3). past = (k_raw, v, pos_k) from a cache. Returns x and this call's (k_raw, v)."""
        B, n, D = x.shape
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(cond).chunk(6, -1)
        h = self.n1(x) * (1 + sc1) + sh1
        q, k, v = self.qkv(h).view(B, n, 3, self.H, self.dh).unbind(2)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        K, V, pk = (k, v, pos) if past is None else (torch.cat([past[0], k], 1), torch.cat([past[1], v], 1), torch.cat([past[2], pos], 0))
        qr, kr = rotate(q, rope_angles(pos, *self.split)), rotate(K, rope_angles(pk, *self.split))
        s = torch.einsum("bnhd,bmhd->bhnm", qr, kr) * self.temp.clamp(0.1, 30.0)[None]
        if mask is not None:
            s = s.masked_fill(~mask[:, None] if mask.dim() == 3 else ~mask[None, None], float("-inf"))
        o = torch.einsum("bhnm,bmhd->bnhd", s.softmax(-1), V).reshape(B, n, D)
        x = x + g1 * self.proj(o)
        x = x + g2 * self.mlp(self.n2(x) * (1 + sc2) + sh2)
        return x, (k, v)


class KVCache:
    """Clean frames' keys (unrotated) and values, per layer. `policy` decides what to keep once a frame is committed."""

    def __init__(self, depth, sink=0, window=None, reindex=False):
        self.depth, self.sink, self.window, self.reindex = depth, sink, window, reindex
        self.k = [[] for _ in range(depth)]; self.v = [[] for _ in range(depth)]; self.ts = []; self._stage = [None] * depth

    def __len__(self):
        return len(self.ts)

    def stage(self, l, k, v):
        self._stage[l] = (k, v)

    def commit(self, t):
        for l in range(self.depth):
            self.k[l].append(self._stage[l][0]); self.v[l].append(self._stage[l][1])
        self.ts.append(t); self._stage = [None] * self.depth; self.evict()

    def evict(self):
        if self.window is None:
            return
        while len(self.ts) > self.sink + self.window:
            for l in range(self.depth):
                del self.k[l][self.sink]; del self.v[l][self.sink]
            del self.ts[self.sink]

    def times(self):
        return list(range(len(self.ts))) if self.reindex else list(self.ts)

    def query_t(self, t):
        return len(self.ts) if self.reindex else t

    def get(self, l, grid):
        if not self.ts:
            return None
        pos = torch.cat([grid(t) for t in self.times()], 0)
        return torch.cat(self.k[l], 1), torch.cat(self.v[l], 1), pos


class LongVidDiT(nn.Module):
    def __init__(self, frame=(32, 32), patch=4, ch=3, dim=128, depth=4, heads=4):
        super().__init__()
        self.fh, self.fw, self.p, self.ch = frame[0], frame[1], patch, ch
        self.gh, self.gw = frame[0] // patch, frame[1] // patch; self.N = self.gh * self.gw; self.pd = ch * patch * patch
        self.depth, self.dim = depth, dim
        dh = dim // heads; dy = dx = (dh * 3 // 8) // 2 * 2; dt = dh - dy - dx
        self.split = (dt, dy, dx)
        self.inp, self.out = nn.Linear(self.pd, dim), nn.Linear(dim, self.pd)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.blocks = nn.ModuleList(Block(dim, heads, self.split) for _ in range(depth))
        self.tmlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)); self.nf = nn.LayerNorm(dim, elementwise_affine=False)
        self.fada = nn.Linear(dim, 2 * dim); nn.init.zeros_(self.fada.weight); nn.init.zeros_(self.fada.bias)
        ys, xs = torch.meshgrid(torch.arange(self.gh), torch.arange(self.gw), indexing="ij")
        self.register_buffer("yx", torch.stack([ys.flatten(), xs.flatten()], 1), persistent=False)

    # ---- layout helpers
    def patchify(self, fr):
        """(B, F, C, H, W) -> (B, F, N, pd)."""
        B, Fn, C, H, W = fr.shape; p = self.p
        return fr.view(B, Fn, C, self.gh, p, self.gw, p).permute(0, 1, 3, 5, 2, 4, 6).reshape(B, Fn, self.N, self.pd)

    def unpatchify(self, tk):
        B, Fn = tk.shape[:2]; p = self.p
        return tk.view(B, Fn, self.gh, self.gw, self.ch, p, p).permute(0, 1, 4, 2, 5, 3, 6).reshape(B, Fn, self.ch, self.fh, self.fw)

    def grid(self, t):
        return torch.cat([torch.full((self.N, 1), int(t), device=self.yx.device), self.yx], 1)

    def temb(self, sigma):
        half = self.dim // 2; f = torch.exp(-math.log(10000.0) * torch.arange(half, device=sigma.device) / half)
        a = (sigma.float() * 1000.0)[..., None] * f
        return self.tmlp(torch.cat([a.sin(), a.cos()], -1))

    def head(self, x, cond):
        sh, sc = self.fada(cond).chunk(2, -1)
        return self.out(self.nf(x) * (1 + sc) + sh)

    # ---- training layout: [clean frames][noisy frames]
    def forward_train(self, noisy, clean, sigma):
        """noisy, clean (B, F, N, pd), sigma (B, F). Returns the predicted velocity for the noisy frames (B, F, N, pd)."""
        B, Fn = noisy.shape[:2]; N = self.N
        x = self.inp(torch.cat([clean, noisy], 1).reshape(B, 2 * Fn * N, self.pd))
        cs = self.temb(torch.cat([torch.zeros_like(sigma), sigma], 1))                       # (B, 2F, D)
        cond = cs[:, :, None].expand(-1, -1, N, -1).reshape(B, 2 * Fn * N, -1)
        t = torch.arange(Fn, device=x.device).repeat_interleave(N); t = torch.cat([t, t])
        pos = torch.cat([t[:, None], self.yx.repeat(2 * Fn, 1)], 1)
        isn = torch.cat([torch.zeros(Fn * N, dtype=torch.bool), torch.ones(Fn * N, dtype=torch.bool)]).to(x.device)
        qt, kt, qn, kn = t[:, None], t[None], isn[:, None], isn[None]
        mask = torch.where(qn, torch.where(kn, qt == kt, kt < qt), torch.where(kn, torch.zeros_like(qt == kt), kt <= qt))
        for b in self.blocks:
            x, _ = b(x, cond, pos, mask=mask)
        return self.head(x, cond)[:, Fn * N:].reshape(B, Fn, N, self.pd)

    # ---- inference: one frame against a cache
    def forward_chunk(self, xt, sigma, t, cache, write=False):
        """xt (B, N, pd), sigma (B,), t the frame's index. write=True commits this (clean) frame to the cache."""
        B = xt.shape[0]; x = self.inp(xt); cond = self.temb(sigma)[:, None].expand(-1, self.N, -1)
        tq = cache.query_t(t); pos = self.grid(tq)
        for l, b in enumerate(self.blocks):
            x, (k, v) = b(x, cond, pos, past=cache.get(l, self.grid))
            if write:
                cache.stage(l, k, v)
        if write:
            cache.commit(t)
        return self.head(x, cond)
