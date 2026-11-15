"""CertViT: a ViT whose attention is local and bounded, so that skipping can be exact on the held copy.

Why these two properties (THEORY.md section 13):

  local windows   a token at layer l depends only on the patches within Chebyshev distance l * r. Whatever a rule holds
                  fixed, the tokens that can differ from a fresh pass are exactly those within that distance of a
                  patch that was sent. Recompute those, hold the rest, and the output is *exactly* the encoder run on
                  the held image V (the frame with every skipped patch replaced by its held copy). Nothing is stale.
  QK-norm         queries and keys are L2-normalised and the logits are scaled by a bounded temperature, so every
                  logit lies in [-s, s]. Dot-product attention without it is not Lipschitz; this is.

What is then left between a streaming pass and a fresh pass on the true frame F is only F - V, and that is what the
skip certificate bounds (Theorem 11: ||F - V||_inf < eps_0 over the frame, for the rule without offset).

`StreamingCertViT.step(frame, keep, halo)`:
  halo="full"  recompute, at layer l, the tokens within l * r of a sent patch (exact on V; costs the dilation)
  halo="none"  recompute only the sent patches' own tokens and hold every other token's state (cheaper; the tokens
               next to a sent patch are stale, and that staleness is what the experiment measures)

The weights are whatever the caller trains or loads; nothing here is pretrained.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def patchify(img: torch.Tensor, P: int) -> torch.Tensor:
    """(H, W) -> (gh * gw, P * P), row-major, cropping to a multiple of P."""
    H, W = img.shape
    gh, gw = H // P, W // P
    return img[: gh * P, : gw * P].reshape(gh, P, gw, P).permute(0, 2, 1, 3).reshape(gh * gw, P * P)


class Block(nn.Module):
    def __init__(self, dim, heads, mlp=4, temp=10.0):
        super().__init__()
        self.h, self.dh = heads, dim // heads
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp * dim), nn.GELU(), nn.Linear(mlp * dim, dim))
        self.temp = nn.Parameter(torch.full((heads, 1, 1), float(temp)))    # bounded: clamped in `scale`

    def scale(self):
        return self.temp.clamp(0.1, 30.0)

    def kv(self, x):
        """Keys and values (N, h, dh) of tokens x (N, D); keys L2-normalised."""
        q, k, v = self.qkv(self.ln1(x)).view(x.shape[0], 3, self.h, self.dh).unbind(1)
        return F.normalize(q, dim=-1), F.normalize(k, dim=-1), v

    def attend(self, q, K, V, allowed):
        """q (M, h, dh) against all keys K, V (N, h, dh); `allowed` (M, N) bool is the window."""
        s = torch.einsum("mhd,nhd->hmn", q, K) * self.scale()
        s = s.masked_fill(~allowed[None], float("-inf"))
        o = torch.einsum("hmn,nhd->mhd", s.softmax(-1), V)
        return self.proj(o.reshape(q.shape[0], -1))

    def finish(self, x, a):
        x = x + a
        return x + self.mlp(self.ln2(x))


class CertViT(nn.Module):
    def __init__(self, grid, patch=16, dim=128, depth=6, heads=4, radius=1, temp=10.0):
        super().__init__()
        self.gh, self.gw = grid
        self.P, self.r, self.depth = patch, radius, depth
        N = self.gh * self.gw
        self.embed = nn.Linear(patch * patch, dim)
        self.pos = nn.Parameter(torch.randn(N, dim) * 0.02)
        self.blocks = nn.ModuleList(Block(dim, heads, temp=temp) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        ys, xs = torch.meshgrid(torch.arange(self.gh), torch.arange(self.gw), indexing="ij")
        yx = torch.stack([ys.flatten(), xs.flatten()], 1)
        self.register_buffer("near", ((yx[:, None] - yx[None]).abs().amax(-1) <= radius), persistent=False)

    def tokens(self, img):
        return self.embed(patchify(img.float() / 255.0 - 0.5, self.P)) + self.pos

    @torch.no_grad()
    def encode(self, img, all_layers=False):
        """Fresh pass on an image: the list of hidden states h_0 .. h_L (each (N, D)), or only h_L."""
        h = self.tokens(img); hs = [h]
        for b in self.blocks:
            q, k, v = b.kv(h)
            h = b.finish(h, b.attend(q, k, v, self.near)); hs.append(h)
        return hs if all_layers else hs[-1]

    def dilate(self, mask, steps):
        """Mask (N,) -> mask of tokens within `steps` * r of a True one (Chebyshev)."""
        m = mask.view(1, 1, self.gh, self.gw).float(); k = 2 * self.r + 1
        for _ in range(steps):
            m = F.max_pool2d(m, k, 1, self.r)
        return m.view(-1) > 0


class StreamingCertViT:
    """Holds the image V and, per layer, every token's hidden state and key/value; updates only what must change."""

    def __init__(self, model: CertViT):
        self.m = model; self.V = None; self.h = None; self.K = None; self.Vv = None; self.Q = None; self.stats = {}

    @torch.no_grad()
    def _full(self, img):
        m = self.m; h = m.tokens(img); self.h = [h]; self.K, self.Vv = [], []
        for b in m.blocks:
            q, k, v = b.kv(h); self.K.append(k); self.Vv.append(v)
            h = b.finish(h, b.attend(q, k, v, m.near)); self.h.append(h)

    @torch.no_grad()
    def step(self, frame, keep, halo="full"):
        """frame (H, W) uint8-valued tensor, keep (gh, gw) bool: patches sent on this frame. Returns h_L (N, D)."""
        m = self.m; P = m.P; keep = torch.as_tensor(keep).flatten()
        frame = torch.as_tensor(frame)
        if self.V is None:
            self.V = frame.clone().float(); self._full(self.V)
            self.stats = dict(frac=[1.0] * (m.depth + 1)); return self.h[-1]
        Hg, Wg = m.gh * P, m.gw * P
        spread = keep.view(m.gh, 1, m.gw, 1).expand(m.gh, P, m.gw, P).reshape(Hg, Wg)
        self.V[:Hg, :Wg] = torch.where(spread, frame[:Hg, :Wg].float(), self.V[:Hg, :Wg])
        x0 = m.tokens(self.V)                                          # cheap; only the sent rows are used
        D = keep.clone(); self.h[0] = torch.where(D[:, None], x0, self.h[0]); frac = [float(D.float().mean())]
        for l, b in enumerate(m.blocks):
            Dout = m.dilate(D, 1) if halo == "full" else D
            if D.any():
                q, k, v = b.kv(self.h[l][D]); self.K[l][D] = k; self.Vv[l][D] = v
            if Dout.any():
                qo, _, _ = b.kv(self.h[l][Dout])
                a = b.attend(qo, self.K[l], self.Vv[l], m.near[Dout])
                self.h[l + 1][Dout] = b.finish(self.h[l][Dout], a)
            D = Dout; frac.append(float(D.float().mean()))
        self.stats = dict(frac=frac)
        return self.h[-1]
