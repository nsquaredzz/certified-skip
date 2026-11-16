"""A small block-causal video diffusion transformer, with the ingredients current long-video models share.

  tokens        a frame is cut into patches; one token per patch, one block of tokens per frame
  attention     bidirectional inside a frame, causal across frames; 3D rotary positions (t, y, x); QK-norm
  conditioning  adaLN-zero on the noise level of each frame separately (diffusion forcing: every frame has its own)
  objective     rectified flow, x_sigma = (1 - sigma) x0 + sigma eps. The network predicts the clean frame x0 (x-prediction): a token holds
                more values than the hidden width, so a network that predicts v = eps - x0 could not pass the noise through it;
                `velocity` turns the output into v = (x_sigma - x0_hat) / sigma, the quantity the loss and the sampler use
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
    """pos (..., 3) = (t, y, x) -> angles (..., (dt + dy + dx) / 2); pos may carry a leading batch dimension."""
    fs = [1.0 / (base ** (torch.arange(0, d, 2, device=pos.device).float() / d)) for d in (dt, dy, dx)]
    return torch.cat([pos[..., i:i + 1].float() * fs[i] for i in range(3)], -1)


def rotate(x, ang):
    """x (B, N, H, dh), ang (N, dh / 2) or (B, N, dh / 2): rotate consecutive pairs."""
    c, s = (ang.cos()[None, :, None], ang.sin()[None, :, None]) if ang.dim() == 2 else (ang.cos()[:, :, None], ang.sin()[:, :, None])
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
        """x (B, n, D); cond (B, n, D); pos (n, 3). past = (k_raw, v, pos_k[, bias]) from a cache; bias (B, E) is added to the
        logits of the cached entries (log of the number of frames an entry stands for; -inf for an entry that is gone).
        Returns x and this call's (k_raw, v)."""
        B, n, D = x.shape
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(cond).chunk(6, -1)
        h = self.n1(x) * (1 + sc1) + sh1
        q, k, v = self.qkv(h).view(B, n, 3, self.H, self.dh).unbind(2)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        amask = None
        if past is None:
            K, V, pk = k, v, pos
        else:
            pp = past[2]
            pk = torch.cat([pp, pos.expand(B, -1, -1) if pp.dim() == 3 else pos], 1 if pp.dim() == 3 else 0)
            K, V = torch.cat([past[0], k], 1), torch.cat([past[1], v], 1)
            if len(past) > 3 and past[3] is not None:
                amask = torch.cat([past[3], torch.zeros(B, n, device=x.device, dtype=past[3].dtype)], 1)[:, None, None, :]
        qr, kr = rotate(q, rope_angles(pos, *self.split)), rotate(K, rope_angles(pk, *self.split))
        qh = qr.transpose(1, 2) * self.temp.clamp(0.1, 30.0)[None].to(qr.dtype)
        if mask is not None:
            amask = mask[None, None] if mask.dim() == 2 else mask[:, None]
        elif amask is not None:
            amask = amask.to(qh.dtype)
        o = F.scaled_dot_product_attention(qh, kr.transpose(1, 2), V.transpose(1, 2), attn_mask=amask, scale=1.0)
        x = x + g1 * self.proj(o.transpose(1, 2).reshape(B, n, D))
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

    def commit(self, t, ref=None):
        self.n_tok = self._stage[0][0].shape[1]
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

    def entries(self):
        """Number of cache entries (tokens) held per sequence."""
        return float(len(self.ts) * getattr(self, "n_tok", 0))

    def times(self):
        return list(range(len(self.ts))) if self.reindex else list(self.ts)

    def query_t(self, t):
        return len(self.ts) if self.reindex else t

    def get(self, l, grid):
        if not self.ts:
            return None
        pos = torch.cat([grid(t) for t in self.times()], 0)
        return torch.cat(self.k[l], 1), torch.cat(self.v[l], 1), pos


def velocity(xs, sigma, x0_hat, floor=0.05):
    """v = (x_sigma - x0_hat) / sigma with sigma floored; xs (..., N, pd), sigma (...,)."""
    return (xs - x0_hat) / sigma.clamp_min(floor)[..., None, None]


class MergeCache:
    """A cache that holds one entry per run of unchanged content, with the run length as a logit bias.

    When a clean frame is committed, each token location is compared with the patch its current entry was made from.
    Within `eps` (sup norm, the anchored rule of the skip certificate) the location is *merged*: no new entry, the
    entry's count goes up by one and its position is moved to the latest frame. Otherwise the frame's token becomes a
    new entry. At attention time an entry adds log(count) to its logit: k identical copies of a key are exactly one
    copy with +log k, so a merged entry weighs what the copies it stands for would have weighed (exact when the
    copies are identical, an approximation within the certificate otherwise).
    `window` drops an entry that was not refreshed for that many frames: static content is refreshed by every merge and
    stays, changed content ages out. Keys are stored before rotation, so a refreshed entry is rotated to its new time.
    """

    def __init__(self, depth, eps, window=None):
        self.depth, self.eps, self.window = depth, float(eps), window
        self.K = [None] * depth; self.V = [None] * depth; self._stage = [None] * depth
        self.alive = self.count = self.tpos = self.ref = self.cur = None

    def __len__(self):
        return 0 if self.alive is None else self.alive.shape[1]

    def stage(self, l, k, v):
        self._stage[l] = (k, v)

    def query_t(self, t):
        return t

    def commit(self, t, ref):
        B, N = ref.shape[:2]; dev = ref.device
        change = torch.ones(B, N, dtype=torch.bool, device=dev) if self.ref is None else (ref - self.ref).abs().amax(-1) > self.eps
        C = 0 if self.alive is None else self.alive.shape[1]
        cat = lambda old, new: new if old is None else torch.cat([old, new], 1)
        for l in range(self.depth):
            k, v = self._stage[l]
            self.K[l], self.V[l] = cat(self.K[l], k[:, None]), cat(self.V[l], v[:, None])
        self.alive = cat(self.alive, change[:, None]); self.count = cat(self.count, torch.ones(B, 1, N, device=dev))
        self.tpos = cat(self.tpos, torch.full((B, 1, N), float(t), device=dev))
        if self.ref is not None:
            keep = ~change
            if keep.any():
                b = torch.arange(B, device=dev)[:, None].expand(B, N)[keep]; p = torch.arange(N, device=dev)[None].expand(B, N)[keep]
                c = self.cur[keep]; self.count[b, c, p] += 1; self.tpos[b, c, p] = float(t)
            self.ref = torch.where(change[..., None], ref, self.ref); self.cur = torch.where(change, torch.full_like(self.cur, C), self.cur)
        else:
            self.ref, self.cur = ref.clone(), torch.full((B, N), C, dtype=torch.long, device=dev)
        if self.window is not None:
            self.alive &= self.tpos >= t - self.window
        self._stage = [None] * self.depth

    def entries(self):
        return 0.0 if self.alive is None else float(self.alive.sum((1, 2)).float().mean())

    def get(self, l, grid):
        if self.alive is None:
            return None
        B, C, N = self.alive.shape; yx = grid(0)[:, 1:]
        pos = torch.cat([self.tpos.reshape(B, C * N, 1), yx.repeat(C, 1)[None].expand(B, -1, -1).to(self.tpos.dtype)], -1)
        bias = torch.where(self.alive, self.count.log(), torch.full_like(self.count, float("-inf"))).reshape(B, C * N)
        return self.K[l].flatten(1, 2), self.V[l].flatten(1, 2), pos, bias


class MotionMergeCache(MergeCache):
    """MergeCache that lets an entry *move*: content that shifted by a whole token keeps its entry, at a new position.

    On commit, each location p of the new frame is compared with the patch held for every location p - d, for every
    whole-token displacement d with |d|_inf <= radius (d = 0 tried first, so a tie keeps the entry in place). If the best
    match is within eps (sup norm, against the patch the entry was made from, carried along with it), the entry
    is claimed by p: no new entry, its count goes up and its position becomes (t, p). An entry can be claimed by one
    location only; a second claimant becomes a new entry. Keys are stored before rotation, so moving an entry is a
    change of its stored (y, x) and nothing is recomputed. radius = 0 is MergeCache.

    Why this can be exact: the model puts position into attention only through rotary encoding, so at layer 0 a
    token's key and value depend on its pixels alone, and a token moved by a whole token is the same entry. Deeper layers
    see neighbours through relative positions, which a translation of the whole neighbourhood leaves unchanged.
    Sub-token motion changes a token's pixels and is not matched; that is a limit of this rule, not an approximation in it.
    """

    def __init__(self, depth, eps, grid_hw, radius=1, window=None):
        super().__init__(depth, eps, window); self.gh, self.gw = grid_hw; self.radius = int(radius)
        self.ypos = self.xpos = self.cur_l = None

    def commit(self, t, ref):
        B, N = ref.shape[:2]; dev = ref.device; gh, gw = self.gh, self.gw
        C = len(self); yy = torch.arange(N, device=dev) // gw; xx = torch.arange(N, device=dev) % gw
        if self.ref is None:
            matched = torch.zeros(B, N, dtype=torch.bool, device=dev); src = torch.zeros(B, N, dtype=torch.long, device=dev)
        else:
            R = self.ref.view(B, gh, gw, -1); X = ref.view(B, gh, gw, -1)
            best = torch.full((B, gh, gw), float("inf"), device=dev); bd = torch.zeros(B, gh, gw, 2, dtype=torch.long, device=dev)
            ds = [(0, 0)] + [(dy, dx) for dy in range(-self.radius, self.radius + 1) for dx in range(-self.radius, self.radius + 1) if (dy, dx) != (0, 0)]
            for dy, dx in ds:
                ty, sy = slice(max(dy, 0), gh + min(dy, 0)), slice(max(-dy, 0), gh + min(-dy, 0))
                tx, sx = slice(max(dx, 0), gw + min(dx, 0)), slice(max(-dx, 0), gw + min(-dx, 0))
                err = (X[:, ty, tx] - R[:, sy, sx]).abs().amax(-1); cur = best[:, ty, tx]; better = err < cur
                best[:, ty, tx] = torch.where(better, err, cur)
                bd[:, ty, tx, 0] = torch.where(better, torch.full_like(err, dy, dtype=torch.long), bd[:, ty, tx, 0])
                bd[:, ty, tx, 1] = torch.where(better, torch.full_like(err, dx, dtype=torch.long), bd[:, ty, tx, 1])
            best, bd = best.view(B, N), bd.view(B, N, 2)
            matched = best <= self.eps; src = (yy[None] - bd[..., 0]) * gw + (xx[None] - bd[..., 1])
            for b in range(B):                                     # one claimant per entry: the closest match wins
                idx = torch.nonzero(matched[b]).flatten()
                if len(idx):
                    taken = set()
                    for p in idx[torch.argsort(best[b, idx])].tolist():
                        s_ = int(src[b, p])
                        if s_ in taken: matched[b, p] = False
                        else: taken.add(s_)
        change = ~matched; cat = lambda old, new: new if old is None else torch.cat([old, new], 1)
        for l in range(self.depth):
            k, v = self._stage[l]; self.K[l], self.V[l] = cat(self.K[l], k[:, None]), cat(self.V[l], v[:, None])
        self.alive = cat(self.alive, change[:, None]); self.count = cat(self.count, torch.ones(B, 1, N, device=dev))
        self.tpos = cat(self.tpos, torch.full((B, 1, N), float(t), device=dev))
        self.ypos = cat(self.ypos, yy.float()[None, None].expand(B, 1, N).clone()); self.xpos = cat(self.xpos, xx.float()[None, None].expand(B, 1, N).clone())
        if self.ref is None:
            self.ref, self.cur = ref.clone(), torch.full((B, N), C, dtype=torch.long, device=dev); self.cur_l = torch.arange(N, device=dev)[None].expand(B, N).clone()
        else:
            bb = torch.arange(B, device=dev)[:, None].expand(B, N)
            cc, ll = self.cur.gather(1, src), self.cur_l.gather(1, src)           # the entry each location would claim
            if matched.any():
                b_, c_, l_, p_ = bb[matched], cc[matched], ll[matched], torch.arange(N, device=dev)[None].expand(B, N)[matched]
                self.count[b_, c_, l_] += 1; self.tpos[b_, c_, l_] = float(t); self.ypos[b_, c_, l_] = yy[p_].float(); self.xpos[b_, c_, l_] = xx[p_].float()
            self.ref = torch.where(matched[..., None], self.ref.gather(1, src[..., None].expand(-1, -1, ref.shape[-1])), ref)
            self.cur = torch.where(matched, cc, torch.full_like(cc, C)); self.cur_l = torch.where(matched, ll, torch.arange(N, device=dev)[None].expand(B, N))
        if self.window is not None:
            self.alive &= self.tpos >= t - self.window
        self._stage = [None] * self.depth

    def get(self, l, grid):
        if self.alive is None:
            return None
        B, C, N = self.alive.shape
        pos = torch.stack([self.tpos, self.ypos, self.xpos], -1).reshape(B, C * N, 3)
        bias = torch.where(self.alive, self.count.log(), torch.full_like(self.count, float("-inf"))).reshape(B, C * N)
        return self.K[l].flatten(1, 2), self.V[l].flatten(1, 2), pos, bias


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
            cache.commit(t, xt)
        return self.head(x, cond)
