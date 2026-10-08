"""Transport-certified skipping: the flat-norm rule.

Idea
----
Treat the patch change d = F - R (after removing its mean c) as a signed mass
distribution on the pixel grid.  Measure it with the *flat norm* at scale ell:

    ||d||_ell = min over flows J on grid edges and residual r :
                   (1/ell) * sum |J|  +  sum |r|      s.t.  div J = d - r

i.e. mass may be transported at cost (L1 distance)/ell per unit, or created /
destroyed at cost 1 per unit.  Mass units are grey-levels x pixels.

By Kantorovich-Rubinstein duality

    ||d||_ell = sup { <f, d> : |f| <= 1,  |f(x)-f(y)| <= |x-y|_1 / ell }.

Rule: drop the patch while an UPPER BOUND on ||F - R - c||_ell is below tau.

Guarantee (conservation law).  If ||d||_ell < tau then for every region S of
the patch, with f_S its ell-ramp (1 on S, decaying linearly to 0 at L1
distance ell), |<f_S, d>| < tau.  So while the patch is dropped
  * no object of mass >= tau (area x contrast) appears or vanishes, unless an
    opposite change of nearly equal mass happens within ell pixels of it, and
  * an object of mass m that moves by delta pixels is kept as soon as
    m * min(delta, 2 ell) >= tau * ell.
Edge jitter (a sub-pixel shift of a high-contrast edge) is mass moving ~1 px
and is nearly free; a person appearing is mass created and is expensive.  The
range rule cannot tell these apart; this one can.

Model-facing bound.  For any linear frontend with filters w_k (e.g. a ViT
patch embedding), |<w_k, d>| <= ||w_k||*_ell * ||d||_ell with
||w||*_ell = max(max|w|, ell * max 4-neighbour |difference|).  So the token
of a dropped patch moves by at most (max_k ||w_k||*_ell) * tau.

Computation.  The exact norm is a min-cost flow (cpp solver, audits only).
The certified statistic is a *quadtree transport* upper bound: mass is moved
up a quadtree (cost = L1 distance between cell centres), cancelled where it
meets opposite mass, destroyed when transport becomes dearer than creation.
The quadtree metric dominates the L1 grid metric, so this is a feasible flow
and hence an upper bound; the minimum over shifted quadtrees tightens it.
All of this is a few reshapes and sums, vectorised over every patch at once.
"""
from __future__ import annotations

import numpy as np

from .core import patch_grid


# ----------------------------------------------------------------- upper bound
def _tree_cost(m: np.ndarray, ell: float) -> np.ndarray:
    """Quadtree transport cost for a stack of square mass grids m[..., Q, Q]
    (Q power of two).  Returns cost[...]."""
    Q = m.shape[-1]
    cost = np.zeros(m.shape[:-2], np.float64)
    j = 1
    alive = np.ones(m.shape[:-2], bool)       # patches still transporting
    cur = m.astype(np.float64)
    while cur.shape[-1] > 1:
        t = (2 ** (j - 1)) / ell              # child centre -> parent centre, L1
        absmass = np.abs(cur).sum(axis=(-2, -1))
        if t >= 1.0:                          # destroying is no dearer than moving
            cost += absmass
            return cost
        cost += t * absmass
        s = cur.shape[-1] // 2
        cur = cur.reshape(*cur.shape[:-2], s, 2, s, 2).sum(axis=(-3, -1))
        j += 1
    cost += np.abs(cur[..., 0, 0])            # root remainder is destroyed
    return cost


def flat_norm_ub(d: np.ndarray, ell: float = 4.0, shifts: int | None = None) -> np.ndarray:
    """Upper bound on the flat norm of each patch in d[..., P, P] (P power of two).
    Quadtree offsets 0..shifts-1 per axis are tried and the minimum taken.
    Transport only happens at levels with 2^(j-1) < ell, whose cells have side
    at most 2^ceil(log2 ell); offsets modulo that side cover every boundary
    placement, so the default shifts = min(P, 2^ceil(log2 ell))."""
    d = np.asarray(d, np.float64)
    P = d.shape[-1]
    Q = 2 * P
    lead = d.shape[:-2]
    if shifts is None:
        shifts = int(min(P, 2 ** int(np.ceil(np.log2(max(ell, 1.0))))))
    best = np.abs(d).sum(axis=(-2, -1))       # destroy everything: always feasible
    for sy in range(shifts):
        for sx in range(shifts):
            pad = np.zeros(lead + (Q, Q), np.float64)
            pad[..., sy:sy + P, sx:sx + P] = d
            best = np.minimum(best, _tree_cost(pad, ell))
    return best


def flat_norm_lb(d: np.ndarray, ell: float = 4.0) -> np.ndarray:
    """Cheap LOWER bound: the mean-free part tested against the best of a few
    1/ell-Lipschitz, |f|<=1 ramp functions (axis-aligned half-plane ramps)."""
    d = np.asarray(d, np.float64)
    P = d.shape[-1]
    x = np.arange(P, dtype=np.float64)
    lb = np.zeros(d.shape[:-2])
    for k in range(P):
        for axis in (0, 1):
            f = np.clip((x - k) / ell, -1, 1)
            F = f[:, None] * np.ones((1, P)) if axis == 0 else np.ones((P, 1)) * f[None, :]
            lb = np.maximum(lb, np.abs((d * F).sum(axis=(-2, -1))))
    return lb


# ------------------------------------------------------------------ test fns
def ramp_functional(S: np.ndarray, ell: float) -> np.ndarray:
    """The ell-ramp of a boolean region S (P x P): 1 on S, decaying linearly
    with L1 distance to 0 at distance ell.  This is a feasible dual test
    function, so <f_S, d> <= ||d||_ell for every d."""
    P = S.shape[0]
    yy, xx = np.mgrid[0:P, 0:P]
    ys, xs = np.where(S)
    if len(ys) == 0:
        return np.zeros((P, P))
    dist = np.min(np.abs(yy[..., None] - ys) + np.abs(xx[..., None] - xs), axis=-1)
    return np.clip(1.0 - dist / ell, 0.0, 1.0)


def dual_norm_of_filter(w: np.ndarray, ell: float) -> float:
    """||w||*_ell = max(max|w|, ell * max 4-neighbour difference): the constant
    such that |<w, d>| <= ||w||*_ell ||d||_ell for all d."""
    w = np.asarray(w, np.float64)
    lip = max(np.abs(np.diff(w, axis=0)).max(initial=0), np.abs(np.diff(w, axis=1)).max(initial=0))
    return float(max(np.abs(w).max(), ell * lip))


def dct_filters(P: int) -> np.ndarray:
    """Orthonormal 2-D DCT-II basis (P*P, P, P): a stand-in linear frontend."""
    n = np.arange(P)
    C = np.cos(np.pi * (2 * n[None, :] + 1) * n[:, None] / (2 * P)) * np.sqrt(2.0 / P)
    C[0] /= np.sqrt(2.0)
    return np.einsum("ui,vj->uvij", C, C).reshape(P * P, P, P)


# ---------------------------------------------------------------- the pruner
class FlatPruner:
    """Transport-certified pruner.  Same interface as core.Pruner.

    Parameters
    ----------
    tau   : mass threshold (grey-levels x pixels).  A 5x5 object of contrast
            40 has mass 1000; tau = 300 keeps anything above ~ a 3x3 blob at
            contrast 33.
    ell   : transport scale in pixels; motion below ell is cheap.
    shifts: quadtree shifts per axis for the bound (4 is plenty).
    """

    def __init__(self, height: int, width: int, patch: int = 16,
                 tau: float = 300.0, ell: float = 4.0, shifts: int | None = None):
        if patch & (patch - 1):
            raise ValueError("patch must be a power of two for the quadtree bound")
        self.H, self.W, self.patch = int(height), int(width), int(patch)
        self.tau, self.ell, self.shifts = float(tau), float(ell), shifts
        self.gh, self.gw = self.H // patch, self.W // patch
        self.ref = None
        self.last_shift = np.zeros((self.gh, self.gw), np.float32)

    @property
    def initialised(self):
        return self.ref is not None

    def reset(self):
        self.ref = None
        self.last_shift[:] = 0

    def scores(self, frame: np.ndarray):
        """(ub[gh,gw], shift[gh,gw]) for the current frame against the reference."""
        acc = np.float32 if frame.dtype.kind == "f" else np.int32
        d = patch_grid(frame.astype(acc) - self.ref.astype(acc), self.patch).astype(np.float64)
        c = d.mean(axis=(2, 3))
        ub = flat_norm_ub(d - c[..., None, None], self.ell, self.shifts)
        return ub.astype(np.float32), c.astype(np.float32)

    def step(self, frame: np.ndarray):
        frame = np.ascontiguousarray(frame)
        if frame.shape != (self.H, self.W):
            raise ValueError("bad frame shape")
        if self.ref is None:
            self.ref = frame.copy()
            z = np.zeros((self.gh, self.gw), np.float32)
            return np.ones((self.gh, self.gw), bool), z, z.copy()
        ub, c = self.scores(frame)
        keep = ub >= self.tau
        rg = patch_grid(self.ref, self.patch)
        rg[keep] = patch_grid(frame, self.patch)[keep]
        self.last_shift = np.where(keep, 0.0, c).astype(np.float32)
        return keep, ub, c

    def run(self, frames):
        keeps, scores, shifts = [], [], []
        for f in frames:
            k, s, c = self.step(f)
            keeps.append(k); scores.append(s); shifts.append(c)
        return {"keep": np.stack(keeps), "spread": np.stack(scores), "shift": np.stack(shifts)}

    def reference(self):
        return self.ref.copy()

    def view(self, compensate_shift: bool = True):
        v = self.ref.astype(np.float32)
        if compensate_shift:
            P = self.patch
            v[: self.gh * P, : self.gw * P] += np.kron(self.last_shift, np.ones((P, P), np.float32))
        if self.ref.dtype.kind in "ui":
            info = np.iinfo(self.ref.dtype)
            v = np.clip(np.rint(v), info.min, info.max)
        return v.astype(self.ref.dtype)


def flat_scores_all(frames: np.ndarray, patch: int, ell: float, shifts: int | None = None):
    """Stateless helper used by the benchmarks: for threshold sweeps we need
    the trajectory of scores under the rule's own reference updates, which
    depends on tau; so this just exposes a factory."""
    def rule(frames, patch, tau):
        T, H, W = frames.shape
        return FlatPruner(H, W, patch, tau, ell, shifts).run(frames)["keep"]
    return rule
