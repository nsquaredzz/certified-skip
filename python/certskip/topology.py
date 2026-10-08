"""0-dimensional persistent homology on image patches and the exact bottleneck
distance, used to (a) define "an object of contrast delta" and (b) check the
guarantee empirically on every dropped patch.

Features
--------
Sublevel-set H0 of f  : dark blobs.  A local minimum is born at its value and
                        dies when its basin merges into an older (deeper) one.
Superlevel-set H0     : bright blobs = sublevel H0 of -f.
Contrast of a feature : its persistence, death - birth.
The global extremum's class is essential; by default its death is capped at
the patch maximum (extended-persistence style) so its contrast is finite and
"an object of contrast delta" means the same thing whether or not the object
is the darkest/brightest thing in its patch.

H1 (holes / rings) is not computed; the stability theorem covers it too, but
dark and bright blobs are what "object appears / vanishes / splits / merges"
means in the paper's claim.
"""
from __future__ import annotations

import numpy as np


def _find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def persistence_h0(patch: np.ndarray, superlevel: bool = False, cap_essential: bool = True):
    """Return an (n, 2) float array of (birth, death) pairs for H0 of the
    sublevel filtration of `patch` (4-connectivity, elder rule).

    With cap_essential=True (default) the essential class (the global
    minimum's component) is given death = max(patch), so a dark blob that is
    the darkest thing in its patch has finite contrast = depth below the
    patch maximum, and a flat patch has no features.  Stability still holds
    for capped diagrams because max f and max g are within ||f-g||_inf.
    With cap_essential=False the essential class has death = +inf.
    With superlevel=True the filtration is of -patch (bright blobs) and
    births/deaths are reported in -patch units."""
    f = np.asarray(patch, dtype=np.float64)
    if superlevel:
        f = -f
    H, W = f.shape
    flat = f.ravel()
    order = np.argsort(flat, kind="stable")
    parent = np.arange(H * W)
    birth = np.full(H * W, np.nan)
    active = np.zeros(H * W, bool)
    pairs = []
    for idx in order:
        v = flat[idx]
        active[idx] = True
        birth[idx] = v
        y, x = divmod(int(idx), W)
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < H and 0 <= nx < W:
                j = ny * W + nx
                if not active[j]:
                    continue
                ra, rb = _find(parent, idx), _find(parent, j)
                if ra == rb:
                    continue
                # elder rule: the younger root (larger birth) dies now
                if birth[ra] < birth[rb] or (birth[ra] == birth[rb] and ra < rb):
                    old, young = ra, rb
                else:
                    old, young = rb, ra
                if birth[young] < v:          # zero-persistence pairs are omitted
                    pairs.append((birth[young], v))
                elif birth[young] == v:
                    pass
                parent[young] = old
    roots = {_find(parent, i) for i in range(H * W)}
    cap = float(flat.max()) if cap_essential else np.inf
    for r in roots:
        if cap_essential and birth[r] >= cap:
            continue                                  # flat patch: no feature
        pairs.append((birth[r], cap))
    return np.array(pairs, dtype=np.float64).reshape(-1, 2)


def persistence(diagram: np.ndarray) -> np.ndarray:
    return diagram[:, 1] - diagram[:, 0]


def features_at_least(diagram: np.ndarray, delta: float, finite_only: bool = True) -> int:
    """Number of features with contrast >= delta."""
    p = persistence(diagram)
    if finite_only:
        p = p[np.isfinite(p)]
    return int(np.sum(p >= delta))


def max_contrast(diagram: np.ndarray) -> float:
    p = persistence(diagram)
    p = p[np.isfinite(p)]
    return float(p.max()) if p.size else 0.0


# ---------------------------------------------------------------- bottleneck
def _perfect_matching_exists(adj, n_left, n_right):
    """Simple augmenting-path bipartite matching. adj[i] = iterable of right ids."""
    match_r = [-1] * n_right

    def try_assign(u, seen):
        for v in adj[u]:
            if seen[v]:
                continue
            seen[v] = True
            if match_r[v] == -1 or try_assign(match_r[v], seen):
                match_r[v] = u
                return True
        return False

    for u in range(n_left):
        if not try_assign(u, [False] * n_right):
            return False
    return True


def bottleneck_distance(D1: np.ndarray, D2: np.ndarray) -> float:
    """Exact bottleneck distance between two H0 diagrams (with essential
    classes matched to each other, finite classes allowed to match the
    diagonal).  Exact via binary search over the candidate distances."""
    D1 = np.asarray(D1, float).reshape(-1, 2)
    D2 = np.asarray(D2, float).reshape(-1, 2)
    e1, e2 = D1[~np.isfinite(D1[:, 1])], D2[~np.isfinite(D2[:, 1])]
    A, B = D1[np.isfinite(D1[:, 1])], D2[np.isfinite(D2[:, 1])]
    ess = 0.0
    if len(e1) != len(e2):
        return np.inf
    if len(e1):
        a, b = np.sort(e1[:, 0]), np.sort(e2[:, 0])
        ess = float(np.max(np.abs(a - b)))
    n, m = len(A), len(B)
    if n == 0 and m == 0:
        return ess
    dA = 0.5 * (A[:, 1] - A[:, 0]) if n else np.zeros(0)
    dB = 0.5 * (B[:, 1] - B[:, 0]) if m else np.zeros(0)
    if n and m:
        C = np.max(np.abs(A[:, None, :] - B[None, :, :]), axis=2)   # (n, m) L_inf
    else:
        C = np.zeros((n, m))
    cands = np.unique(np.concatenate([C.ravel(), dA, dB, [0.0]]))

    def feasible(eps):
        # left: A (n) + B' (m, diagonal projections of B); right: B (m) + A' (n)
        adj = []
        for i in range(n):
            lst = [j for j in range(m) if C[i, j] <= eps]
            if dA[i] <= eps:
                lst.append(m + i)
            adj.append(lst)
        for j in range(m):
            lst = list(range(m, m + n))             # B'_j -- A'_i always allowed
            if dB[j] <= eps:
                lst.append(j)
            adj.append(lst)
        return _perfect_matching_exists(adj, n + m, n + m)

    lo, hi = 0, len(cands) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(cands[mid]):
            hi = mid
        else:
            lo = mid + 1
    return max(ess, float(cands[lo]))


# ---------------------------------------------------------------- checks
def check_patch(new: np.ndarray, ref: np.ndarray, delta: float, tol: float = 1e-6) -> dict:
    """Verify the guarantee on one dropped patch.

    Returns a dict with the spread s, the bottleneck distances (dark/bright)
    between new and shift-compensated ref, and `ok` = both distances <= s/2
    and no finite feature of contrast >= delta exists in `new` that cannot be
    accounted for (which the bound already implies, but is reported
    separately because it is the paper's headline statement)."""
    new = np.asarray(new, np.float64)
    ref = np.asarray(ref, np.float64)
    d = new - ref
    s = float(d.max() - d.min())
    c = 0.5 * float(d.max() + d.min())
    comp = ref + c
    out = {"spread": s, "half_spread": 0.5 * s, "shift": c, "dropped": s < delta}
    for name, sup in (("dark", False), ("bright", True)):
        Dn = persistence_h0(new, sup)
        Dr = persistence_h0(comp, sup)
        bd = bottleneck_distance(Dn, Dr)
        out[f"bottleneck_{name}"] = bd
        out[f"n_features_{name}_new"] = features_at_least(Dn, delta)
        out[f"n_features_{name}_ref"] = features_at_least(Dr, delta)
        # a feature of contrast >= delta in `new` must be within s/2 of a
        # feature of contrast >= delta - s in ref: implied by bd <= s/2 < delta/2
    out["linf_error"] = float(np.abs(new - comp).max())
    out["ok"] = (out["bottleneck_dark"] <= 0.5 * s + tol
                 and out["bottleneck_bright"] <= 0.5 * s + tol
                 and out["linf_error"] <= 0.5 * s + tol)
    return out


def soundness_audit(frames: np.ndarray, keep: np.ndarray, patch: int, delta: float,
                    max_patches: int | None = 2000, rng=None) -> dict:
    """Replay the rule on `frames`, and for a sample of dropped patches check
    the guarantee against the actual last-kept reference.  Returns counts and
    the worst bottleneck/half-spread ratio seen."""
    from .core import patch_grid
    rng = np.random.default_rng(0) if rng is None else rng
    T, H, W = frames.shape
    ref = frames[0].copy()
    rg = patch_grid(ref, patch)
    dropped = [(t, gy, gx) for t in range(1, T) for gy, gx in zip(*np.where(~keep[t]))]
    if max_patches is not None and len(dropped) > max_patches:
        sel = rng.choice(len(dropped), max_patches, replace=False)
        wanted = set(dropped[i] for i in sel)
    else:
        wanted = set(dropped)
    checked = violations = 0
    worst_ratio = 0.0
    worst = None
    for t in range(1, T):
        fg = patch_grid(frames[t], patch)
        for gy, gx in zip(*np.where(~keep[t])):
            if (t, gy, gx) in wanted:
                r = check_patch(fg[gy, gx], rg[gy, gx], delta)
                checked += 1
                if not r["ok"]:
                    violations += 1
                if r["half_spread"] > 0:
                    ratio = max(r["bottleneck_dark"], r["bottleneck_bright"]) / r["half_spread"]
                    if ratio > worst_ratio:
                        worst_ratio, worst = ratio, (t, int(gy), int(gx), r)
        k = keep[t]
        rg[k] = fg[k]
    return {"checked": checked, "violations": violations,
            "worst_bottleneck_over_half_spread": worst_ratio, "worst_case": worst}
