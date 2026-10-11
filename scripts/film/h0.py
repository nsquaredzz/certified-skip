"""Persistent H0 of the path space. Grid points are joined to their neighbours in order of how alike their paths
are. Two components that have both grown to `m` points stay apart if the lightest edge between them is `ratio`
times heavier than the edges inside either: the boundary is sharper than anything within. (The same rule as the
tubes of follow_tubes.py, with path distance in place of grey level.)"""
import numpy as np

def edges(P, gs):
    """(weight, i, j) for the 8-neighbour grid; weight: the largest distance the two points drift apart or together
    at any time of the shot, in pixels (95th percentile over time)."""
    T, N, _ = P.shape; D = (P - P[0]).reshape(T, *gs, 2); out = []; idx = np.arange(N).reshape(gs)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = (slice(0, gs[0] - dy), slice(max(0, -dx), gs[1] - max(0, dx))); b = (slice(dy, gs[0]), slice(max(0, dx), gs[1] + min(0, dx)))
        d = np.sqrt(((D[(slice(None),) + a] - D[(slice(None),) + b]) ** 2).sum(-1)); w = np.sort(d, 0)[int(0.95 * (T - 1))]
        out.append(np.stack([w.ravel(), idx[a].ravel(), idx[b].ravel()], 1))
    return np.concatenate(out)

def cluster(E, N, m, ratio):
    """Labels per point (root ids) and the meetings of two components of m points or more:
    (weight, size a, size b, mean weight inside a, inside b, kept apart)."""
    E = E[np.argsort(E[:, 0], kind="stable")]; parent = list(range(N)); size = [1] * N; tot = [0.0] * N; cnt = [0] * N; events = []; apart = set()
    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]; c = parent[c]
        return c
    for w, i, j in E.tolist():
        a, b = find(int(i)), find(int(j))
        if a == b:
            tot[a] += w; cnt[a] += 1; continue
        if (min(a, b), max(a, b)) in apart:
            continue
        if size[a] < size[b]:
            a, b = b, a
        if size[b] >= m:
            ia, ib = tot[a] / max(cnt[a], 1), tot[b] / max(cnt[b], 1); keep = w > ratio * max(ia, ib)
            events.append((w, size[a], size[b], ia, ib, keep))
            if keep:
                apart.add((min(a, b), max(a, b))); continue
        parent[b] = a; size[a] += size[b]; tot[a] += tot[b] + w; cnt[a] += cnt[b] + 1
        apart = {(min(find(p), find(q)), max(find(p), find(q))) for p, q in apart} if apart else apart
    return np.array([find(c) for c in range(N)]), events
