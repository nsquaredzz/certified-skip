"""Figure and ground without training. A point is figure if it moves unlike the camera or is sharper than the scene;
the figure is then cut into characters by persistent H0 of the paths (h0.py)."""
import numpy as np
from focus_lib import focus
from motion_lib import sample
from h0 import edges, cluster

def otsu(v, bins=256):
    h, e = np.histogram(v, bins); p = h / h.sum(); c = (e[:-1] + e[1:]) / 2; w = p.cumsum(); mu = (p * c).cumsum(); tot = mu[-1]
    s = (tot * w - mu) ** 2 / np.maximum(w * (1 - w), 1e-12); return c[int(np.argmax(s[:-1]))]

def z(v):
    med = np.median(v); return (v - med) / (1.4826 * np.median(np.abs(v - med)) + 1e-9)

def fields(fr, P):
    T, N, _ = P.shape; D = P - P[0]; cam = np.median(D, 1, keepdims=True)
    motion = np.sort(np.hypot(*np.moveaxis(D - cam, -1, 0)), 0)[int(0.95 * (T - 1))]       # furthest the point strays from the camera's path
    sharp = np.mean([sample(focus(fr[t:t + 1])[0], P[t, :, 0], P[t, :, 1]) for t in range(0, T, 4)], 0)
    return motion, sharp

def characters(fr, P, gs, min_share=0.02, ratio=2.5):
    """Returns per point: 0 ground, 1.. characters (largest first); and the fields for inspection."""
    N = P.shape[1]; motion, sharp = fields(fr, P); F = np.clip(np.maximum(z(motion), z(sharp)), -3, 6); cut = otsu(F); fig = F > cut
    E = edges(P, gs); E = E[fig[E[:, 1].astype(int)] & fig[E[:, 2].astype(int)]]                # only edges inside the figure
    L, ev = cluster(E, N, int(min_share * N), ratio); lab = np.zeros(N, int); ids, cnt = np.unique(L[fig], return_counts=True)
    for n, i in enumerate(ids[np.argsort(-cnt)]):
        if cnt[ids == i][0] >= min_share * N:
            lab[fig & (L == i)] = n + 1
    return lab, dict(F=F, cut=cut, fig=fig, events=ev, motion=motion, sharp=sharp)
