"""One look, then carry, then look again when the carry stops being certified.

A look gives every object's mask (here: SAM 2's mask on that frame). Between looks the masks ride on a grid of points
moved by optical flow. A point stays certified while the grey level under it is within DELTA of what it was at the
look and it is inside the frame; only certified points paint the mask. The break signal of an object is the share
of its points that have lost their certificate since the look."""
import numpy as np
from focus_lib import box, gauss
from motion_lib import lk as _lk, sample
DELTA, STEP = 20.0, 2

def lk_init(a, b, vy, vx, r=7, iters=3):
    H, W = a.shape; gy, gx = np.gradient(a); A = gauss(gy * gy, r); B = gauss(gy * gx, r); C = gauss(gx * gx, r); reg = 0.5; det = (A + reg) * (C + reg) - B * B
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for _ in range(iters):
        d = sample(b, yy + vy, xx + vx) - a; p = gauss(gy * d, r); q = gauss(gx * d, r)
        vy = vy - ((C + reg) * p - B * q) / det; vx = vx - ((A + reg) * q - B * p) / det
    return vy, vx

def half(a):
    H, W = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2; return a[:H, :W].reshape(H // 2, 2, W // 2, 2).mean((1, 3))

def flow(a, b, levels=3):
    """Coarse to fine Lucas-Kanade, a -> b, as (dy, dx)."""
    pa, pb = [gauss(a, 1)], [gauss(b, 1)]
    for _ in range(levels - 1):
        pa.append(half(pa[-1])); pb.append(half(pb[-1]))
    vy = np.zeros(pa[-1].shape, np.float32); vx = np.zeros_like(vy)
    for l in range(levels - 1, -1, -1):
        if vy.shape != pa[l].shape:
            H, W = pa[l].shape; uy = np.zeros((H, W), np.float32); ux = np.zeros((H, W), np.float32)
            ry = np.repeat(np.repeat(vy, 2, 0), 2, 1) * 2; rx = np.repeat(np.repeat(vx, 2, 0), 2, 1) * 2; h, w = min(H, ry.shape[0]), min(W, ry.shape[1])
            uy[:h, :w] = ry[:h, :w]; ux[:h, :w] = rx[:h, :w]; uy[h:] = uy[h - 1]; ux[h:] = ux[h - 1]; uy[:, w:] = uy[:, w - 1:w]; ux[:, w:] = ux[:, w - 1:w]; vy, vx = uy, ux
        vy, vx = lk_init(pa[l], pb[l], vy, vx)
    return np.stack([vy, vx]).astype(np.float16)

def run(grey, smooth, flows, ref, looks=None, theta=None, min_gap=3):
    """Simulate. looks: frames at which to look (frame 0 always); or theta: look when any object's break signal
    passes it. Returns masks (T, H, W) of labels, the frames looked at, and the signal (T, K)."""
    T, K, H, W = ref.shape; out = np.zeros((T, H, W), np.uint8); sig = np.zeros((T, K), np.float32); looked = []
    ys, xs = np.mgrid[STEP // 2:H:STEP, STEP // 2:W:STEP]; g0 = np.stack([ys.ravel(), xs.ravel()], 1).astype(np.float32)
    for t in range(T):
        if t > 0:
            v = flows[t - 1].astype(np.float32); p = p + np.stack([sample(v[0], p[:, 0], p[:, 1]), sample(v[1], p[:, 0], p[:, 1])], 1)
            inside = (p[:, 0] >= 0) & (p[:, 0] <= H - 1) & (p[:, 1] >= 0) & (p[:, 1] <= W - 1)
            ok &= inside & (np.abs(sample(smooth[t], p[:, 0], p[:, 1]) - g) <= DELTA)        # a certificate once lost is not regained
            for k in range(K):
                m = lab == k + 1; sig[t, k] = 1 - ok[m].mean() if m.any() else 0.0
        want = t == 0 or (looks is not None and t in looks) or (theta is not None and sig[t].max() > theta and t - looked[-1] >= min_gap)
        if want:
            looked.append(t); p = g0.copy(); yi, xi = p[:, 0].astype(int), p[:, 1].astype(int); lab = np.zeros(len(p), int)
            for k in range(K):
                lab[ref[t, k][yi, xi]] = k + 1
            g = smooth[t][yi, xi].copy(); ok = np.ones(len(p), bool); sig[t] = 0
            for k in range(K):
                out[t][ref[t, k]] = k + 1
            continue
        q = p[ok]; l = lab[ok]; y = q[:, 0].round().astype(int); x = q[:, 1].round().astype(int); cnt = np.zeros((K + 1, H, W), np.float32)
        np.add.at(cnt, (l, y, x), 1.0); cnt = box(cnt, 2); out[t] = cnt.argmax(0) * (cnt.sum(0) > 0)
    return out, looked, sig

def agreement(out, ref):
    """Per frame: mean over the objects present in either of intersection over union."""
    T, K = ref.shape[:2]; a = np.ones(T)
    for t in range(T):
        v = [(m & r).sum() / (m | r).sum() for m, r in ((out[t] == k + 1, ref[t, k]) for k in range(K)) if (m | r).sum() > 0]
        a[t] = np.mean(v) if v else 1.0
    return a
