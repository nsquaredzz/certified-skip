import numpy as np
from focus_lib import box, gauss

def sample(a, y, x):
    """Bilinear sample of a (H, W) at float coordinates."""
    H, W = a.shape; y = np.clip(y, 0, H - 1.001); x = np.clip(x, 0, W - 1.001); y0 = y.astype(int); x0 = x.astype(int); fy = y - y0; fx = x - x0
    return a[y0, x0] * (1 - fy) * (1 - fx) + a[y0, x0 + 1] * (1 - fy) * fx + a[y0 + 1, x0] * fy * (1 - fx) + a[y0 + 1, x0 + 1] * fy * fx

def lk(a, b, r=9, iters=3):
    """Dense Lucas-Kanade flow a -> b (window radius r), a few warped iterations. Returns v (2, H, W) as (dy, dx)
    and the smaller eigenvalue of the structure tensor (how well the window pins the motion down)."""
    H, W = a.shape; a = gauss(a, 1); b = gauss(b, 1); gy, gx = np.gradient(a)
    A = gauss(gy * gy, r); B = gauss(gy * gx, r); C = gauss(gx * gx, r); tr = A + C; det = A * C - B * B
    lam = tr / 2 - np.sqrt(np.maximum(tr * tr / 4 - det, 0)); reg = 0.5; det = (A + reg) * (C + reg) - B * B
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32); vy = np.zeros((H, W), np.float32); vx = np.zeros((H, W), np.float32)
    for _ in range(iters):
        d = sample(b, yy + vy, xx + vx) - a; p = gauss(gy * d, r); q = gauss(gx * d, r)
        vy -= ((C + reg) * p - B * q) / det; vx -= ((A + reg) * q - B * p) / det
    return np.stack([vy, vx]), lam

def trajectories(fr, step=4, r=9):
    """Grid points of the first frame carried through the shot by the flow. Returns P (T, N, 2) as (y, x),
    the grid shape, and per point the weakest structure it sat on."""
    T, H, W = fr.shape; ys, xs = np.mgrid[step // 2:H:step, step // 2:W:step]; gs = ys.shape
    p = np.stack([ys.ravel(), xs.ravel()], 1).astype(np.float32); P = [p.copy()]; conf = np.full(len(p), np.inf, np.float32)
    f = fr.astype(np.float32)
    for t in range(T - 1):
        v, lam = lk(f[t], f[t + 1], r); conf = np.minimum(conf, sample(lam, p[:, 0], p[:, 1]))
        p = p + np.stack([sample(v[0], p[:, 0], p[:, 1]), sample(v[1], p[:, 0], p[:, 1])], 1); P.append(p.copy())
    return np.stack(P), gs, conf
