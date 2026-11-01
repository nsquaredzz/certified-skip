"""Moving cameras: certify modulo the camera's own motion.

The quotient certificate of warp.py holds for any warp for which the warped
reference can be formed.  Here the warp is the camera's motion.

What the model has been sent is kept on a *canvas*, in world coordinates.
For each frame

  1. a pose A (affine, frame -> world) is estimated by aligning the frame to
     the seen part of the canvas (robust Gauss-Newton, coarse to fine);
  2. the canvas is resampled into the frame, W(u) = R(A u);
  3. the quotient + multi-scale rule of warp.py runs on (F, W), on the
     frame's own patch grid: a further sub-pixel shift and a brightness shift
     per patch, then the box statistics on the residual;
  4. the patches that are kept are written back to the canvas.

Guarantee while dropped: exactly that of the quotient rule, with the camera
pose composed into the warp.  F is, up to the certified contrast, the content
the model already holds, moved by the camera.  The certificate is evaluated
against the W actually built, so it is sound whatever the pose estimate does:
a bad pose costs skip rate and never soundness.

A patch whose warped reference is not fully covered by content the model
holds is always kept ("revealed").  When alignment fails (a cut, or the view
has changed too much) the canvas is cleared and the frame is sent whole
("anchor").

What this does not do.  One affine pose per frame explains a camera that
turns, zooms or shakes, and a distant scene.  It does not explain parallax
from a camera that travels through a near scene, and it does not follow
objects that move on their own: their patches are kept while they move.
The canvas is resampled when written and again when read, so one-pixel
detail is attenuated twice; the multi-scale thresholds absorb that and the
single-pixel range rule does not.
"""
from __future__ import annotations

import numpy as np

from .core import patch_grid
from .scalespace import box_max_abs_mean
from .warp import fit_translation, warp


def bilinear(img, x, y, ok=None):
    """Sample img at float pixel coordinates (x to the right, y down).
    Returns (values, inside).  inside is False where (x, y) leaves the image
    or, when a mask `ok` is given, where one of the four neighbours is not ok."""
    H, W = img.shape
    inside = (x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1)
    x0 = np.clip(np.floor(x), 0, W - 2).astype(np.intp); y0 = np.clip(np.floor(y), 0, H - 2).astype(np.intp)
    fx = np.clip(x - x0, 0.0, 1.0); fy = np.clip(y - y0, 0.0, 1.0)
    v = ((1 - fy) * ((1 - fx) * img[y0, x0] + fx * img[y0, x0 + 1])
         + fy * ((1 - fx) * img[y0 + 1, x0] + fx * img[y0 + 1, x0 + 1]))
    if ok is not None:
        inside = inside & ok[y0, x0] & ok[y0, x0 + 1] & ok[y0 + 1, x0] & ok[y0 + 1, x0 + 1]
    return v, inside


def apply(A, x, y):
    """World coordinates of frame point (x, y) under the 2x3 affine A."""
    return A[0, 0] * x + A[0, 1] * y + A[0, 2], A[1, 0] * x + A[1, 1] * y + A[1, 2]


def invert(A):
    L = np.linalg.inv(A[:, :2])
    return np.hstack([L, -L @ A[:, 2:]])


def _box(img, k):
    """k x k box means, dropping the remainder."""
    H, W = img.shape
    return img[: H // k * k, : W // k * k].reshape(H // k, k, W // k, k).mean((1, 3))


class Canvas:
    """What the model has been sent, in world coordinates: pixels and a `seen`
    mask.  A window of fixed size that scrolls to follow the camera; content
    that scrolls out of the window is forgotten."""

    def __init__(self, height, width):
        self.H, self.W = int(height), int(width)
        self.img = np.zeros((self.H, self.W), np.float32)
        self.seen = np.zeros((self.H, self.W), bool)
        self.ox = 0; self.oy = 0                     # world coordinates of window pixel (0, 0)

    def clear(self):
        self.img[:] = 0; self.seen[:] = False; self.ox = 0; self.oy = 0

    def follow(self, x0, y0, x1, y1, slack=2):
        """Scroll so that the world box [x0, x1] x [y0, y1] lies inside the
        window.  False if the box is larger than the window."""
        if x1 - x0 + 2 * slack > self.W - 1 or y1 - y0 + 2 * slack > self.H - 1:
            return False
        if (x0 - slack >= self.ox and y0 - slack >= self.oy
                and x1 + slack <= self.ox + self.W - 1 and y1 + slack <= self.oy + self.H - 1):
            return True
        dx = int(round(0.5 * (x0 + x1) - 0.5 * (self.W - 1))) - self.ox
        dy = int(round(0.5 * (y0 + y1) - 0.5 * (self.H - 1))) - self.oy
        img = np.zeros_like(self.img); seen = np.zeros_like(self.seen)
        ys = slice(max(0, -dy), min(self.H, self.H - dy)); xs = slice(max(0, -dx), min(self.W, self.W - dx))
        if ys.start < ys.stop and xs.start < xs.stop:
            src = (slice(ys.start + dy, ys.stop + dy), slice(xs.start + dx, xs.stop + dx))
            img[ys, xs] = self.img[src]; seen[ys, xs] = self.seen[src]
        self.img, self.seen = img, seen
        self.ox += dx; self.oy += dy
        return True

    def read(self, A, xx, yy, reach=0.5):
        """The canvas resampled at the world positions of frame points (xx, yy).
        Returns (values, ok).  Only seen pixels enter the interpolation, with
        their weights renormalised, and a point is ok while they carry at least
        1 - reach of the weight: the seen content is extended by at most `reach`
        of a pixel, so a camera that trembles by less than that reveals nothing."""
        X, Y = apply(A, xx, yy)
        x = X - self.ox; y = Y - self.oy
        xc = np.clip(x, 0, self.W - 1); yc = np.clip(y, 0, self.H - 1)
        x0 = np.minimum(np.floor(xc), self.W - 2).astype(np.intp); y0 = np.minimum(np.floor(yc), self.H - 2).astype(np.intp)
        fx = xc - x0; fy = yc - y0
        num = np.zeros(x.shape); den = np.zeros(x.shape)
        for dy, wy in ((0, 1 - fy), (1, fy)):
            for dx, wx in ((0, 1 - fx), (1, fx)):
                wgt = wy * wx * self.seen[y0 + dy, x0 + dx]
                num += wgt * self.img[y0 + dy, x0 + dx]; den += wgt
        ok = (den >= 1 - reach) & (np.abs(x - xc) <= reach) & (np.abs(y - yc) <= reach)
        return num / np.maximum(den, 1e-9), ok

    def write(self, frame, A, sent_px):
        """Store the sent pixels of a frame.  A canvas pixel is written only if
        all four frame pixels it is interpolated from were sent."""
        h, w = frame.shape
        cx, cy = apply(A, np.array([0.0, w - 1.0, 0.0, w - 1.0]), np.array([0.0, 0.0, h - 1.0, h - 1.0]))
        j0 = max(0, int(np.floor(cx.min())) - self.ox); j1 = min(self.W, int(np.ceil(cx.max())) - self.ox + 1)
        i0 = max(0, int(np.floor(cy.min())) - self.oy); i1 = min(self.H, int(np.ceil(cy.max())) - self.oy + 1)
        if j0 >= j1 or i0 >= i1:
            return
        ii, jj = np.mgrid[i0:i1, j0:j1].astype(np.float64)
        u, v = apply(invert(A), jj + self.ox, ii + self.oy)
        val, ok = bilinear(frame, u, v, sent_px)
        self.img[i0:i1, j0:j1][ok] = val[ok]
        self.seen[i0:i1, j0:j1][ok] = True


def register(frame, canvas, A0, levels=(8, 4, 2), iters=12, huber=6.0, max_samples=40000, tol=0.02):
    """Affine pose (frame -> world) that aligns the frame to the seen part of
    the canvas: robust Gauss-Newton on brightness constancy with a global
    brightness offset, coarse to fine, started at A0.

    Returns (A, stats) with stats = dict(overlap, resid, corr, contrast), all at
    the finest level: the share of sample points that land on seen canvas, and
    there the median absolute residual (grey levels), the correlation between
    the frame and the warped canvas, and the standard deviation of the frame.
    A is None if the fit broke down."""
    h, w = frame.shape
    F = np.asarray(frame, np.float64)
    A = np.array(A0, np.float64)
    cxf, cyf, s = 0.5 * (w - 1), 0.5 * (h - 1), 0.5 * max(h, w)
    c = 0.0
    stats = {"overlap": 0.0, "resid": np.inf, "corr": 0.0, "contrast": 0.0}
    for k in levels:
        Fk = _box(F, k); Ck = _box(canvas.img, k).astype(np.float64); Sk = _box(canvas.seen, k) > 0.999
        hk, wk = Fk.shape
        gy = np.zeros_like(Fk); gx = np.zeros_like(Fk)
        gy[1:-1] = 0.5 * (Fk[2:] - Fk[:-2]); gx[:, 1:-1] = 0.5 * (Fk[:, 2:] - Fk[:, :-2])
        st = max(1, int(np.ceil(np.sqrt(hk * wk / max_samples))))
        ii, jj = np.mgrid[1:hk - 1:st, 1:wk - 1:st]
        ii = ii.ravel(); jj = jj.ravel()
        f = Fk[ii, jj]; gfx = gx[ii, jj] / k; gfy = gy[ii, jj] / k      # gradients per full-resolution pixel
        u = k * jj + 0.5 * (k - 1); v = k * ii + 0.5 * (k - 1)           # full-resolution frame coordinates
        un = (u - cxf) / s; vn = (v - cyf) / s
        off = 0.5 * (k - 1)
        for _ in range(iters):
            X, Y = apply(A, u, v)
            m, ok = bilinear(Ck, (X - canvas.ox - off) / k, (Y - canvas.oy - off) / k, Sk)
            n = int(ok.sum())
            if n < 200:
                return None, stats
            r = m + c - f
            Li = np.linalg.inv(A[:, :2])
            gX = Li[0, 0] * gfx + Li[1, 0] * gfy; gY = Li[0, 1] * gfx + Li[1, 1] * gfy   # L^-T grad F ~ grad of the canvas at A u
            J = np.stack([gX * un, gX * vn, gX, gY * un, gY * vn, gY, np.ones_like(gX)], 1)
            scale = max(huber, 1.4826 * float(np.median(np.abs(r[ok]))))
            wgt = np.where(ok, np.minimum(1.0, scale / np.maximum(np.abs(r), 1e-9)), 0.0)
            # Tikhonov prior, as in warp.fit_translation: the pose stays put unless the
            # gradient energy is well above a fraction of a grey level per pixel
            Hm = J.T @ (J * wgt[:, None]) + np.diag([0.05 * n] * 6 + [1e-6])
            try:
                d = -np.linalg.solve(Hm, J.T @ (wgt * r))
            except np.linalg.LinAlgError:
                return None, stats
            if not np.all(np.isfinite(d)):
                return None, stats
            # the normalised parameters act on (un, vn, 1); fold the update back into A
            A[0, 0] += d[0] / s; A[0, 1] += d[1] / s; A[0, 2] += d[2] - (d[0] * cxf + d[1] * cyf) / s
            A[1, 0] += d[3] / s; A[1, 1] += d[4] / s; A[1, 2] += d[5] - (d[3] * cxf + d[4] * cyf) / s
            c += d[6]
            det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
            if not 0.2 < det < 5.0:
                return None, stats
            if max(abs(d[0]) + abs(d[1]) + abs(d[2]), abs(d[3]) + abs(d[4]) + abs(d[5])) < tol:
                break
        mo, fo = m[ok], f[ok]
        sm, sf = float(mo.std()), float(fo.std())
        corr = float(((mo - mo.mean()) * (fo - fo.mean())).mean() / (sm * sf)) if sm > 1e-6 and sf > 1e-6 else 0.0
        stats = {"overlap": n / len(ok), "resid": float(np.median(np.abs(r[ok]))), "corr": corr, "contrast": sf}
    return A, stats


def quotient_scores(frame, W, patch, deltas, delta_max=1.0, iters=2):
    """The quotient + multi-scale statistic of warp.WarpPruner against a given
    reference image W.  Returns (score, c, Wimg): a patch may be dropped iff
    score < 1; c is its brightness shift and Wimg the reference after the
    per-patch sub-pixel shift, so the view of a dropped patch is Wimg + c."""
    gh, gw = frame.shape[0] // patch, frame.shape[1] // patch
    if delta_max > 0:
        dy, dx = fit_translation(frame, W, patch, delta_max, iters)
        Wimg = warp(W, dy, dx, patch)
    else:
        Wimg = np.asarray(W, np.float64)
    d = patch_grid(np.asarray(frame, np.float64) - Wimg, patch)
    c = 0.5 * (d.max((2, 3)) + d.min((2, 3)))
    e = d - c[..., None, None]
    score = np.zeros((gh, gw))
    for r, D in deltas.items():
        score = np.maximum(score, box_max_abs_mean(e, r) / (0.5 * D))
    return score, c, Wimg


class MovingPruner:
    """Certified skipping for a camera that moves.

    deltas     box half-width r -> contrast threshold Delta_r, as in scalespace
               ({0: Delta} is the range rule)
    delta_max  further per-patch shift, in pixels, fitted after the camera pose
    window     canvas size as a multiple of the frame size
    min_corr, min_overlap, zoom
               alignment is accepted when the frame and the warped canvas
               correlate at least min_corr where the canvas is seen (a frame
               with almost no contrast is accepted as it is), at least
               min_overlap of the frame lands on seen canvas, and the linear
               scale against the anchor stays within [1/zoom, zoom]; otherwise
               the frame anchors.
    search     0, or a radius in pixels: a held patch that fails the test in
               place is tried again as held content moved by up to `search`
               pixels (block matching, then the same sub-pixel fit and the same
               test).  Such a patch is *followed*: it is not sent, the model is
               told the displacement, and its view of the patch is written to
               the canvas at the new place.
    """

    def __init__(self, height, width, patch=16, deltas=None, delta_max=1.0, iters=2,
                 window=1.5, min_corr=0.5, min_overlap=0.3, zoom=1.3, search=0):
        self.H, self.W, self.patch = int(height), int(width), int(patch)
        self.deltas = dict(deltas) if deltas is not None else {0: 32.0}
        self.delta_max, self.iters = float(delta_max), int(iters)
        self.min_corr, self.min_overlap, self.zoom = float(min_corr), float(min_overlap), float(zoom)
        self.search = int(search)
        self.gh, self.gw = self.H // patch, self.W // patch
        self.canvas = Canvas(int(round(window * self.H)), int(round(window * self.W)))
        self._yy, self._xx = np.mgrid[0:self.H, 0:self.W].astype(np.float64)
        self._cx = np.array([0.0, self.W - 1.0, 0.0, self.W - 1.0]); self._cy = np.array([0.0, 0.0, self.H - 1.0, self.H - 1.0])
        self.pose = None; self._prev = None
        self.last = {}

    def _anchor(self, frame):
        self.canvas.clear()
        A = np.array([[1.0, 0.0, (self.canvas.W - self.W) // 2], [0.0, 1.0, (self.canvas.H - self.H) // 2]])
        self.canvas.write(frame, A, np.ones(frame.shape, bool))
        self.pose, self._prev = A, None
        keep = np.ones((self.gh, self.gw), bool)
        z = np.zeros((self.gh, self.gw))
        self.last = {"anchor": True, "reveal": keep.copy(), "followed": ~keep, "motion": (z, z.copy()), "pose": A.copy(),
                     "resid": 0.0, "overlap": 0.0, "corr": 0.0, "new_area": 1.0, "view": np.asarray(frame, np.float64)}
        return keep

    def _align(self, frame):
        """Pose of this frame, or None when the frame has to anchor."""
        starts = [self.pose]
        if self._prev is not None:                    # constant velocity of the frame centre
            cx, cy = 0.5 * (self.W - 1), 0.5 * (self.H - 1)
            x1, y1 = apply(self.pose, cx, cy); x0, y0 = apply(self._prev, cx, cy)
            P = self.pose.copy(); P[0, 2] += x1 - x0; P[1, 2] += y1 - y0
            starts.insert(0, P)
        def good(st):
            return (st["corr"] >= self.min_corr or st["contrast"] < 1.5) and st["overlap"] >= self.min_overlap

        best = None
        for A0 in starts:
            A, st = register(frame, self.canvas, A0)
            if A is not None and (best is None or st["corr"] > best[1]["corr"]):
                best = (A, st)
            if best is not None and good(best[1]):
                break
        if best is None:
            return None, {"resid": np.inf, "overlap": 0.0, "corr": 0.0}
        A, st = best
        scale = np.sqrt(abs(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]))
        if not good(st) or not 1 / self.zoom <= scale <= self.zoom:
            return None, st
        X, Y = apply(A, self._cx, self._cy)
        if not self.canvas.follow(X.min(), Y.min(), X.max(), Y.max()):
            return None, st
        return A, st

    def _follow(self, frame, W, ok, cand):
        """Try the candidate patches again as held content that has moved.
        Returns (followed[gh, gw], Wf, c, dy, dx): Wf is the reference sampled
        at x + (dy, dx) per patch and c the brightness shift, so the view of a
        followed patch is Wf + c."""
        P, R, m = self.patch, self.search, int(np.ceil(self.delta_max))
        F = np.asarray(frame, np.float64)
        gy, gx = np.nonzero(cand); n = len(gy)
        b = R + m                                     # NaN marks what the model does not hold
        Wp = np.pad(np.where(ok, W, np.nan), b, constant_values=np.nan)
        win = np.lib.stride_tricks.sliding_window_view(Wp, (P + 2 * b, P + 2 * b))[gy * P, gx * P]
        Fp = patch_grid(F, P)[gy, gx]; Fp = Fp - Fp.mean((1, 2), keepdims=True)
        ar = np.arange(P); rows = np.arange(n)[:, None, None]

        def cost(oy, ox):                             # mean-removed SAD at per-patch integer offsets
            blk = win[rows, (b + oy)[:, None, None] + ar[None, :, None], (b + ox)[:, None, None] + ar[None, None, :]]
            s = np.abs(Fp - (blk - blk.mean((1, 2), keepdims=True))).sum((1, 2))
            return np.where(np.isnan(s), np.inf, s)

        best = np.full(n, np.inf); by = np.zeros(n, int); bx = np.zeros(n, int)
        for span in (range(-(R // 2) * 2, R + 1, 2), (-1, 0, 1)):            # every second offset, then the neighbours of the best
            cy, cx = by.copy(), bx.copy()
            for oy in span:
                for ox in span:
                    ty, tx = cy + oy, cx + ox
                    inb = (np.abs(ty) <= R) & (np.abs(tx) <= R)
                    s = np.where(inb, cost(np.clip(ty, -R, R), np.clip(tx, -R, R)), np.inf)
                    hit = s < best
                    best[hit] = s[hit]; by[hit] = ty[hit]; bx[hit] = tx[hit]
        # everything the sub-pixel fit can read around the matched block must be held
        nb = np.arange(-m, P + m)
        src = win[rows, (b + by)[:, None, None] + nb[None, :, None], (b + bx)[:, None, None] + nb[None, None, :]]
        found = np.isfinite(best) & ~np.isnan(src).any((1, 2))
        dy0 = np.zeros((self.gh, self.gw)); dx0 = np.zeros((self.gh, self.gw))
        dy0[gy[found], gx[found]] = by[found]; dx0[gy[found], gx[found]] = bx[found]
        W0 = np.where(ok, W, 0.0)
        ddy, ddx = fit_translation(frame, warp(W0, dy0, dx0, P), P, self.delta_max, self.iters)
        dy, dx = dy0 + ddy, dx0 + ddx
        Wf = warp(W0, dy, dx, P)                      # the reference actually sampled at x + d: what is certified against
        d = patch_grid(F - Wf, P)
        c = 0.5 * (d.max((2, 3)) + d.min((2, 3)))
        e = d - c[..., None, None]
        score = np.zeros((self.gh, self.gw))
        for r, D in self.deltas.items():
            score = np.maximum(score, box_max_abs_mean(e, r) / (0.5 * D))
        followed = np.zeros((self.gh, self.gw), bool)
        followed[gy[found], gx[found]] = True
        followed &= score < 1.0
        return followed, Wf, c, np.where(followed, dy, 0.0), np.where(followed, dx, 0.0)

    def step(self, frame):
        """Process one frame.  Returns keep[gh, gw]; details are in self.last:
        anchor, reveal[gh, gw] (kept because the model holds nothing there),
        followed[gh, gw] and motion (dy, dx) when search > 0, pose, resid,
        overlap, corr, new_area (share of the frame's pixels the canvas does
        not cover) and view."""
        frame = np.ascontiguousarray(frame)
        if frame.shape != (self.H, self.W):
            raise ValueError(f"frame shape {frame.shape} != {(self.H, self.W)}")
        if self.pose is None:
            return self._anchor(frame)
        A, st = self._align(frame)
        if A is None:
            return self._anchor(frame)
        P = self.patch
        W, ok = self.canvas.read(A, self._xx, self._yy)
        # the per-patch shift reads the reference up to delta_max pixels outside the patch,
        # so a patch is held only if that whole neighbourhood is content the model has
        m = int(np.ceil(self.delta_max))
        near = ok
        if m:
            pad = np.pad(ok, m, constant_values=True); near = ok.copy()
            for dy in range(-m, m + 1):
                for dx in range(-m, m + 1):
                    near &= pad[m + dy: m + dy + self.H, m + dx: m + dx + self.W]
        held = patch_grid(near, P).all((2, 3))
        score, c, Wimg = quotient_scores(frame, np.where(ok, W, 0.0), P, self.deltas, self.delta_max, self.iters)
        keep = (score >= 1.0) | ~held
        view = Wimg.copy()
        vg = patch_grid(view, P); vg += np.where(keep, 0.0, c)[..., None, None]
        followed = np.zeros_like(keep); motion = (np.zeros((self.gh, self.gw)), np.zeros((self.gh, self.gw)))
        if self.search > 0 and (keep & held).any():
            followed, Wf, c2, my, mx = self._follow(frame, W, ok, keep & held)
            keep = keep & ~followed; motion = (my, mx)
            vg[followed] = (patch_grid(Wf, P) + c2[..., None, None])[followed]
        vg[keep] = patch_grid(np.asarray(frame, np.float64), P)[keep]
        # the canvas takes the sent patches from the frame and the followed ones from the view
        known = np.zeros(frame.shape, bool)
        known[: self.gh * P, : self.gw * P] = np.kron(keep | followed, np.ones((P, P), bool))
        self.canvas.write(view if followed.any() else frame, A, known)
        self._prev, self.pose = self.pose, A
        self.last = {"anchor": False, "reveal": ~held, "followed": followed, "motion": motion, "pose": A.copy(),
                     "resid": st["resid"], "overlap": st["overlap"], "corr": st["corr"],
                     "new_area": 1.0 - float(ok.mean()), "view": view}
        return keep
