#!/usr/bin/env python3
"""Synthetic fixed-camera benchmark: plant small and slowly-fading objects in a
static scene, pass it through a real H.264 encoder, set every rule to the same
drop rate, and count which rule keeps the patch when the object is present.

    python3 scripts/synthetic_bench.py [--scenes 3] [--seconds 40] [--fps 2]
        [--size 640x360] [--patch 16] [--delta 32] [--crf 23] [--sigma 2.0]
        [--small-trials 32] [--slow-trials 32] [--audit 2000] [--seed 0]

"Caught" for a small object (5x5 px, 0.3 s = 1 sampled frame at 2 fps unless
--small-frames is larger): the patch is kept on at least one frame where the
object is present.  "Caught" for a slow fade-in (linear over --fade-seconds):
the patch is kept at or before the frame where the object's true contrast
first reaches delta.  Drop rates are matched on object-free control clips.
"""
import argparse, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import certskip as cs
from certskip import baselines as bl, topology as tp, video, noise as nz


def make_scene(rng, H, W, kind):
    yy, xx = np.mgrid[0:H, 0:W]
    if kind == 0:   # soft gradient + a few blobs (indoor wall)
        img = 90 + 40 * (xx / W) + 20 * np.sin(yy / 23.0)
        for _ in range(12):
            cy, cx, r, a = rng.uniform(0, H), rng.uniform(0, W), rng.uniform(10, 60), rng.uniform(-40, 40)
            img += a * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))
    elif kind == 1:  # textured (brick / foliage-like)
        img = 110 + 35 * np.sin(xx / 5.0) * np.cos(yy / 7.0) + 15 * rng.normal(size=(H, W))
        img = np.clip(img, 0, 255)
        from numpy.fft import fft2, ifft2
        k = np.exp(-((yy - H / 2) ** 2 + (xx - W / 2) ** 2) / (2 * 1.2 ** 2)); k /= k.sum()
        img = np.real(ifft2(fft2(img) * fft2(np.fft.ifftshift(k))))
    else:            # high contrast geometric (door frames, windows)
        img = np.full((H, W), 140.0)
        for _ in range(8):
            y0, x0 = rng.integers(0, H - 20), rng.integers(0, W - 20)
            h, w = rng.integers(10, H // 2), rng.integers(10, W // 2)
            img[y0:y0 + h, x0:x0 + w] = rng.uniform(30, 230)
    return np.clip(img, 0, 255)


def render(scene, T, sigma, rng, flicker=0.0):
    out = scene[None] + rng.normal(0, sigma, (T,) + scene.shape)
    if flicker:
        out = out + rng.normal(0, flicker, (T, 1, 1))     # global brightness flicker
    return np.clip(out, 0, 255).astype(np.uint8)


def plant_small(frames, rng, P, size, nframes, contrast, margin=2):
    T, H, W = frames.shape
    gy, gx = rng.integers(0, H // P), rng.integers(0, W // P)
    y = gy * P + rng.integers(margin, P - size - margin + 1)
    x = gx * P + rng.integers(margin, P - size - margin + 1)
    t0 = rng.integers(1, T - nframes)
    sign = -1 if rng.random() < 0.5 else 1
    f = frames.copy().astype(np.int32)
    f[t0:t0 + nframes, y:y + size, x:x + size] += sign * contrast
    return np.clip(f, 0, 255).astype(np.uint8), (gy, gx, t0, t0 + nframes)


def plant_slow(frames, rng, P, size, fade_frames, final_contrast, delta):
    T, H, W = frames.shape
    gy, gx = rng.integers(0, H // P), rng.integers(0, W // P)
    y = gy * P + (P - size) // 2; x = gx * P + (P - size) // 2
    t0 = rng.integers(1, max(2, T - fade_frames - 1))
    sign = -1 if rng.random() < 0.5 else 1
    f = frames.copy().astype(np.float64)
    ramp = np.linspace(0, final_contrast, fade_frames)
    for i, a in enumerate(ramp):
        if t0 + i < T:
            f[t0 + i, y:y + size, x:x + size] += sign * a
    f[t0 + fade_frames:, y:y + size, x:x + size] += sign * final_contrast
    # frame at which the planted contrast first reaches delta
    reach = t0 + int(np.argmax(ramp >= delta)) if ramp[-1] >= delta else T - 1
    return np.clip(np.rint(f), 0, 255).astype(np.uint8), (gy, gx, t0, reach)


def caught_small(keep, ev):
    gy, gx, a, b = ev
    return bool(keep[a:b, gy, gx].any())


def caught_slow(keep, ev):
    gy, gx, t0, reach = ev
    return bool(keep[t0:reach + 1, gy, gx].any())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=40)
    ap.add_argument("--fps", type=float, default=2.0, help="sampled frame rate")
    ap.add_argument("--size", default="640x360")
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--delta", type=float, default=32.0)
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--gop", type=int, default=None, help="encoder GOP in sampled frames (default encoder choice)")
    ap.add_argument("--sigma", type=float, default=2.0, help="sensor noise std before encoding")
    ap.add_argument("--flicker", type=float, default=0.0, help="global brightness flicker std")
    ap.add_argument("--small-size", type=int, default=5)
    ap.add_argument("--small-frames", type=int, default=1)
    ap.add_argument("--small-contrast", type=int, default=48)
    ap.add_argument("--small-trials", type=int, default=32)
    ap.add_argument("--slow-size", type=int, default=8)
    ap.add_argument("--fade-seconds", type=float, default=8.0)
    ap.add_argument("--slow-contrast", type=int, default=64)
    ap.add_argument("--slow-trials", type=int, default=32)
    ap.add_argument("--audit", type=int, default=2000)
    ap.add_argument("--target-cap", type=float, default=0.97,
                    help="baselines are calibrated to min(certified drop rate, cap); so when the certified rule drops "
                         "more than the cap the baselines get MORE keep budget than it, never less")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    W, H = (int(x) for x in a.size.lower().split("x"))
    T = int(a.seconds * a.fps)
    P, delta = a.patch, a.delta
    fade_frames = max(2, int(a.fade_seconds * a.fps))
    print(f"{a.scenes} scenes, {T} sampled frames at {a.fps} fps, {W}x{H}, patch {P}, delta {delta}, crf {a.crf}, sigma {a.sigma}")
    print(f"engine: {'C++ (ctypes)' if cs.native_available() else 'numpy'}")

    rules = ["certified", "consecutive_mean", "mean_vs_kept", "uniform"]
    tally = {r: {"small": [0, 0], "slow": [0, 0]} for r in rules}
    worst_err = {"certified": 0.0, "mean_vs_kept": 0.0}
    audited = {"checked": 0, "violations": 0, "worst": 0.0}
    t_start = time.perf_counter()

    for s in range(a.scenes):
        scene = make_scene(rng, H, W, s % 3)
        # --- control clip: calibrate every rule to the certified drop rate
        ctrl = video.encode_roundtrip(render(scene, T, a.sigma, rng, a.flicker), crf=a.crf, gop=a.gop, fps=a.fps)
        ck = bl.certified(ctrl, P, delta, pruner_cls=cs.NativePruner if cs.native_available() else cs.Pruner)
        cert_drop = cs.core.drop_rate(ck)
        target = min(cert_drop, a.target_cap)
        tau_c, _, d_c = bl.calibrate(bl.consecutive_mean, ctrl, P, target)
        tau_k, _, d_k = bl.calibrate(bl.mean_vs_kept, ctrl, P, target, iters=16)
        stride = max(1, int(round(1.0 / max(1e-9, 1.0 - target))))
        sig = nz.estimate_temporal_noise(ctrl, P)["sigma_pixel"]
        print(f"\nscene {s}: post-codec sigma {sig:.2f}; certified drop {cert_drop:.3f} (baseline target {target:.3f}) | consecutive tau {tau_c:.2f} ({d_c:.3f}) | "
              f"mean-vs-kept tau {tau_k:.2f} ({d_k:.3f}) | uniform stride {stride} ({1-1/stride:.3f})")

        def run_all(frames):
            out = {}
            pr = cs.make_pruner(H, W, P, delta, dtype=frames.dtype)
            r = pr.run(frames); out["certified"] = r["keep"]
            worst_err["certified"] = max(worst_err["certified"], cs.core.certified_error(r["keep"], r["spread"]))
            out["consecutive_mean"] = bl.consecutive_mean(frames, P, tau_c)
            out["mean_vs_kept"] = bl.mean_vs_kept(frames, P, tau_k)
            out["uniform"] = bl.uniform(T, H // P, W // P, stride)
            return out, r

        # --- small objects (several per clip to amortise encoding)
        per_clip = 8
        for _ in range(int(np.ceil(a.small_trials / per_clip))):
            clean = render(scene, T, a.sigma, rng, a.flicker)
            evs = []
            for _ in range(per_clip):
                clean, ev = plant_small(clean, rng, P, a.small_size, a.small_frames, a.small_contrast)
                evs.append(ev)
            enc = video.encode_roundtrip(clean, crf=a.crf, gop=a.gop, fps=a.fps)
            keeps, r = run_all(enc)
            for ev in evs:
                for name in rules:
                    tally[name]["small"][1] += 1
                    tally[name]["small"][0] += caught_small(keeps[name], ev)
            if a.audit and audited["checked"] < a.audit:
                rep = tp.soundness_audit(enc, keeps["certified"], P, delta, max_patches=min(400, a.audit - audited["checked"]), rng=rng)
                audited["checked"] += rep["checked"]; audited["violations"] += rep["violations"]
                audited["worst"] = max(audited["worst"], rep["worst_bottleneck_over_half_spread"])

        # --- slow fades
        per_clip = 4
        for _ in range(int(np.ceil(a.slow_trials / per_clip))):
            clean = render(scene, T, a.sigma, rng, a.flicker)
            evs = []
            for _ in range(per_clip):
                clean, ev = plant_slow(clean, rng, P, a.slow_size, fade_frames, a.slow_contrast, delta)
                evs.append(ev)
            enc = video.encode_roundtrip(clean, crf=a.crf, gop=a.gop, fps=a.fps)
            keeps, r = run_all(enc)
            # worst view error of mean-vs-kept: it drops with no bound; measure actual
            ref = enc[0].copy(); rg = cs.patch_grid(ref, P)
            for t in range(1, T):
                fg = cs.patch_grid(enc[t], P)
                k = keeps["mean_vs_kept"][t]
                if (~k).any():
                    e = np.abs(fg[~k].astype(np.int32) - rg[~k].astype(np.int32)).max()
                    worst_err["mean_vs_kept"] = max(worst_err["mean_vs_kept"], float(e))
                rg[k] = fg[k]
            for ev in evs:
                for name in rules:
                    tally[name]["slow"][1] += 1
                    tally[name]["slow"][0] += caught_slow(keeps[name], ev)

    print(f"\n{'rule':<22} {'small ' + str(a.small_size) + 'px, ' + str(a.small_frames) + ' frame(s)':>28} {'slow fade ' + str(a.fade_seconds) + ' s':>18} {'worst view error':>18}")
    for name in rules:
        sm, sl = tally[name]["small"], tally[name]["slow"]
        we = f"{worst_err[name]:.1f} (certified bound)" if name == "certified" else (f"{worst_err[name]:.0f} (measured)" if name in worst_err else "n/a")
        print(f"{name:<22} {sm[0]:>12} / {sm[1]:<13} {sl[0]:>7} / {sl[1]:<8} {we:>18}")
    if a.audit:
        print(f"\nsoundness audit: {audited['checked']} dropped patches, {audited['violations']} violations, worst bottleneck/(s/2) = {audited['worst']:.3f}")
    print(f"elapsed {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
