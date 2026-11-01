#!/usr/bin/env python3
"""Stage 1 audit: do published video token reducers see slow-onset events, at matched budgets?

Objects are planted into real fixed-camera footage (small, tiny, 3-s fade, 12-s fade, mover, faint
persistent), every rule's threshold is calibrated on the untouched clip to a list of drop rates, and
recall per event type is read off at each budget.  The reducers are the decision rules of
python/certskip/reducers.py, i.e. what each published method compares, not its full pipeline.

Two frame rates: the clip's native rate (what a streaming encoder sees) and 2 fps (what a video
LLM is usually fed).  Budgets are reported as drop rate at the given rate and as tokens per second.

    python3 scripts/audit_reducers.py --clips hall,lobby,virat --rates native,2 --out out/audit

Also measured, for Proposition E: the static false-fire rate rho of each consecutive rule on the
fading patches before the fade, and the predicted (1 - rho)^T against the measured miss rate.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import certskip as cs
from certskip import core, video, baselines as bl, reducers as rd, scalespace, warp, sequential

CLIPS = {
    "hall":  dict(path="data/hall_monitor_cif.y4m", native=30, scale=None),
    "lobby": dict(path="data/caviar_LeftBag.mpg", native=25, scale=None),
    "virat": dict(path="data/VIRAT_S_000200_00.mp4", native=10, scale=(640, 352)),
}
TARGETS = [0.80, 0.90, 0.95, 0.97, 0.98, 0.99]


def schedule(delta0, gamma=0.5, radii=(0, 1, 2, 3)):
    return {r: delta0 * (2 * r + 1) ** (-gamma) for r in radii}


def plant(frames, rng, P, fps, n_small=30, n_tiny=30, n_fade=20, n_slow=20, n_move=15, n_faint=15):
    """Events: small 5x5/40 one frame; tiny 3x3/48 one frame; fade 8x8 to 48 over 3 s; slow 12x12 to 48 over
    12 s; mover 6x6/40 at 2 px/frame for 12 frames; faint 6x6/12 persistent 3 s.  Caught: patch kept while
    the object is present (small, tiny, faint within 1 s), or at or before the frame where a fade reaches
    contrast 32 (fade, slow).  Fades keep their final contrast afterwards, so a late keep is still
    recorded as the delay in seconds after reaching 32."""
    f = frames.astype(np.int32).copy()
    T, H, W = f.shape; gh, gw = H // P, W // P
    ev = {k: [] for k in ("small", "tiny", "fade", "slow", "moving", "faint")}

    def spot(size):
        gy, gx = rng.integers(0, gh), rng.integers(0, gw)
        y = gy * P + rng.integers(1, max(2, P - size)); x = gx * P + rng.integers(1, max(2, P - size))
        return gy, gx, y, x

    def sign_at(t, y, x, size):
        return -1 if f[t, y:y + size, x:x + size].mean() > 128 else 1

    for _ in range(n_small):
        gy, gx, y, x = spot(5); t = rng.integers(2, T); s = sign_at(t, y, x, 5)
        f[t, y:y + 5, x:x + 5] += s * 40; ev["small"].append((gy, gx, t, t + 1))
    for _ in range(n_tiny):
        gy, gx, y, x = spot(3); t = rng.integers(2, T); s = sign_at(t, y, x, 3)
        f[t, y:y + 3, x:x + 3] += s * 48; ev["tiny"].append((gy, gx, t, t + 1))
    for kind, secs, size, n in (("fade", 3.0, 8, n_fade), ("slow", 12.0, 12, n_slow)):
        nf = max(3, int(secs * fps))
        for _ in range(n):
            gy, gx, y, x = spot(size); t0 = rng.integers(2, max(3, T - nf // 2)); s = sign_at(t0, y, x, size)
            ramp = np.linspace(0, 48, nf)
            for i, a in enumerate(ramp):
                if t0 + i < T:
                    f[t0 + i, y:y + size, x:x + size] += int(round(s * a))
            if t0 + nf < T:
                f[t0 + nf:, y:y + size, x:x + size] += s * 48
            reach = t0 + int(np.argmax(ramp >= 32))
            ev[kind].append((gy, gx, t0, min(reach, T - 1)))
    for _ in range(n_move):
        gy, gx, y, x = spot(6); t0 = rng.integers(2, max(3, T - 13)); s = sign_at(t0, y, x, 6)
        x = min(x, W - 6 - 2 * 12 - 1)
        cells = []
        for i in range(12):
            if t0 + i < T:
                xi = x + 2 * i
                f[t0 + i, y:y + 6, xi:xi + 6] += s * 40
                cells.append((t0 + i, y // P, xi // P))
        ev["moving"].append(cells)
    persist = max(4, int(3 * fps))
    for _ in range(n_faint):
        gy, gx, y, x = spot(6); t0 = rng.integers(2, max(3, T - persist)); s = sign_at(t0, y, x, 6)
        f[t0:t0 + persist, y:y + 6, x:x + 6] += s * 12
        ev["faint"].append((gy, gx, t0, t0 + persist))
    return np.clip(f, 0, 255).astype(np.uint8), ev


def score(keep, ev, fps):
    out = {}
    T = keep.shape[0]
    for kind in ("small", "tiny"):
        out[kind] = float(np.mean([bool(keep[a:b, gy, gx].any()) for gy, gx, a, b in ev[kind]])) if ev[kind] else np.nan
    for kind in ("fade", "slow"):
        hits, late = [], []
        for gy, gx, t0, reach in ev[kind]:
            hits.append(bool(keep[t0:reach + 1, gy, gx].any()))
            seg = keep[t0:, gy, gx]                                                   # first keep at or after the onset
            late.append(((t0 + int(np.argmax(seg))) - reach) / fps if seg.any() else np.inf)   # seconds relative to reaching 32
        out[kind] = float(np.mean(hits)); out[kind + "_delay_s"] = float(np.median([l for l in late if l < np.inf])) if any(l < np.inf for l in late) else np.inf
        out[kind + "_never"] = float(np.mean([l == np.inf for l in late]))           # never sent from the onset to the end
    cov = [np.mean([keep[t, gy, gx] for t, gy, gx in cells]) for cells in ev["moving"] if cells]
    out["moving_cov"] = float(np.mean(cov)) if cov else np.nan
    horizon = max(2, int(fps))
    out["faint"] = float(np.mean([bool(keep[a:min(b, a + horizon), gy, gx].any()) for gy, gx, a, b in ev["faint"]]))
    return out


def static_fire_rate(keep, ev, kinds=("fade", "slow")):
    """Keep rate on the planted patches before their event starts (the rule's own rho on those patches):
    mean, median, and the fraction of patches on which the rule never fired before the event."""
    rates = []
    for kind in kinds:
        for gy, gx, t0, _ in ev[kind]:
            if t0 > 10:
                rates.append(float(keep[1:t0, gy, gx].mean()))
    if not rates:
        return {"mean": np.nan, "median": np.nan, "zero_frac": np.nan}
    return {"mean": float(np.mean(rates)), "median": float(np.median(rates)), "zero_frac": float(np.mean([r == 0 for r in rates]))}


def certified_rules(H, W, P):
    native = cs.native_available()
    rules = {
        "certified range": (lambda d: cs.make_pruner(H, W, P, d), [16, 20, 24, 28, 32, 40, 48, 64, 80, 96]),
        "certified quotient+MS": (lambda d: warp.WarpPruner(H, W, P, 32.0, 1.0, multiscale=schedule(d)), [24, 32, 40, 48, 64, 80, 96, 128, 160]),
    }
    if native:
        from certskip.native import NativeSequentialPruner
        rules["certified +sequential"] = (lambda d: NativeSequentialPruner(H, W, P, multiscale=schedule(d), z=8.0), [24, 32, 40, 48, 64, 80, 96, 128, 160])
    else:
        rules["certified +sequential"] = (lambda d: sequential.SequentialPruner(H, W, P, multiscale=schedule(d), z=8.0), [32, 48, 64, 80, 96, 128])
    return rules


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default="hall,lobby,virat")
    ap.add_argument("--rates", default="native,2")
    ap.add_argument("--seconds", type=float, default=40.0, help="length analysed at the native rate (2 fps uses 3x)")
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--out", default="out/audit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scores", default=None, help="JSON of precomputed per-patch scores for attention-based reducers (not used here)")
    a = ap.parse_args()
    P = a.patch
    results = {}
    t_all = time.perf_counter()
    for cname in a.clips.split(","):
        spec = CLIPS[cname]
        for rate in a.rates.split(","):
            fps = spec["native"] if rate == "native" else float(rate)
            secs = a.seconds if rate == "native" else 3 * a.seconds
            frames = video.read_gray(spec["path"], fps=fps, scale=spec["scale"], max_frames=int(secs * fps))
            T, H, W = frames.shape; gh, gw = H // P, W // P
            if T < 20 * fps:
                print(f"skip {cname} at {fps:g} fps: only {T} frames ({T / fps:.0f} s)"); continue
            rng = np.random.default_rng(a.seed)
            planted, ev = plant(frames, rng, P, fps)
            key = f"{cname}@{fps:g}fps"
            print(f"\n=== {key}: {T} frames {W}x{H}, {sum(len(v) for v in ev.values())} events; tokens/s at full rate {fps * gh * gw:.0f}")
            res = {"fps": fps, "T": T, "H": H, "W": W, "tokens_per_s_full": fps * gh * gw, "rules": {}}
            # certified rules: sweep their own thresholds on the planted clip, drop measured on the untouched clip
            for rname, (factory, grid) in certified_rules(H, W, P).items():
                rows = []; t0 = time.perf_counter()
                for thr in grid:
                    keep = factory(thr).run(planted)["keep"]
                    drop_ctrl = core.drop_rate(factory(thr).run(frames)["keep"])
                    r = score(keep, ev, fps); r.update(thr=thr, drop=drop_ctrl, drop_planted=core.drop_rate(keep),
                                                     tokens_per_s=(1 - drop_ctrl) * fps * gh * gw, rho=static_fire_rate(keep, ev))
                    rows.append(r)
                res["rules"][rname] = rows
                print(f"  {rname:36s} [{time.perf_counter()-t0:4.0f}s] " + "  ".join(f"d{r['drop']:.3f}:f{r['fade']:.2f}/s{r['slow']:.2f}" for r in rows))
            # reducers: calibrate on the untouched clip to each target drop rate
            for rname, (fn, _, desc) in rd.REGISTRY.items():
                rows = []; t0 = time.perf_counter()
                hi = 2.0 if "cosine" in rname or "dycoke" in rname else 255.0
                for target in TARGETS:
                    thr, _, d = bl.calibrate(fn, frames, P, target, hi=hi, iters=18)
                    if abs(d - target) > 0.02:
                        continue                                   # the method cannot reach this budget at this rate
                    keep = fn(planted, P, thr)
                    r = score(keep, ev, fps); r.update(thr=thr, drop=d, drop_planted=core.drop_rate(keep),
                                                     tokens_per_s=(1 - d) * fps * gh * gw, rho=static_fire_rate(keep, ev))
                    rows.append(r)
                res["rules"][rname] = rows
                print(f"  {rname:36s} [{time.perf_counter()-t0:4.0f}s] " + "  ".join(f"d{r['drop']:.3f}:f{r['fade']:.2f}/s{r['slow']:.2f}" for r in rows))
            # uniform frame sampling
            rows = []
            for target in TARGETS:
                keep = bl.uniform_for_drop(T, gh, gw, target); d = core.drop_rate(keep)
                r = score(keep, ev, fps); r.update(thr=None, drop=d, drop_planted=d, tokens_per_s=(1 - d) * fps * gh * gw, rho=1 - d)
                rows.append(r)
            res["rules"]["uniform frames"] = rows
            results[key] = res
            Path(a.out + ".json").write_text(json.dumps(results, indent=1, default=float))
    print(f"\nwrote {a.out}.json in {time.perf_counter()-t_all:.0f}s")


if __name__ == "__main__":
    main()
