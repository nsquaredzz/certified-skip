#!/usr/bin/env python3
"""Plant objects into REAL fixed-camera footage and sweep every rule's
threshold: drop rate versus recall, so rules with different certificates can
be compared on the only axis that matters operationally.

Events planted (additive, clipped, random sign where sensible):
  small   5x5 px, contrast 40, one frame
  tiny    3x3 px, contrast 48, one frame
  fade    8x8 px, contrast rising linearly to 48 over 3 s then static;
          caught = patch kept at or before the frame where contrast reaches 32
  moving  6x6 px, contrast 40, moving 2 px/frame for 12 frames;
          coverage = fraction of those frames whose containing patch is kept

    python3 scripts/real_bench.py --clips hall,leftbag,virat --out out/real_bench
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import certskip as cs
from certskip import core, flatnorm, scalespace, warp, video, sequential

CLIPS = {
    "hall":    dict(path="data/hall_monitor_cif.y4m", fps=30, scale=None, max_frames=300),
    "leftbag": dict(path="data/caviar_LeftBag.mpg", fps=12.5, scale=None, max_frames=800),
    "virat":   dict(path="data/VIRAT_S_000200_00.mp4", fps=10, scale=(640, 352), max_frames=300),
    "akiyo":   dict(path="data/akiyo_cif.y4m", fps=30, scale=None, max_frames=300),
}


def schedule(delta0, gamma, radii=(0, 1, 2, 3)):
    return {r: delta0 * (2 * r + 1) ** (-gamma) for r in radii}


def make_rules(H, W, P):
    return {
        "range":        (lambda t: core.Pruner(H, W, P, t),                       [12, 16, 20, 24, 28, 32, 40, 48, 64, 80]),
        "flat l=4":     (lambda t: flatnorm.FlatPruner(H, W, P, t, 4.0),           [150, 200, 300, 400, 600, 800, 1200, 1600, 2400, 3200]),
        "multiscale g=.5": (lambda t: scalespace.MultiScalePruner(H, W, P, schedule(t, 0.5)), [16, 24, 32, 40, 48, 64, 80, 96, 128, 160]),
        "multiscale g=1":  (lambda t: scalespace.MultiScalePruner(H, W, P, schedule(t, 1.0)), [16, 24, 32, 40, 48, 64, 80, 96, 128, 160]),
        "quotient+range":  (lambda t: warp.WarpPruner(H, W, P, t, 1.0),           [8, 12, 16, 20, 24, 28, 32, 40, 48, 64]),
        "quotient+ms g=.5": (lambda t: warp.WarpPruner(H, W, P, 32.0, 1.0, multiscale=schedule(t, 0.5)), [16, 24, 32, 40, 48, 64, 80, 96, 128, 160]),
        "sequential z (ms80)": (lambda t: sequential.SequentialPruner(H, W, P, multiscale=schedule(80.0, 0.5), z=t), [4.0, 5.0, 6.0, 8.0, 10.0, 12.0]),
    }


def plant(frames, rng, P, fps, n_small=40, n_tiny=40, n_fade=20, n_move=20, n_faint=15, faint_contrasts=(8, 12, 16)):
    f = frames.astype(np.int32).copy()
    T, H, W = f.shape; gh, gw = H // P, W // P
    ev = {"small": [], "tiny": [], "fade": [], "moving": [], "faint": []}
    def spot(size):
        gy, gx = rng.integers(0, gh), rng.integers(0, gw)
        y = gy * P + rng.integers(1, P - size); x = gx * P + rng.integers(1, P - size)
        return gy, gx, y, x
    def sign_at(t, y, x, size):
        # push away from saturation so the planted contrast is not clipped away
        m = f[t, y:y + size, x:x + size].mean()
        return -1 if m > 128 else 1
    for _ in range(n_small):
        gy, gx, y, x = spot(5); t = rng.integers(2, T); s = sign_at(t, y, x, 5)
        f[t, y:y+5, x:x+5] += s * 40; ev["small"].append((gy, gx, t, t + 1))
    for _ in range(n_tiny):
        gy, gx, y, x = spot(3); t = rng.integers(2, T); s = sign_at(t, y, x, 3)
        f[t, y:y+3, x:x+3] += s * 48; ev["tiny"].append((gy, gx, t, t + 1))
    nf = max(3, int(3 * fps))
    for _ in range(n_fade):
        gy, gx, y, x = spot(8); t0 = rng.integers(2, max(3, T - nf - 1)); s = sign_at(t0, y, x, 8)
        ramp = np.linspace(0, 48, nf)
        for i, a in enumerate(ramp):
            if t0 + i < T: f[t0 + i, y:y+8, x:x+8] += int(round(s * a))
        f[t0 + nf:, y:y+8, x:x+8] += s * 48
        reach = t0 + int(np.argmax(ramp >= 32))
        ev["fade"].append((gy, gx, t0, reach))
    for _ in range(n_move):
        gy, gx, y, x = spot(6); t0 = rng.integers(2, max(3, T - 13)); s = sign_at(t0, y, x, 6)
        x = min(x, W - 6 - 2 * 12 - 1)
        cells = []
        for i in range(12):
            if t0 + i < T:
                xi = x + 2 * i
                f[t0 + i, y:y+6, xi:xi+6] += s * 40
                cells.append((t0 + i, y // P, xi // P))
        ev["moving"].append(cells)
    persist = max(4, int(3 * fps))
    for cst in faint_contrasts:
        for _ in range(n_faint):
            gy, gx, y, x = spot(6); t0 = rng.integers(2, max(3, T - persist)); s = sign_at(t0, y, x, 6)
            f[t0:t0 + persist, y:y+6, x:x+6] += s * cst
            ev["faint"].append((gy, gx, t0, t0 + persist, cst))
    return np.clip(f, 0, 255).astype(np.uint8), ev


def score(keep, ev, fps=None):
    out = {}
    for kind in ("small", "tiny", "fade"):
        hits = sum(bool(keep[a:b + (1 if kind == "fade" else 0), gy, gx].any()) for gy, gx, a, b in ev[kind])
        out[kind] = hits / max(1, len(ev[kind]))
    cov = [np.mean([keep[t, gy, gx] for t, gy, gx in cells]) for cells in ev["moving"] if cells]
    out["moving_cov"] = float(np.mean(cov)) if cov else 0.0
    # faint persistent objects: recall within 1 s of appearance, and median delay (frames) among detections
    horizon = max(2, int(fps)) if fps else 10
    for cst in sorted(set(e[4] for e in ev["faint"])):
        hits, delays = 0, []
        for gy, gx, a, b, c in ev["faint"]:
            if c != cst:
                continue
            seg = keep[a:min(b, a + horizon), gy, gx]
            if seg.any():
                hits += 1; delays.append(int(np.argmax(seg)))
        n = sum(1 for e in ev["faint"] if e[4] == cst)
        out[f"faint{cst}"] = hits / max(1, n)
        out[f"faint{cst}_delay"] = float(np.median(delays)) if delays else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default="hall,leftbag,virat")
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--out", default="out/real_bench")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rules", default=None, help="comma-separated subset of rule names")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    P = a.patch
    results = {}
    t_all = time.perf_counter()
    for name in a.clips.split(","):
        spec = CLIPS[name]
        frames = video.read_gray(spec["path"], fps=spec["fps"], scale=spec["scale"], max_frames=spec["max_frames"])
        T, H, W = frames.shape
        planted, ev = plant(frames, rng, P, spec["fps"])
        print(f"\n=== {name}: {T} frames {W}x{H} @ {spec['fps']} fps, {sum(len(v) for v in ev.values())} events")
        rules = make_rules(H, W, P)
        if a.rules:
            want = [r.strip() for r in a.rules.split(",")]
            rules = {k: v for k, v in rules.items() if k in want}
        results[name] = {}
        for rname, (factory, thrs) in rules.items():
            rows = []
            t0 = time.perf_counter()
            for thr in thrs:
                keep = factory(thr).run(planted)["keep"]
                r = score(keep, ev, spec["fps"]); r["thr"] = thr; r["drop"] = core.drop_rate(keep)
                rows.append(r)
            results[name][rname] = rows
            # best drop rate with small=100% and fade=100%
            ok = [r for r in rows if r["small"] >= 1.0 and r["fade"] >= 1.0]
            best = max(ok, key=lambda r: r["drop"]) if ok else None
            ok95 = [r for r in rows if r["small"] >= 0.95 and r["tiny"] >= 0.95 and r["fade"] >= 0.95]
            best95 = max(ok95, key=lambda r: r["drop"]) if ok95 else None
            def fmt(b):
                if b is None:
                    return "none"
                faint = "  faint8/12/16 recall " + "/".join(f"{b.get(f'faint{c}', float('nan')):.2f}" for c in (8, 12, 16)) + \
                        " delay " + "/".join(f"{b.get(f'faint{c}_delay', float('nan')):.0f}" for c in (8, 12, 16))
                return f"drop {b['drop']:.3f} @thr {b['thr']} (small {b['small']:.2f} tiny {b['tiny']:.2f} fade {b['fade']:.2f} move-cov {b['moving_cov']:.2f}){faint}"
            print(f"  {rname:18s} [{time.perf_counter()-t0:4.0f}s]  100% small+fade: {fmt(best)}\n  {'':18s}          >=95% all:      {fmt(best95)}")
    Path(a.out + ".json").write_text(json.dumps(results, indent=1))
    # plot: drop vs recall(small) and drop vs recall(tiny), per clip
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    clips = list(results)
    fig, axes = plt.subplots(len(clips), 4, figsize=(20, 4 * len(clips)), squeeze=False)
    for i, cname in enumerate(clips):
        for j, kind in enumerate(("small", "tiny", "moving_cov", "faint12")):
            ax = axes[i, j]
            for rname, rows in results[cname].items():
                ax.plot([r["drop"] for r in rows], [r.get(kind, np.nan) for r in rows], "o-", ms=3, label=rname)
            ax.set_xlabel("drop rate"); ax.set_ylabel({"small": "recall 5x5/40", "tiny": "recall 3x3/48", "moving_cov": "tracking coverage 6x6/40", "faint12": "recall faint 6x6/12 within 1 s"}[kind])
            ax.set_title(cname); ax.grid(alpha=.3); ax.set_ylim(-0.02, 1.02)
            if i == 0 and j == 0: ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(a.out + ".png", dpi=120)
    print(f"\nwrote {a.out}.json / .png in {time.perf_counter()-t_all:.0f}s")


if __name__ == "__main__":
    main()
