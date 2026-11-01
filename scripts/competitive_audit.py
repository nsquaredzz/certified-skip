#!/usr/bin/env python3
"""Theorem D on footage: the causal anchored rule against the clairvoyant optimum.

For every patch of each clip, the number of keeps of the range rule at Delta, the fewest keeps of
any Delta-sound schedule (OPT_Delta, by dynamic programming with full knowledge of the future), and
OPT_{Delta/2}.  Checks (i) OPT_Delta <= m_Delta and (ii) m_Delta <= 2 OPT_{Delta/2} + 1 on every
patch and reports the ratio m_Delta / OPT_Delta.  The same for the L1-mean statistic (the anchored
mean rule).

    python3 scripts/competitive_audit.py --clips hall,lobby,virat --deltas 16,32,48 --out out/competitive
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from certskip import core, video, optimal as op

CLIPS = {
    "hall":  dict(path="data/hall_monitor_cif.y4m", fps=30, scale=None),
    "lobby": dict(path="data/caviar_LeftBag.mpg", fps=25, scale=None),
    "virat": dict(path="data/VIRAT_S_000200_00.mp4", fps=10, scale=(640, 352)),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default="hall,lobby,virat")
    ap.add_argument("--deltas", default="16,32,48")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--patches", type=int, default=0, help="random subset of patches (0 = all)")
    ap.add_argument("--out", default="out/competitive")
    a = ap.parse_args()
    P = 16
    rng = np.random.default_rng(0)
    results = {}
    for cname in a.clips.split(","):
        spec = CLIPS[cname]
        fr = video.read_gray(spec["path"], fps=spec["fps"], scale=spec["scale"], max_frames=a.frames)
        T, H, W = fr.shape
        g = np.stack([core.patch_grid(f, P) for f in fr]).astype(np.float64)
        gh, gw = g.shape[1:3]
        idx = np.arange(gh * gw)
        if a.patches:
            idx = rng.choice(idx, min(a.patches, len(idx)), replace=False)
        results[cname] = {"T": T, "patches": int(len(idx))}
        for stat_name, stat, deltas in (("range", op.stat_range, [float(d) for d in a.deltas.split(",")]),
                                        ("mean", op.stat_mean_abs, [2.0, 4.0, 8.0])):
            for delta in deltas:
                t0 = time.perf_counter()
                m = o = oh = 0; viol = 0; ratios = []
                for s in idx:
                    X = g[:, s // gw, s % gw]
                    r = op.competitive_check(X, delta, stat)
                    m += r["m"]; o += r["opt_delta"]; oh += r["opt_half"]
                    viol += (not r["i_holds"]) + (not r["ii_holds"]); ratios.append(r["ratio"])
                rec = {"m": m, "opt_delta": o, "opt_half": oh, "violations": viol, "ratio_total": m / max(o, 1),
                       "ratio_patch_median": float(np.median(ratios)), "ratio_patch_max": float(np.max(ratios)),
                       "drop_rule": 1 - m / (T * len(idx)), "drop_opt": 1 - o / (T * len(idx)), "seconds": time.perf_counter() - t0}
                results[cname][f"{stat_name}@{delta:g}"] = rec
                print(f"{cname:6s} {stat_name:5s} theta={delta:4g}: rule {m:7d} keeps, OPT {o:7d}, OPT at theta/2 {oh:7d}; "
                      f"ratio {rec['ratio_total']:.3f} (patch median {rec['ratio_patch_median']:.2f}, max {rec['ratio_patch_max']:.1f}); "
                      f"violations {viol}; drop rule/OPT {rec['drop_rule']:.3f}/{rec['drop_opt']:.3f}  [{rec['seconds']:.0f}s]")
        Path(a.out + ".json").write_text(json.dumps(results, indent=1))
    print(f"wrote {a.out}.json")


if __name__ == "__main__":
    main()
