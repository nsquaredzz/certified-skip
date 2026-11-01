#!/usr/bin/env python3
"""What a cached token cannot carry: the cost of refusing the brightness shift c.

A dropped patch is certified equal to its reference *up to a brightness shift c* (the midrange of the
change).  The model's view adds c back; a token cached inside a model cannot, unless the pipeline is
affine (Proposition F: for a linear patch embedding E, E(W + c) = E(W) + c E(1) exactly, and nothing
after the first nonlinearity has that property).  The alternative is the keep rule: send the patch
whenever |c| exceeds a cap, so that every reused token is within Delta/2 + cap of the truth pointwise
with no shift to apply.  This script measures what that costs in drop rate on fixed cameras, as a
function of the cap, for the range rule and for quotient + multi-scale.

    python3 scripts/shift_cost.py --clips hall,lobby,virat --caps 0,2,4,8,16,32 --out out/shift_cost
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from certskip import core, video, warp

CLIPS = {
    "hall":  dict(path="data/hall_monitor_cif.y4m", fps=30, scale=None),
    "lobby": dict(path="data/caviar_LeftBag.mpg", fps=25, scale=None),
    "virat": dict(path="data/VIRAT_S_000200_00.mp4", fps=10, scale=(640, 352)),
}


def schedule(delta0, gamma=0.5, radii=(0, 1, 2, 3)):
    return {r: delta0 * (2 * r + 1) ** (-gamma) for r in radii}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default="hall,lobby,virat")
    ap.add_argument("--caps", default="0,1,2,4,8,16,32,inf")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--out", default="out/shift_cost")
    a = ap.parse_args()
    P = 16
    caps = [float(c) for c in a.caps.split(",")]
    results = {}
    for cname in a.clips.split(","):
        spec = CLIPS[cname]
        fr = video.read_gray(spec["path"], fps=spec["fps"], scale=spec["scale"], max_frames=a.frames)
        T, H, W = fr.shape
        results[cname] = {}
        r_range = core.Pruner(H, W, P, 32.0).run(fr)
        r_qms = warp.WarpPruner(H, W, P, 32.0, 1.0, multiscale=schedule(80.0)).run(fr)
        for rname, r in (("range d=32", r_range), ("quotient+MS d0=80", r_qms)):
            keep, shift = r["keep"], r["shift"]
            row = {}
            for cap in caps:
                k = keep | (np.abs(shift) > cap) if np.isfinite(cap) else keep
                k[0] = True
                row[str(cap)] = core.drop_rate(k)
            # distribution of |c| on dropped patches
            cs = np.abs(shift[1:][~keep[1:]])
            row["abs_c_median"] = float(np.median(cs)); row["abs_c_p90"] = float(np.percentile(cs, 90)); row["abs_c_p99"] = float(np.percentile(cs, 99))
            results[cname][rname] = row
            print(f"{cname:6s} {rname:18s} drop by cap: " + "  ".join(f"{c}:{row[str(c)]:.3f}" for c in caps) +
                  f"   |c| on dropped: median {row['abs_c_median']:.1f} p90 {row['abs_c_p90']:.1f} p99 {row['abs_c_p99']:.1f}")
    Path(a.out + ".json").write_text(json.dumps(results, indent=1))
    print(f"wrote {a.out}.json")


if __name__ == "__main__":
    main()
