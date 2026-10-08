#!/usr/bin/env python3
"""Breadth: every fixed-camera clip in data/, one table.

Per clip: resolution, fps, robust temporal noise, oracle real-change fraction, drop rate of the
three certified rules at their Part-II/III operating points (range Delta=32; quotient+multi-scale
Delta0=80; + sequential z'=8), and planted-object recall (small 5x5/40, tiny 3x3/48, fade 8x8->48,
faint persistent 6x6 at contrast 12) for the quotient+multi-scale and sequential rules.

    python3 scripts/breadth_bench.py [--max-frames 600] [--out out/breadth]
"""
import argparse, glob, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import certskip as cs
from certskip import noise as nz, video
from certskip.core import drop_rate, patch_grid
from certskip.native import NativeQuotientPruner, NativeSequentialPruner
from real_bench import plant, score, schedule

P = 16
EXCLUDE = ("tears_of_steel",)                       # moving camera, not in scope


def clip_list():
    out = []
    for f in sorted(glob.glob("data/*")):
        name = Path(f).name
        if any(e in name for e in EXCLUDE) or not name.endswith((".mp4", ".y4m", ".mpg", ".mov")):
            continue
        out.append(f)
    return out


def oracle_fraction(frames, thr=30, min_px=8):
    fr = frames.astype(np.int16); bg = np.median(fr[::3], axis=0)
    diff = np.abs(fr - bg) > thr
    maj = np.zeros_like(diff); maj[1:-1] = (diff[:-2].astype(int) + diff[1:-1] + diff[2:]) >= 2
    return float(np.mean([patch_grid(m, P).sum(axis=(2, 3)).__ge__(min_px).mean() for m in maj[1:]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=600); ap.add_argument("--out", default="out/breadth")
    ap.add_argument("--hd-scale", default="960x544")
    a = ap.parse_args()
    sched = schedule(80.0, 0.5)
    rows = []
    rng = np.random.default_rng(0)
    for f in clip_list():
        info = video.probe(f)
        W0, H0, fps = info["width"], info["height"], info["fps"] or 25.0
        scale = tuple(int(x) for x in a.hd_scale.split("x")) if W0 > 1000 else None
        t0 = time.perf_counter()
        fr = video.read_gray(f, fps=fps, scale=scale, max_frames=a.max_frames)
        T, H, W = fr.shape
        sigma = nz.estimate_temporal_noise(fr, P)["sigma_pixel"]
        orc = oracle_fraction(fr)
        d_range = drop_rate(NativeQuotientPruner(H, W, P, 32.0, 1.0).run(fr)["keep"])
        d_ms = drop_rate(NativeSequentialPruner(H, W, P, multiscale=sched, z=1e9).run(fr)["keep"])
        d_seq = drop_rate(NativeSequentialPruner(H, W, P, multiscale=sched, z=8.0).run(fr)["keep"])
        planted, ev = plant(fr, rng, P, fps, n_small=20, n_tiny=20, n_fade=10, n_move=0, n_faint=10, faint_contrasts=(12,))
        r_ms = score(NativeSequentialPruner(H, W, P, multiscale=sched, z=1e9).run(planted)["keep"], ev, fps)
        r_seq = score(NativeSequentialPruner(H, W, P, multiscale=sched, z=8.0).run(planted)["keep"], ev, fps)
        row = dict(clip=Path(f).name, res=f"{W}x{H}", fps=round(fps, 1), frames=T, sigma=round(sigma, 2), oracle=round(orc, 3),
                   drop_range=round(d_range, 3), drop_ms=round(d_ms, 3), drop_seq=round(d_seq, 3),
                   ms_small=r_ms["small"], ms_tiny=r_ms["tiny"], ms_fade=r_ms["fade"], ms_faint12=r_ms["faint12"],
                   seq_small=r_seq["small"], seq_tiny=r_seq["tiny"], seq_fade=r_seq["fade"], seq_faint12=r_seq["faint12"],
                   secs=round(time.perf_counter() - t0, 1))
        rows.append(row)
        print(f"{row['clip']:36s} {row['res']:>9s} {row['fps']:5.1f} σ={row['sigma']:4.2f} oracle={row['oracle']:.3f} | drop range {row['drop_range']:.3f} ms {row['drop_ms']:.3f} seq {row['drop_seq']:.3f} | "
              f"ms recall s/t/f/faint {row['ms_small']:.2f}/{row['ms_tiny']:.2f}/{row['ms_fade']:.2f}/{row['ms_faint12']:.2f} | seq {row['seq_small']:.2f}/{row['seq_tiny']:.2f}/{row['seq_fade']:.2f}/{row['seq_faint12']:.2f}  [{row['secs']}s]", flush=True)
    Path(a.out + ".json").write_text(json.dumps(rows, indent=1))
    d = np.array([[r["drop_range"], r["drop_ms"], r["drop_seq"]] for r in rows])
    print(f"\n{len(rows)} clips. drop rate median (range / ms / seq): {np.median(d[:,0]):.3f} / {np.median(d[:,1]):.3f} / {np.median(d[:,2]):.3f}; min {d[:,0].min():.3f} / {d[:,1].min():.3f} / {d[:,2].min():.3f}")
    print(f"faint-12 recall: ms mean {np.mean([r['ms_faint12'] for r in rows]):.2f}, seq mean {np.mean([r['seq_faint12'] for r in rows]):.2f}; "
          f"small/tiny/fade recall seq mean {np.mean([r['seq_small'] for r in rows]):.2f}/{np.mean([r['seq_tiny'] for r in rows]):.2f}/{np.mean([r['seq_fade'] for r in rows]):.2f}")


if __name__ == "__main__":
    main()
