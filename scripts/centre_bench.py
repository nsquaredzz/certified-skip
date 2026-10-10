#!/usr/bin/env python3
"""What to hold: the fewest sends a certificate allows, and what a little latency recovers (THEORY.md section 11).

Per clip and per certificate, the share of patch-frames sent after the first frame by

  bound       the greedy split into runs of diameter < 2: no policy with any latency, holding anything, sends
              fewer (Theorem 13); attained for the sup norm
  L = 0       send-on-delta: hold the current frame (the rule in use, here without the sub-pixel shift)
  L > 0       on a break, hold the centre of the longest run among the next L frames that the centre certifies;
              the model's picture lags L frames and the certificate is unchanged (Theorem 16)
  in use      the C++ rule of section 9, quotient + multi-scale without offset, at the same schedule, for reference

Certificates: the sup norm with eps = 40 grey levels, and the multi-scale norm of the benchmarks (Delta_0 = 80,
gamma = 1/2) without offset. The largest certificate norm met on any frame is printed; it must stay below 1.

    python3 scripts/centre_bench.py [--clips hall_monitor,caviar_] [--max-frames 600] [--out out/centre]
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from certskip import video, centre as ct
from certskip.native import NativeSequentialPruner
from real_bench import schedule

P = 16
LOOK = (0, 1, 2, 4, 8, 15)


def plot(rows, path, norm="multi-scale"):
    """Sends relative to holding the current frame, against latency: median over clips and the quartiles."""
    import logging
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    th = dict(surface="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7", s1="#2a78d6")
    plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})
    rel = np.array([[100 * r[norm]["sends"][str(L)] / r[norm]["sends"]["0"] for L in LOOK] for r in rows.values()])
    low = np.array([100 * r[norm]["bound"] / r[norm]["sends"]["0"] for r in rows.values()])
    med = np.median(rel, 0); x = np.arange(len(LOOK))
    fig, ax = plt.subplots(figsize=(7.2, 4.5)); fig.patch.set_facecolor(th["surface"]); ax.set_facecolor(th["surface"])
    for sp in ax.spines.values():
        sp.set_color(th["axis"]); sp.set_linewidth(0.8)
    ax.tick_params(colors=th["muted"], length=0, labelsize=9); ax.grid(True, axis="y", color=th["grid"], linewidth=0.8); ax.set_axisbelow(True)
    ax.axhline(float(np.median(low)), color=th["muted"], linewidth=1.0)
    ax.annotate(f"fewest sends any method can make: {np.median(low):.0f}", (x[-1], float(np.median(low))), xytext=(0, -6), textcoords="offset points", ha="right", va="top", fontsize=9, color=th["ink2"])
    ax.fill_between(x, np.quantile(rel, 0.25, 0), np.quantile(rel, 0.75, 0), color=th["s1"], alpha=0.10, linewidth=0)
    ax.plot(x, med, color=th["s1"], linewidth=2, solid_capstyle="round", solid_joinstyle="round", marker="o", markersize=6.5, markeredgecolor=th["surface"], markeredgewidth=1.6, clip_on=False)
    for i in (1, len(x) - 1):
        ax.annotate(f"{med[i]:.0f}", (x[i], med[i]), xytext=(0, 9), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, color=th["ink2"])
    ax.set_xticks(x); ax.set_xticklabels([str(L) for L in LOOK]); ax.set_xlim(-0.3, len(x) - 0.7); ax.set_ylim(0, 105); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("frames of latency (frames looked ahead before a patch is sent)", fontsize=9.5, color=th["ink2"])
    ax.set_ylabel("patches sent, holding the current frame = 100", fontsize=9.5, color=th["ink2"])
    fig.text(0.085, 0.955, "Holding the centre of the next few frames halves what is sent", fontsize=13, color=th["ink"], weight="bold", va="top")
    fig.text(0.085, 0.895, f"Same certificate at every latency. Median of {len(rows)} fixed cameras, band between the quartiles.", fontsize=9, color=th["ink2"], va="top")
    fig.subplots_adjust(left=0.085, right=0.97, bottom=0.115, top=0.815)
    fig.savefig(path, facecolor=th["surface"]); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="hall_monitor,akiyo,caviar_,xiph_bridge"); ap.add_argument("--out", default="out/centre")
    ap.add_argument("--max-frames", type=int, default=600); ap.add_argument("--replot", action="store_true", help="redraw the figure from the saved JSON")
    a = ap.parse_args()
    if a.replot:
        plot(json.loads(Path(a.out + ".json").read_text()), a.out + ".png"); return
    norms = {"sup": {0: 80.0}, "multi-scale": schedule(80.0, 0.5)}
    rows = {}
    print(f"sends per patch-frame after the first frame, %   (latency L in frames)\n{'clip':34s} {'norm':11s} | {'bound':>6s} | " + " ".join(f"L={L:<4d}" for L in LOOK) + f" | {'in use':>6s} | largest norm")
    for f in [f for f in sorted(glob.glob("data/*")) if any(Path(f).name.startswith(c) for c in a.clips.split(","))]:
        info = video.probe(f); fps = info["fps"] or 25.0
        fr = video.read_gray(f, fps=fps, max_frames=a.max_frames); T, H, W = fr.shape
        pct = lambda n: float(100 * (n.sum() - n.size) / (n.size * (T - 1)))
        name = Path(f).stem; rows[name] = {"fps": fps, "frames": T}
        for label, sched in norms.items():
            low = ct.fewest_sends(fr, P, multiscale=sched); sends, worst = {}, 0.0
            for L in LOOK:
                out = ct.CentrePruner(H, W, P, multiscale=sched, lookahead=L).run(fr)
                n = out["keep"].sum(0); sends[str(L)] = pct(n); worst = max(worst, out["worst"])
                assert (low <= n).all() and out["worst"] < 1.0, (name, label, L)       # Theorem 13 and the certificate, on every patch
            inuse = pct(NativeSequentialPruner(H, W, P, multiscale=sched, z=1e9, offset="none").run(fr)["keep"].sum(0))
            rows[name][label] = dict(bound=pct(low), sends=sends, in_use=inuse, worst=worst)
            print(f"{name:34s} {label:11s} | {pct(low):6.2f} | " + " ".join(f"{sends[str(L)]:6.2f}" for L in LOOK) + f" | {inuse:6.2f} | {worst:.4f}", flush=True)
        Path(a.out + ".json").write_text(json.dumps(rows, indent=1))
    print(f"\n{len(rows)} clips. Median over clips:")
    for label in norms:
        s0 = np.array([r[label]["sends"]["0"] for r in rows.values()])
        print(f"  {label:11s} bound / L=0: {np.median([r[label]['bound'] for r in rows.values()] / s0):.2f}; sends relative to L=0: " +
              ", ".join(f"L={L}: {np.median(np.array([r[label]['sends'][str(L)] for r in rows.values()]) / s0):.2f}" for L in LOOK[1:]) +
              f"; rule in use / L=0: {np.median(np.array([r[label]['in_use'] for r in rows.values()]) / s0):.2f}")
    plot(rows, a.out + ".png")
    print(f"wrote {a.out}.json and {a.out}.png")


if __name__ == "__main__":
    main()
