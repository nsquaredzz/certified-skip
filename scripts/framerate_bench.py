#!/usr/bin/env python3
"""Frame rate: what a consecutive-frame rule and a rule that compares with the held copy leave the model, as the
same footage is read faster.

Each clip is read at its native rate and subsampled to lower rates. Only real frames are used, nothing is
interpolated. At every rate two rules run at the same skip rate:

  certified   quotient + multi-scale without the per-patch offset (THEORY.md section 9), Delta_0 = 80
  heuristic   keep a patch iff the mean |F_t - F_(t-1)| over it >= tau, the criterion of EVS and of run-length
              tokenisation; `matched` re-tunes tau at every rate to the certified rule's skip rate there,
              `fixed` tunes it once, at the rate closest to 2 fps, and leaves it alone

After every frame the copy the model holds is compared with the truth, pixel by pixel:
  object pixels wrong   share of moving-object pixels off by more than 30 grey levels
  ghost pixels          share of the rest of the frame off by more than 30 grey levels: what a rule leaves behind
Moving-object pixels are those more than 30 grey levels from the temporal median of the clip, an oracle no
online rule has.

    python3 scripts/framerate_bench.py [--clips hall_monitor,caviar_LeftBag] [--out out/framerate]
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from certskip import video, baselines as bl
from certskip.core import drop_rate, patch_grid
from certskip.native import NativeSequentialPruner
from real_bench import schedule

P = 16
WRONG = 30                                # grey levels; also the oracle's threshold for a moving-object pixel
STRIDES = (30, 15, 10, 6, 3, 2, 1)
KEYS = ("obj_err", "obj_wrong", "obj_n", "bg_wrong", "bg_n")


def tally(held, truth, moving, acc):
    e = np.abs(held.astype(np.int16) - truth.astype(np.int16))
    acc["obj_err"] += float(e[moving].sum()); acc["obj_wrong"] += int((e[moving] > WRONG).sum()); acc["obj_n"] += int(moving.sum())
    acc["bg_wrong"] += int((e[~moving] > WRONG).sum()); acc["bg_n"] += int((~moving).sum())


def summary(acc, keep):
    n = max(acc["obj_n"], 1)
    return dict(skip=float(drop_rate(keep)), obj_err=acc["obj_err"] / n, obj_wrong=100 * acc["obj_wrong"] / n, ghost=100 * acc["bg_wrong"] / max(acc["bg_n"], 1))


def run_certified(sub, mov, sched):
    T, H, W = sub.shape
    pr = NativeSequentialPruner(H, W, P, multiscale=sched, z=1e9, offset="none"); acc = dict.fromkeys(KEYS, 0); keeps = []
    for t in range(T):
        keeps.append(pr.step(sub[t])[0])
        if t:
            tally(pr.view(False), sub[t], mov[t], acc)
    return summary(acc, np.stack(keeps))


def run_heuristic(sub, mov, tau, scores):
    keep = bl.consecutive_mean(sub, P, tau, scores=scores); held = sub[0].copy(); acc = dict.fromkeys(KEYS, 0)
    for t in range(1, len(sub)):
        hg = patch_grid(held, P); hg[keep[t]] = patch_grid(sub[t], P)[keep[t]]
        tally(held, sub[t], mov[t], acc)
    return summary(acc, keep)


def tau_for_skip(scores, target):
    """Threshold at which the consecutive-frame rule skips the target share of patch-frames (first frame excluded, as in drop_rate)."""
    return float(np.quantile(scores, min(max(target, 0.0), 1.0)))


def plot(rows, path):
    import logging
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    th = dict(surface="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7", cert="#2a78d6", heur="#eb6834")
    plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})
    names = [n for n in rows if len(rows[n]) >= 4]
    cols = min(4, len(names)); nrow = -(-len(names) // cols)
    fig, axes = plt.subplots(nrow, cols, figsize=(2.75 * cols + 0.4, 2.15 * nrow + 1.35), squeeze=False, sharey=True)
    fig.patch.set_facecolor(th["surface"])
    ymax = max(max(r["heuristic_matched"]["ghost"], r["certified"]["ghost"]) for n in names for r in rows[n])
    for ax, name in zip(axes.ravel(), names):
        r = rows[name]; x = [q["fps"] for q in r]
        ax.set_facecolor(th["surface"])
        for s in ax.spines.values():
            s.set_color(th["axis"]); s.set_linewidth(0.8)
        ax.tick_params(colors=th["muted"], length=0, labelsize=8); ax.grid(True, axis="y", color=th["grid"], linewidth=0.8); ax.set_axisbelow(True); ends = {}
        for key, col in (("heuristic_matched", th["heur"]), ("certified", th["cert"])):
            y = [q[key]["ghost"] for q in r]
            ax.plot(x, y, color=col, linewidth=2, solid_capstyle="round", solid_joinstyle="round", marker="o", markersize=5.5, markeredgecolor=th["surface"], markeredgewidth=1.4, clip_on=False)
            ends[key] = y[-1]
        close = ends["heuristic_matched"] - ends["certified"] < 0.09 * ymax
        for key in ends:
            up = key == "heuristic_matched"
            ax.annotate(f"{ends[key]:.2f}" if ends[key] < 1 else f"{ends[key]:.1f}", (x[-1], ends[key]), xytext=(7, (5 if up else -4) if close else 0), textcoords="offset points",
                        ha="left", va="center", fontsize=8, color=th["ink2"], annotation_clip=False)
        ax.set_xscale("log"); ax.set_xticks([1, 3, 10, 30]); ax.set_xticklabels(["1", "3", "10", "30"]); ax.minorticks_off(); ax.set_xlim(0.7, 62); ax.set_ylim(0, ymax * 1.12)
        ax.set_title(name.replace("caviar_", "").replace("_cif", "").replace("xiph_", "").replace("_", " "), fontsize=9, color=th["ink"], loc="left", pad=5)
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("frames per second", fontsize=8.5, color=th["ink2"])
    for ax in axes[:, 0]:
        ax.set_ylabel("ghost pixels, %", fontsize=8.5, color=th["ink2"])
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=th["heur"], linewidth=2, marker="o", markersize=5.5, markeredgecolor=th["surface"], label="consecutive-frame heuristic, re-tuned at every frame rate"),
               Line2D([0], [0], color=th["cert"], linewidth=2, marker="o", markersize=5.5, markeredgecolor=th["surface"], label="certified rule (compares with the copy the model holds)")]
    top = 1 - 0.62 / fig.get_figheight()
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.035, top), ncol=2, frameon=False, fontsize=8.5, labelcolor=th["ink2"], handlelength=1.8, columnspacing=2.2)
    fig.text(0.04, 1 - 0.20 / fig.get_figheight(), "A consecutive-frame rule leaves more behind as the same footage is read faster", fontsize=11.5, color=th["ink"], weight="bold", va="top")
    fig.text(0.04, 1 - 0.43 / fig.get_figheight(), "Share of the static part of the frame that is wrong (off by more than 30 grey levels) in what the model holds. Both rules skip the same share of patches at every rate.",
             fontsize=8.3, color=th["ink2"], va="top")
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.42 / fig.get_figheight() + 0.02, top=1 - 1.2 / fig.get_figheight(), wspace=0.12, hspace=0.42)
    fig.savefig(path, facecolor=th["surface"]); plt.close(fig)


def plot_summary(rows, path):
    """One panel: the median over clips at each subsampling step, with the band between the quartiles."""
    import logging
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    th = dict(surface="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7", cert="#2a78d6", heur="#eb6834")
    plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})
    by = {}                                                        # step from the native rate (1 = native) -> per-clip values
    for r in rows.values():
        native = r[-1]["fps"]
        for q in r:
            by.setdefault(round(native / q["fps"]), []).append(q)
    steps = sorted(by, reverse=True); x = [30.0 / s for s in steps]
    fig, ax = plt.subplots(figsize=(7.2, 4.5)); fig.patch.set_facecolor(th["surface"]); ax.set_facecolor(th["surface"])
    for sp in ax.spines.values():
        sp.set_color(th["axis"]); sp.set_linewidth(0.8)
    ax.tick_params(colors=th["muted"], length=0, labelsize=9); ax.grid(True, axis="y", color=th["grid"], linewidth=0.8); ax.set_axisbelow(True)
    for key, col in (("heuristic_matched", th["heur"]), ("certified", th["cert"])):
        v = [np.array([q[key]["ghost"] for q in by[s]]) for s in steps]
        med = [float(np.median(a)) for a in v]
        ax.fill_between(x, [np.quantile(a, 0.25) for a in v], [np.quantile(a, 0.75) for a in v], color=col, alpha=0.10, linewidth=0)
        ax.plot(x, med, color=col, linewidth=2, solid_capstyle="round", solid_joinstyle="round", marker="o", markersize=6.5, markeredgecolor=th["surface"], markeredgewidth=1.6, clip_on=False)
        ax.annotate(f"{med[-1]:.2f}%", (x[-1], med[-1]), xytext=(9, 0), textcoords="offset points", ha="left", va="center", fontsize=9.5, color=th["ink2"], annotation_clip=False)
    ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels(["1", "2", "3", "5", "10", "15", "25 to 30"][-len(x):]); ax.minorticks_off(); ax.set_xlim(x[0] * 0.82, x[-1] * 1.75)
    ax.set_ylim(0, None); ax.set_yticks(np.arange(0, ax.get_ylim()[1], 0.5)); ax.set_xlabel("frames per second read from the same footage", fontsize=9.5, color=th["ink2"]); ax.set_ylabel("ghost pixels, % of the static part of the frame", fontsize=9.5, color=th["ink2"])
    handles = [Line2D([0], [0], color=th["heur"], linewidth=2, marker="o", markersize=6.5, markeredgecolor=th["surface"], label="consecutive-frame heuristic"),
               Line2D([0], [0], color=th["cert"], linewidth=2, marker="o", markersize=6.5, markeredgecolor=th["surface"], label="certified rule")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.075, 0.80), ncol=2, frameon=False, fontsize=9.5, labelcolor=th["ink2"], handlelength=1.8, columnspacing=2.0)
    fig.text(0.085, 0.955, "The heuristic leaves more ghosts behind the faster it looks", fontsize=13, color=th["ink"], weight="bold", va="top")
    fig.text(0.085, 0.895, f"Median of {len(rows)} fixed cameras, band between the quartiles.\nBoth rules skip the same share of patches at every frame rate.", fontsize=9, color=th["ink2"], va="top", linespacing=1.35)
    fig.subplots_adjust(left=0.085, right=0.97, bottom=0.115, top=0.725)
    fig.savefig(path, facecolor=th["surface"]); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="hall_monitor,akiyo,caviar_,xiph_bridge"); ap.add_argument("--out", default="out/framerate")
    ap.add_argument("--max-frames", type=int, default=2400); ap.add_argument("--replot", action="store_true", help="redraw the figures from the saved JSON")
    a = ap.parse_args()
    if a.replot:
        rows = json.loads(Path(a.out + ".json").read_text()); plot(rows, a.out + ".png"); plot_summary(rows, a.out + "_summary.png"); return
    sched = schedule(80.0, 0.5); rows = {}
    files = [f for f in sorted(glob.glob("data/*")) if any(Path(f).name.startswith(c) for c in a.clips.split(","))]
    for f in files:
        info = video.probe(f); fps0 = info["fps"] or 25.0
        fr = video.read_gray(f, fps=fps0, max_frames=a.max_frames)
        T, H, W = fr.shape; fr = fr[:, : H // P * P, : W // P * P]
        moving = np.abs(fr.astype(np.int16) - np.median(fr[::5], axis=0).astype(np.int16)) > WRONG
        name = Path(f).stem; todo = []
        for s in STRIDES:
            sub, mov = fr[::s], moving[::s]
            if len(sub) >= 12:
                todo.append((fps0 / s, sub, mov, run_certified(sub, mov, sched), bl.consecutive_mean_scores(sub, P)))
        fps_fix, _, _, c_fix, sc_fix = min(todo, key=lambda q: abs(q[0] - 2.0))
        tau_fixed = tau_for_skip(sc_fix, c_fix["skip"])
        print(f"\n=== {name}: {T} frames at {fps0:g} fps, {W}x{H}; moving-object pixels {100 * moving.mean():.1f}% of the footage; fixed heuristic tuned at {fps_fix:.1f} fps, tau = {tau_fixed:.2f}")
        print(f"  {'fps':>5s} | skipped, %: {'cert':>5s} {'match':>6s} {'fixed':>6s} | object pixels wrong, %: {'cert':>5s} {'match':>6s} {'fixed':>6s} | ghost pixels, %: {'cert':>5s} {'match':>6s} {'fixed':>6s}")
        rows[name] = []
        for fps, sub, mov, c, scores in todo:
            hm = run_heuristic(sub, mov, tau_for_skip(scores, c["skip"]), scores); hf = run_heuristic(sub, mov, tau_fixed, scores)
            print(f"  {fps:5.1f} | {'':11s} {100 * c['skip']:5.1f} {100 * hm['skip']:6.1f} {100 * hf['skip']:6.1f} | {'':23s} {c['obj_wrong']:5.1f} {hm['obj_wrong']:6.1f} {hf['obj_wrong']:6.1f} |"
                  f" {'':16s} {c['ghost']:5.2f} {hm['ghost']:6.2f} {hf['ghost']:6.2f}", flush=True)
            rows[name].append(dict(fps=fps, frames=len(sub), certified=c, heuristic_matched=hm, heuristic_fixed=hf, tau_fixed=tau_fixed))
    Path(a.out + ".json").write_text(json.dumps(rows, indent=1))
    lo = {n: min(r, key=lambda q: abs(q["fps"] - 2.0)) for n, r in rows.items()}; hi = {n: r[-1] for n, r in rows.items()}
    print(f"\n{len(rows)} clips. Median over clips, at about 2 fps and at the native rate:")
    for key, label in (("ghost", "ghost pixels, %"), ("obj_wrong", "object pixels wrong, %")):
        for rule in ("certified", "heuristic_matched", "heuristic_fixed"):
            print(f"  {label:24s} {rule:18s} {np.median([lo[n][rule][key] for n in rows]):6.2f} -> {np.median([hi[n][rule][key] for n in rows]):6.2f}")
    plot(rows, a.out + ".png"); plot_summary(rows, a.out + "_summary.png")
    print(f"wrote {a.out}.json, {a.out}.png and {a.out}_summary.png")


if __name__ == "__main__":
    main()
