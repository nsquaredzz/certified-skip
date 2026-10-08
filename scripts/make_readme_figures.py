#!/usr/bin/env python3
"""Draw the static figures of README.md from the result files in out/ and one clip in data/.

    python3 scripts/make_readme_figures.py            # all three, light and dark
    python3 scripts/make_readme_figures.py breadth    # breadth | faint | rule

Each figure is written for a light and for a dark page (assets/fig_<name>_{light,dark}.png); the README
picks one with a <picture> element. Colours are a fixed, colour-blind-checked palette: blue / orange /
aqua for the three rules, one blue in two shades for before and after, blue-grey-red for signed change.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
ASSETS = ROOT / "assets"

THEMES = {
    "light": dict(surface="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  s1="#2a78d6", s2="#eb6834", s3="#1baf7a", before="#86b6ef", after="#2a78d6", neg="#2a78d6", mid="#f0efec", pos="#e34948"),
    "dark":  dict(surface="#0d1117", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                  s1="#3987e5", s2="#d95926", s3="#199e70", before="#184f95", after="#3987e5", neg="#3987e5", mid="#383835", pos="#e66767"),
}
plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 9.5,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})


def style(ax, th, grid="x"):
    ax.set_facecolor(th["surface"])
    for s in ax.spines.values():
        s.set_color(th["axis"]); s.set_linewidth(0.8)
    ax.tick_params(colors=th["muted"], length=0, labelsize=8.5)
    ax.grid(False)
    if grid:
        ax.grid(True, axis=grid, color=th["grid"], linewidth=0.8); ax.set_axisbelow(True)


def save(fig, name, mode, th):
    ASSETS.mkdir(exist_ok=True)
    out = ASSETS / f"fig_{name}_{mode}.png"
    fig.savefig(out, facecolor=th["surface"]); plt.close(fig)
    print(f"  {out.relative_to(ROOT)}  {out.stat().st_size / 1e3:.0f} kB")


def pretty(clip):
    s = clip.rsplit(".", 1)[0].replace("_cif", "").replace("_1080p", "")
    for pre, name in (("caviar_", "CAVIAR  "), ("virat_", "VIRAT  "), ("VIRAT_S_", "VIRAT  "), ("xiph_", "Xiph  ")):
        if s.startswith(pre):
            return name + s[len(pre):]
    return "Xiph  " + s


# ----------------------------------------------------------------------------- 25 cameras, before and after
def fig_breadth(mode):
    th = THEMES[mode]
    rows = json.loads((ROOT / "out/breadth.json").read_text())
    rows.sort(key=lambda r: r["drop_range"])                      # worst camera for the one-line rule on top
    before = np.array([100 * (1 - r["drop_range"]) for r in rows]); after = np.array([100 * (1 - r["drop_seq"]) for r in rows])
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(8.4, 6.3)); fig.patch.set_facecolor(th["surface"])
    fig.subplots_adjust(left=0.285, right=0.965, top=0.845, bottom=0.085)
    style(ax, th, "x")
    ax.hlines(y, after, before, color=th["axis"], linewidth=1.4, zorder=2)
    ring = dict(markeredgecolor=th["surface"], markeredgewidth=1.3, linestyle="none", zorder=3)
    ax.plot(before, y, "o", color=th["before"], markersize=7.5, **ring)
    ax.plot(after, y, "o", color=th["after"], markersize=7.5, **ring)
    ax.set_yticks(y); ax.set_yticklabels([pretty(r["clip"]) for r in rows], color=th["ink2"], fontsize=8.3)
    ax.set_ylim(-0.8, len(rows) - 0.2); ax.set_xlim(-0.5, 26)
    ax.set_xticks([0, 5, 10, 15, 20, 25]); ax.set_xticklabels(["0", "5 %", "10 %", "15 %", "20 %", "25 %"])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("patches still sent to the model, as a share of all patches", color=th["ink2"], fontsize=9)
    for i in (0, 2):                                               # two rows carry their numbers; the axis carries the rest
        ax.text(before[i] + 0.45, y[i], f"{before[i]:.1f} %", va="center", ha="left", color=th["ink2"], fontsize=8.3)
        ax.text(after[i] - 0.45, y[i], f"{after[i]:.1f} %", va="center", ha="right", color=th["ink2"], fontsize=8.3)
    fig.text(0.03, 0.955, "Patches still sent to the model on 25 fixed cameras", color=th["ink"], fontsize=12.5, fontweight="bold", va="center")
    fig.text(0.03, 0.915, f"One row per camera, thresholds fixed in advance. Median camera: {np.median(before):.1f} % with the one-line rule, "
                          f"{np.median(after):.1f} % with the full rule.", color=th["ink2"], fontsize=9.3, va="center")
    h = [Line2D([], [], marker="o", color=th["before"], markersize=7.5, linestyle="none", label="one-line range rule, Δ = 32"),
         Line2D([], [], marker="o", color=th["after"], markersize=7.5, linestyle="none", label="quotient + multi-scale + sequential")]
    leg = fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.022, 0.895), ncol=2, frameon=False, fontsize=9, handletextpad=0.3, columnspacing=1.8)
    for t in leg.get_texts():
        t.set_color(th["ink2"])
    save(fig, "breadth", mode, th)


# ----------------------------------------------------------------------------- faint objects: recall against cost
def fig_faint(mode):
    th = THEMES[mode]
    a = json.loads((ROOT / "out/real_bench_seq.json").read_text())          # memoryless sweeps
    b = json.loads((ROOT / "out/real_bench_seq3.json").read_text())         # sequential sweep, final rule
    cams = [("hall", "hallway, analog CCTV"), ("leftbag", "lobby, MPEG-1 CCTV"), ("virat", "parking lot, 720p")]
    series = [("range", a, th["s1"], "o", "one-line range rule"), ("quotient+ms g=.5", a, th["s2"], "s", "quotient + multi-scale"),
              ("sequential z (ms80)", b, th["s3"], "D", "+ sequential test")]
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.35), sharey=True); fig.patch.set_facecolor(th["surface"])
    fig.subplots_adjust(left=0.075, right=0.985, top=0.70, bottom=0.165, wspace=0.09)
    for ax, (cam, title) in zip(axes, cams):
        style(ax, th, "y")
        for key, src, col, mk, _ in series:
            pts = sorted(((100 * (1 - p["drop"]), p["faint12"]) for p in src[cam][key]), reverse=True)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, linewidth=2, marker=mk, markersize=5.8,
                    markeredgecolor=th["surface"], markeredgewidth=1.0, solid_joinstyle="round", solid_capstyle="round", zorder=3)
        ax.set_xscale("log"); ax.set_xlim(60, 0.6)
        ax.set_xticks([30, 10, 3, 1]); ax.set_xticklabels(["30 %", "10 %", "3 %", "1 %"]); ax.minorticks_off()
        ax.set_ylim(-0.04, 1.06); ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "50 %", "100 %"])
        ax.set_title(title, color=th["ink2"], fontsize=9.3, loc="left", pad=6)
        ax.spines["left"].set_visible(False)
    axes[1].set_xlabel("patches sent to the model  (log scale; cheaper to the right)", color=th["ink2"], fontsize=9)
    axes[0].set_ylabel("faint objects kept", color=th["ink2"], fontsize=9)
    fig.text(0.03, 0.935, "Faint objects need memory", color=th["ink"], fontsize=12.5, fontweight="bold", va="center")
    fig.text(0.03, 0.868, "Share of planted 6×6 px objects of contrast 12 that are kept within one second, as each rule is made cheaper.",
             color=th["ink2"], fontsize=9.3, va="center")
    h = [Line2D([], [], color=c, marker=m, linewidth=2, markersize=5.8, label=l) for _, _, c, m, l in series]
    leg = fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.022, 0.845), ncol=3, frameon=False, fontsize=9, handlelength=1.8, columnspacing=1.8)
    for t in leg.get_texts():
        t.set_color(th["ink2"])
    save(fig, "faint", mode, th)


# ----------------------------------------------------------------------------- the rule on three real patches
def fig_rule(mode, t=150, delta=32.0, P=16):
    from certskip import video
    from certskip.core import Pruner, patch_grid, spread_and_shift
    from certskip.warp import fit_translation, warp
    th = THEMES[mode]
    fr = video.read_gray(ROOT / "data/hall_monitor_cif.y4m", fps=30, max_frames=t + 1)
    pr = Pruner(fr.shape[1], fr.shape[2], P, delta)
    for f in fr[:t]:
        pr.step(f)
    R = pr.reference(); F = fr[t]
    spread, _ = spread_and_shift(F, R, P)
    dy, dx = fit_translation(F, R, P)
    res = patch_grid(F.astype(np.float64) - warp(R, dy, dx, P), P)
    spread_q = res.max((2, 3)) - res.min((2, 3))
    moving = patch_grid(np.abs(F.astype(int) - np.median(fr[::5], 0)) > 30, P).sum((2, 3)) >= 8      # where something really is
    gh, gw = spread.shape; inner = np.zeros_like(moving); inner[1:-1, 1:-1] = True
    quiet = np.argwhere((spread < delta) & ~moving & inner)
    A = tuple(quiet[np.argsort(spread[tuple(quiet.T)])[len(quiet) // 2]])                           # a typical skipped patch
    B = np.unravel_index(np.argmax(np.where(moving, spread, -1)), spread.shape)                     # the largest real change
    jit = (spread >= delta) & ~moving & (spread_q < delta) & inner
    C = np.unravel_index(np.argmax(np.where(jit, spread - spread_q, -1)), spread.shape)             # jitter the quotient explains
    picks = [("A", A), ("B", B), ("C", C)]
    print("  rule figure patches:", {k: (tuple(int(v) for v in p), round(float(spread[p]), 1), round(float(spread_q[p]), 1)) for k, p in picks},
          "shift of C:", round(float(np.hypot(dy[C], dx[C])), 2), "px")
    cmap = LinearSegmentedColormap.from_list("div", [th["neg"], th["mid"], th["pos"]])
    fig = plt.figure(figsize=(8.4, 3.75)); fig.patch.set_facecolor(th["surface"])
    fig.text(0.03, 0.94, "The rule on three patches of one frame", color=th["ink"], fontsize=12.5, fontweight="bold", va="center")
    fig.text(0.03, 0.872, "Spread = largest minus smallest change of a patch since the copy the model already holds. Skip while it is below Δ = 32 grey levels.",
             color=th["ink2"], fontsize=9.1, va="center")
    ax = fig.add_axes([0.03, 0.06, 0.36, 0.72]); ax.imshow(F, cmap="gray", vmin=0, vmax=255, interpolation="nearest"); ax.axis("off")
    for name, (gy, gx) in picks:
        ax.add_patch(Rectangle((gx * P - 0.5, gy * P - 0.5), P, P, fill=False, edgecolor="#ffffff", linewidth=2.6))
        ax.add_patch(Rectangle((gx * P - 0.5, gy * P - 0.5), P, P, fill=False, edgecolor="#0b0b0b", linewidth=1.1))
        ax.text(gx * P + P + 5, gy * P + P / 2, name, color="#0b0b0b", fontsize=10.5, fontweight="bold", va="center", ha="left",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="#ffffff", edgecolor="none"))
    cols = ["held copy", "this frame", "change"]
    x0, w, gap = 0.425, 0.068, 0.012
    for j, c in enumerate(cols):
        fig.text(x0 + j * (w + gap) + w / 2, 0.775, c, color=th["muted"], fontsize=7.8, ha="center", va="bottom")
    def verdict(name, p):
        s, q = float(spread[p]), float(spread_q[p])
        if name == "A":
            return f"Spread {s:.0f}, below 32: skip.", "The model keeps its token. Proved: nothing of\ncontrast 32 or more appeared or vanished here."
        if name == "B":
            return f"Spread {s:.0f}, above 32: send.", "A person walked in. The patch is sent and\nbecomes the new held copy."
        return f"Spread {s:.0f}, above 32: sent for nothing.", (f"A sharp edge moved {np.hypot(dy[p], dx[p]):.1f} px. With that shift taken\n"
                                                                 f"out the spread is {q:.0f}: the quotient rule skips it.")
    for i, (name, p) in enumerate(picks):
        yb = 0.595 - i * 0.25; gy, gx = p
        tiles = [R[gy * P:(gy + 1) * P, gx * P:(gx + 1) * P], F[gy * P:(gy + 1) * P, gx * P:(gx + 1) * P]]
        d = tiles[1].astype(int) - tiles[0].astype(int)
        for j in range(3):
            a = fig.add_axes([x0 + j * (w + gap), yb, w, w * 8.4 / 3.75])
            if j < 2:
                a.imshow(tiles[j], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            else:
                a.imshow(d, cmap=cmap, vmin=-64, vmax=64, interpolation="nearest")
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(True); s.set_color(th["axis"]); s.set_linewidth(0.8)
        fig.text(x0 - 0.022, yb + w * 8.4 / 3.75 / 2, name, color=th["ink"], fontsize=10.5, fontweight="bold", va="center", ha="center")
        head, why = verdict(name, p)
        xt = x0 + 3 * (w + gap) + 0.012
        fig.text(xt, yb + 0.122, head, color=th["ink"], fontsize=9.6, va="center", fontweight="bold")
        fig.text(xt, yb + 0.048, why, color=th["ink2"], fontsize=8.6, va="center", linespacing=1.35)
    cax = fig.add_axes([x0 + 2 * (w + gap), 0.036, w, 0.016])
    cax.imshow(np.linspace(-1, 1, 64)[None], cmap=cmap, aspect="auto"); cax.set_xticks([]); cax.set_yticks([])
    for s in cax.spines.values():
        s.set_visible(False)
    fig.text(x0 + 2 * (w + gap) - 0.006, 0.044, "−64", color=th["muted"], fontsize=7.2, ha="right", va="center")
    fig.text(x0 + 2 * (w + gap) + w + 0.006, 0.044, "+64 grey levels", color=th["muted"], fontsize=7.2, ha="left", va="center")
    save(fig, "rule", mode, th)


def main():
    what = sys.argv[1:] or ["breadth", "faint", "rule"]
    for name in what:
        for mode in ("light", "dark"):
            {"breadth": fig_breadth, "faint": fig_faint, "rule": fig_rule}[name](mode)


if __name__ == "__main__":
    main()
