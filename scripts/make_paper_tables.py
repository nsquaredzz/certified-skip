#!/usr/bin/env python3
"""Build the paper's tables (LaTeX) and figures (PDF) from the JSON outputs in out/.

    python3 scripts/make_paper_tables.py

Reads  out/warp_davis.json, out/warp_film.json, out/warp_audit.json, out/competitive.json,
       out/shift_cost.json, out/audit.json
Writes paper/tables/*.tex and paper/figures/*.pdf, and prints the headline numbers used in the text.
"""
import json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"; TAB = ROOT / "paper/tables"; FIG = ROOT / "paper/figures"
TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
                     "figure.dpi": 150, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})
COLORS = {"fixed range": "#9e9e9e", "fixed quotient+MS": "#4d4d4d", "warp range": "#8fb3e0", "warp+refine range": "#3d7dc8", "warp+refine MS": "#0b3d91"}


def load(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def tex_escape(s):
    return s.replace("%", r"\%").replace("_", r"\_").replace("&", r"\&").replace("+", "+")


headline = {}

# ---------------------------------------------------------------- DAVIS
davis = load("warp_davis.json")
if davis:
    names = ["fixed range", "fixed quotient+MS", "warp range", "warp+refine range", "warp+refine MS"]
    clips = sorted(davis)
    rows = {n: [davis[c][n] for c in clips if n in davis[c]] for n in names}
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"rule & drop & drop, object & drop, backgr. & revealed & cuts/clip & $\iota$ median & ms/frame \\", r"\midrule"]
    for n in names:
        R = rows[n]
        if not R:
            continue
        aud = [r.get("audit") for r in R if r.get("audit")]
        lines.append(f"{tex_escape(n)} & {np.median([r['drop'] for r in R]):.3f} & {np.nanmedian([r.get('drop_object', np.nan) for r in R]):.3f} & "
                     f"{np.nanmedian([r.get('drop_background', np.nan) for r in R]):.3f} & {np.median([r['noref'] for r in R]):.3f} & "
                     f"{np.mean([r['cuts'] for r in R]):.2f} & {np.median([r['iota_median'] for r in R]):.0f} & {np.median([r['ms_per_frame'] for r in R]):.0f} \\\\")
        if aud:
            headline[f"davis_audit_{n}"] = (sum(x["checked"] for x in aud), sum(x["violations"] for x in aud), max(x["worst"] for x in aud))
        headline[f"davis_drop_{n}"] = float(np.median([r["drop"] for r in R]))
        headline[f"davis_min_{n}"] = float(np.min([r["drop"] for r in R]))
        headline[f"davis_bg_{n}"] = float(np.nanmedian([r.get("drop_background", np.nan) for r in R]))
        headline[f"davis_obj_{n}"] = float(np.nanmedian([r.get("drop_object", np.nan) for r in R]))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "davis.tex").write_text("\n".join(lines) + "\n")
    headline["davis_clips"] = len(clips)
    # per-pixel iota distribution for the full rule
    pix = [r.get("iota_pix_frac_le") for r in rows["warp+refine MS"] if r.get("iota_pix_frac_le")]
    if pix:
        keys = list(pix[0]); med = {k: float(np.median([p[k] for p in pix])) for k in keys}
        headline["iota_pix_frac_le"] = med
    # figure: per-clip drop, fixed quotient+MS vs warp+refine MS, sorted
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    order = np.argsort([davis[c]["warp+refine MS"]["drop"] for c in clips])
    x = np.arange(len(clips))
    for n in ("fixed range", "fixed quotient+MS", "warp+refine MS"):
        ax.plot(x, [davis[clips[i]][n]["drop"] for i in order], ".", ms=4, color=COLORS[n], label=n)
    ax.set_xlabel("DAVIS 2017 clip, sorted by the warp rule's drop rate"); ax.set_ylabel("share of patches dropped"); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.legend(loc="upper left", frameon=False)
    fig.savefig(FIG / "davis_drop.pdf"); plt.close(fig)
    # figure: background vs object drop for the warp rule
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    ax.scatter([davis[c]["warp+refine MS"].get("drop_background", np.nan) for c in clips], [davis[c]["warp+refine MS"].get("drop_object", np.nan) for c in clips], s=8, color=COLORS["warp+refine MS"])
    ax.plot([0, 1], [0, 1], "-", lw=0.5, color="#bbb"); ax.set_xlabel("drop rate on background patches"); ax.set_ylabel("drop rate on object patches"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.savefig(FIG / "davis_object_vs_background.pdf"); plt.close(fig)

# ---------------------------------------------------------------- film
film = load("warp_film.json")
if film:
    names = ["fixed range", "fixed quotient+MS", "warp range", "warp+refine range", "warp+refine MS"]
    keys = [k for k in film if k.startswith("film@")]
    lines = [r"\begin{tabular}{l" + "r" * len(keys) + "rr}", r"\toprule",
             "rule & " + " & ".join(f"drop, {k[5:]}" for k in keys) + r" & cuts & $\iota$ median \\", r"\midrule"]
    for n in names:
        vals = [film[k][n]["drop"] for k in keys if n in film[k]]
        if not vals:
            continue
        cuts = sum(film[k][n]["cuts"] for k in keys if n in film[k])
        io = np.median([film[k][n]["iota_median"] for k in keys if n in film[k]])
        lines.append(f"{tex_escape(n)} & " + " & ".join(f"{v:.3f}" for v in vals) + f" & {cuts} & {io:.0f} \\\\")
        headline[f"film_drop_{n}"] = float(np.mean(vals))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "film.tex").write_text("\n".join(lines) + "\n")
    aud = [film[k]["warp+refine MS"].get("audit") for k in keys if film[k].get("warp+refine MS", {}).get("audit")]
    if aud:
        headline["film_audit"] = (sum(x["checked"] for x in aud), sum(x["violations"] for x in aud))

# ---------------------------------------------------------------- warp audit (synthetic)
wa = load("warp_audit.json")
if wa:
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"warp & scene & dropped & Thm.~\ref{thm:warp} viol. & Prop.~\ref{prop:iota} viol. & $\iota$ median & $\iota$ p90 & $\Delta$-features surviving \\", r"\midrule"]
    tot_checked = tot_viol = 0
    for k, r in wa.items():
        w, sc = k.split("/")
        lines.append(f"{w} & {sc} & {r['dropped']}/{r['patches']} & {r['A_violations']} & {r['B_pixel_violations'] + r['B_diagram_violations']} & {r['iota_median']:.1f} & {r['iota_p90']:.1f} & {r['C_feature_survival']:.2f} of {r['C_features']} \\\\")
        tot_checked += r["dropped"]; tot_viol += r["A_violations"] + r["B_pixel_violations"] + r["B_diagram_violations"]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "warp_audit.tex").write_text("\n".join(lines) + "\n")
    headline["warp_audit"] = (tot_checked, tot_viol)

# ---------------------------------------------------------------- competitive
comp = load("competitive.json")
if comp:
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"clip & statistic, $\theta$ & rule keeps $m_\theta$ & $\OPT_\theta$ & $\OPT_{\theta/2}$ & $m_\theta/\OPT_\theta$ & worst patch & viol. \\", r"\midrule"]
    ratios = []
    for clip, d in comp.items():
        for k, v in d.items():
            if not isinstance(v, dict):
                continue
            st, th = k.split("@")
            lines.append(f"{clip} & {st}, {th} & {v['m']} & {v['opt_delta']} & {v['opt_half']} & {v['ratio_total']:.3f} & {v['ratio_patch_max']:.1f} & {v['violations']} \\\\")
            ratios.append(v["ratio_total"])
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "competitive.tex").write_text("\n".join(lines) + "\n")
    headline["competitive_ratio_range"] = (min(ratios), max(ratios))

# ---------------------------------------------------------------- shift cost
sc = load("shift_cost.json")
if sc:
    caps = ["0.0", "1.0", "2.0", "4.0", "8.0", "16.0", "32.0", "inf"]
    lines = [r"\begin{tabular}{ll" + "r" * len(caps) + "rr}", r"\toprule",
             r"clip & rule & " + " & ".join(r"$\kappa=" + (c.rstrip("0").rstrip(".") if c != "inf" else r"\infty") + "$" for c in caps) + r" & $|c|$ p90 & $|c|$ p99 \\", r"\midrule"]
    for clip, d in sc.items():
        for rname, row in d.items():
            lines.append(f"{clip} & {tex_escape(rname)} & " + " & ".join(f"{row[c]:.3f}" for c in caps) + f" & {row['abs_c_p90']:.0f} & {row['abs_c_p99']:.0f} \\\\")
            headline[f"shift_{clip}_{rname}"] = (row["16.0"], row["inf"])
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "shift_cost.tex").write_text("\n".join(lines) + "\n")
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    for clip, d in sc.items():
        for rname, row in d.items():
            xs = [float(c) for c in caps if c != "inf"]; ys = [row[c] for c in caps if c != "inf"]
            ax.plot(xs, ys, "-o", ms=3, lw=1, label=f"{clip}, {rname}")
    ax.set_xscale("symlog", linthresh=1); ax.set_xlabel(r"cap $\kappa$ on $|c|$ (grey levels)"); ax.set_ylabel("drop rate with keep rule"); ax.legend(frameon=False, fontsize=6, loc="lower right")
    fig.savefig(FIG / "shift_cost.pdf"); plt.close(fig)

# ---------------------------------------------------------------- audit
aud = load("audit.json")
if aud:
    EVENTS = [("small", "small 5$\\times$5"), ("tiny", "tiny 3$\\times$3"), ("fade", "fade 3 s"), ("slow", "fade 12 s"), ("moving_cov", "mover"), ("faint", "faint 12")]
    order = ["certified range", "certified quotient+MS", "certified +sequential", "mean vs last kept", "consecutive (EVS/RLT)",
             "cosine consecutive (DyCoke/FrameFusion)", "dycoke groups of 4", "segment static (PruneVid)", "frame consecutive (LongVU)",
             "consecutive + refresh/60", "uniform frames"]
    short = {"certified range": "certified range", "certified quotient+MS": "certified quotient+MS", "certified +sequential": "certified +sequential",
             "mean vs last kept": "anchored mean (no certificate)", "consecutive (EVS/RLT)": "consecutive mean (EVS, RLT)",
             "cosine consecutive (DyCoke/FrameFusion)": "consecutive cosine (DyCoke, FrameFusion)", "dycoke groups of 4": "DyCoke groups of 4",
             "segment static (PruneVid)": "segment static (PruneVid)", "frame consecutive (LongVU)": "frame similarity (LongVU)",
             "consecutive + refresh/60": "consecutive + refresh every 60", "uniform frames": "uniform frames"}

    def at_budget(rows, target):
        """Row closest to the target drop rate (within 0.02), else None."""
        ok = [r for r in rows if abs(r["drop"] - target) <= 0.02]
        return min(ok, key=lambda r: abs(r["drop"] - target)) if ok else None

    for key, res in aud.items():
        # operating point: the certified quotient+MS row with Delta0 = 80
        cert = [r for r in res["rules"]["certified quotient+MS"] if r["thr"] == 80]
        if not cert:
            continue
        target = cert[0]["drop"]
        lines = [r"\begin{tabular}{lrr" + "r" * len(EVENTS) + "r}", r"\toprule",
                 r"rule & drop & tokens/s & " + " & ".join(e[1] for e in EVENTS) + r" & $\rho$ median \\", r"\midrule"]
        for rname in order:
            rows = res["rules"].get(rname)
            if not rows:
                continue
            if rname.startswith("certified"):
                r = [x for x in rows if x["thr"] == 80][0] if rname != "certified range" else min(rows, key=lambda x: abs(x["drop"] - target))
            else:
                r = at_budget(rows, target)
            if r is None:
                lines.append(f"{short[rname]} & \\multicolumn{{{2 + len(EVENTS) + 1}}}{{l}}{{cannot reach the budget at this frame rate}} \\\\")
                continue
            rho = r["rho"]["median"] if isinstance(r["rho"], dict) else r["rho"]
            vals = " & ".join(f"{r[e[0]]:.2f}" if np.isfinite(r[e[0]]) else "--" for e in EVENTS)
            lines.append(f"{short[rname]} & {r['drop']:.3f} & {r['tokens_per_s']:.0f} & {vals} & {rho:.3f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (TAB / f"audit_{key.replace('@', '_').replace('.', 'p')}.tex").write_text("\n".join(lines) + "\n")
        headline[f"audit_{key}_target"] = target
    # figure: recall vs drop for slow fades and small objects, native rate, one panel per clip
    keys = [k for k in aud if "fps" in k and not k.endswith("@2fps")]
    if keys:
        fig, axes = plt.subplots(2, len(keys), figsize=(2.6 * len(keys), 4.2), squeeze=False)
        pal = plt.cm.tab20(np.linspace(0, 1, len(order)))
        for j, key in enumerate(keys):
            for i, ev in enumerate(("slow", "small")):
                ax = axes[i, j]
                for ci, rname in enumerate(order):
                    rows = aud[key]["rules"].get(rname, [])
                    if rows:
                        ax.plot([r["drop"] for r in rows], [r[ev] for r in rows], "-o", ms=2.5, lw=1, color=pal[ci], label=short[rname])
                ax.set_xlim(0.78, 1.0); ax.set_ylim(-0.02, 1.02)
                ax.set_title(f"{key}: {'12-s fade' if ev == 'slow' else 'small object, one frame'}")
                ax.set_xlabel("drop rate"); ax.set_ylabel("recall")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, fontsize=5.5, loc="lower center", ncol=3)
        fig.tight_layout(rect=(0, 0.14, 1, 1)); fig.savefig(FIG / "audit_curves.pdf"); plt.close(fig)

(TAB / "headline.json").write_text(json.dumps(headline, indent=1, default=float))
for k, v in headline.items():
    print(f"{k}: {v}")
