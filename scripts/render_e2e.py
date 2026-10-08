#!/usr/bin/env python3
"""Render the end-to-end blind-spot experiment (scripts/e2e_vlm.py, E1) as a video.

Reproduces the exact planted trials of a finished run (same RNG stream, same calibration) and shows,
side by side, what the camera saw and what each pruning rule left the video LLM to look at:
truth / EVS-style view / certified view / certified + sequential view. The model's recorded answers
are burned in on a hold card at the end of every trial.

    .venv/bin/python scripts/render_e2e.py --json out/e2e_e1b.json --trials 16 \
        --pick hall:0,lobby:0,virat:4 --out out/e2e_blindspot.mp4
"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from certskip import video, baselines as bl
from certskip.core import patch_grid, drop_rate
from certskip.native import NativeSequentialPruner
from real_bench import schedule
from e2e_vlm import CLIPS, LastKeptView, P

PW, PH, HDR, FTR, OUT_FPS = 960, 576, 52, 72, 25
GREEN, RED, YEL, WHITE, GREY = (80, 230, 80), (255, 80, 80), (255, 215, 0), (255, 255, 255), (170, 170, 170)


def font(sz):
    for f in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(f, sz)
        except Exception:
            pass
    return ImageFont.load_default()


F_T, F_P, F_S = font(24), font(22), font(19)


def run_views_keeps(frames, sched, tau_evs):
    """Like e2e_vlm.run_views but also returns the per-frame keep masks."""
    T, H, W = frames.shape
    views = {"evs": [], "certified": [], "sequential": []}; keeps = {k: [] for k in views}
    pr_ms = NativeSequentialPruner(H, W, P, multiscale=sched, z=1e9)
    pr_sq = NativeSequentialPruner(H, W, P, multiscale=sched, z=8.0)
    ev = LastKeptView(frames[0]); prev = frames[0]
    for t in range(T):
        k, _, _ = pr_ms.step(frames[t]); views["certified"].append(pr_ms.view()); keeps["certified"].append(k)
        k, _, _ = pr_sq.step(frames[t]); views["sequential"].append(pr_sq.view()); keeps["sequential"].append(k)
        if t == 0:
            views["evs"].append(frames[0].copy()); keeps["evs"].append(np.ones_like(k))
        else:
            k = bl.consecutive_mean(np.stack([prev, frames[t]]), P, tau_evs)[1]
            views["evs"].append(ev.update(frames[t], k)); keeps["evs"].append(k)
        prev = frames[t]
    return views, keeps


def replay_trials(rng, fr, spec, n_trials, fade_s_max, contrast):
    """Replays e2e_vlm's RNG draws for one clip; returns the trial geometry list."""
    T = len(fr); fps = spec["fps"]; box = spec["box"]
    hold = int(2 * fps); pre = int(2 * fps)
    fade_s = min(fade_s_max, (T - pre - hold - 2) / fps); nf = int(fade_s * fps)
    y0f, y1f, x0f, x1f = spec["floor"]; out = []
    for trial in range(n_trials):
        t0 = int(rng.integers(pre, max(pre + 1, T - nf - hold)))
        y = int(rng.integers(y0f, max(y0f + 1, y1f - box))); x = int(rng.integers(x0f, max(x0f + 1, x1f - box)))
        sign = -1 if fr[t0, y:y + box, x:x + box].mean() > 110 else 1
        out.append(dict(trial=trial, t0=t0, y=y, x=x, sign=sign, pre=pre, nf=nf, hold=hold, fade_s=fade_s, box=box))
    return out


def plant(fr, g, contrast):
    seg = fr[g["t0"] - g["pre"]: g["t0"] + g["nf"] + g["hold"]].astype(np.int32).copy()
    ramp = np.concatenate([np.zeros(g["pre"]), np.linspace(0, contrast, g["nf"]), np.full(g["hold"], contrast)])
    b = g["box"]
    for i, c in enumerate(ramp):
        seg[i, g["y"]:g["y"] + b, g["x"]:g["x"] + b] += int(round(g["sign"] * c))
    return np.clip(seg, 0, 255).astype(np.uint8), ramp


def panel(img, keep, up, g, label, sub, outline=True):
    """One PW x PH RGB panel: up-scaled grey image centred, kept patches in green, planted box in yellow."""
    H, W = img.shape
    im = Image.fromarray(np.repeat(np.repeat(img, up, 0), up, 1)).convert("RGB")
    d = ImageDraw.Draw(im)
    if keep is not None:
        for gy, gx in zip(*np.where(keep)):
            d.rectangle([gx * P * up, gy * P * up, (gx + 1) * P * up - 1, (gy + 1) * P * up - 1], outline=GREEN, width=1)
    if outline:
        m = 5; b = g["box"] * up
        d.rectangle([g["x"] * up - m, g["y"] * up - m, g["x"] * up + b + m, g["y"] * up + b + m], outline=YEL, width=2)
    can = Image.new("RGB", (PW, PH), (0, 0, 0))
    ox, oy = (PW - W * up) // 2, (PH - H * up) // 2
    can.paste(im, (ox, oy))
    d = ImageDraw.Draw(can)
    d.rectangle([0, 0, PW, 60], fill=(0, 0, 0))
    d.text((10, 6), label, font=F_P, fill=WHITE); d.text((10, 34), sub, font=F_S, fill=GREY)
    return can


def ans_colour(a, truth_yes):
    y = a.lower().startswith("yes")
    return GREEN if y == truth_yes else RED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="out/e2e_e1b.json"); ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--pick", default="hall:0,lobby:0,virat:4"); ap.add_argument("--out", default="out/e2e_blindspot.mp4")
    ap.add_argument("--fade-s", type=float, default=8.0); ap.add_argument("--contrast", type=int, default=70)
    ap.add_argument("--hold-card", type=float, default=4.0); ap.add_argument("--stills", default="out/e2e_blindspot")
    a = ap.parse_args()
    res = json.loads(Path(a.json).read_text())
    picks = {}
    for p in a.pick.split(","):
        c, i = p.split(":"); picks.setdefault(c, []).append(int(i))
    rng = np.random.default_rng(0); sched = schedule(80.0, 0.5)
    CW, CH = 2 * PW, HDR + 2 * PH + FTR
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{CW}x{CH}", "-r", str(OUT_FPS),
           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", a.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n_out = 0
    for name in ("hall", "lobby", "virat"):          # same order as the experiment: the RNG stream is shared
        spec = CLIPS[name]
        fr = video.read_gray(spec["path"], fps=spec["fps"], scale=spec["scale"])
        geoms = replay_trials(rng, fr, spec, a.trials, a.fade_s, a.contrast)
        if name not in picks:
            continue
        T, H, W = fr.shape; fps = spec["fps"]; up = spec["up"]
        d_cert = drop_rate(NativeSequentialPruner(H, W, P, multiscale=sched, z=1e9).run(fr)["keep"])
        tau, _, d_evs = bl.calibrate(bl.consecutive_mean, fr[: min(T, 600)], P, d_cert)
        for ti in picks[name]:
            g = geoms[ti]; rec = res[name]["E1"][ti]
            assert rec["sign"] == g["sign"], f"RNG replay mismatch on {name} trial {ti}"
            seg, ramp = plant(fr, g, a.contrast)
            views, keeps = run_views_keeps(seg, sched, tau)
            word = "dark" if g["sign"] < 0 else "bright"
            q = f"Is there a {word} square patch on the floor or wall in this image? Answer yes or no."
            answers = rec["answers"]; truth_yes = answers["truth"].lower().startswith("yes")
            tot = {k: [0, 0] for k in keeps}
            n_seg = len(seg); n_play = int(round(n_seg / fps * OUT_FPS)); n_hold = int(a.hold_card * OUT_FPS)
            last_canvas = None
            for i in range(n_play + n_hold):
                t = min(int(i * fps / OUT_FPS), n_seg - 1) if i < n_play else n_seg - 1
                card = i >= n_play
                if not card and t > 0 and (i == 0 or int((i - 1) * fps / OUT_FPS) != t):
                    for k in keeps:
                        tot[k][0] += int(keeps[k][t].sum()); tot[k][1] += keeps[k][t].size
                if card and last_canvas is not None:
                    canvas = last_canvas.copy()
                else:
                    labels = {"evs": ("EVS-style view  (consecutive-frame mean, same drop rate)", ),
                              "certified": ("CERTIFIED view  (quotient + multi-scale range rule)", ),
                              "sequential": ("CERTIFIED + SEQUENTIAL view  (space-time scan)", )}
                    pn = {}
                    pn["truth"] = panel(seg[t], None, up, g, "TRUTH  (what the camera sees)",
                                        f"square fades in over {g['fade_s']:.0f} s, contrast now {ramp[t]:.0f} / {a.contrast} grey levels")
                    for k in ("evs", "certified", "sequential"):
                        kept_now = int(keeps[k][t].sum()); d_so_far = 1 - tot[k][0] / max(tot[k][1], 1)
                        pn[k] = panel(views[k][t], keeps[k][t], up, g, labels[k][0],
                                      f"kept now {kept_now} / {keeps[k][t].size} patches (green)   dropped so far {100 * d_so_far:.1f}%")
                    canvas = Image.new("RGB", (CW, CH), (0, 0, 0))
                    canvas.paste(pn["truth"], (0, HDR)); canvas.paste(pn["evs"], (PW, HDR))
                    canvas.paste(pn["certified"], (0, HDR + PH)); canvas.paste(pn["sequential"], (PW, HDR + PH))
                    d = ImageDraw.Draw(canvas)
                    d.text((12, 12), f"End-to-end blind spot   |   {name}: {W}x{H} @ {fps} fps, real footage   |   all rules at drop rate {100 * d_cert:.1f}%   |   t = {t / fps:5.1f} s",
                           font=F_T, fill=WHITE)
                    # footer: question + progress bar
                    y0 = HDR + 2 * PH
                    d.text((12, y0 + 8), f"Question to Qwen2-VL-2B at the end of the fade:  \"{q}\"", font=F_S, fill=GREY)
                    frac = (t - g["pre"]) / max(g["nf"], 1)
                    bx0, bx1, by = 12, CW - 12, y0 + 44
                    d.rectangle([bx0, by, bx1, by + 14], outline=GREY)
                    d.rectangle([bx0, by, bx0 + int((bx1 - bx0) * min(max(frac, 0), 1)), by + 14], fill=YEL if frac < 1 else GREEN)
                    stage = "before onset" if frac < 0 else ("fading in" if frac < 1 else "fade complete, model is asked")
                    d.text((bx1 - 330, by - 24), stage, font=F_S, fill=WHITE)
                    last_canvas = canvas
                if card:
                    d = ImageDraw.Draw(canvas)
                    # answer stamps on each panel
                    def stamp(x, y, txt, col):
                        d.rectangle([x, y, x + 520, y + 46], fill=(0, 0, 0)); d.text((x + 10, y + 8), txt, font=F_T, fill=col)
                    stamp(PW - 540, HDR + PH - 60, f"model: {answers['truth']}   (control frame without square: {answers['control']})", GREEN if truth_yes else GREY)
                    stamp(2 * PW - 540, HDR + PH - 60, f"model: {answers['evs']}   object in view: {'yes' if rec['present_in_view']['evs'] else 'NO'}",
                          ans_colour(answers["evs"], truth_yes))
                    stamp(PW - 540, HDR + 2 * PH - 60, f"model: {answers['certified']}   object in view: {'yes' if rec['present_in_view']['certified'] else 'NO'}",
                          ans_colour(answers["certified"], truth_yes))
                    stamp(2 * PW - 540, HDR + 2 * PH - 60, f"model: {answers['sequential']}   object in view: {'yes' if rec['present_in_view']['sequential'] else 'NO'}",
                          ans_colour(answers["sequential"], truth_yes))
                    if i == n_play and a.stills:
                        canvas.save(f"{a.stills}_{name}_t{ti}.png")
                proc.stdin.write(np.asarray(canvas, np.uint8).tobytes()); n_out += 1
            print(f"{name} trial {ti}: t0={g['t0']} box at ({g['x']},{g['y']}) sign {g['sign']}  answers {answers}  "
                  f"drop evs/cert/seq so far {[round(1 - tot[k][0] / max(tot[k][1], 1), 3) for k in ('evs', 'certified', 'sequential')]}", flush=True)
    proc.stdin.close(); proc.wait()
    print(f"wrote {a.out}: {n_out} frames at {OUT_FPS} fps ({n_out / OUT_FPS:.0f} s)")


if __name__ == "__main__":
    main()
