#!/usr/bin/env python3
"""A short video of the findings of THEORY.md sections 9 to 11 (RESULTS.md Parts VIII to X): out/findings.mp4.

Four scenes, each showing what a rule leaves the model holding, side by side with the camera:
  1  real footage at its native 25 fps, nothing planted: the consecutive-frame heuristic and the certified rule at
     the same skip rate; pixels off by more than 30 grey levels are tinted red (Part IX)
  2  the lights are dimmed to 60 % (applied to real footage): the rule with the per-patch offset and the rule
     without it (Part VIII)
  3  a square fades in on the parking lot (planted, as in Part VII): the same two rules (Part VIII)
  4  the river camera: which patches are sent when a frame is held and when the centre of the next 8 frames is
     held, at the same certificate (Part X)
  5  the measured numbers, read from the result files in out/ that exist
Needs ffmpeg with libx264 on PATH, and the footage of `fetch_data.py all`.

    python3 scripts/render_findings.py [--out out/findings.mp4]
"""
import argparse, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from certskip import video, baselines as bl, centre as ct
from certskip.core import drop_rate, patch_grid
from certskip.native import NativeSequentialPruner
from real_bench import schedule

W, H, FPS, P = 1280, 720, 30, 16
BG, INK, MUTED, RED, GREEN = (14, 14, 13), (240, 239, 236), (165, 163, 155), (232, 76, 76), (110, 215, 95)
GOOD, BAD = (96, 205, 118), (232, 76, 76)                         # verdicts: always with a tick or a cross and a sentence
SCHED = schedule(80.0, 0.5)


def font(size, bold=False):
    from matplotlib import font_manager as fm
    return ImageFont.truetype(fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal")), size)


F_TITLE, F_SUB, F_LABEL, F_STAT, F_NOTE, F_BIG = font(33, True), font(19), font(20, True), font(18), font(15), font(27)


def rgb(gray, wrong=None, keep=None):
    """Grey frame as RGB, with wrong pixels tinted red or sent patches outlined green."""
    g = np.clip(gray, 0, 255).astype(np.float32); out = np.stack([g, g, g], -1)
    if wrong is not None:
        out[wrong] = 0.35 * out[wrong] + 0.65 * np.array(RED, np.float32)
    out = out.astype(np.uint8)
    if keep is not None:
        for gy, gx in zip(*np.where(keep)):
            y, x = gy * P, gx * P
            out[y, x:x + P] = GREEN; out[y + P - 1, x:x + P] = GREEN; out[y:y + P, x] = GREEN; out[y:y + P, x + P - 1] = GREEN
    return out


def card(title, sub, panels, note):
    """One 1280x720 frame: title, subtitle, a row of panels and a footnote. A panel is (image, label, stat) or
    (image, label, stat, (ok, sentence)): the verdict frames the panel in green or red and is written under the label."""
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    d.text((40, 30), title, font=F_TITLE, fill=INK); d.multiline_text((40, 82), sub, font=F_SUB, fill=MUTED, spacing=7)
    n = len(panels); gap = 14; pw = (W - 80 - gap * (n - 1)) // n; ph = min(int(pw * 0.75), 420)
    below = 46 + 25 * max(len(q[2].split("\n")) + (len(q) > 3) for q in panels)   # label, verdict and the lines of the stat
    y0 = 150 + max(0, (H - 50 - 150 - ph - below) // 2)            # the row of panels sits in the middle of what is left
    for i, q in enumerate(panels):
        img, label, stat = q[:3]; x0 = 40 + i * (pw + gap); ys = y0 + ph + 42
        im.paste(Image.fromarray(img).resize((pw, ph), Image.BICUBIC), (x0, y0))
        d.text((x0, y0 + ph + 12), label, font=F_LABEL, fill=INK)
        if len(q) > 3:
            ok, sentence = q[3]; col = GOOD if ok else BAD
            d.rectangle((x0 - 4, y0 - 4, x0 + pw + 3, y0 + ph + 3), outline=col, width=4)
            d.text((x0, ys), "\u2713" if ok else "\u2717", font=F_LABEL, fill=col); d.text((x0 + 26, ys + 1), sentence, font=F_STAT, fill=INK); ys += 27
        d.multiline_text((x0, ys), stat, font=F_STAT, fill=MUTED, spacing=6)
    d.text((40, H - 34), note, font=F_NOTE, fill=MUTED)
    return np.asarray(im)


def text_card(title, lines, note):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    d.text((40, 30), title, font=F_TITLE, fill=INK); y = 150
    for head, body in lines:
        d.text((40, y), head, font=F_BIG, fill=INK); d.text((40, y + 40), body, font=F_SUB, fill=MUTED); y += 118 if len(lines) <= 4 else 100
    d.text((40, H - 34), note, font=F_NOTE, fill=MUTED)
    return np.asarray(im)


class Writer:
    def __init__(self, path):
        self.proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                                      "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
        self.n = 0

    def add(self, frame, repeat=1):
        for _ in range(repeat):
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes()); self.n += 1

    def close(self):
        self.proc.stdin.close(); self.proc.wait()


def held_copies(frames, offset):
    """Copy held by the certified rule after every frame, and its keep masks."""
    T, h, w = frames.shape; pr = NativeSequentialPruner(h, w, P, multiscale=SCHED, z=1e9, offset=offset); held, keep = [], []
    for f in frames:
        keep.append(pr.step(f)[0]); held.append(pr.view(False))
    return held, np.stack(keep)


def scene_ghosts(wr, seconds=8):
    fr = video.read_gray("data/caviar_LeftBag.mpg", fps=25); T = len(fr); n = seconds * 25
    cert, keep = held_copies(fr, "none"); skip = drop_rate(keep)
    scores = bl.consecutive_mean_scores(fr, P); tau = float(np.quantile(scores, skip)); hk = bl.consecutive_mean(fr, P, tau, scores=scores)
    heur, ref = [fr[0].copy()], fr[0].copy()
    for t in range(1, T):
        patch_grid(ref, P)[hk[t]] = patch_grid(fr[t], P)[hk[t]]; heur.append(ref.copy())
    bad = lambda a, t: np.abs(a[t].astype(int) - fr[t].astype(int)) > 30
    wrong = np.array([bad(heur, t).mean() for t in range(T)]); c = np.concatenate([[0], np.cumsum(wrong)])
    s = int(np.argmax(c[n:] - c[:-n]))                              # the stretch where the heuristic is most wrong
    for i in range(int(seconds * FPS)):
        t = s + int(i * 25 / FPS); bh, bc = bad(heur, t), bad(cert, t)
        wr.add(card("Same skip rate, real footage, nothing planted",
                    f"What each rule leaves the model holding at 25 frames per second. Both skip {100 * skip:.1f}% of patches.\nRed: pixels off by more than 30 grey levels.",
                    [(rgb(fr[t]), "camera", ""), (rgb(heur[t], bh), "previous-frame heuristic", f"pixels wrong now: {100 * bh.mean():.1f}%"),
                     (rgb(cert[t], bc), "certified rule", f"pixels wrong now: {100 * bc.mean():.1f}%")],
                    "CAVIAR lobby camera (EC project IST 2001 37540). The heuristic is the consecutive-frame criterion re-implemented here."))


def scene_lights(wr):
    fr = video.read_gray("data/hall_monitor_cif.y4m", fps=30); seg = fr.astype(np.float32); seg[150:] *= 0.6; seg = np.clip(np.rint(seg), 0, 255).astype(np.uint8)
    old, _ = held_copies(seg, "patch"); new, _ = held_copies(seg, "none")
    def bright(a, t):
        e = patch_grid(a[t].astype(np.float32) - seg[t][: a[t].shape[0], : a[t].shape[1]], P).mean((2, 3)); return 100 * (e > 30).mean()
    for t in range(105, 300):
        wr.add(card("The lights are dimmed to 60%", "Dimming applied to real footage at the five-second mark.\nWhat each rule leaves the model holding.",
                    [(rgb(seg[t]), "camera", ""), (rgb(old[t]), "rule as published", f"patches still over 30 levels too bright: {bright(old, t):.0f}%"),
                     (rgb(new[t]), "rule after the fix", f"patches still over 30 levels too bright: {bright(new, t):.0f}%")],
                    "Xiph hall_monitor. The published rule forgives a brightness shift per patch, so flat patches are never resent."))


def scene_square(wr):
    fr = video.read_gray("data/VIRAT_S_000200_00.mp4", fps=10, scale=(960, 544), start=10, max_frames=120)
    y, x, box, contrast, pre, nf = 300, 420, 96, 70, 20, 80
    sign = -1 if fr[pre, y:y + box, x:x + box].mean() > 110 else 1
    seg = fr.astype(np.int32); ramp = np.concatenate([np.zeros(pre), np.linspace(0, contrast, nf), np.full(len(fr) - pre - nf, contrast)])
    for i, c in enumerate(ramp):
        seg[i, y:y + box, x:x + box] += int(round(sign * c))
    seg = np.clip(seg, 0, 255).astype(np.uint8)
    old, _ = held_copies(seg, "patch"); new, _ = held_copies(seg, "none")
    ys, xs = slice(y - 96, y + 192), slice(x - 144, x + 240)
    def present(a, t):
        return 100 * (np.abs(a[t][y:y + box, x:x + box].astype(int) - fr[t][y:y + box, x:x + box].astype(int)) > 0.5 * ramp[t] + 1).mean() if ramp[t] > 8 else 0.0
    for t in range(len(seg)):
        wr.add(card("A square fades in over eight seconds", "Planted on real footage, as in the end-to-end experiment.\nWhat each rule leaves the model holding.",
                    [(rgb(seg[t][ys, xs]), "camera", f"contrast so far: {ramp[t]:.0f} grey levels"), (rgb(old[t][ys, xs]), "rule as published", f"square pixels that reached the copy: {present(old, t):.0f}%"),
                     (rgb(new[t][ys, xs]), "rule after the fix", f"square pixels that reached the copy: {present(new, t):.0f}%")],
                    "VIRAT parking lot (Kitware public release). Shown at 1.5 times real speed."), repeat=2)


def scene_centre(wr, seconds=8, look=8):
    fr = video.read_gray("data/xiph_bridge_close_cif.y4m", fps=30, max_frames=seconds * 30 + 60); T, h, w = fr.shape
    k0 = ct.CentrePruner(h, w, P, multiscale=SCHED, lookahead=0).run(fr)["keep"]; k8 = ct.CentrePruner(h, w, P, multiscale=SCHED, lookahead=look).run(fr)["keep"]
    for t in range(30, 30 + seconds * 30):
        s0, s8 = 100 * k0[1:t + 1].mean(), 100 * k8[1:t + 1].mean()
        wr.add(card("Same guarantee, fewer patches", f"Green: patches sent on this frame. Holding the centre of the next {look} frames, the picture lags\n{look} frames and the certificate is unchanged.",
                    [(rgb(fr[t], keep=k0[t]), "hold the current frame", f"patches sent so far: {s0:.1f}%"), (rgb(fr[t], keep=k8[t]), f"hold the centre of the next {look} frames", f"patches sent so far: {s8:.1f}%")],
                    "Xiph bridge_close: a river whose water moves. Multi-scale certificate without offset or sub-pixel shift, the same in both panels."))


def scene_numbers(wr, seconds=5):
    """The measured numbers behind the scenes, read from the result files that exist."""
    import json
    lines = []
    f = Path("out/e2e_e1b_held_cuda.json")
    if f.exists() and len(json.loads(f.read_text())) == 3:          # all three cameras of the run
        e1 = [r for res in json.loads(f.read_text()).values() for r in res["E1"]]
        lines.append((f"The square reaches the model's copy in {int(sum(r['present_in_view']['certified'] for r in e1))} of {len(e1)} trials, up from {int(sum(r['present_in_view']['published'] for r in e1))}",
                      "A model that reuses tokens holds the last copy it was sent. The rule is now certified against that copy."))
    f = Path("out/framerate.json")
    if f.exists():
        rows = json.loads(f.read_text()); med = lambda k: float(np.median([r[-1][k]["ghost"] for r in rows.values()]))
        lines.append((f"At the camera's own frame rate the heuristic leaves {med('heuristic_matched') / med('certified'):.0f} times more ghost pixels",
                      f"Median of {len(rows)} fixed cameras at equal skip rates: {med('heuristic_matched'):.2f}% of the static scene wrong, against {med('certified'):.2f}%."))
    f = Path("out/centre.json")
    if f.exists():
        rows = [r for r in json.loads(f.read_text()).values() if "multi-scale" in r]
        cut = 100 * (1 - float(np.median([r["multi-scale"]["sends"]["8"] / r["multi-scale"]["sends"]["0"] for r in rows])))
        gap = float(np.median([r["multi-scale"]["sends"]["0"] / r["multi-scale"]["bound"] for r in rows]))
        lines.append((f"Holding the centre of the next 8 frames sends {cut:.0f}% fewer patches, same guarantee",
                      f"Median of {len(rows)} cameras. Holding the current frame sends {gap:.1f} times the provable minimum for any method."))
    frame = text_card("What was measured", lines, "Code, tests and proofs: github.com/nsquaredzz/certified-skip   (THEORY.md sections 9 to 12)")
    wr.add(frame, repeat=seconds * FPS)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out/findings.mp4"); ap.add_argument("--scenes", default="1,2,3,4,5")
    a = ap.parse_args(); wr = Writer(a.out)
    for s in a.scenes.split(","):
        {"1": scene_ghosts, "2": scene_lights, "3": scene_square, "4": scene_centre, "5": scene_numbers}[s](wr); print(f"scene {s} done, {wr.n / FPS:.1f} s so far", flush=True)
    wr.close(); print(f"wrote {a.out}: {wr.n / FPS:.1f} s")


if __name__ == "__main__":
    main()
