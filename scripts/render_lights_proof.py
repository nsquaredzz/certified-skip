#!/usr/bin/env python3
"""One clip, no mask: the hallway lights go down and you see the picture each rule leaves the model with.

  old rule   the certified rule as published: a brightness offset fitted per patch (THEORY.md sections 2 to 4)
  new rule   the same rule without the offset (section 9): certified against the copy the model holds
Both at the benchmark schedule (Delta_0 = 80, gamma = 1/2), without the sequential test and without look-ahead.

A 2 x 2 grid: the camera, four lines of text, and under them the two held pictures exactly as held. The share
under each is the pixels more than 20 grey levels away from the camera on that frame. The dimming (to 60 %
over one second) is applied in software to real footage. Needs ffmpeg with libx264 on PATH.

    python3 scripts/render_lights_proof.py [--out out/lights_proof.mp4] [--still 200]
"""
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from certskip import video
from certskip.native import NativeSequentialPruner
from PIL import Image, ImageDraw
from render_findings import BG, FPS, GREEN, INK, MUTED, P, RED, SCHED, font
from render_before_after import Writer, rgb

UP, GAP, START, DIM0, DIM1, LEVEL = 2, 4, 45, 90, 120, 0.6
F_TAG, F_BIG, F_TEXT, F_NOTE = font(22, True), font(40, True), font(27), font(19)


def held(frames, offset):
    T, h, w = frames.shape; pr = NativeSequentialPruner(h, w, P, multiscale=SCHED, z=1e9, offset=offset); out, keep = [], []
    for f in frames:
        keep.append(pr.step(f)[0]); out.append(pr.view(False))
    return out, np.stack(keep)


def wrong(picture, truth):
    return 100 * (np.abs(picture.astype(np.int16) - truth.astype(np.int16)) > 20).mean()


def text_panel(w, h):
    im = Image.new("RGB", (w, h), BG); d = ImageDraw.Draw(im); x, y = 44, 70
    d.text((x, y), "The lights dim.", font=F_BIG, fill=INK); y += 84
    for line, colour in (("The model is only sent the parts", INK), ("of the picture a rule says changed.", INK), ("", INK),
                         ("Old rule: most of the room stays lit.", RED), ("New rule: the whole room dims.", GREEN)):
        d.text((x, y), line, font=F_TEXT, fill=colour); y += 44
    d.text((x, h - 84), "Real hallway footage. The dimming to 60%", font=F_NOTE, fill=MUTED); d.text((x, h - 56), "is applied in software.", font=F_NOTE, fill=MUTED)
    return im


def frame(cam, old, new, panel):
    h, w = cam.shape; W, H = w * UP, h * UP
    im = Image.new("RGB", (2 * W + GAP, 2 * H + GAP), BG); d = ImageDraw.Draw(im, "RGBA"); im.paste(panel, (W + GAP, 0))
    for pic, (x0, y0), tag, colour in ((cam, (0, 0), "camera", None), (old, (0, H + GAP), "what the model has: old rule", RED), (new, (W + GAP, H + GAP), "what the model has: new rule", GREEN)):
        im.paste(Image.fromarray(rgb(pic)).resize((W, H), Image.NEAREST), (x0, y0))
        d.rectangle((x0 + 10, y0 + 10, x0 + 26 + d.textlength(tag, font=F_TAG), y0 + 44), fill=(0, 0, 0, 190)); d.text((x0 + 18, y0 + 13), tag, font=F_TAG, fill=INK)
        if colour:
            note = f"{wrong(pic, cam):.0f}% of the picture is wrong"
            d.rectangle((x0 + 10, y0 + H - 46, x0 + 26 + d.textlength(note, font=F_TAG), y0 + H - 12), fill=(0, 0, 0, 190)); d.text((x0 + 18, y0 + H - 43), note, font=F_TAG, fill=colour)
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out/lights_proof.mp4"); ap.add_argument("--still", type=int, help="write this one frame as a PNG next to --out and stop")
    a = ap.parse_args(); out = Path(a.out)
    fr = video.read_gray("data/hall_monitor_cif.y4m", fps=30)
    gain = np.concatenate([np.ones(DIM0), np.linspace(1, LEVEL, DIM1 - DIM0), np.full(len(fr) - DIM1, LEVEL)])
    seg = np.clip(np.rint(fr.astype(np.float32) * gain[:, None, None]), 0, 255).astype(np.uint8)
    (old, ko), (new, kn) = held(seg, "patch"), held(seg, "none"); panel = text_panel(fr.shape[2] * UP, fr.shape[1] * UP)
    if a.still is not None:
        Image.fromarray(frame(seg[a.still], old[a.still], new[a.still], panel)).save(out.with_suffix(".png")); return
    wr = None
    for t in range(START, len(seg)):
        f = frame(seg[t], old[t], new[t], panel); wr = wr or Writer(out, (f.shape[1], f.shape[0])); wr.add(f)
    wr.close(); t = len(seg) - 1
    print(f"wrote {out} ({wr.n / FPS:.1f} s); wrong at the end: old {wrong(old[t], seg[t]):.0f}%, new {wrong(new[t], seg[t]):.0f}%; "
          f"picture re-sent after the first frame: old {100 * ko[1:].mean():.1f}%, new {100 * kn[1:].mean():.1f}%")


if __name__ == "__main__":
    main()
