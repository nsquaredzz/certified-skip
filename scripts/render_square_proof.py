#!/usr/bin/env python3
"""One clip: a large dark band fades in on the hallway floor, and how much of it each rule gets to the model.

The band is drawn as in Part VII (`e2e_vlm.py`): flat, 70 grey levels darker than the floor, fading in linearly
over the clip, except that here people walking over it stay in front of it. It is a diagonal band 80 pixels wide
across the floor instead of a 48 x 48 square at a random place, so that it can be seen in a feed.

  old method   the certified rule as published: a brightness offset fitted per patch (THEORY.md sections 2 to 4)
  new method   the same rule without the offset (section 9)

The rules work on the grey footage. What is drawn on top is for the viewer only: every pixel of the band is
tinted orange by how much of the band is in that picture, and a white line marks where the band is in the camera
frame. The share under each picture is the pixels of the band that are at least half as dark there as in the
camera frame. "Sends" is the share of patches sent over the clip after its first frame. The score line is the model's answers in the Part VII rerun on held copies (smaller squares). On
this large band the 2B model's answers were not stable from one placement to the next, so none is shown.

The fade plays at twice its speed and the last frame is held with the shares on it. Needs ffmpeg with libx264.

    python3 scripts/render_square_proof.py [--out out/square_proof.mp4] [--still]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from certskip import video
from PIL import Image, ImageDraw
from e2e_vlm import CLIPS
from render_findings import BG, FPS, GREEN, INK, MUTED, RED, font
from render_before_after import Writer, rgb
from render_lights_proof import held

UP, GAP, SPEED, HOLD_S, CONTRAST, RUN = 2, 4, 2, 5, 70, "out/e2e_e1b_held_cuda.json"
Y, X, TALL, WIDE, PRE, HOLD = 176, 0, 112, 256, 60, 60
SQUARE = (slice(Y, Y + TALL), slice(X, X + WIDE))                 # the box around the band
_y, _x = np.indices((TALL, WIDE))
SHAPE = np.abs(_x - (205 - _y * 160 / TALL)) <= 40                 # a band 80 wide, from the far end of the floor to the near left corner
EDGE = SHAPE & ~(np.roll(SHAPE, 1, 1) & np.roll(SHAPE, -1, 1) & np.roll(SHAPE, 1, 0))
ORANGE = np.array((255, 130, 0), np.float32)
F_TAG, F_BIG, F_TEXT, F_ANS, F_NOTE = font(24, True), font(40, True), font(28), font(28, True), font(19)
PANELS = (("truth", "REAL SCENE (camera)"), ("old", "WHAT THE AI SEES: old method"), ("new", "WHAT THE AI SEES: new method"))


def planted():
    """The hallway clip without the band and with it, and for each frame what stands in front of the band."""
    spec = CLIPS["hall"]; fr = video.read_gray(spec["path"], fps=spec["fps"], scale=spec["scale"]); nf = len(fr) - PRE - HOLD
    seg = fr.astype(np.int32).copy(); ramp = np.concatenate([np.zeros(PRE), np.linspace(0, CONTRAST, nf), np.full(HOLD, CONTRAST)])
    floor = np.median(fr[::5], axis=0); fronts = []                 # the empty hallway: whatever differs from it stands on the band
    for i, c in enumerate(ramp):
        front = np.abs(fr[i][SQUARE] - floor[SQUARE]) > 25
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):           # one pixel wider, so no dark rim is drawn around legs
            front |= np.roll(front, (dy, dx), (0, 1))
        seg[i][SQUARE] -= np.where(front | ~SHAPE, 0, int(round(c))); fronts.append(front)
    return fr, np.clip(seg, 0, 255).astype(np.uint8), fronts


def tinted(picture, before, front, outline):
    """The picture in colour: inside the band, orange by how much darker than the scene without the band; with
    `outline`, a white line where the band is in the camera frame."""
    out = rgb(picture).astype(np.float32)
    a = (0.8 * SHAPE * np.clip((before[SQUARE].astype(np.float32) - picture[SQUARE]) / CONTRAST, 0, 1))[..., None]
    box = (1 - a) * out[SQUARE] + a * ORANGE
    if outline:
        box[EDGE & ~front] = 255
    out[SQUARE] = box
    return out.astype(np.uint8)


def deepest(mask):
    """A pixel of the mask far from its border: where to point at a hole."""
    m = mask.copy(); last = m
    while m.any():
        last = m; m = m & np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)
    ys, xs = np.where(last); return int(ys[len(ys) // 2]), int(xs[len(xs) // 2])


def text_panel(w, h, score, sent):
    """What the clip shows, in the corner the pictures leave free."""
    im = Image.new("RGB", (w, h), BG); d = ImageDraw.Draw(im); x, y = 40, 28
    for line in ("Does the AI see", "what changed?"):
        d.text((x, y), line, font=F_BIG, fill=INK); y += 50
    y += 12; d.rectangle((x, y + 5, x + 26, y + 31), fill=tuple(int(c) for c in ORANGE))
    d.text((x + 38, y), "Something new appears on the floor", font=F_TEXT, fill=INK); y += 40
    for line, colour in (("(a dark band, shown here in orange).", INK), ("", INK),
                         ("To save compute, the AI is sent only", INK), ("about 6% of the video. A method picks", INK), ("which pieces.", INK), ("", INK),
                         (f"Old method: sends {sent['old']:.1f}%, loses part of it.", RED), (f"New method: sends {sent['new']:.1f}%, loses nothing.", GREEN)):
        d.text((x, y), line, font=F_TEXT, fill=colour); y += 39 if line else 14
    y = h - 96
    for line in ("The band is added to real footage. Fade at 2x speed. On smaller", "patches, asked “is there a dark square patch?”, the model says",
                 f"yes {score[0]} of 27 times with the old method and {score[1]} with the new."):
        d.text((x, y), line, font=F_NOTE, fill=MUTED); y += 27
    return im


def frame(pics, panel, sent, shares=None, hole=None):
    h, w = pics["truth"].shape[:2]; W, H = w * UP, h * UP
    im = Image.new("RGB", (2 * W + GAP, 2 * H + GAP), BG); d = ImageDraw.Draw(im, "RGBA"); im.paste(panel, (W + GAP, 0))
    for (key, tag), (x0, y0) in zip(PANELS, ((0, 0), (0, H + GAP), (W + GAP, H + GAP))):
        im.paste(Image.fromarray(pics[key]).resize((W, H), Image.NEAREST), (x0, y0))
        d.rectangle((x0 + 10, y0 + 10, x0 + 28 + d.textlength(tag, font=F_TAG), y0 + 48), fill=(0, 0, 0, 200)); d.text((x0 + 19, y0 + 14), tag, font=F_TAG, fill=INK)
        if key != "truth":
            note = f"sends {sent[key]:.1f}% of the video"
            d.rectangle((x0 + 10, y0 + 52, x0 + 28 + d.textlength(note, font=F_TAG), y0 + 88), fill=(0, 0, 0, 200)); d.text((x0 + 19, y0 + 55), note, font=F_TAG, fill=INK)
        if shares and key != "truth":
            ok = shares[key] > 99; colour = GREEN if ok else RED
            note = "✓ nothing is missing" if ok else f"✗ {100 - shares[key]:.0f}% of it is missing"; xr = x0 + W - 12
            d.rectangle((xr - 22 - d.textlength(note, font=F_ANS), y0 + H - 60, xr, y0 + H - 12), fill=(0, 0, 0, 220)); d.text((xr - 11 - d.textlength(note, font=F_ANS), y0 + H - 54), note, font=F_ANS, fill=colour)
            d.rectangle((x0 + 2, y0 + 2, x0 + W - 3, y0 + H - 3), outline=colour, width=6)
            if hole and not ok:                                      # point at the largest hole
                px, py = x0 + UP * (X + hole[1]), y0 + UP * (Y + hole[0]); lx, ly = x0 + 372, y0 + 448
                d.line((lx, ly + 20, px, py), fill=(255, 255, 255, 255), width=3); d.ellipse((px - 6, py - 6, px + 6, py + 6), fill=(255, 255, 255, 255))
                d.rectangle((lx - 8, ly, lx + 12 + d.textlength("missing", font=F_ANS), ly + 42), fill=(0, 0, 0, 220)); d.text((lx + 2, ly + 4), "missing", font=F_ANS, fill=RED)
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out/square_proof.mp4")
    ap.add_argument("--still", action="store_true", help="write the held last frame as a PNG next to --out and stop")
    a = ap.parse_args(); out = Path(a.out)
    fr, seg, fronts = planted(); (old, ko), (new, kn) = held(seg, "patch"), held(seg, "none"); grey = {"truth": seg, "old": old, "new": new}; last = len(seg) - 1
    sent = {"old": 100 * ko[1:].mean(), "new": 100 * kn[1:].mean()}        # share of the patches sent after the first frame
    dark = (fr[last][SQUARE].astype(int) - seg[last][SQUARE]) > CONTRAST / 2
    there = {k: (fr[last][SQUARE].astype(int) - grey[k][last][SQUARE]) > CONTRAST / 2 for k in ("old", "new")}
    shares = {k: 100 * v[dark].mean() for k, v in there.items()}
    e1 = [r for res in json.loads(Path(RUN).read_text()).values() for r in res["E1"] if r["yes"]["truth"] == 1]
    score = [sum(r["yes"][k] == 1 for r in e1) for k in ("published", "certified")]
    panel = text_panel(seg.shape[2] * UP, seg.shape[1] * UP, score, sent)
    shot = lambda t, *more: frame({k: tinted(v[t], fr[t], fronts[t], k != "truth") for k, v in grey.items()}, panel, sent, *more)
    final = shot(last, shares, deepest(dark & ~there["old"]))
    if a.still:
        Image.fromarray(final).save(out.with_suffix(".png")); return
    wr = Writer(out, (final.shape[1], final.shape[0]))
    for t in range(30, len(seg), SPEED):
        wr.add(shot(t))
    for _ in range(int(HOLD_S * FPS)):
        wr.add(final)
    wr.close()
    print(f"wrote {out} ({wr.n / FPS:.1f} s); share of the band in the model's picture at the end: old {shares['old']:.0f}%, new {shares['new']:.0f}%; video sent: old {sent['old']:.2f}%, new {sent['new']:.2f}%; model, of {len(e1)}: old {score[0]}, new {score[1]}")


if __name__ == "__main__":
    main()
