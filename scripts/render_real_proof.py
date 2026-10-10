#!/usr/bin/env python3
"""Old method against new method on real footage with nothing added: a sunset time-lapse, out/real_proof.mp4.

  old method   the certified rule as published: a brightness offset fitted per patch (THEORY.md sections 2 to 4)
  new method   the same rule without the offset (section 9)
Both at the benchmark schedule (Delta_0 = 80, gamma = 1/2), without the sequential test, on the grey frames.

The footage is "Sunset timelapse in Funchal - 2014" by valunik, CC BY 3.0, from Wikimedia Commons
(https://commons.wikimedia.org/wiki/File:Sunset_timelapse_in_Funchal_-_2014.webm), resized to 704 x 400. Put it
at data/commons/funchal.webm.

A 2 x 2 grid: the camera, a few lines of text, and under them the picture each method leaves the model with:
every patch as it was the last time it was sent, with the wrong pixels tinted red and the patches sent since the last frame shown boxed in green (`--plain` leaves them as they are). The methods decide on brightness; the colour pictures are built
from the colour frames with the same decisions. "Wrong" is the share of pixels of that picture more than 20 grey
levels from the camera frame, in brightness. "Sends" is the share of patches sent after the first frame.
The clip is replayed at three times its speed and the last frame is held. Needs ffmpeg with libx264 on PATH.

    python3 scripts/render_real_proof.py [--out out/real_proof.mp4] [--still 1500]
    python3 scripts/render_real_proof.py --src data/commons/olympiaturm.ogv --size 480x272 --fps 24     # numbers only
"""
import argparse, subprocess, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from certskip import video
from certskip.native import NativeSequentialPruner
from PIL import Image, ImageDraw
from render_findings import BG, FPS, GREEN, INK, MUTED, P, RED, SCHED, font
from render_before_after import Writer

SRC, SIZE, SRC_FPS, GAP, SPEED, HOLD_S, OFF = "data/commons/funchal.webm", (704, 400), 30, 4, 3, 4, 20
TINT, SENT = (255, 40, 40), (60, 230, 110)
F_TAG, F_BIG, F_TEXT, F_ANS, F_NOTE = font(24, True), font(34, True), font(26), font(26, True), font(13)
PANELS = (("truth", "REAL SCENE (camera)"), ("old", "WHAT THE MODEL SEES: old way"), ("new", "WHAT THE MODEL SEES: new way"))


def colour_frames():
    """The frames of iter_gray, in colour."""
    W, H = SIZE
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-nostdin", "-i", SRC, "-vf", f"fps={SRC_FPS},scale={W}:{H}:flags=area", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    while True:
        buf = proc.stdout.read(W * H * 3)
        if len(buf) < W * H * 3:
            break
        yield np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    proc.wait()


def paste(held, frame, keep):
    """Copy the patches sent on this frame into the held picture."""
    for gy, gx in zip(*np.where(keep)):
        held[gy * P:(gy + 1) * P, gx * P:(gx + 1) * P] = frame[gy * P:(gy + 1) * P, gx * P:(gx + 1) * P]


def decisions():
    """Patches sent on every frame by each method, and how much of the held picture is wrong after each frame."""
    W, H = SIZE; pr = {"old": NativeSequentialPruner(H, W, P, multiscale=SCHED, z=1e9, offset="patch"), "new": NativeSequentialPruner(H, W, P, multiscale=SCHED, z=1e9, offset="none")}
    keeps = {k: [] for k in pr}; wrong = {k: [] for k in pr}; held = {}
    for t, f in enumerate(video.iter_gray(SRC, fps=SRC_FPS, scale=SIZE)):
        for k, p in pr.items():
            keep = p.step(f)[0].astype(bool).copy(); keeps[k].append(keep)
            if t == 0:
                held[k] = f.copy()
            paste(held[k], f, keep)
            wrong[k].append(100 * (np.abs(held[k].astype(np.int16) - f) > OFF).mean())
    return {k: np.stack(v) for k, v in keeps.items()}, wrong


def marked(held, grey, truth, plain, now):
    """The held pictures with the wrong pixels tinted red and the patches sent just now (`now`) boxed in green,
    unless `plain`."""
    if plain:
        return held
    out = {}
    for k, pic in held.items():
        bad = np.abs(grey[k].astype(np.int16) - truth) > OFF; pic = pic.astype(np.float32)
        pic[bad] = 0.45 * pic[bad] + 0.55 * np.array(TINT, np.float32)
        for gy, gx in zip(*np.where(now[k])):
            y, x = gy * P, gx * P
            pic[y:y + P, x:x + P] = 0.7 * pic[y:y + P, x:x + P] + 0.3 * np.array(SENT, np.float32)
            pic[y:y + 2, x:x + P] = SENT; pic[y + P - 2:y + P, x:x + P] = SENT; pic[y:y + P, x:x + 2] = SENT; pic[y:y + P, x + P - 2:x + P] = SENT
        out[k] = pic.astype(np.uint8)
    return out


def text_panel():
    """What the clip is, in plain words, with the credit small underneath."""
    W, H = SIZE; im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im); x, y = 36, 10
    for line in ("Can the model keep up", "with the sunset?"):
        d.text((x, y), line, font=F_BIG, fill=INK); y += 39
    y += 6
    for line in ("The model keeps a full picture of the scene.", "To save work, only the parts that change", "are refreshed."):
        d.text((x, y), line, font=F_TEXT, fill=INK); y += 30
    y += 7; d.rectangle((x, y + 4, x + 24, y + 28), fill=SENT); d.text((x + 34, y), "Green = refreshed just now.", font=F_TEXT, fill=INK); y += 30
    d.text((x + 34, y), "The rest is kept from before.", font=F_TEXT, fill=INK); y += 30
    d.rectangle((x, y + 4, x + 24, y + 28), fill=TINT); d.text((x + 34, y), "Red = changed, but never refreshed.", font=F_TEXT, fill=INK); y += 37
    for line, colour in (("Old way: the sky goes stale.", RED), ("New way: it keeps up.", GREEN)):
        d.text((x, y), line, font=F_TEXT, fill=colour); y += 30
    y = H - 38
    for line in ("Footage: “Sunset timelapse in Funchal - 2014” by valunik, CC BY 3.0, Wikimedia Commons;", "resized, replayed at 3x. Red = brightness off by more than 20 on a scale of 0 to 255."):
        d.text((x, y), line, font=F_NOTE, fill=MUTED); y += 17
    return im


def frame(pics, panel, sent, wrong, final=False):
    W, H = SIZE; im = Image.new("RGB", (2 * W + GAP, 2 * H + GAP), BG); d = ImageDraw.Draw(im, "RGBA"); im.paste(panel, (W + GAP, 0))
    for (key, tag), (x0, y0) in zip(PANELS, ((0, 0), (0, H + GAP), (W + GAP, H + GAP))):
        im.paste(Image.fromarray(pics[key]), (x0, y0))
        d.rectangle((x0 + 10, y0 + 10, x0 + 28 + d.textlength(tag, font=F_TAG), y0 + 48), fill=(0, 0, 0, 200)); d.text((x0 + 19, y0 + 14), tag, font=F_TAG, fill=INK)
        if key == "truth":
            continue
        note = f"refreshes {sent[key]:.1f}% of the video"
        d.rectangle((x0 + 10, y0 + H - 96, x0 + 28 + d.textlength(note, font=F_TAG), y0 + H - 60), fill=(0, 0, 0, 200)); d.text((x0 + 19, y0 + H - 93), note, font=F_TAG, fill=INK)
        bad = wrong[key] >= 5; colour = RED if bad else GREEN; note = f"{'✗' if bad else '✓'} {wrong[key]:.0f}% of the picture is stale"
        d.rectangle((x0 + 10, y0 + H - 56, x0 + 32 + d.textlength(note, font=F_ANS), y0 + H - 12), fill=(0, 0, 0, 220)); d.text((x0 + 21, y0 + H - 51), note, font=F_ANS, fill=colour)
        if final:
            d.rectangle((x0 + 2, y0 + 2, x0 + W - 3, y0 + H - 3), outline=colour, width=6)
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out/real_proof.mp4"); ap.add_argument("--src", help="another clip: only the numbers are printed, at --size and --fps"); ap.add_argument("--size", default="704x400"); ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--plain", action="store_true", help="do not tint the wrong pixels red"); ap.add_argument("--still", type=int, help="write this one frame as a PNG next to --out and stop")
    a = ap.parse_args(); out = Path(a.out)
    if a.src:
        global SRC, SIZE, SRC_FPS
        SRC, SIZE, SRC_FPS = a.src, tuple(int(v) for v in a.size.split("x")), a.fps
    keeps, wrong = decisions(); T = len(wrong["old"]); sent = {k: 100 * v[1:].mean() for k, v in keeps.items()}; panel = text_panel()
    print(f"{T} frames; sent: old {sent['old']:.2f}%, new {sent['new']:.2f}%; wrong, mean over the clip: old {np.mean(wrong['old']):.1f}%, new {np.mean(wrong['new']):.1f}%; "
          f"at the end: old {wrong['old'][-1]:.1f}%, new {wrong['new'][-1]:.1f}%; largest: old {max(wrong['old']):.1f}%, new {max(wrong['new']):.1f}%", flush=True)
    if a.src:
        return
    held, grey, wr, last = {}, {}, None, None
    for t, (f, g) in enumerate(zip(colour_frames(), video.iter_gray(SRC, fps=SRC_FPS, scale=SIZE))):
        for k in keeps:
            if t == 0:
                held[k] = f.copy(); grey[k] = g.copy()
            paste(held[k], f, keeps[k][t]); paste(grey[k], g, keeps[k][t])
        now = {k: keeps[k][max(t - SPEED + 1, 1):t + 1].any(0) if t else np.zeros_like(keeps[k][0]) for k in keeps}   # sent since the last frame shown; the first frame is sent whole and not boxed
        show = t % SPEED == 0 or t == T - 1
        if a.still is not None:
            if t == a.still:
                Image.fromarray(frame({"truth": f, **marked(held, grey, g, a.plain, now)}, panel, sent, {k: wrong[k][t] for k in keeps}, True)).save(out.with_suffix(".png")); return
            continue
        if show:
            last = frame({"truth": f, **marked(held, grey, g, a.plain, now)}, panel, sent, {k: wrong[k][t] for k in keeps}, t == T - 1)
            wr = wr or Writer(out, (last.shape[1], last.shape[0])); wr.add(last)
    for _ in range(int(HOLD_S * FPS)):
        wr.add(last)
    wr.close(); print(f"wrote {out} ({wr.n / FPS:.1f} s)")


if __name__ == "__main__":
    main()
