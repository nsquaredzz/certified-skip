#!/usr/bin/env python3
"""Old rule against new rule as plain footage with a mask on it: out/before_after.mp4, and one clip per scene.

  old rule   the certified rule as published: quotient + multi-scale, with a brightness offset and a sub-pixel
             shift fitted per patch (THEORY.md sections 2 to 4)
  new rule   no offset, no shift, and the centre of the next 8 frames is held (sections 9 and 11): certified
             against the copy the model holds, for the whole frame; the picture lags 8 frames
Both at the benchmark schedule (Delta_0 = 80, gamma = 1/2), without the sequential test.

No cards and no captions. A clip is two panels, the picture each rule leaves the model with, played at the
footage's own speed with a mask on top, a tag in the corner of each and one line of legend underneath.

  square   a dark square fades in on the parking lot over eight seconds (planted on real footage)
  lights   the hallway lights go down to 60 % (applied to real footage)
  river    the river, untouched: the mask is the patches sent to the model on that frame

In the first two the mask covers every pixel the change moved by more than 10 grey levels: green where the
model's picture followed it at least half-way, red stripes where it did not. The numbers printed at the end are
computed on the frames shown. Needs ffmpeg with libx264 on PATH, and the footage of `fetch_data.py all`.

    python3 scripts/render_before_after.py [--scenes square,lights,river] [--camera] [--out out/before_after.mp4]
"""
import argparse, subprocess, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from certskip import video, centre as ct
from certskip.native import NativeSequentialPruner
from PIL import Image, ImageDraw
from render_findings import BG, FPS, INK, MUTED, P, SCHED, font

LOOK, UP, GAP, BAND = 8, 2, 4, 44                                  # look-ahead; panels are doubled, with a legend band below
GOT, MISS = (60, 220, 120), (240, 60, 60)
F_TAG, F_LEGEND = font(22, True), font(20)
CHANGE = [(GOT, "change that reached the model"), (MISS, "change that has not")]
facts = {}


def both(frames):
    """Picture held and patches sent after every frame, for the old and for the new rule."""
    T, h, w = frames.shape; gh, gw = h // P, w // P
    pr = NativeSequentialPruner(h, w, P, multiscale=SCHED, z=1e9, offset="patch"); old, ko = [], []
    for f in frames:
        ko.append(pr.step(f)[0]); old.append(pr.view(False))
    new = []
    out = ct.CentrePruner(h, w, P, multiscale=SCHED, lookahead=LOOK).run(frames, on_frame=lambda s, k, c: new.append(np.clip(np.rint(c), 0, 255).astype(np.uint8)))
    assert out["worst"] < 1.0
    new = [np.pad(c, ((0, h - gh * P), (0, w - gw * P)), mode="edge") for c in new]
    return old, np.stack(ko), new, out["keep"]


def sent(keep, t0, t1):
    return 100 * keep[t0:t1].mean()


def rgb(gray):
    return np.repeat(gray[..., None], 3, -1)


def change_mask(picture, truth, before):
    """The model's picture with a mask over everything that really changed (truth against `before`, the scene
    without the change): green where the change reached the picture, red stripes where it is missing.
    Returns the image and the share of the change that reached the picture, in %."""
    size = np.abs(truth.astype(np.int16) - before.astype(np.int16)); changed = size > 10
    got = changed & (np.abs(picture.astype(np.int16) - truth.astype(np.int16)) <= 0.5 * size); miss = changed & ~got
    out = rgb(picture).astype(np.float32)
    yy, xx = np.indices(picture.shape); a = np.where(((yy + xx) // 4) % 2 == 0, 0.7, 0.4)[..., None]
    out[got] = 0.55 * out[got] + 0.45 * np.array(GOT, np.float32)
    out[miss] = ((1 - a) * out + a * np.array(MISS, np.float32))[miss]
    return out.astype(np.uint8), 100 * got.sum() / max(changed.sum(), 1)


def sent_mask(frame, keep):
    """The frame with the patches sent on it filled and outlined."""
    out = rgb(frame).copy()
    for gy, gx in zip(*np.where(keep)):
        y, x = gy * P, gx * P
        out[y:y + P, x:x + P] = (0.5 * out[y:y + P, x:x + P] + 0.5 * np.array(GOT)).astype(np.uint8)
        out[y, x:x + P] = GOT; out[y + P - 1, x:x + P] = GOT; out[y:y + P, x] = GOT; out[y:y + P, x + P - 1] = GOT
    return out


def tile(panels, tags, legend, note):
    """Panels side by side at twice their size, a tag in the corner of each, and under them a band with the
    legend on the left and where the footage comes from on the right."""
    h, w = panels[0].shape[:2]; n = len(panels)
    im = Image.new("RGB", (n * w * UP + (n - 1) * GAP, h * UP + BAND), BG); d = ImageDraw.Draw(im, "RGBA")
    for i, (p, tag) in enumerate(zip(panels, tags)):
        x0 = i * (w * UP + GAP)
        im.paste(Image.fromarray(p).resize((w * UP, h * UP), Image.NEAREST), (x0, 0))
        d.rectangle((x0 + 10, 10, x0 + 26 + d.textlength(tag, font=F_TAG), 44), fill=(0, 0, 0, 170)); d.text((x0 + 18, 13), tag, font=F_TAG, fill=INK)
    x, y = 12, h * UP + (BAND - 22) // 2
    for colour, text in legend:
        d.rectangle((x, y + 2, x + 18, y + 20), fill=colour); d.text((x + 28, y - 1), text, font=F_LEGEND, fill=INK); x += 64 + d.textlength(text, font=F_LEGEND)
    d.text((im.width - 12 - d.textlength(note, font=F_LEGEND), y - 1), note, font=F_LEGEND, fill=MUTED)
    return np.asarray(im)


class Writer:
    def __init__(self, path, size):
        self.proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(FPS), "-i", "-",
                                      "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
        self.n = 0

    def add(self, frame):
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes()); self.n += 1

    def close(self):
        self.proc.stdin.close(); self.proc.wait()


def scene_square():
    fr = video.read_gray("data/VIRAT_S_000200_00.mp4", fps=30, scale=(960, 544), start=10, max_frames=345)
    y, x, box, contrast, pre, nf = 300, 420, 96, 70, 60, 240
    sign = -1 if fr[pre, y:y + box, x:x + box].mean() > 110 else 1
    seg = fr.copy(); ramp = np.concatenate([np.zeros(pre), np.linspace(0, contrast, nf), np.full(len(fr) - pre - nf, contrast)])
    for i, c in enumerate(ramp):
        seg[i, y:y + box, x:x + box] = np.clip(fr[i, y:y + box, x:x + box].astype(np.int32) + int(round(sign * c)), 0, 255)
    old, ko, new, kn = both(seg)
    ys, xs = slice(y - 96, y + 192), slice(x - 128, x + 224)       # a 352 x 288 window with the square in the middle
    for t in range(45, len(seg)):
        (io, go), (im, gn) = change_mask(old[t][ys, xs], seg[t][ys, xs], fr[t][ys, xs]), change_mask(new[t][ys, xs], seg[t][ys, xs], fr[t][ys, xs])
        yield [rgb(seg[t][ys, xs]), io, im], CHANGE, "parking lot: a dark square added to real footage, fading in"
    facts["square"] = f"share of the square in the model's picture at the end: old {go:.0f}%, new {gn:.0f}%; picture re-sent: old {sent(ko, 1, len(seg)):.2f}%, new {sent(kn, 1, len(seg)):.2f}%"


def scene_lights():
    fr = video.read_gray("data/hall_monitor_cif.y4m", fps=30); seg = fr.astype(np.float32); seg[150:] *= 0.6; seg = np.clip(np.rint(seg), 0, 255).astype(np.uint8)
    old, ko, new, kn = both(seg)
    for t in range(105, len(seg)):
        (io, go), (im, gn) = change_mask(old[t], seg[t], fr[t]), change_mask(new[t], seg[t], fr[t])
        yield [rgb(seg[t]), io, im], CHANGE, "hallway: lights dimmed to 60% in software"
    facts["lights"] = f"share of the dimming in the model's picture at the end: old {go:.0f}%, new {gn:.0f}%; picture re-sent: old {sent(ko, 1, len(seg)):.1f}%, new {sent(kn, 1, len(seg)):.1f}%"


def scene_river(seconds=8):
    fr = video.read_gray("data/xiph_bridge_close_cif.y4m", fps=30, max_frames=seconds * 30 + 30)
    old, ko, new, kn = both(fr)
    for t in range(30, len(fr)):
        yield [rgb(fr[t]), sent_mask(fr[t], ko[t]), sent_mask(fr[t], kn[t])], [(GOT, "sent to the model on this frame")], "river: untouched footage"
    facts["river"] = f"picture sent on the frames shown: old {sent(ko, 30, len(fr)):.1f}%, new {sent(kn, 30, len(fr)):.1f}%"


SCENES = {"square": scene_square, "lights": scene_lights, "river": scene_river}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out/before_after.mp4"); ap.add_argument("--scenes", default="square,lights,river")
    ap.add_argument("--camera", action="store_true", help="add the untouched camera as a first panel (too wide for most feeds)")
    a = ap.parse_args(); out = Path(a.out); tags = ["camera", "old rule", "new rule"][0 if a.camera else 1:]; joined = None
    for s in a.scenes.split(","):
        clip = None
        for panels, legend, note in SCENES[s]():
            frame = tile(panels[0 if a.camera else 1:], tags, legend, note)
            if clip is None:
                size = (frame.shape[1], frame.shape[0]); clip = Writer(out.with_name(f"{out.stem}_{s}.mp4"), size); joined = joined or Writer(out, size)
            clip.add(frame); joined.add(frame)
        clip.close(); print(f"{s}: {clip.n / FPS:.1f} s; {facts[s]}", flush=True)
    joined.close(); print(f"wrote {out} ({joined.n / FPS:.1f} s) and one clip per scene next to it")


if __name__ == "__main__":
    main()
