#!/usr/bin/env python3
"""Position linking against position linking with the tubes consulted, on CAVIAR: out/follow_proof.mp4.

Two parts, each following one labelled person with both linkers (scripts/follow_tubes.py, same settings):

  Meet_Crowd, person 0, 1 frame per second    position alone swaps to someone else; with the tubes it stays
  Walk1, person 4, 2 frames per second        the other way round: the tubes break a track position alone kept

Both linkers get the labelled boxes without identities on the sampled frames only. The clip is played at its own
25 fps; between two samples the box drawn is the labelled box of whoever the track was put on at the last sample:
green while that is the person followed, red once it is someone else. The white outline is the person followed.
On the right the tubes (cells of the block above the threshold, one colour per tube) are tinted in.
Needs ffmpeg with libx264 on PATH.

    .venv/bin/python scripts/render_follow_proof.py [--out out/follow_proof.mp4] [--still 0:200]
"""
import argparse, subprocess, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from certskip import video
from PIL import Image, ImageDraw
from follow_from_skips import FPS, load_gt
from follow_tubes import DS, change, linked, tubes
from render_findings import BG, GREEN, INK, MUTED, RED, font

PARTS = (("Meet_Crowd", 0, 1, "Position alone swaps to someone else. With the tubes it stays on the person."),
         ("Walk1", 4, 2, "The other way round: here the tubes break a track that position alone kept."))
TAU, COEXIST, MARGIN, ZOOM, GAP, HEAD, FOOT, HOLD_S = 25.0, 25, 24.0, 2, 6, 92, 64, 4
F_BIG, F_TEXT, F_TAG, F_ANS, F_NOTE = font(28, True), font(20), font(22, True), font(24, True), font(14)
METHODS = (("pos", "POSITION ONLY"), ("comb", "POSITION + TUBES"))


class Writer:
    def __init__(self, path, size):
        self.proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(FPS), "-i", "-",
                                      "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)

    def add(self, frame):
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.proc.stdin.close(); self.proc.wait()


def palette(c):
    """A colour per tube, away from the green and red of the verdicts."""
    r = np.random.default_rng(c); h = r.uniform(0.5, 0.92); x = 1 - abs((h * 6) % 2 - 1)
    rgb = [(0, x, 1), (x, 0, 1), (1, 0, x)][int(h * 6) - 3]             # cyan to blue to magenta
    return np.array(rgb, np.float32) * 255


def prepare(name, pid, fps):
    path = ROOT / "data" / f"caviar_{name}.mpg"; gt = load_gt(ROOT / "data" / "caviar_gt" / f"{name}.xml"); fr = video.read_gray(path)
    g = change(path, "causal"); t, y, x, cl = tubes(g, TAU, COEXIST); T = len(g); del g
    o = np.argsort(t, kind="stable"); ts, ys, xs, cs = t[o], y[o], x[o], cl[o]; cut = np.nonzero(np.diff(ts))[0] + 1
    tz = {int(ts[i]): (ys[i:j], xs[i:j], cs[i:j]) for i, j in zip(np.r_[0, cut], np.r_[cut, len(ts)])}
    stride = FPS // fps; on = {"pos": {}, "comb": {}}
    score = {"pos": linked(gt, stride, T, trace=on["pos"])[pid], "comb": linked(gt, stride, T, tz, MARGIN, trace=on["comb"])[pid]}
    samples = [s for s in sorted(on["pos"]) if pid in gt.get(s, {})]
    track = {k: next(q for q, p in on[k][samples[0]].items() if p == pid) for k in on}
    return dict(name=name, pid=pid, fps=fps, gt=gt, fr=fr, tz=tz, stride=stride, on=on, score=score, samples=samples, track=track,
                last=max(s for s in gt if pid in gt[s] and s < T))


def picture(grey, tz_t):
    pic = np.repeat(grey[:, :, None], 3, 2).astype(np.float32)
    if tz_t is not None:
        y, x, cl = tz_t
        for c in np.unique(cl):
            m = cl == c; col = palette(int(c))
            for dy in range(DS):
                for dx in range(DS):
                    pic[y[m] * DS + dy, x[m] * DS + dx] = 0.45 * pic[y[m] * DS + dy, x[m] * DS + dx] + 0.55 * col
    return Image.fromarray(pic.astype(np.uint8)).resize((grey.shape[1] * ZOOM, grey.shape[0] * ZOOM), Image.BILINEAR)


def frame(S, t, final=False):
    H, W = S["fr"].shape[1] * ZOOM, S["fr"].shape[2] * ZOOM; gt = S["gt"]; pid = S["pid"]
    im = Image.new("RGB", (2 * W + GAP, HEAD + H + FOOT), BG); d = ImageDraw.Draw(im, "RGBA")
    d.text((16, 8), f"Following one person with boxes seen {S['fps']} time{'s' if S['fps'] > 1 else ''} a second", font=F_BIG, fill=INK)
    d.text((16, 48), PARTS[S["i"]][3], font=F_TEXT, fill=INK)
    s = t // S["stride"] * S["stride"]; done = [q for q in S["samples"] if q <= t]
    note = f"CAVIAR {S['name']}, frame {t}. The linker last saw the boxes at frame {s}."
    d.text((2 * W + GAP - 16 - d.textlength(note, font=F_NOTE), 14), note, font=F_NOTE, fill=MUTED)
    for j, (key, tag) in enumerate(METHODS):
        x0 = j * (W + GAP); im.paste(picture(S["fr"][t], S["tz"].get(t) if key == "comb" else None), (x0, HEAD))
        who = S["on"][key].get(s, {}).get(S["track"][key]) if s >= S["samples"][0] else pid
        true = gt.get(t, {}).get(pid)
        if true is not None and who != pid:
            d.rectangle([x0 + true[0] * ZOOM, HEAD + true[1] * ZOOM, x0 + true[2] * ZOOM, HEAD + true[3] * ZOOM], outline=(255, 255, 255, 255), width=2)
            d.text((x0 + true[0] * ZOOM, HEAD + true[3] * ZOOM + 3), "the person", font=F_NOTE, fill=(255, 255, 255, 255))
        ok = who == pid; colour = GREEN if ok else RED
        box = gt.get(t, {}).get(who) or gt.get(s, {}).get(who) if who is not None else None
        if box is not None:
            d.rectangle([x0 + box[0] * ZOOM, HEAD + box[1] * ZOOM, x0 + box[2] * ZOOM, HEAD + box[3] * ZOOM], outline=colour + (255,), width=4)
        d.rectangle((x0 + 10, HEAD + 10, x0 + 28 + d.textlength(tag, font=F_TAG), HEAD + 46), fill=(0, 0, 0, 200)); d.text((x0 + 19, HEAD + 14), tag, font=F_TAG, fill=INK)
        right = sum(S["on"][key][q].get(S["track"][key]) == pid for q in done); n = max(len(done), 1)
        state = "on the right person" if ok else ("track lost" if who is None else "on someone else")
        d.text((x0 + 16, HEAD + H + 8), f"{'✓' if ok else '✗'} {state}", font=F_ANS, fill=colour)
        tail = f"right on {right} of {len(done)} looks ({100 * right / n:.0f}%)"
        d.text((x0 + W - 16 - d.textlength(tail, font=F_TEXT), HEAD + H + 12), tail, font=F_TEXT, fill=INK)
        if final:
            good = S["score"][key]["right"] >= 0.9
            d.rectangle((x0 + 2, HEAD + 2, x0 + W - 3, HEAD + H - 3), outline=(GREEN if good else RED) + (255,), width=6)
    d.text((16, HEAD + H + 42), "Green box: the track is on the person. Red box: it is on someone else; the white outline is the person. Colours on the right: the tubes.", font=F_NOTE, fill=MUTED)
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT / "out" / "follow_proof.mp4"))
    ap.add_argument("--still", help="part:frame, written as a PNG next to --out")
    a = ap.parse_args(); out = Path(a.out); wr = None
    for i, (name, pid, fps, _) in enumerate(PARTS):
        if a.still and int(a.still.split(":")[0]) != i:
            continue
        S = prepare(name, pid, fps); S["i"] = i
        print(f"{name}, person {pid}, {fps} fps: position only right {100 * S['score']['pos']['right']:.1f} %, with the tubes {100 * S['score']['comb']['right']:.1f} %", flush=True)
        if a.still:
            t = int(a.still.split(":")[1]); Image.fromarray(frame(S, t, t == S["last"])).save(out.with_name(f"{out.stem}_{i}_{t}.png")); return
        for t in range(S["samples"][0], S["last"] + 1):
            f = frame(S, t, t == S["last"]); wr = wr or Writer(out, (f.shape[1], f.shape[0])); wr.add(f)
        for _ in range(HOLD_S * FPS):
            wr.add(f)
        del S
    wr.close(); print(f"wrote {out}")


if __name__ == "__main__":
    main()
