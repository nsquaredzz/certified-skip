#!/usr/bin/env python3
"""Render the moving-camera rule on a clip: [frame + what happens to each patch] [the model's view].

  green   sent because the certificate failed (refresh)
  yellow  sent because the model holds nothing there (frontier)
  blue    followed: held content that moved, not sent
  red     frame border: alignment failed and the whole frame was sent (anchor)
  white   with --masks: outline of the hand-labelled objects (DAVIS)

A bar on top names the clip and counts, for this frame and so far, the patches sent and followed.

    python3 scripts/render_moving.py data/tears_of_steel_720p.mov out/moving_film.mp4 --start 60 --max-frames 240
    python3 scripts/render_moving.py "data/davis/DAVIS/JPEGImages/480p/bear/%05d.jpg" out/moving_bear.mp4 \\
        --masks data/davis/DAVIS/Annotations/480p/bear --out-fps 24
"""
import argparse, subprocess, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from certskip import moving as mv, video

P = 16
SCHED = {r: 80.0 * (2 * r + 1) ** -0.5 for r in (0, 1, 2, 3)}
GREEN, YELLOW, BLUE, RED, WHITE = (60, 220, 90), (240, 200, 40), (70, 150, 255), (235, 60, 60), (255, 255, 255)
BAR = 32


def outline(img, mask, colour):
    """Draw the border of every patch of the grid mask."""
    for gy, gx in zip(*np.nonzero(mask)):
        y, x = gy * P, gx * P
        img[y, x:x + P] = colour; img[y + P - 1, x:x + P] = colour
        img[y:y + P, x] = colour; img[y:y + P, x + P - 1] = colour


def bar(width, title, t, now, total):
    """The caption strip: legend on the left, counts on the right."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGB", (width, BAR), (18, 18, 18)); d = ImageDraw.Draw(im); font = ImageFont.load_default(size=15)
    x = 10
    for colour, label in ((GREEN, "sent"), (YELLOW, "new area, sent"), (BLUE, "followed, not sent"), (RED, "cut: whole frame sent")):
        d.rectangle([x, 9, x + 13, 22], outline=colour, width=2); d.text((x + 20, 7), label, fill=(225, 225, 225), font=font)
        x += 34 + int(d.textlength(label, font=font))
    sent, fol = now
    n, ssum, fsum = total
    right = (f"{title}   frame {t}   this frame: sent {100 * sent:.0f} %, followed {100 * fol:.0f} %   "
             f"so far: skipped {100 * (1 - ssum / max(1, n)):.1f} %")
    d.text((width - 10 - d.textlength(right, font=font), 7), right, fill=(255, 255, 255), font=font)
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--start", type=float, default=None); ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--search", type=int, default=16, help="0 turns the follow step off")
    ap.add_argument("--masks", default=None, help="folder of DAVIS annotation PNGs for this clip")
    ap.add_argument("--out-fps", type=float, default=24.0); ap.add_argument("--crf", type=int, default=24)
    ap.add_argument("--still", default=None, help="also save frame N as a PNG: N:path")
    ap.add_argument("--title", default="", help="shown in the bar")
    a = ap.parse_args()
    masks = sorted(Path(a.masks).glob("*.png")) if a.masks else None
    pr = proc = None; stats = []; total = [0, 0.0, 0.0]
    for t, f in enumerate(video.iter_gray(a.inp, start=a.start, max_frames=a.max_frames)):
        if pr is None:
            H, W = f.shape[0] // P * P, f.shape[1] // P * P                # crop to the patch grid
            pr = mv.MovingPruner(H, W, P, SCHED, search=a.search)
            cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * W + 8}x{H + BAR}", "-r", str(a.out_fps),
                   "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(a.crf), "-pix_fmt", "yuv420p", a.out]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        f = np.ascontiguousarray(f[:H, :W])
        keep = pr.step(f); L = pr.last
        left = np.repeat(f[..., None], 3, 2)
        if masks is not None and t < len(masks):
            from PIL import Image
            g = np.array(Image.open(masks[t]))[:H, :W] > 0
            edge = g & ~(np.roll(g, 1, 0) & np.roll(g, -1, 0) & np.roll(g, 1, 1) & np.roll(g, -1, 1))
            left[edge] = WHITE
        fol = L["followed"]
        if fol.any():
            tint = np.kron(fol, np.ones((P, P), bool))
            left[tint] = (0.6 * left[tint] + 0.4 * np.array(BLUE)).astype(np.uint8)
            outline(left, fol, BLUE)
        outline(left, keep & L["reveal"], YELLOW)
        outline(left, keep & ~L["reveal"], GREEN)
        if L["anchor"] and t:
            left[:4] = RED; left[-4:] = RED; left[:, :4] = RED; left[:, -4:] = RED
        right = np.repeat(np.clip(np.rint(L["view"]), 0, 255).astype(np.uint8)[..., None], 3, 2)
        if t:                                                            # the first frame is always sent whole
            total[0] += 1; total[1] += keep.mean(); total[2] += fol.mean()
        frame = np.concatenate([left, np.zeros((H, 8, 3), np.uint8), right], 1)
        frame = np.concatenate([bar(frame.shape[1], a.title, t, (keep.mean(), fol.mean()), total), frame], 0)
        proc.stdin.write(frame.tobytes())
        if a.still and int(a.still.split(":", 1)[0]) == t:
            from PIL import Image
            Image.fromarray(frame).save(a.still.split(":", 1)[1])
        if t:
            stats.append((keep.mean(), fol.mean(), float(L["anchor"])))
    proc.stdin.close(); proc.wait()
    s = np.array(stats)
    print(f"{len(s) + 1} frames -> {a.out}: sent {s[:, 0].mean():.3f}, followed {s[:, 1].mean():.3f}, anchors {int(s[:, 2].sum())}")


if __name__ == "__main__":
    main()
