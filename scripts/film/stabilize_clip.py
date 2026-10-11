#!/usr/bin/env python3
"""Video proof for stabilize_test.py: each window plays slowly; under it the raw probability with all tokens new and
with still tokens reused, and the answer the change detector declares.

    .venv/bin/python scripts/film/stabilize_clip.py <dir> <out.mp4> 201
"""
import sys, json, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[1] / "python"))
import numpy as np
from render_findings import BG, INK, MUTED, font
from stabilize_test import WINDOWS, QS, detector, H_THRESH
from PIL import Image, ImageDraw
d, dst, tag = sys.argv[1], sys.argv[2], sys.argv[3]
col = np.load(f"{d}/sam{tag}.npz")["col"]; res = json.load(open(HERE.parents[1] / "out" / f"film_stabilize_{tag}.json"))
SER = (("full", "all tokens new", (57, 135, 229)), ("still", "still tokens reused", (217, 89, 38))); DET = (25, 158, 112)
W, H, Z, FPS, SLOW, HOLD_S = 640, 268, 1.2, 24, 6, 5; ZW, ZH = int(W * Z) // 2 * 2, int(H * Z) // 2 * 2; HEAD, CH, FOOT = 78, 300, 56; size = (ZW * 2 + 6, HEAD + ZH + CH + FOOT)
FB, FT, FG, FN, FS = font(26, True), font(18), font(16, True), font(14), font(13)
ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst], stdin=subprocess.PIPE)
for a, b in WINDOWS[tag]:
    ts = list(range(a, b + 1))
    for t in ts:
        im = Image.new("RGB", size, BG); dr = ImageDraw.Draw(im, "RGBA")
        dr.text((14, 6), "Can processing steps stop the model's answer flickering from frame to frame?", font=FB, fill=INK)
        dr.text((14, 44), f"Frames {a + 1}–{b + 1}, played at a sixth of the speed. Green: the answer a sequential change detector (threshold {H_THRESH:g}) declares from the reused-token stream.", font=FT, fill=INK)
        im.paste(Image.fromarray(col[t]).resize((ZW, ZH), Image.BILINEAR), ((size[0] - ZW) // 2, HEAD))
        pw = (size[0] - 60) // 2
        for i, q in enumerate(QS[tag]):
            x0 = 20 + i * (pw + 20); y0 = HEAD + ZH + 60; ph = CH - 110
            dr.text((x0, HEAD + ZH + 8), q.split(" Answer")[0], font=FG, fill=INK); dr.text((x0, HEAD + ZH + 30), "probability the model puts on “yes”, every frame", font=FN, fill=MUTED)
            for v in (0, 0.5, 1):
                yy = y0 + ph - v * ph; dr.line((x0 + 30, yy, x0 + pw, yy), fill=(50, 50, 48)); dr.text((x0, yy - 8), f"{v:g}", font=FS, fill=MUTED)
            xs = lambda f: x0 + 30 + (pw - 40) * (f - a) / (b - a)
            dr.line((xs(t), y0 - 4, xs(t), y0 + ph + 4), fill=(120, 120, 115), width=2)
            lx = x0 + 30
            for k, name, c in SER:
                p = [res[q][k][str(f)] for f in ts]; pts = [(xs(f), y0 + ph - v * ph) for f, v in zip(ts, p) if f <= t]
                if len(pts) > 1:
                    dr.line(pts, fill=c, width=2)
                for pt in pts:
                    dr.ellipse((pt[0] - 4, pt[1] - 4, pt[0] + 4, pt[1] + 4), fill=c, outline=BG, width=2)
                dr.rectangle((lx, y0 + ph + 26, lx + 12, y0 + ph + 38), fill=c); dr.text((lx + 18, y0 + ph + 22), name, font=FN, fill=INK); lx += 30 + dr.textlength(name, font=FN)
            dec = detector(np.array([res[q]["still"][str(f)] for f in ts]))
            for f, s in zip(ts, dec):
                if f <= t:
                    dr.rectangle((xs(f) - 5, y0 + ph + 6, xs(f) + 5, y0 + ph + 16), fill=DET if s else (60, 60, 58))
            dr.text((lx + 10, y0 + ph + 22), "■ declared “yes”", font=FN, fill=DET)
            dr.text((x0 + 30, y0 + ph + 42), f"frame {a + 1}", font=FS, fill=MUTED); dr.text((x0 + pw - 50, y0 + ph + 42), f"frame {b + 1}", font=FS, fill=MUTED)
        dr.text((14, size[1] - 24), f"Qwen2-VL-2B, frozen. Tears of Steel, (CC) Blender Foundation. Shot at {tag} s, frame {t + 1}.", font=FN, fill=MUTED)
        f = np.asarray(im).tobytes()
        for _ in range(HOLD_S * FPS if t == b else SLOW):
            ff.stdin.write(f)
ff.stdin.close(); ff.wait(); print("wrote", dst)
