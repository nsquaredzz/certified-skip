#!/usr/bin/env python3
"""Video proof for token_probe.py: the shot, and under it the probability the model puts on each answer over the
shot, for the four ways of feeding it. The curves are drawn up to the frame being shown.

    .venv/bin/python scripts/film/token_probe_clip.py <dir> <out.mp4> 201 356
"""
import sys, json, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[1] / "python"))
import numpy as np
from render_findings import BG, INK, MUTED, font
from PIL import Image, ImageDraw
d, dst, tags = sys.argv[1], sys.argv[2], sys.argv[3:]
SERIES = (("full", "all tokens new", (57, 135, 229)), ("still", "still reused", (217, 89, 38)), ("moved", "still + moved reused", (25, 158, 112)), ("stale", "never refreshed", (201, 133, 0)))
EVENTS = {"201": "A loudspeaker comes into view near frame 45; the woman is hidden behind the man from about frame 110.", "356": "The young man's mouth opens near the end of the shot."}
PLOT = {"201": (("Is there a grey loudspeaker visible?", "yes"), ("Is the woman visible?", "yes"), ("How many people are visible?", "2")),
        "356": (("Is the young man's mouth open?", "yes"), ("Is the young man smiling?", "yes"), ("Is the robot hand touching the young man?", "yes"))}
W, H, Z, FPS, HOLD_S = 640, 268, 1.2, 24, 6; ZW, ZH = int(W * Z) // 2 * 2, int(H * Z) // 2 * 2; HEAD, CH, FOOT = 78, 270, 60; CW = ZW
size = (ZW * 2 + 6, HEAD + ZH + CH + FOOT); FB, FT, FG, FN, FS = font(26, True), font(18), font(16, True), font(14), font(13)
ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst], stdin=subprocess.PIPE)
for tag in tags:
    col = np.load(f"{d}/sam{tag}.npz")["col"]; T = len(col); rows = json.load(open(HERE.parents[1] / "out" / f"film_token_probe_{tag}.json"))
    def series(qtext, ans):
        R = sorted((r for r in rows if r["q"].startswith(qtext.rstrip("?"))), key=lambda r: r["frame"]); fr = [r["frame"] for r in R]
        return fr, {k: [r[k] if r["kind"] == "yesno" else r[k][int(ans)] for r in R] for k, _, _ in SERIES}
    plots = [(q, ans) + series(q, ans) for q, ans in PLOT[tag]]
    for t in range(T):
        im = Image.new("RGB", size, BG); dr = ImageDraw.Draw(im, "RGBA")
        dr.text((14, 6), "Does the model notice what changes, when its tokens are reused?", font=FB, fill=INK); dr.text((14, 44), EVENTS[tag], font=FT, fill=INK)
        im.paste(Image.fromarray(col[t]).resize((ZW, ZH), Image.BILINEAR), ((size[0] - ZW) // 2, HEAD))
        lx = (size[0] - ZW) // 2 + 10
        for k, name, c in SERIES:
            dr.rectangle((lx, HEAD + 14, lx + 14, HEAD + 28), fill=c); dr.text((lx + 20, HEAD + 10), name, font=FN, fill=INK); lx += 34 + dr.textlength(name, font=FN)
        pw = (size[0] - 40) // 3 - 20
        for i, (q, ans, fr, ys) in enumerate(plots):
            x0 = 20 + i * (pw + 30); y0 = HEAD + ZH + 54; ph = CH - 90
            dr.text((x0, HEAD + ZH + 8), q, font=FG, fill=INK); dr.text((x0, HEAD + ZH + 30), f"probability the model puts on “{ans}”", font=FN, fill=MUTED)
            for v in (0, 0.5, 1):
                yy = y0 + ph - v * ph; dr.line((x0 + 30, yy, x0 + pw, yy), fill=(50, 50, 48)); dr.text((x0, yy - 8), f"{v:g}", font=FS, fill=MUTED)
            xs = lambda f: x0 + 30 + (pw - 30) * f / (T - 1)
            dr.line((xs(t), y0 - 4, xs(t), y0 + ph + 4), fill=(120, 120, 115), width=2)
            for k, name, c in SERIES:
                pts = [(xs(f), y0 + ph - v * ph) for f, v in zip(fr, ys[k]) if f <= t]
                if len(pts) > 1:
                    dr.line(pts, fill=c, width=2)
                for p in pts:
                    dr.ellipse((p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4), fill=c, outline=BG, width=2)
            dr.text((x0 + 30, y0 + ph + 8), "frame 1", font=FS, fill=MUTED); dr.text((x0 + pw - 50, y0 + ph + 8), f"frame {T}", font=FS, fill=MUTED)
        dr.text((14, size[1] - 24), f"Qwen2-VL-2B, frozen; probability read from its first answer token. Tears of Steel, (CC) Blender Foundation. Shot at {tag} s, frame {t + 1} of {T}.", font=FN, fill=MUTED)
        f = np.asarray(im).tobytes()
        for _ in range(HOLD_S * FPS if t == T - 1 else 1):
            ff.stdin.write(f)
ff.stdin.close(); ff.wait(); print("wrote", dst)
