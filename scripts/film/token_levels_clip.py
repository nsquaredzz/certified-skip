#!/usr/bin/env python3
"""Video proof for token_levels.py: the level of every token cell through the shot, and the model's answers.

    .venv/bin/python scripts/film/token_levels_clip.py <dir> <out.mp4> 201 356
"""
import sys, json, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[1] / "python"))
import numpy as np
from render_findings import BG, INK, MUTED, RED, GREEN, font
from PIL import Image, ImageDraw
d, dst, tags = sys.argv[1], sys.argv[2], sys.argv[3:]
Z, GAP, HEAD, FOOT, FPS, HOLD_S = 1.2, 6, 78, 236, 24, 7
W, H = 640, 268; ZW, ZH = int(W * Z) // 2 * 2, int(H * Z) // 2 * 2; size = (2 * ZW + GAP, HEAD + ZH + FOOT)
FB, FT, FG, FN, FQ = font(26, True), font(18), font(18, True), font(14), font(16)
TINT = {1: np.array((70, 140, 255), np.float32), 2: np.array((255, 150, 40), np.float32)}       # moved, new
ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst], stdin=subprocess.PIPE)
for tag in tags:
    col = np.load(f"{d}/sam{tag}.npz")["col"]; z = np.load(f"{d}/levels{tag}.npz"); rows = [r for r in json.loads(str(z["rows"])) if r["full"] != "the"]; T = len(col)      # "The man ..." cut off before it said left or right
    panels = (("STILL TOKENS REUSED", z["lev_still"], "still"), ("STILL AND MOVED TOKENS REUSED", z["lev_moved"], "moved")); asked = sorted({r["frame"] for r in rows}); shown = []
    for t in range(T):
        im = Image.new("RGB", size, BG); dr = ImageDraw.Draw(im, "RGBA")
        dr.text((14, 6), "Which tokens does the model have to compute again?", font=FB, fill=INK)
        dr.text((14, 44), "Clear: kept from before.   Blue: carried along with what moved.   Orange: computed again on this frame.", font=FT, fill=INK)
        for j, (name, lev, key) in enumerate(panels):
            x0 = j * (ZW + GAP); pic = col[t].astype(np.float32); L = np.asarray(Image.fromarray(lev[t]).resize((W, H), Image.NEAREST))
            for k, c in TINT.items():
                pic[L == k] = 0.55 * pic[L == k] + 0.45 * c
            im.paste(Image.fromarray(pic.astype(np.uint8)).resize((ZW, ZH), Image.BILINEAR), (x0, HEAD))
            dr.rectangle((x0 + 8, HEAD + 8, x0 + 24 + dr.textlength(name, font=FG), HEAD + 38), fill=(0, 0, 0, 200)); dr.text((x0 + 16, HEAD + 11), name, font=FG, fill=INK)
            note = f"computed again so far: {100 * (lev[1:t + 1] == 2).mean() if t else 100:.0f} % of tokens"
            dr.rectangle((x0 + 8, HEAD + ZH - 38, x0 + 24 + dr.textlength(note, font=FG), HEAD + ZH - 8), fill=(0, 0, 0, 200)); dr.text((x0 + 16, HEAD + ZH - 35), note, font=FG, fill=INK)
        if t in asked:
            shown = [r for r in rows if r["frame"] == t]
        y = HEAD + ZH + 10; cx = (760, 930, 1100, 1270, 1420)
        dr.text((14, y), f"Questions put to the model at frame {shown[0]['frame'] + 1}" if shown else "Questions come up when the model is asked.", font=FG, fill=INK)
        for x, head in zip(cx[1:], ("all tokens new", "still reused", "still + moved", "never refreshed")):
            dr.text((x - 60, y + 2), head, font=FN, fill=MUTED)
        for i, r in enumerate(shown):
            yy = y + 30 + i * 30; dr.text((14, yy), r["q"].split(" Answer")[0], font=FQ, fill=INK)
            for x, k in zip(cx[1:], ("full", "still", "moved", "stale")):
                same = r[k] == r["full"]; dr.text((x - 60, yy), (r[k] or "?") + ("" if k == "full" else "  ✓" if same else "  ✗"), font=FG, fill=INK if k == "full" else GREEN if same else RED)
        dr.text((14, size[1] - 24), f"Qwen2-VL-2B, frozen. Tears of Steel, (CC) Blender Foundation, mango.blender.org. Shot at {tag} s, frame {t + 1} of {T}. ✓ = same answer as with all tokens new.", font=FN, fill=MUTED)
        f = np.asarray(im).tobytes()
        for _ in range(HOLD_S * FPS if t in asked else 2):
            ff.stdin.write(f)
ff.stdin.close(); ff.wait(); print("wrote", dst)
