"""Video proof for looks_test.py: one look only against looks when the break signal passes a threshold."""
import sys, subprocess
sys.path.insert(0, sys.argv[1] + "/keep"); sys.path.insert(0, "scripts")
import numpy as np
from looks_lib import *
from render_findings import BG, INK, MUTED, RED, GREEN, font
from PIL import Image, ImageDraw
tag, theta, dst = sys.argv[2], float(sys.argv[3]), sys.argv[4]
z = np.load(f"{sys.argv[1]}/sam{tag}.npz"); ref, col = z["masks"], z["col"]; T, K, H, W = ref.shape; flows = np.load(f"{sys.argv[1]}/flows{tag}.npy")
grey = col.astype(np.float32).mean(3); smooth = np.stack([gauss(g, 1) for g in grey])
runs = [("ONE LOOK, THEN CARRY",) + run(grey, smooth, flows, ref), (f"LOOK AGAIN WHEN THE SIGNAL PASSES {100 * theta:.0f} %",) + run(grey, smooth, flows, ref, theta=theta)]
agr = [agreement(r[1], ref) for r in runs]
COLS = [(235, 70, 70), (70, 130, 240), (240, 200, 40), (70, 210, 100)]; Z, GAP, HEAD, FOOT, FPS, SLOW = 1.2, 6, 78, 86, 24, 2
ZW, ZH = int(W * Z) // 2 * 2, int(H * Z) // 2 * 2; FB, FT, FG, FN = font(26, True), font(18), font(18, True), font(14)
size = (2 * ZW + GAP, HEAD + ZH + FOOT)
ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst], stdin=subprocess.PIPE)
for t in range(T):
    im = Image.new("RGB", size, BG); d = ImageDraw.Draw(im, "RGBA")
    d.text((14, 6), "Three people, a panning camera, one walks in front of the others", font=FB, fill=INK)
    d.text((14, 44), "Colour: masks carried from the last look. White line: SAM 2 run on this frame. Played at half speed.", font=FT, fill=INK)
    for j, (name, out, looked, sig) in enumerate(runs):
        x0 = j * (ZW + GAP); pic = col[t].astype(np.float32) * 0.6
        for k in range(K):
            m = out[t] == k + 1; pic[m] = 0.5 * col[t][m] + 0.5 * np.array(COLS[k], np.float32)
            s = ref[t, k]; pic[s & ~(np.roll(s, 1, 0) & np.roll(s, -1, 0) & np.roll(s, 1, 1) & np.roll(s, -1, 1))] = 255
        im.paste(Image.fromarray(pic.astype(np.uint8)).resize((ZW, ZH), Image.BILINEAR), (x0, HEAD))
        d.rectangle((x0 + 8, HEAD + 8, x0 + 24 + d.textlength(name, font=FG), HEAD + 38), fill=(0, 0, 0, 200)); d.text((x0 + 16, HEAD + 11), name, font=FG, fill=INK)
        if t in looked:
            d.rectangle((x0 + 3, HEAD + 3, x0 + ZW - 4, HEAD + ZH - 4), outline=(255, 255, 255, 255), width=6)
        since = t - max(q for q in looked if q <= t)
        if since < 4:
            d.rectangle((x0 + ZW - 120, HEAD + 8, x0 + ZW - 8, HEAD + 40), fill=(255, 255, 255, 235)); d.text((x0 + ZW - 106, HEAD + 11), "LOOK", font=FB.font_variant(size=22), fill=(0, 0, 0))
        a = agr[j][t]; n = sum(q <= t for q in looked); y = HEAD + ZH + 8
        d.text((x0 + 14, y), f"agreement with SAM 2: {100 * a:.0f} %", font=FG, fill=GREEN if a >= 0.9 else RED)
        tail = f"looks so far: {n} of {t + 1} frames"; d.text((x0 + ZW - 14 - d.textlength(tail, font=FT), y), tail, font=FT, fill=INK)
        bw = ZW - 150; d.text((x0 + 14, y + 30), "break signal", font=FN, fill=MUTED); bx = x0 + 110
        d.rectangle((bx, y + 32, bx + bw, y + 44), fill=(60, 60, 58)); d.rectangle((bx, y + 32, bx + int(bw * float(sig[t].max())), y + 44), fill=(240, 200, 40))
        if j == 1:
            d.line((bx + int(bw * theta), y + 28, bx + int(bw * theta), y + 48), fill=(255, 255, 255), width=2)
        d.text((x0 + 14, y + 54), f"so far: mean {100 * agr[j][:t + 1].mean():.0f} %, lowest {100 * agr[j][:t + 1].min():.0f} %", font=FN, fill=MUTED)
    f = np.asarray(im).tobytes()
    for _ in range(SLOW * (FPS * 3 if t == T - 1 else 1)):
        ff.stdin.write(f)
ff.stdin.close(); ff.wait()
for (name, out, looked, sig), a in zip(runs, agr):
    print(f"{name}: {len(looked)} looks, mean {100 * a.mean():.1f} %, lowest {100 * a.min():.1f} %")
