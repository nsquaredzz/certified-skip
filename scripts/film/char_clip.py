import sys, subprocess
import numpy as np
sys.path.insert(0, "python"); sys.path.insert(0, "scripts")
from render_findings import BG, INK, font
from PIL import Image, ImageDraw
SRC, START, W, H, Z, FPS, HEAD = "data/tears_of_steel_720p.mov", 356.4, 640, 268, 1.5, 24, 70
z = np.load(sys.argv[1]); P, lab = z["P"], z["lab"]; T = len(P)
p = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-ss", str(START), "-i", SRC, "-frames:v", str(T), "-vf", f"scale={W}:{H}:flags=area", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, check=True)
col = np.frombuffer(p.stdout, np.uint8).reshape(-1, H, W, 3)
cols = {1: np.array((235, 70, 70), np.float32), 2: np.array((70, 210, 100), np.float32)}
ZW, ZH = int(W * Z), int(H * Z); FB, FT, FN = font(26, True), font(18), font(14)
ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{ZW}x{HEAD + 2 * ZH + 4}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", sys.argv[2]], stdin=subprocess.PIPE)
frames = []
for t in range(T):
    grey = col[t].mean(2, keepdims=True).repeat(3, 2).astype(np.float32) * 0.45; mask = np.zeros((H, W), np.uint8)
    for k in cols:                                       # every tracked point carries its label to where it is now
        q = P[t][lab == k]; y = np.clip(q[:, 0].round().astype(int), 2, H - 3); x = np.clip(q[:, 1].round().astype(int), 2, W - 3)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                mask[y + dy, x + dx] = k
    pic = grey.copy()
    for k, c in cols.items():
        m = mask == k; pic[m] = 0.5 * col[t][m] + 0.5 * c
    im = Image.new("RGB", (ZW, HEAD + 2 * ZH + 4), BG); d = ImageDraw.Draw(im, "RGBA")
    d.text((14, 6), "Two people, no training: motion and sharpness", font=FB, fill=INK)
    d.text((14, 42), "Red: the woman, whole. Green: the man's torso plus her robot hand; his face and hair are missed.", font=FT, fill=INK)
    im.paste(Image.fromarray(col[t]).resize((ZW, ZH), Image.BILINEAR), (0, HEAD)); im.paste(Image.fromarray(pic.astype(np.uint8)).resize((ZW, ZH), Image.BILINEAR), (0, HEAD + ZH + 4))
    note = f"Tears of Steel, (CC) Blender Foundation, mango.blender.org; {START:.0f} s, frame {t + 1} of {T}"
    d.text((ZW - 12 - d.textlength(note, font=FN), HEAD + 2 * ZH - 18), note, font=FN, fill=(255, 255, 255, 220))
    frames.append(np.asarray(im).tobytes())
for _ in range(3):
    for f in frames:
        ff.stdin.write(f)
ff.stdin.close(); ff.wait()
