import sys, subprocess
sys.path.insert(0, sys.argv[1] + "/keep"); sys.path.insert(0, "scripts")
import numpy as np
from focus_lib import box
from render_findings import BG, INK, font
from PIL import Image, ImageDraw
z = np.load(sys.argv[1] + "/sam356.npz"); ref, col = z["masks"], z["col"]; c = np.load(sys.argv[1] + "/carry356.npz"); P, lab, R = c["P"], c["lab"], c["R"]
T, _, H, W = ref.shape; Z, HEAD, FPS = 1.5, 96, 24; ZW, ZH = int(W * Z), int(H * Z); FB, FT = font(26, True), font(18)
ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{ZW}x{HEAD + ZH}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", sys.argv[2]], stdin=subprocess.PIPE)
frames = []
for t in range(T):
    y = np.clip(P[t, :, 0].round().astype(int), 0, H - 1); x = np.clip(P[t, :, 1].round().astype(int), 0, W - 1); cnt = np.zeros((3, H, W), np.float32)
    np.add.at(cnt, (lab, y, x), 1.0); cnt = box(cnt, 2); m = cnt.argmax(0) * (cnt.sum(0) > 0); pic = col[t].astype(np.float32) * 0.6
    for k, colr in ((1, (70, 210, 100)), (2, (235, 70, 70))):
        pic[m == k] = 0.5 * col[t][m == k] + 0.5 * np.array(colr, np.float32)
        s = ref[t, k - 1]; pic[s & ~(np.roll(s, 1, 0) & np.roll(s, -1, 0) & np.roll(s, 1, 1) & np.roll(s, -1, 1))] = 255
    im = Image.new("RGB", (ZW, HEAD + ZH), BG); d = ImageDraw.Draw(im)
    d.text((14, 6), "The model looks once; the paths carry the masks", font=FB, fill=INK)
    d.text((14, 42), "Colour: masks from frame 1, carried along tracked points. White line: SAM 2 run on this frame.", font=FT, fill=INK)
    d.text((14, 66), f"Frame {t + 1} of {T}. Agreement with SAM 2: man {100 * R[t, 0]:.0f} %, woman {100 * R[t, 1]:.0f} %", font=FT, fill=INK)
    im.paste(Image.fromarray(pic.astype(np.uint8)).resize((ZW, ZH), Image.BILINEAR), (0, HEAD)); frames.append(np.asarray(im).tobytes())
for _ in range(3):
    for f in frames:
        ff.stdin.write(f)
ff.stdin.close(); ff.wait()
