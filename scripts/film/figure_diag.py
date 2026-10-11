import sys; sys.path.insert(0, "python"); sys.path.insert(0, sys.argv[1])
import numpy as np
from certskip import video
from motion_lib import trajectories
from figure import *
from PIL import Image
start, n, tag = float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
fr = video.read_gray("data/tears_of_steel_720p.mov", scale=(640, 268), start=start, max_frames=n)
P, gs, conf = trajectories(fr); lab, info = characters(fr, P, gs); N = len(lab)
print(tag, "figure share", round(float(info["fig"].mean()), 3), "otsu cut", round(float(info["cut"]), 2))
for w, sa, sb, ia, ib, keep in info["events"]:
    print(f"   meet at {w:.2f} px: {100 * sb / N:.1f} % and {100 * sa / N:.1f} %, inside {ib:.2f} and {ia:.2f} px -> {'kept apart' if keep else 'merged'}")
print("   characters (share of frame):", [round(float((lab == k).mean()), 3) for k in range(1, lab.max() + 1)])
cols = np.array([(0, 0, 0), (230, 60, 60), (60, 200, 90), (70, 110, 240), (240, 200, 40), (220, 80, 220), (60, 220, 220)] + [(200, 200, 200)] * 20, np.uint8)
up = lambda a: np.asarray(Image.fromarray(a).resize((640, 268), Image.NEAREST))
g3 = np.repeat(fr[0][:, :, None], 3, 2); Fm = up((np.clip((info["F"] + 3) / 15, 0, 1) * 255).astype(np.uint8).reshape(gs)); seg = up(cols[lab].reshape(*gs, 3))
over = np.where(seg.sum(2, keepdims=True) > 0, 0.45 * g3 + 0.55 * seg, 0.35 * g3).astype(np.uint8)
Image.fromarray(np.vstack([g3, np.repeat(Fm[:, :, None], 3, 2), over])).save(f"{sys.argv[1]}/figure_{tag}.png")
np.savez(f"{sys.argv[1]}/fig_{tag}.npz", P=P, gs=gs, lab=lab)
