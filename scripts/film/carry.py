"""Carry test: SAM 2's masks on frame 0, carried through the shot along tracked paths, against SAM 2 on every frame."""
import sys, time
sys.path.insert(0, sys.argv[1] + "/keep"); sys.path.insert(0, "scripts")
import numpy as np
from motion_lib import trajectories
from focus_lib import box
from PIL import Image
z = np.load(sys.argv[1] + "/sam356.npz"); ref, col = z["masks"], z["col"]; T, _, H, W = ref.shape
grey = col.astype(np.float32).mean(3)
t0 = time.time(); P, gs, conf = trajectories(grey, step=2); dt = time.time() - t0
print(f"paths: {P.shape[1]} points through {T} frames in {dt:.1f} s, {1000 * dt / T:.0f} ms a frame (CPU)")
y0 = P[0, :, 0].astype(int); x0 = P[0, :, 1].astype(int); lab = np.zeros(P.shape[1], int); lab[ref[0, 0][y0, x0]] = 1; lab[ref[0, 1][y0, x0]] = 2

def carried(t):
    y = np.clip(P[t, :, 0].round().astype(int), 0, H - 1); x = np.clip(P[t, :, 1].round().astype(int), 0, W - 1); cnt = np.zeros((3, H, W), np.float32)
    np.add.at(cnt, (lab, y, x), 1.0); cnt = box(cnt, 2); return cnt.argmax(0) * (cnt.sum(0) > 0)

iou = lambda a, b: (a & b).sum() / max((a | b).sum(), 1)
rows = []; keep = {}
for t in range(T):
    c = carried(t); rows.append([iou(c == k + 1, ref[t, k]) for k in (0, 1)] + [iou(ref[0, k], ref[t, k]) for k in (0, 1)])
    if t in (0, T // 2, T - 1):
        keep[t] = c
R = np.array(rows); names = ("man", "woman")
print("agreement with SAM 2 on that frame (IoU), carried | frame-0 mask left where it was")
for t in (0, 24, 48, 72, 96, 120, T - 1):
    print(f"  frame {t:>3} ({t / 24:.1f} s): " + "   ".join(f"{names[k]} {R[t, k]:.2f} | {R[t, 2 + k]:.2f}" for k in (0, 1)))
for k in (0, 1):
    below = np.nonzero(R[:, k] < 0.9)[0]; print(f"  {names[k]}: mean {R[:, k].mean():.3f}, lowest {R[:, k].min():.3f}, first frame under 0.90: {below[0] if len(below) else 'never'}; held still: mean {R[:, 2 + k].mean():.3f}, lowest {R[:, 2 + k].min():.3f}")
np.savez(sys.argv[1] + "/carry356.npz", R=R, P=P, lab=lab)
tiles = []
for t, c in keep.items():
    pic = col[t].astype(np.float32) * 0.6
    for k, colr in ((1, (70, 210, 100)), (2, (235, 70, 70))):
        pic[c == k] = 0.5 * col[t][c == k] + 0.5 * np.array(colr, np.float32)
        m = ref[t, k - 1]; edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)); pic[edge] = 255
    tiles.append(pic.astype(np.uint8))
Image.fromarray(np.hstack(tiles)).save(sys.argv[1] + "/carry356.png")
