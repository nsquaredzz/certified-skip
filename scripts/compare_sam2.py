#!/usr/bin/env python3
"""Side by side with a learned tracker: the moving-camera rule against SAM 2 on labelled clips.

The two do different jobs and are given different things, and the output says so:

  ours    no prompt, nothing learned.  It marks the patches whose content changed or moved
          (sent because the certificate failed, or followed).  Frontier and anchor patches are
          not counted: they are sent because the camera moved, not because anything was found.
  SAM 2   unmodified, given the hand-labelled mask of every object on the first frame (the
          semi-supervised protocol of DAVIS) and asked to follow them.

Reported per clip: the region similarity J of SAM 2 (IoU with the labels, per object, the first
frame excluded), and for both methods the IoU, precision and recall of "this 16 x 16 patch is on a
labelled object" (a patch counts as object when at least half of it is).

    python3 -m venv .venv && .venv/bin/pip install torch torchvision sam2 pillow numpy
    .venv/bin/python scripts/compare_sam2.py --ckpt data/sam2/sam2.1_hiera_base_plus.pt \\
        --clips mallard-water,snowboard,car-shadow,train,planes-water,drift-chicane --out out/sam2_compare
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse, json, subprocess, sys, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from certskip import moving as mv
from certskip.core import patch_grid
from render_moving import outline, GREEN, YELLOW, BLUE, RED, WHITE, SCHED, P

BAR = 32
PALETTE = [(255, 90, 90), (90, 200, 255), (255, 200, 60), (160, 255, 120), (230, 120, 255), (255, 150, 60), (120, 160, 255), (200, 200, 200)]


def iou(a, b):
    u = (a | b).sum()
    return 1.0 if u == 0 else float((a & b).sum() / u)


def patch_scores(pred, gt):
    """IoU, precision and recall of a patch set against the labelled-object patches, summed over frames."""
    tp = sum(int((p & g).sum()) for p, g in zip(pred, gt)); fp = sum(int((p & ~g).sum()) for p, g in zip(pred, gt))
    fn = sum(int((~p & g).sum()) for p, g in zip(pred, gt))
    return {"iou": tp / max(1, tp + fp + fn), "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn)}


def edge(g):
    return g & ~(np.roll(g, 1, 0) & np.roll(g, -1, 0) & np.roll(g, 1, 1) & np.roll(g, -1, 1))


def caption(width, left, right):
    im = Image.new("RGB", (width, BAR), (18, 18, 18)); d = ImageDraw.Draw(im); font = ImageFont.load_default(size=15)
    d.text((10, 7), left, fill=(255, 255, 255), font=font)
    d.text((width // 2 + 14, 7), right, fill=(255, 255, 255), font=font)
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/davis/DAVIS"); ap.add_argument("--ckpt", default="data/sam2/sam2.1_hiera_base_plus.pt")
    ap.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    ap.add_argument("--clips", default="mallard-water,snowboard,car-shadow,train,planes-water,drift-chicane")
    ap.add_argument("--device", default=None, help="mps, cuda or cpu (default: the best available)")
    ap.add_argument("--search", type=int, default=16); ap.add_argument("--out", default="out/sam2_compare")
    ap.add_argument("--no-video", action="store_true"); ap.add_argument("--out-fps", type=float, default=12.0)
    a = ap.parse_args()
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    dev = a.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    predictor = build_sam2_video_predictor(a.config, a.ckpt, device=dev)
    params = sum(p.numel() for p in predictor.parameters())
    head = f"SAM 2.1 ({Path(a.ckpt).name}, {params / 1e6:.0f} M parameters, {dev}) against the moving-camera rule (numpy on the CPU)"
    print(head)
    results = {}; lines = [head]
    for name in a.clips.split(","):
        jd, ad = Path(a.root) / "JPEGImages" / "480p" / name, Path(a.root) / "Annotations" / "480p" / name
        files = sorted(jd.glob("*.jpg"))
        rgb = [np.array(Image.open(f).convert("RGB")) for f in files]
        gt = [np.array(Image.open(ad / (f.stem + ".png"))) for f in files]
        ids = [int(k) for k in np.unique(gt[0]) if k > 0]
        T = len(files); H, W = gt[0].shape[0] // P * P, gt[0].shape[1] // P * P
        # ---------------- SAM 2: the first-frame masks, then propagation
        t0 = time.perf_counter()
        with torch.inference_mode():
            state = predictor.init_state(video_path=str(jd))
            for k in ids:
                predictor.add_new_mask(inference_state=state, frame_idx=0, obj_id=k, mask=gt[0] == k)
            sam = [None] * T
            for fi, obj_ids, logits in predictor.propagate_in_video(state):
                m = (logits > 0.0).cpu().numpy()[:, 0]
                lab = np.zeros(gt[0].shape, np.uint8)
                for k, mk in zip(obj_ids, m):
                    lab[mk] = k
                sam[fi] = lab
            predictor.reset_state(state)
        t_sam = (time.perf_counter() - t0) / T
        J = float(np.mean([iou(sam[t] == k, gt[t] == k) for t in range(1, T) for k in ids]))
        # ---------------- ours
        pr = mv.MovingPruner(H, W, P, SCHED, search=a.search)
        t0 = time.perf_counter(); ours = []; recs = []
        for t in range(T):
            f = np.ascontiguousarray(np.array(Image.fromarray(rgb[t]).convert("L"))[:H, :W])
            keep = pr.step(f); L = pr.last
            ours.append(((keep & ~L["reveal"]) | L["followed"]) if not L["anchor"] else np.zeros_like(keep))
            recs.append((keep.copy(), L["reveal"].copy(), L["followed"].copy(), bool(L["anchor"])))
        t_ours = (time.perf_counter() - t0) / T
        gobj = [patch_grid(g[:H, :W] > 0, P).mean((2, 3)) >= 0.5 for g in gt]
        sobj = [patch_grid(s[:H, :W] > 0, P).mean((2, 3)) >= 0.5 for s in sam]
        res = {"frames": T, "objects": len(ids), "sam2_J": J, "sam2_patch": patch_scores(sobj[1:], gobj[1:]),
               "ours_patch": patch_scores(ours[1:], gobj[1:]), "sam2_s_per_frame": t_sam, "ours_s_per_frame": t_ours,
               "ours_sent": float(np.mean([r[0].mean() for r in recs[1:]]))}
        results[name] = res
        line = (f"  {name:16s} {T:3d} frames, {len(ids)} object(s) | SAM 2: J {J:.3f}, patch IoU {res['sam2_patch']['iou']:.3f}, {t_sam:.2f} s/frame"
                f" | ours: patch IoU {res['ours_patch']['iou']:.3f} (precision {res['ours_patch']['precision']:.2f}, recall {res['ours_patch']['recall']:.2f}),"
                f" {t_ours:.2f} s/frame, {100 * res['ours_sent']:.0f} % of patches sent")
        print(line, flush=True); lines.append(line)
        # ---------------- video
        if a.no_video:
            continue
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * W + 8}x{H + BAR}", "-r", str(a.out_fps),
               "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "22", "-pix_fmt", "yuv420p", f"{a.out}_{name}.mp4"]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for t in range(T):
            keep, reveal, fol, anchor = recs[t]
            g = gt[t][:H, :W] > 0
            left = np.repeat(np.array(Image.fromarray(rgb[t]).convert("L"))[:H, :W, None], 3, 2)
            if fol.any():
                tint = np.kron(fol, np.ones((P, P), bool)); left[tint] = (0.6 * left[tint] + 0.4 * np.array(BLUE)).astype(np.uint8)
                outline(left, fol, BLUE)
            outline(left, keep & reveal, YELLOW); outline(left, keep & ~reveal, GREEN)
            if anchor and t:
                left[:4] = RED; left[-4:] = RED; left[:, :4] = RED; left[:, -4:] = RED
            right = rgb[t][:H, :W].copy()
            for i, k in enumerate(ids):
                mk = sam[t][:H, :W] == k
                right[mk] = (0.45 * right[mk] + 0.55 * np.array(PALETTE[i % len(PALETTE)])).astype(np.uint8)
            left[edge(g)] = WHITE; right[edge(g)] = WHITE
            frame = np.concatenate([left, np.zeros((H, 8, 3), np.uint8), right], 1)
            cap = caption(frame.shape[1],
                          f"OURS, no prompt: green sent, blue followed.   {name}: patches on the object, IoU {100 * res['ours_patch']['iou']:.0f} %",
                          f"SAM 2, given the first-frame mask.   IoU with the labels (J) {100 * J:.0f} %    white = hand label    frame {t}")
            proc.stdin.write(np.concatenate([cap, frame], 0).tobytes())
        proc.stdin.close(); proc.wait()
    n = len(results)
    mean = lambda f: float(np.mean([f(r) for r in results.values()]))
    summ = {"clips": n, "sam2_J": mean(lambda r: r["sam2_J"]), "sam2_patch_iou": mean(lambda r: r["sam2_patch"]["iou"]),
            "ours_patch_iou": mean(lambda r: r["ours_patch"]["iou"]), "ours_patch_precision": mean(lambda r: r["ours_patch"]["precision"]),
            "ours_patch_recall": mean(lambda r: r["ours_patch"]["recall"]), "sam2_s_per_frame": mean(lambda r: r["sam2_s_per_frame"]),
            "ours_s_per_frame": mean(lambda r: r["ours_s_per_frame"]), "parameters": params, "device": dev}
    line = (f"mean over {n} clips | SAM 2: J {summ['sam2_J']:.3f}, patch IoU {summ['sam2_patch_iou']:.3f}, {summ['sam2_s_per_frame']:.2f} s/frame"
            f" | ours: patch IoU {summ['ours_patch_iou']:.3f} (precision {summ['ours_patch_precision']:.2f}, recall {summ['ours_patch_recall']:.2f}), {summ['ours_s_per_frame']:.2f} s/frame")
    print(line); lines.append(line)
    Path(a.out + ".json").write_text(json.dumps({"summary": summ, "clips": results}, indent=1))
    Path(a.out + ".txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
