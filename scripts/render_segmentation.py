#!/usr/bin/env python3
"""Three-panel video: frame | overlay | flat object mask (blue background, one colour per object).

    python3 scripts/render_segmentation.py in.mp4 out.mp4 [--fps F] [--scale WxH] [--start s]
        [--max-frames N] [--init 50] [--lam 1.2] [--z0 3] [--min-area 40] [--upscale 2] [--no-shadows]
"""
import argparse, subprocess, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import certskip as cs
from certskip.segment import ForegroundSegmenter, RealtimeSegmenter, render_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--fps", type=float, default=None); ap.add_argument("--scale", default=None)
    ap.add_argument("--start", type=float, default=None); ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--init", type=int, default=50); ap.add_argument("--lam", type=float, default=1.2)
    ap.add_argument("--z0", type=float, default=3.0); ap.add_argument("--gamma", type=float, default=0.6)
    ap.add_argument("--min-area", type=int, default=40); ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--upscale", type=int, default=1); ap.add_argument("--crf", type=int, default=26)
    ap.add_argument("--no-shadows", action="store_true"); ap.add_argument("--certify", action="store_true")
    ap.add_argument("--dump-frames", default=None, help="comma-separated frame indices to save as PNG (out stem)")
    ap.add_argument("--rt", action="store_true", help="real-time tracker (prediction-correction) instead of per-frame solve")
    ap.add_argument("--steps", type=int, default=3); ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--compare-full", action="store_true", help="also solve each frame fully and report IoU (slow)")
    ap.add_argument("--native", action="store_true", help="C++ real-time segmenter")
    ap.add_argument("--close", type=int, default=2, help="morphological closing radius")
    ap.add_argument("--rho", type=float, default=0.6, help="Lagrangian evidence accumulation factor (0 = off)")
    ap.add_argument("--snap", type=int, default=3, help="guided-filter boundary snap radius (0 = off)")
    ap.add_argument("--snap-eps", type=float, default=36.0, help="guided-filter regularisation (lower = follows image edges more)")
    a = ap.parse_args()
    scale = tuple(int(x) for x in a.scale.lower().split("x")) if a.scale else None
    frames = cs.video.read_gray(a.inp, fps=a.fps, scale=scale, start=a.start, max_frames=a.max_frames)
    T, H, W = frames.shape
    out_fps = a.fps or 24
    sched = {r: 80.0 * (2 * r + 1) ** -0.5 for r in (1, 2, 3)} if a.certify else None
    kw = dict(init_frames=frames[: min(a.init, T)], z0=a.z0, gamma=a.gamma, lam=a.lam, iters=a.iters,
              min_area=a.min_area, shadows=not a.no_shadows, certify=sched, close_r=a.close, rho=a.rho, snap_r=a.snap, snap_eps=a.snap_eps)
    if a.native:
        from certskip.native import NativeRtSegmenter
        seg = NativeRtSegmenter(H, W, 16, steps=a.steps, mu=a.mu, **kw)
    else:
        seg = RealtimeSegmenter(H, W, 16, steps=a.steps, mu=a.mu, compare_full=a.compare_full, **kw) if a.rt else ForegroundSegmenter(H, W, 16, **kw)
    import time
    t_seg = 0.0
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{3*W*a.upscale}x{(H+24)*a.upscale}",
           "-r", str(out_fps), "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", str(a.crf), "-pix_fmt", "yuv420p", a.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    dump = set(int(x) for x in a.dump_frames.split(",")) if a.dump_frames else set()
    from PIL import Image, ImageDraw
    for t in range(T):
        t0 = time.perf_counter(); mask = seg.step(frames[t]); t_seg += time.perf_counter() - t0
        rgb = np.repeat(frames[t][..., None], 3, -1)
        flat = render_mask(seg.labels)
        over = rgb.copy().astype(np.int32)
        over[mask] = (0.45 * over[mask] + 0.55 * flat[mask]).astype(np.int32)
        over = over.astype(np.uint8)
        for pid in np.unique(seg.labels[seg.labels > 0]):           # bounding boxes per tracked object
            ys, xs = np.where(seg.labels == pid)
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            col = flat[ys[0], xs[0]]
            over[y0, x0:x1 + 1] = col; over[y1, x0:x1 + 1] = col; over[y0:y1 + 1, x0] = col; over[y0:y1 + 1, x1] = col
        panel = np.concatenate([rgb, over, flat], axis=1)
        canvas = np.concatenate([np.zeros((24, 3 * W, 3), np.uint8), panel], axis=0)
        im = Image.fromarray(canvas); d = ImageDraw.Draw(im)
        d.text((6, 6), f"t={t/out_fps:5.1f}s  objects {len(seg.ids)}  jitter ({seg.motion[0]:+.2f},{seg.motion[1]:+.2f}) px", fill=(0, 255, 0))
        d.text((W + 6, 6), "overlay", fill=(255, 255, 255)); d.text((2 * W + 6, 6), "object mask (no labels)", fill=(255, 255, 255))
        canvas = np.asarray(im)
        if a.upscale > 1:
            canvas = np.repeat(np.repeat(canvas, a.upscale, 0), a.upscale, 1)
        if t in dump:
            Image.fromarray(canvas).save(f"{Path(a.out).with_suffix('')}_f{t}.png")
        proc.stdin.write(np.ascontiguousarray(canvas).tobytes())
    proc.stdin.close(); proc.wait()
    print(f"wrote {a.out}: {T} frames; segmentation {T / t_seg:.1f} fps ({1000 * t_seg / T:.1f} ms/frame) at {W}x{H}, rendering excluded")
    if a.rt or a.native:
        st = seg.stats
        print(f"  active set {100 * np.mean(st['active_frac']):.1f}% of pixels on average; KKT residual max on frozen region "
              f"{np.max(st['kkt_frozen_max']):.3f} (mean {np.mean(st['kkt_frozen_max']):.3f}), on active region mean {np.mean(st['kkt_active_max']):.3f}")
        if st["iou_vs_full"]:
            iou = np.array(st['iou_vs_full'])
            print(f"  IoU of tracked mask vs fully converged per-frame solve: mean {iou.mean():.3f}, median {np.median(iou):.3f}, frames below 0.8: {100*(iou < 0.8).mean():.1f}%")


if __name__ == "__main__":
    main()
