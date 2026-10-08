#!/usr/bin/env python3
"""Render a side-by-side video: [truth + kept patches in green] [model's view] [|view - truth|].

    python3 scripts/render_overlay.py in.mp4 out.mp4 [--fps 30] [--patch 16] [--delta 32]
        [--scale WxH] [--start s] [--max-frames N] [--slow 1]
"""
import argparse, subprocess, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import certskip as cs


def heat(err, vmax):
    x = np.clip(err / vmax, 0, 1)
    r = np.clip(3 * x, 0, 1); g = np.clip(3 * x - 1, 0, 1); b = np.clip(3 * x - 2, 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--fps", type=float, default=None, help="decode/sample rate (default source)")
    ap.add_argument("--out-fps", type=float, default=None, help="playback fps (default = --fps or 24)")
    ap.add_argument("--patch", type=int, default=16); ap.add_argument("--delta", type=float, default=32)
    ap.add_argument("--scale", default=None); ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--upscale", type=int, default=1)
    ap.add_argument("--crf", type=int, default=26)
    ap.add_argument("--rule", default="range", choices=["range", "quotient", "multiscale", "flat", "quotient+ms", "sequential"])
    ap.add_argument("--delta-max", type=float, default=1.0, help="quotient rule: max sub-pixel motion")
    a = ap.parse_args()
    scale = tuple(int(x) for x in a.scale.lower().split("x")) if a.scale else None
    P = a.patch
    it = cs.video.iter_gray(a.inp, fps=a.fps, scale=scale, start=a.start, max_frames=a.max_frames)
    first = next(it)
    H, W = first.shape[0] // P * P, first.shape[1] // P * P          # crop to the patch grid
    def frames_iter():
        yield first[:H, :W]
        for f in it:
            yield f[:H, :W]
    if a.rule == "range":
        pr = cs.make_pruner(H, W, P, a.delta)
    elif a.rule == "quotient":
        from certskip.warp import WarpPruner
        pr = WarpPruner(H, W, P, a.delta, a.delta_max)
    elif a.rule == "multiscale":
        from certskip.scalespace import MultiScalePruner
        pr = MultiScalePruner(H, W, P, {r: a.delta * (2 * r + 1) ** -0.5 for r in (0, 1, 2, 3)})
    elif a.rule == "sequential":
        from certskip.sequential import SequentialPruner
        pr = SequentialPruner(H, W, P, multiscale={r: 80.0 * (2 * r + 1) ** -0.5 for r in (0, 1, 2, 3)}, z=a.delta)
    elif a.rule == "quotient+ms":
        from certskip.warp import WarpPruner
        pr = WarpPruner(H, W, P, 32.0, a.delta_max, multiscale={r: a.delta * (2 * r + 1) ** -0.5 for r in (0, 1, 2, 3)})
    else:
        from certskip.flatnorm import FlatPruner
        pr = FlatPruner(H, W, P, a.delta, 4.0)
    out_fps = a.out_fps or a.fps or 24
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{3*W*a.upscale}x{(H+28)*a.upscale}",
           "-r", str(out_fps), "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", str(a.crf), "-pix_fmt", "yuv420p", a.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    tot_kept = tot = 0
    font = None
    try:
        from PIL import Image, ImageDraw
        pil = True
    except Exception:
        pil = False
    T = 0
    for t, frame in enumerate(frames_iter()):
        T += 1
        frame = np.ascontiguousarray(frame)
        k, s, c = pr.step(frame)
        view = pr.view()
        err = np.abs(view.astype(np.int32) - frame.astype(np.int32))
        left = np.repeat(frame[..., None], 3, -1).copy()
        for gy, gx in zip(*np.where(k)):
            y0, x0 = gy * P, gx * P
            left[y0, x0:x0 + P] = (0, 255, 0); left[y0 + P - 1, x0:x0 + P] = (0, 255, 0)
            left[y0:y0 + P, x0] = (0, 255, 0); left[y0:y0 + P, x0 + P - 1] = (0, 255, 0)
        mid = np.repeat(view[..., None], 3, -1)
        right = heat(err, 32)
        panel = np.concatenate([left, mid, right], axis=1)
        bar = np.zeros((28, 3 * W, 3), np.uint8)
        canvas = np.concatenate([bar, panel], axis=0)
        if t > 0:
            tot_kept += k.sum(); tot += k.size
        if pil:
            im = Image.fromarray(canvas); d = ImageDraw.Draw(im)
            bound = (0.5 * s[~k].max() if (~k).any() else 0) if a.rule in ("range", "quotient") else float("nan")
            d.text((6, 7), f"t={t/(a.fps or out_fps):5.1f}s  kept {k.sum()}/{k.size}  dropped so far {100*(1-tot_kept/max(tot,1)):.1f}%   {a.rule} thr={a.delta:.0f}", fill=(0, 255, 0))
            d.text((W + 6, 7), "model's view: last kept copy of each patch (+shift" + (" +sub-px motion)" if a.rule == "quotient" else ")"), fill=(255, 255, 255))
            d.text((2 * W + 6, 7), f"|view-truth|  max {err.max()}  bound {bound:.1f} on dropped (scale 0..32)", fill=(255, 200, 0))
            canvas = np.asarray(im)
        if a.upscale > 1:
            canvas = np.repeat(np.repeat(canvas, a.upscale, 0), a.upscale, 1)
        proc.stdin.write(np.ascontiguousarray(canvas).tobytes())
    proc.stdin.close(); proc.wait()
    print(f"wrote {a.out}: {T} frames, overall drop {1 - tot_kept / max(tot, 1):.3f}")


if __name__ == "__main__":
    main()
