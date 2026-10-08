#!/usr/bin/env python3
"""Ceiling estimate: what fraction of patches contain REAL change (moving
people, etc.), measured by an offline oracle that no online rule can use:
temporal-median background, |frame - background| > thr on >= min_px pixels
of a patch, after a 3-frame temporal majority to suppress single-frame noise.
Any rule that keeps fewer patches than this is dropping real motion."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from certskip import video, core

CLIPS = {
    "hall":    dict(path="data/hall_monitor_cif.y4m", fps=30, scale=None, max_frames=300),
    "leftbag": dict(path="data/caviar_LeftBag.mpg", fps=12.5, scale=None, max_frames=800),
    "virat":   dict(path="data/VIRAT_S_000200_00.mp4", fps=10, scale=(640, 352), max_frames=300),
}
P = 16
for name, spec in CLIPS.items():
    fr = video.read_gray(spec["path"], fps=spec["fps"], scale=spec["scale"], max_frames=spec["max_frames"]).astype(np.int16)
    bg = np.median(fr, axis=0)
    out = []
    for thr, min_px in ((30, 8), (20, 8), (40, 4)):
        diff = np.abs(fr - bg) > thr
        # temporal majority over 3 frames
        maj = np.zeros_like(diff); maj[1:-1] = (diff[:-2].astype(int) + diff[1:-1] + diff[2:]) >= 2
        g = core.patch_grid(maj[0], P)  # shape probe
        cnt = np.stack([core.patch_grid(m, P).sum(axis=(2, 3)) for m in maj])
        frac = (cnt >= min_px)[1:].mean()
        out.append(f"thr {thr}/{min_px}px: {frac:.3f}")
    print(f"{name:8s} oracle real-change patch fraction: " + " | ".join(out))
