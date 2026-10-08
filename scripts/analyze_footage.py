#!/usr/bin/env python3
"""The gating experiment: does the certified rule still drop enough on REAL
fixed-camera footage, given its noise?

    python3 scripts/analyze_footage.py cam.mp4 [--fps 2] [--patch 16] \
        [--deltas 16,24,32,48,64] [--scale 1280x720] [--max-frames 600] \
        [--audit 500] [--plot out.png] [--json out.json]

Prints: estimated temporal noise, predicted spread distribution, the measured
drop rate / certified error for each delta, keyframe spikes, matched-drop-rate
baselines, and (optionally) a soundness audit on dropped patches.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import certskip as cs
from certskip import noise as nz, baselines as bl, topology as tp, video


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--fps", type=float, default=2.0, help="sampling rate fed to the model (default 2)")
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--deltas", default="16,24,32,48,64")
    ap.add_argument("--scale", default=None, help="WxH to resize to before tiling (model input size)")
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--audit", type=int, default=0, help="soundness-audit this many dropped patches at the first delta")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    scale = tuple(int(x) for x in a.scale.lower().split("x")) if a.scale else None
    deltas = [float(x) for x in a.deltas.split(",")]
    info = video.probe(a.video)
    print(f"source: {info['width']}x{info['height']} @ {info['fps']:.2f} fps, {info['codec']}, {info['duration']} s")
    t0 = time.perf_counter()
    frames = video.read_gray(a.video, fps=a.fps, scale=scale, start=a.start, max_frames=a.max_frames)
    T, H, W = frames.shape
    print(f"decoded {T} frames of {W}x{H} grey in {time.perf_counter()-t0:.1f}s; grid {H//a.patch}x{W//a.patch} patches of {a.patch}px")

    est = nz.estimate_temporal_noise(frames, a.patch)
    sig = est["sigma_pixel"]
    print(f"\ntemporal noise (static half of patches): sigma_pixel = {sig:.2f} grey levels "
          f"(diff sigma median over all patches {est['sigma_diff_median_all_patches']:.2f}, p90 {est['sigma_diff_p90_all_patches']:.2f})")
    sp = nz.simulate_spread(sig, a.patch)
    print(f"Gaussian-model spread of a static patch: mean {sp.mean():.1f}, p50 {np.quantile(sp,.5):.1f}, p90 {np.quantile(sp,.9):.1f}, p99 {np.quantile(sp,.99):.1f}")
    print(f"delta needed for 90% / 95% / 99% drop on a perfectly static scene: "
          f"{nz.delta_for_drop_rate(sig,.90,a.patch):.0f} / {nz.delta_for_drop_rate(sig,.95,a.patch):.0f} / {nz.delta_for_drop_rate(sig,.99,a.patch):.0f}")

    rows = nz.sweep_delta(frames, deltas, a.patch)
    print("\n delta | measured drop | Gaussian ceiling | certified max view error")
    for r in rows:
        r["gaussian_ceiling"] = nz.predicted_drop_rate(sig, r["delta"], a.patch, sp)
        print(f" {r['delta']:5.0f} | {r['drop_rate']:12.3f} | {r['gaussian_ceiling']:16.3f} | {r['certified_max_error']:6.1f}")

    # keyframe spikes at the middle delta
    mid = rows[len(rows) // 2]
    ks = nz.keyframe_spikes(mid["per_frame_keep_fraction"])
    print(f"\nkeep-fraction spikes at delta={mid['delta']:.0f}: {len(ks['spikes'])} frames ({100*ks['fraction_of_frames']:.1f}%), "
          f"median keep {ks.get('median_keep_fraction', 0):.3f}, spike keep {ks.get('spike_mean_keep_fraction')}, period ~{ks['period']} sampled frames")
    if ks["period"]:
        print("  (a regular period usually means codec I-frames re-quantising the picture; "
              "compare with the source GOP)")

    # baselines at the measured drop rate of the middle delta
    target = mid["drop_rate"]
    print(f"\nbaselines calibrated to the certified rule's drop rate at delta={mid['delta']:.0f} ({target:.3f}):")
    tau, _, d = bl.calibrate(bl.consecutive_mean, frames, a.patch, target)
    print(f"  consecutive-frame mean  tau={tau:.2f}  drop={d:.3f}   (no guarantee)")
    tau2, _, d2 = bl.calibrate(bl.mean_vs_kept, frames, a.patch, target, iters=18)
    print(f"  mean vs last kept       tau={tau2:.2f}  drop={d2:.3f}   (no guarantee)")

    audit = None
    if a.audit:
        pr = cs.make_pruner(H, W, a.patch, deltas[0], dtype=frames.dtype)
        out = pr.run(frames)
        audit = tp.soundness_audit(frames, out["keep"], a.patch, deltas[0], max_patches=a.audit)
        print(f"\nsoundness audit at delta={deltas[0]:.0f}: checked {audit['checked']} dropped patches, "
              f"violations {audit['violations']}, worst bottleneck/(s/2) = {audit['worst_bottleneck_over_half_spread']:.3f}")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot([r["delta"] for r in rows], [r["drop_rate"] for r in rows], "o-", label="measured")
        ax[0].plot([r["delta"] for r in rows], [r["gaussian_ceiling"] for r in rows], "s--", label=f"Gaussian ceiling (σ={sig:.1f})")
        ax[0].set_xlabel("Δ (grey levels)"); ax[0].set_ylabel("drop rate"); ax[0].set_ylim(0, 1); ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].plot(mid["per_frame_keep_fraction"]); ax[1].set_xlabel("sampled frame"); ax[1].set_ylabel(f"keep fraction, Δ={mid['delta']:.0f}"); ax[1].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(a.plot, dpi=130)
        print(f"plot -> {a.plot}")

    if a.json:
        dump = {"video": a.video, "fps": a.fps, "patch": a.patch, "frames": T, "size": [W, H],
                "sigma_pixel": sig, "rows": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()} for r in rows],
                "keyframe_spikes": {"n": int(len(ks["spikes"])), "period": ks["period"]},
                "baselines": {"consecutive_mean_tau": tau, "mean_vs_kept_tau": tau2},
                "audit": None if audit is None else {k: v for k, v in audit.items() if k != "worst_case"}}
        Path(a.json).write_text(json.dumps(dump, indent=2))
        print(f"json -> {a.json}")


if __name__ == "__main__":
    main()
