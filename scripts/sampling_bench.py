#!/usr/bin/env python3
"""Kill test for predictive look scheduling.

For each clip (native frame rate): run the certified per-patch rule under
  * uniform sampling with horizon h in {1, 2, 5, 10, 15, 30}
  * the adaptive sampler with alpha in {0.05, 0.1, 0.2, 0.3}
and report, per policy: looks per second, token refreshes per second, the
realised miss rate (should match alpha: the theorem), the mean/95th-percentile
staleness of real motion (frames since the last look, over oracle-motion
patch-frames), and the onset delay of real events (first look after an oracle
onset). Oracle motion: temporal-median background, offline, generous.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import certskip as cs
from certskip.core import patch_grid
from certskip.native import NativeQuotientPruner, NativeSequentialPruner

SCHED = {r: 80.0 * (2 * r + 1) ** -0.5 for r in (0, 1, 2, 3)}


def make_pruner(H, W, delta, rule):
    if rule == "range":
        return NativeQuotientPruner(H, W, P, delta, 1.0)
    return NativeSequentialPruner(H, W, P, multiscale=SCHED, z=1e9)      # quotient + multi-scale certificate, no sequential firing
from certskip.sampler import AdaptiveSampler, UniformSampler

CLIPS = {
    "corridor": dict(path="data/caviar_WalkByShop1cor.mpg", fps=25, scale=None, max_frames=2360),
    "lobby":    dict(path="data/caviar_LeftBag.mpg", fps=25, scale=None, max_frames=1446),
    "virat":    dict(path="data/VIRAT_S_000200_00.mp4", fps=30, scale=(640, 352), max_frames=2114),
}
P = 16


def oracle_motion(frames, thr=20, min_px=4):
    fr = frames.astype(np.int16); bg = np.median(fr[::5], axis=0)
    diff = np.abs(fr - bg) > thr
    maj = np.zeros_like(diff); maj[1:-1] = (diff[:-2].astype(int) + diff[1:-1] + diff[2:]) >= 2
    return np.stack([patch_grid(m, P).sum(axis=(2, 3)) >= min_px for m in maj])      # (T, gh, gw)


def onsets(motion, quiet=25):
    """(t, gy, gx) where a patch turns on after >= `quiet` frames off."""
    T = motion.shape[0]; out = []
    on = motion.astype(np.int8)
    for gy in range(motion.shape[1]):
        for gx in range(motion.shape[2]):
            s = on[:, gy, gx]; last_on = -10**9
            for t in range(T):
                if s[t]:
                    if t - last_on > quiet:
                        out.append((t, gy, gx))
                    last_on = t
    return out


def evaluate(look_t, keeps, motion, onset_list, fps, T):
    look_t = np.array(sorted(look_t))
    # staleness of real motion: for every oracle-motion patch-frame, frames since the last look
    last_look = np.full(T, -1); j = -1
    for t in range(T):
        while j + 1 < len(look_t) and look_t[j + 1] <= t:
            j += 1
        last_look[t] = look_t[j] if j >= 0 else 0
    stale_t = np.arange(T) - last_look                                   # per frame
    mot_frames = np.where(motion.any(axis=(1, 2)))[0]
    stale = stale_t[mot_frames] if len(mot_frames) else np.array([0])
    # onset delay: frames from an oracle onset to the first look at or after it
    delays = []
    for (t, gy, gx) in onset_list:
        nxt = look_t[np.searchsorted(look_t, t)] if np.searchsorted(look_t, t) < len(look_t) else None
        if nxt is not None:
            delays.append(nxt - t)
    refreshes = sum(int(k.sum()) for t, k in keeps.items() if t > 0)
    return {"looks_per_s": len(look_t) / (T / fps), "refresh_per_s": refreshes / (T / fps),
            "stale_mean": float(stale.mean()), "stale_p95": float(np.quantile(stale, 0.95)),
            "onset_delay_mean": float(np.mean(delays)) if delays else float("nan"),
            "onset_delay_p95": float(np.quantile(delays, 0.95)) if delays else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="corridor,lobby,virat"); ap.add_argument("--delta", type=float, default=32.0)
    ap.add_argument("--H", type=int, default=30); ap.add_argument("--gamma", type=float, default=0.3); ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--out", default="out/sampling_bench"); ap.add_argument("--rule", default="ms", choices=["range", "ms"])
    ap.add_argument("--decay", type=float, default=0.5, help="step gamma/n^decay (0 = constant)")
    a = ap.parse_args()
    results = {}
    for name in a.clips.split(","):
        spec = CLIPS[name]
        fr = cs.video.read_gray(spec["path"], fps=spec["fps"], scale=spec["scale"], max_frames=spec["max_frames"])
        T, H, W = fr.shape; fps = spec["fps"]
        motion = oracle_motion(fr); ons = onsets(motion, quiet=fps)
        print(f"\n=== {name}: {T} frames @ {fps} fps, oracle motion in {100*motion.mean():.1f}% of patch-frames, {len(ons)} onsets")
        print(f"{'policy':22s} {'looks/s':>8s} {'refresh/s':>10s} {'miss rate':>10s} {'stale mean':>11s} {'stale p95':>10s} {'onset mean':>11s} {'onset p95':>10s}")
        results[name] = {}
        for h in (1, 2, 5, 10, 15, 30):
            pr = make_pruner(H, W, a.delta, a.rule); s = UniformSampler(pr, h); keeps = s.run(fr)
            r = evaluate(s.log["look_t"], keeps, motion, ons, fps, T)
            on = np.array(s.log["onset"]); r["miss_rate"] = float(on[1:].mean()) if h >= 2 and len(on) > 1 else 0.0
            results[name][f"uniform h={h}"] = r
            print(f"{'uniform h=%d' % h:22s} {r['looks_per_s']:8.2f} {r['refresh_per_s']:10.1f} {r['miss_rate']:10.3f} {r['stale_mean']:11.2f} {r['stale_p95']:10.1f} {r['onset_delay_mean']:11.2f} {r['onset_delay_p95']:10.1f}")
        uni = {k: v for k, v in results[name].items() if k.startswith("uniform")}
        def matched(looks):
            # uniform policy at the same cost, by log-linear interpolation of its curve
            xs = np.array([v["looks_per_s"] for v in uni.values()]); order = np.argsort(xs); xs = xs[order]
            out = {}
            for key in ("miss_rate", "stale_mean", "onset_delay_mean"):
                ys = np.array([v[key] for v in uni.values()])[order]
                out[key] = float(np.interp(np.log(looks), np.log(xs), ys))
            return out
        for alpha in (0.02, 0.05, 0.1, 0.2):
            pr = make_pruner(H, W, a.delta, a.rule); s = AdaptiveSampler(pr, alpha=alpha, gamma=a.gamma, H=a.H, m_refresh=a.m, decay=a.decay); keeps = s.run(fr)
            r = evaluate(s.log["look_t"], keeps, motion, ons, fps, T)
            err = np.array(s.log["err"]); r["miss_rate"] = float(err.mean()) if len(err) else 0.0
            r["bound"] = s.bound(); r["n_decisions"] = int(len(err))
            r["err_curve"] = np.cumsum(err).tolist(); r["h_mean"] = float(np.mean(s.log["h"]))
            results[name][f"adaptive a={alpha}"] = r
            ok = r['miss_rate'] <= alpha + r['bound']; mt = matched(r['looks_per_s']); r["matched_uniform"] = mt
            print(f"{'adaptive a=%.2f' % alpha:22s} {r['looks_per_s']:8.2f} {r['refresh_per_s']:10.1f} {r['miss_rate']:10.3f} {r['stale_mean']:11.2f} {r['stale_p95']:10.1f} {r['onset_delay_mean']:11.2f} {r['onset_delay_p95']:10.1f}   guarantee {'holds' if ok else 'VIOLATED'} (slack {r['bound']:.3f})  | uniform at same cost: miss {mt['miss_rate']:.3f} stale {mt['stale_mean']:.2f} onset {mt['onset_delay_mean']:.2f}  | h by state {s.rate_summary()}")
    Path(a.out + ".json").write_text(json.dumps(results, indent=1))
    # figure: realised cumulative miss rate vs alpha (theorem), and staleness vs looks/s (efficiency)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    clips = list(results); fig, axes = plt.subplots(2, len(clips), figsize=(5 * len(clips), 8), squeeze=False)
    for j, c in enumerate(clips):
        ax = axes[0, j]
        for key, r in results[c].items():
            if key.startswith("adaptive"):
                cum = np.array(r["err_curve"]); n = np.arange(1, len(cum) + 1)
                alpha = float(key.split("=")[1]); line, = ax.plot(n, cum / n, label=f"α={alpha}"); ax.axhline(alpha, color=line.get_color(), ls=":", lw=1)
        ax.set_title(f"{c}: realised miss rate → α"); ax.set_xlabel("look decisions"); ax.set_ylabel("cumulative miss rate"); ax.set_ylim(0, 0.6); ax.legend(fontsize=8); ax.grid(alpha=.3)
        ax = axes[1, j]
        for key, r in results[c].items():
            mk = "o" if key.startswith("uniform") else "D"; col = "C0" if key.startswith("uniform") else "C3"
            ax.plot(r["looks_per_s"], r["stale_mean"], mk, color=col, ms=6); ax.annotate(key.replace("uniform ", "u ").replace("adaptive ", "a "), (r["looks_per_s"], r["stale_mean"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xscale("log"); ax.set_xlabel("looks per second"); ax.set_ylabel("mean staleness of real motion (frames)"); ax.set_title(f"{c}: cost vs staleness"); ax.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(a.out + ".png", dpi=120); print(f"\nwrote {a.out}.json / .png")


if __name__ == "__main__":
    main()
