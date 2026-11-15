#!/usr/bin/env python3
"""Phase 0 of the KV-cache programme: how many cache entries does a certificate need, and does it depend on frame rate?

No model is run. A cache entry is one kept block: a skipped block adds nothing to the cache and the model
keeps attending to the entry of its last kept copy (one entry per run, THEORY.md sections 11-12). So the
cache size of a rule is its number of keeps, first frame included, and the cache of any rule that is
certified at tolerance eps has at least C_eps entries (Theorem 13).

Per clip, per frame rate (the clip subsampled from its native rate, real frames only), per block size
(16 px, the size of every earlier result, and 28 px, one token of Qwen2-VL):

  certified   quotient + multi-scale without the per-patch offset (Part VIII); the rule the results recommend
  send-delta  the same certificate without the sub-pixel shift (centre.CentrePruner, lookahead 0), the rule the
              lower bound is stated for
  centre-L    the same, holding the centre of the next L frames (Theorem 16)
  bound       centre.fewest_sends: no policy certified at the same tolerance can have fewer entries (Theorem 13)
  consec      keep iff the mean |F_t - F_(t-1)| over the block >= tau (EVS / run-length tokenisation / InfiniPot-V's
              temporal criterion in pixel space); `fixed` tunes tau once at the rate nearest 2 fps,
              `matched` re-tunes it at every rate to the certified rule's entry count

and, for the rules that hold a frame, the held copy is compared with the truth (ghost pixels, as in Part IX).

The cache sizes are also given as GB per hour for a reference model: Qwen2-VL-7B, 28 layers x 4 KV heads x 128 dims
x 2 (K and V) x 2 bytes = 57,344 bytes per token, one token per 28 px block and per pair of frames, 448 x 448 frames at 2 fps (256 tokens per second).

    python3 scripts/kv_phase0.py [--clips hall,leftbag,virat] [--blocks 16,28] [--lookahead 4] [--out out/kv_phase0]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from certskip import video, baselines as bl
from certskip.centre import CentrePruner, fewest_sends
from certskip.core import patch_grid
from certskip.native import NativeSequentialPruner
from real_bench import schedule

WRONG = 30
STRIDES = (30, 15, 10, 6, 3, 2, 1)
BYTES_PER_TOKEN = 28 * 4 * 128 * 2 * 2
REF_TOKENS_PER_S = 256                       # 448x448 = 256 tokens per 2 frames at 2 fps: Qwen2-VL merges two frames into one token
CLIPS = {                                         # name -> (path, frames, scale)
    "hall":    ("data/hall_monitor_cif.y4m", 600, None),
    "leftbag": ("data/caviar_LeftBag.mpg", 600, None),
    "virat":   ("data/VIRAT_S_000200_00.mp4", 600, (960, 540)),
}


def gb_per_hour(entries_per_block_s):
    """Entries per block per second -> GB per hour, for 448x448 frames: 256 blocks (the 2-frame merge halves the full-cache rate at 2 fps)."""
    return entries_per_block_s * 256 * 3600 * BYTES_PER_TOKEN / 1e9


def ghost_of(keep, sub, moving, P):
    """Share of the static part of the frame wrong in the held copy, for a rule given by its keep masks."""
    held = sub[0].copy(); bad = tot = 0
    for t in range(1, len(sub)):
        m = np.kron(keep[t], np.ones((P, P), bool)); held[m] = sub[t][m]
        e = np.abs(held.astype(np.int16) - sub[t].astype(np.int16)); bg = ~moving[t]
        bad += int((e[bg] > WRONG).sum()); tot += int(bg.sum())
    return 100.0 * bad / max(tot, 1)


def cert_keep(sub, P, sched):
    T, H, W = sub.shape
    pr = NativeSequentialPruner(H, W, P, multiscale=sched, z=1e9, offset="none")
    return np.stack([pr.step(sub[t])[0] for t in range(T)])


def tau_for(scores, entries_frac, T):
    """tau at which the consecutive rule has the given share of entries (first frame counts as T-independent constant)."""
    target_skip = 1.0 - entries_frac
    return float(np.quantile(scores, min(max(target_skip, 0.0), 1.0)))


def frac(keep):
    """Steady state: share of block-frames that add an entry, first frame excluded (it adds the whole first image)."""
    return float(keep[1:].sum()) / keep[1:].size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="hall,leftbag,virat"); ap.add_argument("--blocks", default="16,28")
    ap.add_argument("--lookahead", type=int, default=4); ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", default="out/kv_phase0")
    a = ap.parse_args()
    sched = schedule(80.0, 0.5); rows = {}
    for name in a.clips.split(","):
        path, nmax, scale = CLIPS[name]; nmax = a.max_frames or nmax
        info = video.probe(path); fps0 = info["fps"] or 25.0
        fr = video.read_gray(path, fps=fps0, scale=scale, max_frames=nmax)
        for P in map(int, a.blocks.split(",")):
            T, H, W = fr.shape; f = fr[:, : H // P * P, : W // P * P]
            moving = np.abs(f.astype(np.int16) - np.median(f[::5], axis=0).astype(np.int16)) > WRONG
            key = f"{name}@{P}"; rows[key] = []; todo = []
            for s in STRIDES:
                sub, mov = f[::s], moving[::s]
                if len(sub) >= 12:
                    todo.append((fps0 / s, sub, mov))
            # tau for the fixed heuristic: tuned at the rate nearest 2 fps to the certified rule's entries there
            fps_fix, sub_fix, _ = min(todo, key=lambda q: abs(q[0] - 2.0))
            tau_fixed = tau_for(bl.consecutive_mean_scores(sub_fix, P), frac(cert_keep(sub_fix, P, sched)), len(sub_fix))
            print(f"\n=== {key}: {T} frames at {fps0:g} fps, {W}x{H} -> {f.shape[2]}x{f.shape[1]}, {f.shape[1] // P}x{f.shape[2] // P} blocks; fixed tau {tau_fixed:.2f} (tuned at {fps_fix:.1f} fps)")
            print(f"  {'fps':>5s} | entries, % of full cache: {'cert':>6s} {'sod':>6s} {'ctr' + str(a.lookahead):>6s} {'bound':>6s} {'cons-fx':>7s} {'cons-mt':>7s} | ratio cert/bound | ghost %: {'cert':>5s} {'cons-fx':>7s} {'cons-mt':>7s} | GB/h ref: {'cert':>5s} {'bound':>5s}")
            for fps, sub, mov in todo:
                t0 = time.time(); n = sub.shape[0]
                kc = cert_keep(sub, P, sched)
                sc = bl.consecutive_mean_scores(sub, P)
                kf = bl.consecutive_mean(sub, P, tau_fixed, scores=sc)
                km = bl.consecutive_mean(sub, P, tau_for(sc, frac(kc), n), scores=sc)
                bound = (fewest_sends(sub, P, multiscale=sched) - 1).sum() / kc[1:].size
                sod = CentrePruner(*sub.shape[1:], P, multiscale=sched, lookahead=0).run(sub); ks = sod["keep"]
                ctr = CentrePruner(*sub.shape[1:], P, multiscale=sched, lookahead=a.lookahead).run(sub)["keep"] if a.lookahead else ks
                r = dict(fps=fps, frames=n, blocks=int(kc.shape[1] * kc.shape[2]),
                         cert=frac(kc), sod=frac(ks), centre=frac(ctr), bound=float(bound), cons_fixed=frac(kf), cons_matched=frac(km),
                         ghost_cert=ghost_of(kc, sub, mov, P), ghost_fixed=ghost_of(kf, sub, mov, P), ghost_matched=ghost_of(km, sub, mov, P),
                         sod_worst=float(sod["worst"]), sec=time.time() - t0)
                r["per_s"] = {k: r[k] * fps for k in ("cert", "sod", "centre", "bound", "cons_fixed", "cons_matched")}; r["gb_cert"] = gb_per_hour(r["per_s"]["cert"]); r["gb_bound"] = gb_per_hour(r["per_s"]["bound"]); r["gb_full"] = gb_per_hour(1.0)
                rows[key].append(r)
                print(f"  {fps:5.1f} | {'':24s} {100 * r['cert']:6.2f} {100 * r['sod']:6.2f} {100 * r['centre']:6.2f} {100 * r['bound']:6.2f} {100 * r['cons_fixed']:7.2f} {100 * r['cons_matched']:7.2f} |"
                      f" {r['cert'] / r['bound']:16.2f} | {'':9s} {r['ghost_cert']:5.2f} {r['ghost_fixed']:7.2f} {r['ghost_matched']:7.2f} | {'':11s} {r['gb_cert']:5.2f} {r['gb_bound']:5.2f}   ({r['sec']:.0f}s)", flush=True)
    Path(a.out + ".json").write_text(json.dumps(rows, indent=1))
    print("\n=== Summary: share of the full cache kept, and projected GB per hour (Qwen2-VL-7B, 448x448 at 2 fps, full cache %.1f GB)" % gb_per_hour(1.0))
    for P in map(int, a.blocks.split(",")):
        ks = [k for k in rows if k.endswith(f"@{P}")]
        lo = {k: min(rows[k], key=lambda q: abs(q["fps"] - 2.0)) for k in ks}; hi = {k: rows[k][-1] for k in ks}
        print(f"  {P} px blocks, median over {len(ks)} clips, at ~2 fps -> native rate:")
        for rule in ("cert", "sod", "centre", "bound", "cons_fixed"):
            print(f"    entries {rule:11s} {100 * np.median([lo[k][rule] for k in ks]):6.2f} % -> {100 * np.median([hi[k][rule] for k in ks]):6.2f} %")
        for rule in ("cert", "bound", "cons_fixed"):
            print(f"    entries per block-second, {rule:11s} {np.median([lo[k]['per_s'][rule] for k in ks]):6.3f} -> {np.median([hi[k]['per_s'][rule] for k in ks]):6.3f}")
        print(f"    GB per hour, cert: {np.median([lo[k]['gb_cert'] for k in ks]):.2f} -> {np.median([hi[k]['gb_cert'] for k in ks]):.2f}; bound {np.median([lo[k]['gb_bound'] for k in ks]):.2f} -> {np.median([hi[k]['gb_bound'] for k in ks]):.2f}")
        for rule in ("ghost_cert", "ghost_fixed", "ghost_matched"):
            print(f"    {rule:17s} {np.median([lo[k][rule] for k in ks]):6.2f} % -> {np.median([hi[k][rule] for k in ks]):6.2f} %")
    print(f"wrote {a.out}.json")


if __name__ == "__main__":
    main()
