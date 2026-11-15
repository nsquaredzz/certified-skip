#!/usr/bin/env python3
"""Stage 1 of CertViT, structure only: what a streaming local-attention encoder costs and drifts on real footage.

The encoder has random weights. Nothing here says anything about accuracy; it measures the architecture's
bookkeeping on real keep masks. For each clip and frame rate, two rules with the same number of entries decide which
patches are sent (certified: the send-on-delta rule of Theorem 11 without shift; heuristic: consecutive-frame mean,
tau matched to the certified rule's entries). The streaming encoder holds the image V (sent patches replaced, others
held) and recomputes either the halo (exact on V) or only the sent patches (halo none).

  drift     ||h_L(F) - h_L(stream)|| / ||h_L(F) - h_L(F_0)||: the distance of the model's output from a fresh pass on the
            true frame, in units of what a model that never updated would be off by (1 = no better than never updating)
  compute   share of token updates a full pass would do, averaged over layers (the cost of the halo)

    python3 scripts/certvit_drift.py [--clips hall,leftbag] [--seeds 3] [--out out/certvit_drift]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from certskip import video, baselines as bl
from certskip.centre import CentrePruner
from certskip.certvit import CertViT, StreamingCertViT
from real_bench import schedule

P = 16
CLIPS = {"hall": ("data/hall_monitor_cif.y4m", 300), "leftbag": ("data/caviar_LeftBag.mpg", 600)}


def hold(V, f, keep):
    m = keep.view(*keep.shape[:2], 1, 1) if False else None
    gh, gw = keep.shape; k = keep[:, None, :, None].expand(gh, P, gw, P).reshape(gh * P, gw * P)
    return torch.where(k, f, V)


def evaluate(model, frames, keep):
    T, H, W = frames.shape; gh, gw = H // P, W // P
    ft = torch.from_numpy(frames[:, : gh * P, : gw * P].astype(np.float32)); kp = torch.from_numpy(keep)
    f0 = model.encode(ft[0], True); out = {}
    for halo in ("full", "none"):
        s = StreamingCertViT(model); V = ft[0].clone(); s.step(ft[0], kp[0], halo); err, comp, layer = [], [], []
        for t in range(1, T):
            V = hold(V, ft[t], kp[t]); hL = s.step(ft[t], kp[t], halo)
            fr = model.encode(ft[t], True); stale = (fr[-1] - f0[-1]).norm()
            err.append(float((fr[-1] - hL).norm() / stale.clamp_min(1e-9)))
            layer.append([float((fr[l] - s.h[l]).norm() / (fr[l] - f0[l]).norm().clamp_min(1e-9)) for l in range(1, model.depth + 1)])
            comp.append(float(np.mean(s.stats["frac"][1:])))
            if halo == "full" and t % 50 == 0:
                assert torch.allclose(hL, model.encode(V), atol=1e-4), "streaming is not the encoder on the held image"
        out[halo] = dict(drift=float(np.mean(err)), drift_max=float(np.max(err)), compute=float(np.mean(comp)), by_layer=np.mean(layer, 0).tolist())
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--clips", default="hall,leftbag"); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--strides", default="1,3"); ap.add_argument("--out", default="out/certvit_drift"); a = ap.parse_args()
    sched = schedule(80.0, 0.5); rows = {}; torch.set_num_threads(4)
    for name in a.clips.split(","):
        path, nmax = CLIPS[name]; info = video.probe(path); fps0 = info["fps"] or 25.0
        fr = video.read_gray(path, fps=fps0, max_frames=nmax); T, H, W = fr.shape; fr = fr[:, : H // P * P, : W // P * P]
        for stride in map(int, a.strides.split(",")):
            sub = fr[::stride]; key = f"{name}@{fps0 / stride:.1f}fps"; t0 = time.time()
            kc = CentrePruner(sub.shape[1], sub.shape[2], P, multiscale=sched, lookahead=0).run(sub)["keep"]
            sc = bl.consecutive_mean_scores(sub, P); tau = float(np.quantile(sc, 1 - float(kc[1:].mean())))
            kh = bl.consecutive_mean(sub, P, tau, scores=sc); rows[key] = dict(entries_cert=float(kc[1:].mean()), entries_heur=float(kh[1:].mean()), frames=len(sub))
            for rule, keep in (("certified", kc), ("heuristic", kh)):
                rs = []
                for seed in range(a.seeds):
                    torch.manual_seed(seed); m = CertViT((sub.shape[1] // P, sub.shape[2] // P), patch=P, dim=128, depth=6, heads=4, radius=1).eval()
                    rs.append(evaluate(m, sub, keep))
                rows[key][rule] = {h: {k: (np.mean([r[h][k] for r in rs], 0).tolist() if k == "by_layer" else float(np.mean([r[h][k] for r in rs]))) for k in rs[0][h]} for h in ("full", "none")}
            r = rows[key]
            print(f"{key:16s} entries {100 * r['entries_cert']:5.2f}% | halo full: compute {100 * r['certified']['full']['compute']:5.1f}% (heur {100 * r['heuristic']['full']['compute']:5.1f}%)"
                  f" | drift cert {r['certified']['full']['drift']:.3f} heur {r['heuristic']['full']['drift']:.3f}"
                  f" | halo none: compute {100 * r['certified']['none']['compute']:4.1f}%, drift cert {r['certified']['none']['drift']:.3f} heur {r['heuristic']['none']['drift']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
            print("    drift by layer, certified, halo full:", " ".join(f"{x:.3f}" for x in r["certified"]["full"]["by_layer"]), "| halo none:", " ".join(f"{x:.3f}" for x in r["certified"]["none"]["by_layer"]), flush=True)
    Path(a.out + ".json").write_text(json.dumps(rows, indent=1)); print(f"wrote {a.out}.json")


if __name__ == "__main__":
    main()
