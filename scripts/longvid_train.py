#!/usr/bin/env python3
"""Train the tiny block-causal video DiT on the moving-discs world, then roll it out far past its training length.

Training: clean-context diffusion forcing with rectified flow. Each of the F frames gets its own noise level (half the
time one shared level), the loss is the velocity error on the noisy copy. Evaluation: the model is given the first
frame, then generates N frames with a KV cache under several policies; a policy is (sink, window, reindex).

Coherence metrics, per frame of the rollout, against what the episode is known to contain:
  bg_err   median over pixels of the distance to the true background (robust to the discs, which cover <= 10 %)
  fg       share of pixels further than 0.3 from the background: the discs, or artifacts if it grows
  colour   among those pixels, the share within 0.35 of a colour the episode actually has (a drifting palette scores low)

    python3 scripts/longvid_train.py --steps 3000 --out out/longvid
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from longvid.model import LongVidDiT
from longvid import data
from longvid.sample import rollout

F_TRAIN = 8


def train(model, steps, B, lr, seed, log):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01); t0 = time.time(); ema = None
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=max(0.05, 2.0 / steps))
    for it in range(steps):
        fr, _, _ = data.batch(rng, B, F_TRAIN); x0 = model.patchify(torch.from_numpy(fr))
        shared = torch.rand(B, 1) < 0.5
        s = torch.sigmoid(torch.randn(B, F_TRAIN) * 1.0 + 0.0); s = torch.where(shared, s[:, :1].expand(-1, F_TRAIN), s)
        eps = torch.randn_like(x0); xs = (1 - s)[..., None, None] * x0 + s[..., None, None] * eps
        loss = ((model.forward_train(xs, x0, s) - (eps - x0)) ** 2).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if it % 100 == 0 or it == steps - 1: log(f"step {it:5d}  loss {ema:.4f}  ({time.time() - t0:.0f}s)")


def metrics(gen, bg, cols):
    """gen (N, 3, H, W) numpy in [-1, 1]."""
    d = np.abs(gen - bg[None]).max(1); out = []
    for t in range(len(gen)):
        fgm = d[t] > 0.3; px = gen[t][:, fgm].T
        near = (np.abs(px[:, None] - cols[None]).max(-1) < 0.35).any(1).mean() if len(px) else 1.0
        out.append((float(np.median(np.abs(gen[t] - bg).mean(0))), float(fgm.mean()), float(near)))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--frames", type=int, default=64); ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="out/longvid"); ap.add_argument("--load", default=""); ap.add_argument("--patch", type=int, default=8)
    a = ap.parse_args(); torch.set_num_threads(4); Path(a.out).parent.mkdir(exist_ok=True)
    lf = open(a.out + ".txt", "a")
    def log(s): print(s, flush=True); lf.write(s + "\n"); lf.flush()
    torch.manual_seed(a.seed); model = LongVidDiT(frame=(32, 32), patch=a.patch, dim=128, depth=4, heads=4)
    log(f"parameters {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M, {model.N} tokens per frame, trained on {F_TRAIN} frames")
    if a.load: model.load_state_dict(torch.load(a.load))
    else: train(model, a.steps, a.batch, a.lr, a.seed, log); torch.save(model.state_dict(), a.out + ".pt")
    model.eval(); rng = np.random.default_rng(1234); fr, bg, cols = data.batch(rng, a.episodes, 1)
    policies = {"full": dict(), "sink1+win7": dict(sink=1, window=7), "sink1+win7 reindex": dict(sink=1, window=7, reindex=True), "win8 reindex": dict(sink=0, window=8, reindex=True)}
    res = {}
    for name, kw in policies.items():
        g = torch.Generator().manual_seed(7); t0 = time.time()
        gen = rollout(model, torch.from_numpy(fr), a.frames, steps=8, gen=g, **kw).numpy()
        m = np.stack([metrics(gen[i], bg[i], cols[i]) for i in range(a.episodes)]).mean(0)         # (N, 3)
        res[name] = m.tolist(); T = len(m); seg = [(0, 8), (8, 24), (24, 48), (48, T)]
        log(f"{name:20s} ({time.time() - t0:.0f}s) " + " | ".join(f"frames {s}-{e}: bg {m[s:e, 0].mean():.3f} fg {m[s:e, 1].mean():.3f} col {m[s:e, 2].mean():.2f}" for s, e in seg if s < T))
        np.save(f"{a.out}_{name.replace(' ', '_').replace('+', '_')}.npy", gen)
    Path(a.out + ".json").write_text(json.dumps(res)); log(f"wrote {a.out}.json")


if __name__ == "__main__":
    main()
