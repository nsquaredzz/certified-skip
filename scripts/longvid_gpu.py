#!/usr/bin/env python3
"""Scaled-up toy world on one GPU: train a ~15M-parameter block-causal video DiT for a fixed time, then compare KV-cache
policies on long rollouts with error bars. Built to finish in about 15-20 minutes on an A100.

Training   clean-context diffusion forcing, x-prediction with a velocity loss (see longvid/model.py), bf16, AdamW, a
           learning-rate schedule on *elapsed time* (so the budget is respected whatever the speed), weight averaging (EMA),
           a checkpoint every --save-every seconds and --resume to continue after an interruption.
Evaluation the first frame of each episode is the prompt; the model generates --eval-frames frames (default 128, eight
           times its training length) under each cache policy, the same episodes and noise for every policy (paired).

Policies   full           every clean frame kept
           sink1+winK     the first frame plus the last K frames (what StreamingVLM / Self-Forcing-style rolling caches do)
           reindex        the same with positions renumbered after eviction
           merge e=E      MergeCache: unchanged locations share one entry with a log-count bias (the idea under test)
           merge e=E wK   the same, with entries dropped K frames after their last refresh
           motion e=E     MotionMergeCache: an entry whose content moved by a whole token moves with it (no new entry)
World      static (fixed camera) or pan (--world pan: the camera pans at --pan-speeds pixels per frame over a wrapping
           background; two prompt frames, so the speed is visible). With patch 8, a pan of 8 is one token per frame.
Metrics    mse     mean squared error against the true continuation
           bg      median distance of generated pixels to the true background (static content drifting)
           fgdev   |share of pixels off the background - true disc area|  (discs vanishing or smearing)
           col     mean distance of off-background pixels to the nearest colour the episode has (identity drifting)
           entries cache entries held (tokens), averaged over the rollout; a full cache at frame t holds (t + 1) * tokens per frame
Error bars are standard errors over episodes. The model is trained once: they do not include training-seed variance.

    python3 scripts/longvid_gpu.py --minutes 10 --out out/longvid_gpu
"""
import argparse, copy, json, math, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from longvid.model import LongVidDiT, KVCache, MergeCache, MotionMergeCache, velocity
from longvid import data
from longvid.sample import rollout

AREA = math.pi * data.R ** 2 / data.S ** 2


class Stream(torch.utils.data.IterableDataset):
    def __init__(self, B, T, seed, world="static", speeds=(-8, -4, 4, 8)):
        self.B, self.T, self.seed, self.world, self.speeds = B, T, seed, world, speeds

    def __iter__(self):
        w = torch.utils.data.get_worker_info(); rng = np.random.default_rng([self.seed, 0 if w is None else w.id])
        while True:
            yield torch.from_numpy(data.batch_world(rng, self.B, self.T, self.world, self.speeds)[0])


def lr_at(frac, peak, warm=0.03, floor=0.05):
    return peak * frac / warm if frac < warm else peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * (frac - warm) / (1 - warm))))


def train(a, model, dev, log):
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.01); used, step, ck = 0.0, 0, Path(a.out + ".ckpt")
    if a.resume and ck.exists():
        s = torch.load(ck, map_location=dev); model.load_state_dict(s["model"]); ema.load_state_dict(s["ema"]); opt.load_state_dict(s["opt"]); used, step = s["used"], s["step"]
        log(f"resumed at step {step}, {used:.0f}s of {a.minutes * 60:.0f}s used")
    dl = torch.utils.data.DataLoader(Stream(a.batch, a.frames_train, a.seed + step, a.world, tuple(a.pan_speeds)), batch_size=None, num_workers=a.workers, prefetch_factor=4, persistent_workers=a.workers > 0)
    budget = a.minutes * 60; t_last, t_save, ema_loss = time.time(), time.time(), None; mp, ep = list(model.parameters()), list(ema.parameters()); model.train()
    for fr in dl:
        now = time.time(); used += now - t_last; t_last = now
        if used >= budget:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(used / budget, a.lr)
        x0 = model.patchify(fr.to(dev, non_blocking=True)); B, Fn = x0.shape[:2]
        s = torch.sigmoid(torch.randn(B, Fn, device=dev)); s = torch.where(torch.rand(B, 1, device=dev) < 0.5, s[:, :1].expand(-1, Fn), s)
        eps = torch.randn_like(x0); xs = (1 - s)[..., None, None] * x0 + s[..., None, None] * eps
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out = model.forward_train(xs, x0, s)
        loss = ((velocity(xs, s, out.float()) - (eps - x0)) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
        with torch.no_grad():
            torch._foreach_lerp_(ep, mp, 1 - a.ema if step > 200 else 1.0)
        if step % 50 == 0:
            ema_loss = loss.item() if ema_loss is None else 0.9 * ema_loss + 0.1 * loss.item()
            if step % 200 == 0: log(f"step {step:6d}  loss {ema_loss:.4f}  lr {g['lr']:.2e}  {used:5.0f}s / {budget:.0f}s  {step / max(used, 1):.1f} it/s")
        if time.time() - t_save > a.save_every:
            torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), used=used, step=step), ck); t_save = time.time()
    torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), used=used, step=step), ck)
    log(f"trained {step} steps in {used:.0f}s ({step * a.batch / max(used, 1):.0f} clips/s)"); return ema


def metrics(gen, gt, bg, cols):
    """gen, gt, bg (E, N, 3, H, W) in [-1, 1] -> (E, N, 4): bg, fgdev, col, mse, per episode and frame.
    mse is against the true continuation, which the prompt frames determine (velocities are visible in two frames)."""
    out = np.zeros(gen.shape[:2] + (4,), np.float32)
    for e in range(len(gen)):
        d = np.abs(gen[e] - bg[e]).max(1)
        for t in range(gen.shape[1]):
            m = d[t] > 0.3; px = gen[e, t][:, m].T
            out[e, t] = (np.median(np.abs(gen[e, t] - bg[e, t]).mean(0)), abs(m.mean() - AREA * len(cols[e])),
                         np.abs(px[:, None] - cols[e][None]).max(-1).min(1).mean() if len(px) else 0.0, ((gen[e, t] - gt[e, t]) ** 2).mean())
    return out


def policies(a, model):
    P = {"full": lambda d: KVCache(d), "sink1+win3": lambda d: KVCache(d, 1, 3), "sink1+win7": lambda d: KVCache(d, 1, 7),
         "sink1+win15": lambda d: KVCache(d, 1, 15), "sink1+win7 reindex": lambda d: KVCache(d, 1, 7, True)}
    for e in a.eps:
        P[f"merge e={e}"] = lambda d, e=e: MergeCache(d, e)
    for e, w in ((0.1, 7), (0.1, 15), (0.2, 7)):
        P[f"merge e={e} w{w}"] = lambda d, e=e, w=w: MergeCache(d, e, w)
    hw = (model.gh, model.gw)
    for e in a.eps:
        P[f"motion e={e}"] = lambda d, e=e: MotionMergeCache(d, e, hw, a.radius)
    for e, w in ((0.1, 7), (0.2, 7)):
        P[f"motion e={e} w{w}"] = lambda d, e=e, w=w: MotionMergeCache(d, e, hw, a.radius, w)
    if a.policies:
        keep = [x.strip() for x in a.policies.split(",")]; P = {k: v for k, v in P.items() if k in keep}
    return P


def evaluate(a, model, dev, log):
    Pn = a.prompt_frames or (2 if a.world == "pan" else 1); N = a.eval_frames
    rng = np.random.default_rng(1234); fr, bgs, cols = data.batch_world(rng, a.eval_episodes, Pn + N, a.world, tuple(a.pan_speeds))
    prompt = torch.from_numpy(fr[:, :Pn]).to(dev); gt, bg = fr[:, Pn:], bgs[:, Pn:]
    seg = [(0, 16), (16, 48), (48, 96), (96, N)]; seg = [(s, min(e, N)) for s, e in seg if s < N]; res = {}
    for name, mk in policies(a, model).items():
        t0 = time.time(); ent = []; M = []
        for i in range(0, a.eval_episodes, a.eval_batch):
            sl = slice(i, i + a.eval_batch); g = torch.Generator(device=dev).manual_seed(7 + i); e_b = []
            gen = rollout(model, prompt[sl], N, steps=a.steps, gen=g, make_cache=mk, on_frame=lambda n, e: e_b.append(e)).float().cpu().numpy()
            M.append(metrics(gen, gt[sl], bg[sl], cols[sl])); ent.append(e_b)
        M = np.concatenate(M); ent = np.mean(ent, 0); res[name] = dict(metrics=M.tolist(), entries=ent.tolist())
        row = []
        for s, e in seg:
            m = M[:, s:e].mean(1); se = m.std(0, ddof=1) / math.sqrt(len(m)); row.append(f"{s + 1}-{e}: bg {m[:, 0].mean():.3f}±{se[0]:.3f} mse {m[:, 3].mean():.3f}±{se[3]:.3f} fgdev {m[:, 1].mean():.3f} col {m[:, 2].mean():.2f} ent {ent[s:e].mean():.0f}")
        log(f"{name:20s} ({time.time() - t0:.0f}s) " + " | ".join(row))
    ref = "sink1+win7" if "sink1+win7" in res else next(iter(res)); L = min(48, N // 2); base = np.array(res[ref]["metrics"])[:, L:].mean(1)
    log(f"\npaired against {ref}, frames {L + 1}-{N} (negative = better; mean ± SE over {a.eval_episodes} episodes, same prompts and noise):")
    for name, r in res.items():
        d = np.array(r["metrics"])[:, L:].mean(1) - base; se = d.std(0, ddof=1) / math.sqrt(len(d)); ent = np.mean(r["entries"][L:])
        log(f"  {name:20s} entries {ent:6.0f}   d(bg) {d[:, 0].mean():+.4f}±{se[0]:.4f}   d(mse) {d[:, 3].mean():+.4f}±{se[3]:.4f}   d(fgdev) {d[:, 1].mean():+.4f}±{se[1]:.4f}")
    return res


def main():
    ap = argparse.ArgumentParser(); A = ap.add_argument
    A("--minutes", type=float, default=10); A("--batch", type=int, default=64); A("--lr", type=float, default=3e-4); A("--ema", type=float, default=0.995)
    A("--dim", type=int, default=384); A("--depth", type=int, default=8); A("--heads", type=int, default=6); A("--patch", type=int, default=4)
    A("--frames-train", type=int, default=16); A("--workers", type=int, default=8); A("--seed", type=int, default=0); A("--save-every", type=float, default=120)
    A("--eval-episodes", type=int, default=32); A("--eval-batch", type=int, default=32); A("--eval-frames", type=int, default=128); A("--steps", type=int, default=8)
    A("--eps", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.4]); A("--world", choices=["static", "pan"], default="static"); A("--pan-speeds", type=int, nargs="+", default=[-8, -4, 4, 8])
    A("--prompt-frames", type=int, default=0); A("--radius", type=int, default=1); A("--policies", default="")
    A("--resume", action="store_true"); A("--eval-only", action="store_true"); A("--out", default="out/longvid_gpu")
    a = ap.parse_args(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.manual_seed(a.seed)
    lf = open(a.out + ".txt", "a")
    def log(s): print(s, flush=True); lf.write(s + "\n"); lf.flush()
    model = LongVidDiT(frame=(32, 32), patch=a.patch, dim=a.dim, depth=a.depth, heads=a.heads).to(dev)
    log(f"device {dev}, {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters, {model.N} tokens per frame, {a.frames_train}-frame clips, batch {a.batch}, {a.minutes:g} min budget, world {a.world}")
    if a.eval_only: model.load_state_dict(torch.load(a.out + ".ckpt", map_location=dev)["ema"]); ema = model
    else: ema = train(a, model, dev, log)
    ema.eval(); t0 = time.time()
    with torch.no_grad():
        res = evaluate(a, ema, dev, log)
    Path(a.out + ".json").write_text(json.dumps(res)); log(f"evaluation took {time.time() - t0:.0f}s; wrote {a.out}.json")


if __name__ == "__main__":
    main()
