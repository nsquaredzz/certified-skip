#!/usr/bin/env python3
"""Ride a learned tracker: the moving-camera rule decides on which frames a frozen SAM 2 has to look.

On every frame the rule runs first.  The share of patches in the object's region that it could not
certify as held or moved is added to a running total; when the total passes --tau, SAM 2 is called
on that frame and the total is reset.  On the other frames SAM 2 is not run and the mask is carried
from the previous frame by block motion.  SAM 2 itself is untouched: it is shown only the frames it
is called on, as if they were a shorter video.

Four schedules per clip, all scored against the hand labels (region similarity J, first frame excluded):

  every frame   SAM 2 on every frame: the reference
  gated         the rule decides
  uniform       SAM 2 on evenly spaced frames, the same number of calls as the gated run, same carry
  never         the first-frame mask carried by motion alone

If the gate does not beat the uniform schedule at equal calls, it adds nothing.

With --agreement there are no hand labels beyond a first-frame mask: the every-frame run becomes
the reference and J reads as agreement with it.

    .venv/bin/python scripts/gate_sam2.py --dry                     # call rates for a few tau, no SAM 2
    .venv/bin/python scripts/gate_sam2.py --tau 1.0,2.0 --out out/gate_sam2
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
from render_moving import outline, GREEN, WHITE, SCHED, P
from moving_bench import dilate

B, R = 8, 16                # carry: block size and search radius, pixels
BAR = 32
PALETTE = [(255, 90, 90), (90, 200, 255), (255, 200, 60), (160, 255, 120), (230, 120, 255), (255, 150, 60), (120, 160, 255), (200, 200, 200)]


# ------------------------------------------------------------------ carry: block motion on luma
def block_motion(prev, cur):
    """Per-block (dy, dx): the block of `cur` at x came from `prev` at x + d.
    Full search at half resolution with a small penalty on length, then +-1 at full resolution."""
    H, W = cur.shape; gh, gw = H // B, W // B
    half = lambda a: a[: H // 2 * 2, : W // 2 * 2].reshape(H // 2, 2, W // 2, 2).mean((1, 3))
    p2, c2 = half(prev), half(cur); r2, b2 = R // 2, B // 2
    pp = np.pad(p2, r2, mode="edge"); h2, w2 = c2.shape
    best = np.full((gh, gw), np.inf); by = np.zeros((gh, gw), int); bx = np.zeros((gh, gw), int)
    for oy in range(-r2, r2 + 1):
        for ox in range(-r2, r2 + 1):
            d = np.abs(c2 - pp[r2 + oy: r2 + oy + h2, r2 + ox: r2 + ox + w2])
            s = d[: gh * b2, : gw * b2].reshape(gh, b2, gw, b2).sum((1, 3)) + 0.15 * b2 * b2 * (abs(oy) + abs(ox)) ** 0.5
            hit = s < best; best[hit] = s[hit]; by[hit] = 2 * oy; bx[hit] = 2 * ox
    pf = np.pad(prev, R + 1, mode="edge"); best[:] = np.inf; fy, fx = by.copy(), bx.copy()
    yy = (np.arange(gh) * B)[:, None, None, None] + np.arange(B)[None, None, :, None]
    xx = (np.arange(gw) * B)[None, :, None, None] + np.arange(B)[None, None, None, :]
    cb = cur[: gh * B, : gw * B].reshape(gh, B, gw, B).swapaxes(1, 2)
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            ty, tx = by + oy, bx + ox
            s = np.abs(cb - pf[yy + ty[..., None, None] + R + 1, xx + tx[..., None, None] + R + 1]).sum((2, 3))
            hit = s < best; best[hit] = s[hit]; fy[hit] = ty[hit]; fx[hit] = tx[hit]
    return fy, fx


def carry(lab, dy, dx):
    H, W = lab.shape; gh, gw = dy.shape
    out = lab.copy()
    yy = np.clip(np.arange(gh * B)[:, None] + np.kron(dy, np.ones((B, B), int)), 0, H - 1)
    xx = np.clip(np.arange(gw * B)[None, :] + np.kron(dx, np.ones((B, B), int)), 0, W - 1)
    out[: gh * B, : gw * B] = lab[yy, xx]
    return out


def iou(a, b):
    u = (a | b).sum()
    return 1.0 if u == 0 else float((a & b).sum() / u)


def jmean(labs, gt, ids):
    return float(np.mean([iou(labs[t] == k, gt[t] == k) for t in range(1, len(gt)) for k in ids]))


# ------------------------------------------------------------------ the gate
def uncertified(keep, anchor, lab, Hc, Wc):
    """Share of the object's region (its patches and their neighbours) that the rule had to send."""
    if anchor:
        return 1.0
    region = dilate(patch_grid(lab[:Hc, :Wc] > 0, P).any((2, 3)))
    return float((keep & region).sum() / max(1, region.sum()))


class Gate:
    def __init__(self, rec, tau, max_gap, Hc, Wc):
        self.rec, self.tau, self.max_gap, self.Hc, self.Wc = rec, tau, max_gap, Hc, Wc
        self.acc = 0.0; self.gap = 0; self.trace = {}

    def __call__(self, t, carried):
        keep, anchor = self.rec[t]
        self.acc += uncertified(keep, anchor, carried, self.Hc, self.Wc); self.gap += 1
        self.trace[t] = self.acc
        if self.acc >= self.tau or anchor or self.gap >= self.max_gap:
            self.acc = 0.0; self.gap = 0
            return True
        return False


# ------------------------------------------------------------------ SAM 2 on a chosen subset of frames
def track(predictor, torch, jd, gt0, ids, flows, decide):
    """Labels for every frame.  decide(t, carried_labels) says whether SAM 2 looks at frame t.
    The frames SAM 2 is called on are fed to it as consecutive frames of a shorter video."""
    T = len(flows)
    labs = [gt0.copy()]; called = [True]; t_sam = 0.0
    with torch.inference_mode():
        state = predictor.init_state(video_path=str(jd), offload_video_to_cpu=T > 150)
        for k in ids:
            predictor.add_new_mask(inference_state=state, frame_idx=0, obj_id=k, mask=gt0 == k)
        j = 0
        for t in range(1, T):
            carried = carry(labs[-1], *flows[t])
            if not decide(t, carried):
                labs.append(carried); called.append(False)
                continue
            t0 = time.perf_counter()
            j += 1
            if j != t:
                state["images"][j] = state["images"][t]          # j < t: later frames are still untouched
            if j == 1:                                            # the prompted frame has to be processed once
                next(iter(predictor.propagate_in_video(state, start_frame_idx=0, max_frame_num_to_track=0)))
            _, obj_ids, logits = next(iter(predictor.propagate_in_video(state, start_frame_idx=j, max_frame_num_to_track=0)))
            m = logits[:, 0].float().cpu().numpy()
            lab = np.zeros(gt0.shape, gt0.dtype); top = m.max(0); arg = m.argmax(0)
            lab[top > 0] = np.asarray(obj_ids, gt0.dtype)[arg[top > 0]]
            t_sam += time.perf_counter() - t0
            labs.append(lab); called.append(True)
        predictor.reset_state(state)
    return labs, called, t_sam


def caption(width, left, right):
    im = Image.new("RGB", (width, BAR), (18, 18, 18)); d = ImageDraw.Draw(im); font = ImageFont.load_default(size=15)
    d.text((10, 7), left, fill=(255, 255, 255), font=font); d.text((width // 2 + 14, 7), right, fill=(255, 255, 255), font=font)
    return np.asarray(im)


def paint(rgb, lab, ids, g):
    out = rgb.copy()
    for i, k in enumerate(ids):
        m = lab == k
        out[m] = (0.45 * out[m] + 0.55 * np.array(PALETTE[i % len(PALETTE)])).astype(np.uint8)
    out[g & ~(np.roll(g, 1, 0) & np.roll(g, -1, 0) & np.roll(g, 1, 1) & np.roll(g, -1, 1))] = WHITE
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/davis/DAVIS"); ap.add_argument("--ckpt", default="data/sam2/sam2.1_hiera_base_plus.pt")
    ap.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    ap.add_argument("--clips", default="mallard-water,snowboard,car-shadow,train,planes-water,drift-chicane")
    ap.add_argument("--tau", default="1.0", help="comma-separated thresholds on the accumulated uncertified share")
    ap.add_argument("--max-gap", type=int, default=12, help="call SAM 2 at least this often, whatever the rule says")
    ap.add_argument("--dry", action="store_true", help="no SAM 2: call rates with the hand labels standing in for the mask")
    ap.add_argument("--agreement", action="store_true", help="only a first-frame mask exists: score against the every-frame run")
    ap.add_argument("--fixed-rate", type=float, default=None,
                    help="run only a uniform schedule at this share of frames on every clip: blind skipping that is not told each clip's rate")
    ap.add_argument("--video-tau", type=float, default=None, help="render the comparison video at this tau")
    ap.add_argument("--device", default=None); ap.add_argument("--out", default="out/gate_sam2"); ap.add_argument("--out-fps", type=float, default=12.0)
    a = ap.parse_args()
    taus = [float(x) for x in a.tau.split(",")]
    predictor = torch = None
    if not a.dry:
        import torch
        from sam2.build_sam import build_sam2_video_predictor
        dev = a.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        predictor = build_sam2_video_predictor(a.config, a.ckpt, device=dev)
    results = {}; lines = []
    for name in a.clips.split(","):
        jd, ad = Path(a.root) / "JPEGImages" / "480p" / name, Path(a.root) / "Annotations" / "480p" / name
        files = sorted(jd.glob("*.jpg")); T = len(files)
        rgb = [np.array(Image.open(f).convert("RGB")) for f in files]
        gray = [np.array(Image.fromarray(x).convert("L")) for x in rgb]
        gt0 = np.array(Image.open(ad / (files[0].stem + ".png")))
        gt = None if a.agreement else [np.array(Image.open(ad / (f.stem + ".png"))) for f in files]
        ids = [int(k) for k in np.unique(gt0) if k > 0]
        Hc, Wc = gray[0].shape[0] // P * P, gray[0].shape[1] // P * P
        # the rule on every frame, and the block motion used to carry masks; both once per clip
        t0 = time.perf_counter(); pr = mv.MovingPruner(Hc, Wc, P, SCHED, search=16); rec = []
        for t in range(T):
            keep = pr.step(np.ascontiguousarray(gray[t][:Hc, :Wc])); rec.append((keep.copy(), bool(pr.last["anchor"])))
        t_rule = (time.perf_counter() - t0) / T
        t0 = time.perf_counter()
        flows = [None] + [block_motion(gray[t - 1].astype(float), gray[t].astype(float)) for t in range(1, T)]
        t_carry = (time.perf_counter() - t0) / (T - 1)
        never = [gt0]
        for t in range(1, T):
            never.append(carry(never[-1], *flows[t]))
        if a.dry:
            stand_in = gt if gt is not None else never
            row = f"  {name:16s} {T:3d} frames:"
            for tau in taus:
                g = Gate(rec, tau, a.max_gap, Hc, Wc); n = sum(g(t, stand_in[t]) for t in range(1, T))
                row += f"  tau {tau:g}: SAM 2 on {100 * n / (T - 1):.0f} %"
            print(row, flush=True)
            continue
        if a.fixed_rate is not None:
            n = max(1, int(round(a.fixed_rate * (T - 1))))
            pick = set(np.unique(np.round(np.linspace(1, T - 1, n)).astype(int)).tolist())
            ul, uc, ts = track(predictor, torch, jd, gt0, ids, flows, lambda t, c: t in pick)
            results[name] = {"frames": T, "calls": sum(uc[1:]), "J": jmean(ul, gt, ids), "sam_s": ts}
            print(f"  {name:16s} fixed rate: SAM 2 on {sum(uc[1:])} of {T - 1} frames, J {results[name]['J']:.3f}", flush=True)
            continue
        res = {"frames": T, "objects": len(ids), "rule_s_per_frame": t_rule, "carry_s_per_frame": t_carry, "runs": {}}
        full, _, ts = track(predictor, torch, jd, gt0, ids, flows, lambda t, c: True)
        if gt is None:
            gt = full
        res["runs"]["every frame"] = {"calls": T - 1, "J": jmean(full, gt, ids), "sam_s": ts}
        res["runs"]["never"] = {"calls": 0, "J": jmean(never, gt, ids), "sam_s": 0.0}
        keepvid = None
        for tau in taus:
            gate = Gate(rec, tau, a.max_gap, Hc, Wc)
            gl, gc, ts = track(predictor, torch, jd, gt0, ids, flows, gate)
            n = sum(gc[1:])
            res["runs"][f"gated {tau:g}"] = {"calls": n, "J": jmean(gl, gt, ids), "sam_s": ts,
                                             "agree_with_every_frame": float(np.mean([iou(gl[t] == k, full[t] == k) for t in range(1, T) for k in ids]))}
            pick = set(np.unique(np.round(np.linspace(1, T - 1, max(n, 1))).astype(int)).tolist()) if n else set()
            ul, uc, ts = track(predictor, torch, jd, gt0, ids, flows, lambda t, c: t in pick)
            res["runs"][f"uniform {tau:g}"] = {"calls": sum(uc[1:]), "J": jmean(ul, gt, ids), "sam_s": ts}
            if a.video_tau is not None and abs(tau - a.video_tau) < 1e-9:
                keepvid = (gl, gc, gate.trace, tau)
        results[name] = res
        per = res["runs"]["every frame"]["sam_s"] / (T - 1)
        row = f"  {name:16s} {T:3d} frames, {len(ids)} obj, SAM 2 {per:.2f} s/frame, rule {t_rule:.2f}, carry {t_carry:.2f} | every frame J {res['runs']['every frame']['J']:.3f} | never {res['runs']['never']['J']:.3f}"
        for tau in taus:
            g, u = res["runs"][f"gated {tau:g}"], res["runs"][f"uniform {tau:g}"]
            row += f" | tau {tau:g}: SAM 2 on {100 * g['calls'] / (T - 1):.0f} %, gated J {g['J']:.3f}, uniform J {u['J']:.3f}"
        print(row, flush=True); lines.append(row)
        if keepvid:
            gl, gc, trace, tau = keepvid
            W2 = rgb[0].shape[1] // 2 * 2; H2 = rgb[0].shape[0] // 2 * 2
            cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * W2 + 8}x{H2 + BAR}", "-r", str(a.out_fps),
                   "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "22", "-pix_fmt", "yuv420p", f"{a.out}_{name}.mp4"]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE); calls = 0
            for t in range(T):
                calls += int(gc[t]) if t else 0
                g = gt[t] > 0
                left = paint(rgb[t], gl[t], ids, g); right = paint(rgb[t], full[t], ids, g)
                if t:                                                    # the patches in the object's region the rule could not certify
                    region = dilate(patch_grid(gl[t][:Hc, :Wc] > 0, P).any((2, 3)))
                    outline(left, rec[t][0] & region, GREEN)
                tag = (235, 60, 60) if gc[t] else (60, 200, 90)
                left[8:40, 8:40] = tag
                frame = np.concatenate([left[:H2, :W2], np.zeros((H2, 8, 3), np.uint8), right[:H2, :W2]], 1)
                state = "SAM 2 CALLED" if gc[t] else "SAM 2 skipped, mask carried"
                cap = caption(frame.shape[1],
                              f"GATED  {name}  frame {t}: {state}.   SAM 2 used on {calls} of {t} frames so far",
                              f"SAM 2 ON EVERY FRAME.   white = {'its own outline, the reference' if a.agreement else 'hand label'}"
                              f"   red square = SAM 2 ran, green = skipped")
                proc.stdin.write(np.concatenate([cap, frame], 0).tobytes())
            proc.stdin.close(); proc.wait()
    if a.dry:
        return
    if a.fixed_rate is not None:
        tot = sum(r["calls"] for r in results.values()); fr = sum(r["frames"] - 1 for r in results.values())
        line = f"fixed rate {a.fixed_rate:g}: mean J {np.mean([r['J'] for r in results.values()]):.3f}, SAM 2 on {tot} of {fr} frames"
        print(line)
        Path(a.out + "_fixed.json").write_text(json.dumps({"rate": a.fixed_rate, "clips": results}, indent=1))
        return
    names = list(results)
    summ = {}
    for key in results[names[0]]["runs"]:
        summ[key] = {"J": float(np.mean([results[n]["runs"][key]["J"] for n in names])),
                     "sam_share": float(np.mean([results[n]["runs"][key]["calls"] / (results[n]["frames"] - 1) for n in names])),
                     "sam_frames": int(sum(results[n]["runs"][key]["calls"] for n in names)),
                     "sam_s": float(sum(results[n]["runs"][key]["sam_s"] for n in names))}
    frames = sum(results[n]["frames"] - 1 for n in names)
    rule_s = sum(results[n]["rule_s_per_frame"] * results[n]["frames"] for n in names)
    carry_s = sum(results[n]["carry_s_per_frame"] * (results[n]["frames"] - 1) for n in names)
    lines.append(f"mean over {len(names)} clips, {frames} frames; the rule costs {rule_s:.0f} s and the carry {carry_s:.0f} s in total on this machine")
    for key, s in summ.items():
        extra = rule_s + carry_s if key.startswith("gated") else carry_s if key.startswith(("uniform", "never")) else 0.0
        lines.append(f"  {key:12s} J {s['J']:.3f}   SAM 2 on {100 * s['sam_share']:5.1f} % of frames ({s['sam_frames']} calls)   "
                     f"time: SAM 2 {s['sam_s']:.0f} s + ours {extra:.0f} s = {s['sam_s'] + extra:.0f} s")
    print("\n".join(lines[-len(summ) - 1:]))
    Path(a.out + ".json").write_text(json.dumps({"summary": summ, "frames": frames, "rule_s": rule_s, "carry_s": carry_s, "clips": results}, indent=1))
    Path(a.out + ".txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
