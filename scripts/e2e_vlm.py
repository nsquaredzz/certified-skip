#!/usr/bin/env python3
"""End-to-end with a real video LLM (Qwen2-VL-2B-Instruct, Apple MPS, CUDA or CPU).

The model is fed the VIEW a pruning rule leaves it: kept patches show the truth, dropped patches show
what the rule says they still are. This isolates the information content of the pruning decisions
from the token-reuse mechanics, and runs on an unmodified model. Token savings are the drop
rates reported elsewhere.

Two choices decide what a dropped patch shows (THEORY.md section 9):
  --view shown   the last kept copy, moved by the sub-pixel shift and re-levelled by the brightness offset
                 that the rule fitted on the current frame (the default, and what RESULTS Part VII reports)
  --view held    the last kept copy as it is, which is what a model that reuses tokens holds
  --offset patch the rule removes a brightness offset per patch before testing it (the default)
  --offset none  the rule has no brightness freedom, so its certificate is about the held copy
With --offset none --view held the blind-spot trials also ask the model about the held copy of the
rule with the per-patch offset ("published"), on the same trials.

E1  Blind spot, end to end.  A 36x36 (CIF) / 72x72 (HD) square fades in linearly over 8 s on a
    real clip. Views: consecutive-frame mean rule (EVS/RLT style) at the certified rule's drop
    rate; certified quotient+multi-scale; + sequential. Question at the end of the fade:
    "Is there a dark/bright square patch on the floor or wall? Answer yes or no."
    Also asked on the truth frame and on a no-object control.
E2  No regression.  On untouched footage, "How many people are visible? Answer with a number."
    and "Is there a person visible? yes/no", on the truth frame and on each rule's view at the
    same drop rate; we report agreement with the truth-frame answer.

    .venv/bin/python scripts/e2e_vlm.py [--trials 12] [--e2-frames 15] [--out out/e2e] [--offset none --view held]
"""
import argparse, json, re, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import certskip as cs
from certskip import video, baselines as bl
from certskip.core import patch_grid, drop_rate
from certskip.native import NativeSequentialPruner
from real_bench import schedule

P = 16
CLIPS = {
    "hall":  dict(path="data/hall_monitor_cif.y4m", fps=30, scale=None, up=2, box=48, floor=(150, 288, 40, 320)),
    "lobby": dict(path="data/caviar_LeftBag.mpg", fps=25, scale=None, up=2, box=48, floor=(100, 280, 60, 330)),
    "virat": dict(path="data/VIRAT_S_000200_00.mp4", fps=10, scale=(960, 544), up=1, box=96, floor=(260, 500, 120, 900)),
}


# ------------------------------------------------------------------ model
class VLM:
    def __init__(self, name="Qwen/Qwen2-VL-2B-Instruct", max_pixels=640 * 28 * 28):
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        self.torch = torch
        self.dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32 if self.dev == "cpu" else torch.float16
        if self.dev == "cuda":
            # The half-precision weights are about 4.2 GiB. On a smaller card the layers that do not fit are held in
            # main memory and copied to the card for each call; 0.5 GiB of the card is left free for activations.
            room = {0: torch.cuda.mem_get_info()[0] - (512 << 20), "cpu": "8GiB"}
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(name, torch_dtype=dtype, device_map="auto", max_memory=room).eval()
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(name, torch_dtype=dtype).to(self.dev).eval()
        self.proc = AutoProcessor.from_pretrained(name, min_pixels=128 * 28 * 28, max_pixels=max_pixels)
        self.n = 0; self.t = 0.0

    def ask(self, image, question, max_new_tokens=6):
        from PIL import Image
        t0 = time.perf_counter()
        img = Image.fromarray(image).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        text = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = self.proc(text=[text], images=[img], return_tensors="pt").to(self.dev)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        ans = self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        self.n += 1; self.t += time.perf_counter() - t0
        return ans


def yes_no(ans):
    a = ans.lower()
    return 1 if a.startswith("yes") or " yes" in a[:8] else (0 if a.startswith("no") or " no" in a[:8] else -1)


def number(ans):
    m = re.search(r"\d+", ans)
    words = {"zero": 0, "none": 0, "no": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
    if m:
        return int(m.group())
    for w, v in words.items():
        if w in ans.lower():
            return v
    return -1


def upscale(img, k):
    return img if k == 1 else np.repeat(np.repeat(img, k, 0), k, 1)


# ------------------------------------------------------------------ views
class LastKeptView:
    """View for a rule given as a keep-mask sequence: last kept content per patch."""
    def __init__(self, first):
        self.ref = first.copy()

    def update(self, frame, keep):
        rg = patch_grid(self.ref, P); fg = patch_grid(frame, P); rg[keep] = fg[keep]
        return self.ref.copy()


def run_views(frames, sched, tau_evs, offset="patch", held=False, published=False):
    """Returns dict rule -> list of views (one per frame). held: dropped patches show the copy the model holds."""
    T, H, W = frames.shape
    out = {"evs": [], "certified": [], "sequential": []}
    pr_ms = NativeSequentialPruner(H, W, P, multiscale=sched, z=1e9, offset=offset)
    pr_sq = NativeSequentialPruner(H, W, P, multiscale=sched, z=8.0, offset=offset)
    pr_pub = NativeSequentialPruner(H, W, P, multiscale=sched, z=1e9) if published else None
    if published:
        out["published"] = []
    ev = LastKeptView(frames[0]); prev = frames[0]
    for t in range(T):
        pr_ms.step(frames[t]); out["certified"].append(pr_ms.view(not held))
        pr_sq.step(frames[t]); out["sequential"].append(pr_sq.view(not held))
        if published:
            pr_pub.step(frames[t]); out["published"].append(pr_pub.view(False))
        if t == 0:
            out["evs"].append(frames[0].copy())
        else:
            k = bl.consecutive_mean(np.stack([prev, frames[t]]), P, tau_evs)[1]
            out["evs"].append(ev.update(frames[t], k))
        prev = frames[t]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=12); ap.add_argument("--e2-frames", type=int, default=15)
    ap.add_argument("--clips", default="hall,lobby,virat"); ap.add_argument("--out", default="out/e2e")
    ap.add_argument("--fade-s", type=float, default=8.0); ap.add_argument("--contrast", type=int, default=70)
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--skip-e2", action="store_true")
    ap.add_argument("--offset", choices=("patch", "none"), default="patch"); ap.add_argument("--view", choices=("shown", "held"), default="shown")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    sched = schedule(80.0, 0.5)
    held = a.view == "held"; pub = held and a.offset == "none"
    rules = ("evs", "certified", "sequential") + (("published",) if pub else ())
    vlm = VLM(a.model)
    print(f"model on {vlm.dev}")
    if (a.offset, a.view) != ("patch", "shown"):
        print(f"rule offset: {a.offset}; dropped patches show the {a.view} copy")
    results = {}
    for name in a.clips.split(","):
        spec = CLIPS[name]
        fr = video.read_gray(spec["path"], fps=spec["fps"], scale=spec["scale"])
        T, H, W = fr.shape; fps = spec["fps"]; up = spec["up"]; box = spec["box"]
        # calibrate the EVS-style rule to the certified drop rate on the untouched clip
        d_cert = drop_rate(NativeSequentialPruner(H, W, P, multiscale=sched, z=1e9, offset=a.offset).run(fr)["keep"])
        tau, _, d_evs = bl.calibrate(bl.consecutive_mean, fr[: min(T, 600)], P, d_cert)
        print(f"\n=== {name}: {T} frames {W}x{H}; certified drop {d_cert:.3f}, EVS-style tau {tau:.2f} (drop {d_evs:.3f})")
        res = {"drop_cert": d_cert, "drop_evs": d_evs, "E1": [], "E2": []}
        # ---------------- E1: slow fade-in, end to end
        hold = int(2 * fps); pre = int(2 * fps)
        fade_s = min(a.fade_s, (T - pre - hold - 2) / fps)        # short clips get a shorter fade
        nf = int(fade_s * fps)
        print(f"  fade {fade_s:.1f} s over {nf} frames (per-frame step {a.contrast / nf:.2f} grey levels)")
        y0f, y1f, x0f, x1f = spec["floor"]
        for trial in range(a.trials):
            t0 = int(rng.integers(pre, max(pre + 1, T - nf - hold)))
            y = int(rng.integers(y0f, max(y0f + 1, y1f - box))); x = int(rng.integers(x0f, max(x0f + 1, x1f - box)))
            sign = -1 if fr[t0, y:y + box, x:x + box].mean() > 110 else 1          # contrast against the local surroundings
            seg = fr[t0 - pre: t0 + nf + hold].astype(np.int32).copy()
            ramp = np.concatenate([np.zeros(pre), np.linspace(0, a.contrast, nf), np.full(hold, a.contrast)])
            for i, c in enumerate(ramp):
                seg[i, y:y + box, x:x + box] += int(round(sign * c))
            seg = np.clip(seg, 0, 255).astype(np.uint8)
            views = run_views(seg, sched, tau, a.offset, held, pub)
            last = len(seg) - 1
            word = "dark" if sign < 0 else "bright"
            q = f"Is there a {word} square patch on the floor or wall in this image? Answer yes or no."
            answers = {"truth": vlm.ask(upscale(seg[last], up), q), "control": vlm.ask(upscale(fr[t0 + nf + hold - 1], up), q)}
            for rule in rules:
                answers[rule] = vlm.ask(upscale(views[rule][last], up), q)
            # whether the object is present in each view, by pixels (ground truth of what the model was shown)
            present = {rule: float(np.abs(views[rule][last][y:y + box, x:x + box].astype(int) - fr[t0 + nf + hold - 1][y:y + box, x:x + box].astype(int)).mean() > a.contrast / 2) for rule in rules}
            rec = {"trial": trial, "sign": sign, "answers": answers, "yes": {k: yes_no(v) for k, v in answers.items()}, "present_in_view": present}
            res["E1"].append(rec)
            print(f"  E1 trial {trial:2d} {word:6s}: truth={answers['truth']!r:6} control={answers['control']!r:6} evs={answers['evs']!r:6} cert={answers['certified']!r:6} seq={answers['sequential']!r:6}{f" pub={answers['published']!r:6}" if pub else ""}  | object pixels in view: evs {int(present['evs'])} cert {int(present['certified'])} seq {int(present['sequential'])}{f" pub {int(present['published'])}" if pub else ""}", flush=True)
        # ---------------- E2: agreement on untouched footage
        views = run_views(fr[: min(T, 900)], sched, tau, a.offset, held) if not a.skip_e2 else None
        idx = np.linspace(30, min(T, 900) - 1, a.e2_frames).astype(int) if not a.skip_e2 else []
        for t in idx:
            q1 = "How many people are visible in this image? Answer with a single number."
            q2 = "Is there a person visible in this image? Answer yes or no."
            rec = {"t": int(t)}
            for key, img in (("truth", fr[t]),) + tuple((rule, views[rule][t]) for rule in ("evs", "certified", "sequential")):
                rec[key] = {"count": number(vlm.ask(upscale(img, up), q1)), "person": yes_no(vlm.ask(upscale(img, up), q2))}
            res["E2"].append(rec)
            print(f"  E2 t={t:4d}: count truth/evs/cert/seq = {rec['truth']['count']}/{rec['evs']['count']}/{rec['certified']['count']}/{rec['sequential']['count']}  person = {rec['truth']['person']}/{rec['evs']['person']}/{rec['certified']['person']}/{rec['sequential']['person']}", flush=True)
        results[name] = res
        # summary
        e1 = res["E1"]
        print(f"  E1 yes-rate: truth {np.mean([r['yes']['truth'] == 1 for r in e1]):.2f}  control {np.mean([r['yes']['control'] == 1 for r in e1]):.2f}  "
              f"evs {np.mean([r['yes']['evs'] == 1 for r in e1]):.2f}  certified {np.mean([r['yes']['certified'] == 1 for r in e1]):.2f}  sequential {np.mean([r['yes']['sequential'] == 1 for r in e1]):.2f}   "
              f"(object actually in view: evs {np.mean([r['present_in_view']['evs'] for r in e1]):.2f} cert {np.mean([r['present_in_view']['certified'] for r in e1]):.2f} seq {np.mean([r['present_in_view']['sequential'] for r in e1]):.2f})")
        perc = [r for r in e1 if r['yes']['truth'] == 1 and r['yes']['control'] != 1]
        if perc:
            print(f"  E1 conditional on the model perceiving the object in the truth frame ({len(perc)} trials): "
                  f"evs {np.mean([r['yes']['evs'] == 1 for r in perc]):.2f}  certified {np.mean([r['yes']['certified'] == 1 for r in perc]):.2f}  sequential {np.mean([r['yes']['sequential'] == 1 for r in perc]):.2f}")
        if pub:
            print(f"  E1 published rule, held copy: object in view {np.mean([r['present_in_view']['published'] for r in e1]):.2f}, yes-rate {np.mean([r['yes']['published'] == 1 for r in e1]):.2f}"
                  + (f", conditional {np.mean([r['yes']['published'] == 1 for r in perc]):.2f}" if perc else ""))
        e2 = res["E2"]
        if not e2:
            Path(a.out + ".json").write_text(json.dumps(results | {name: res}, indent=1)); results[name] = res; continue
        for rule in ("evs", "certified", "sequential"):
            agree_c = np.mean([r[rule]["count"] == r["truth"]["count"] for r in e2]); agree_p = np.mean([r[rule]["person"] == r["truth"]["person"] for r in e2])
            mad = np.mean([abs(r[rule]["count"] - r["truth"]["count"]) for r in e2 if r[rule]["count"] >= 0 and r["truth"]["count"] >= 0])
            print(f"  E2 {rule:10s}: count agreement {agree_c:.2f} (MAD {mad:.2f}), person agreement {agree_p:.2f}")
        Path(a.out + ".json").write_text(json.dumps(results, indent=1))
    print(f"\n{vlm.n} queries, {vlm.t / max(vlm.n, 1):.1f} s per query on {vlm.dev}; wrote {a.out}.json")


if __name__ == "__main__":
    main()
