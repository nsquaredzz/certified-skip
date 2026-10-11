#!/usr/bin/env python3
"""Two processing steps against the model's frame-to-frame flicker, on dense windows of one shot, no training.

  input step    certified token reuse (the still level of token_levels.py): the model's input changes only where a
                certificate failed, so its answer can only move where something certified changed
  output step   a sequential change detector on the answer's log-odds (the quickest-change idea of THEORY.md
                section 6): the declared answer flips only when the evidence for the other answer, summed over frames,
                passes a threshold h. False flips are then rare; the price is a delay of a few frames at real events.

Every frame of each window is probed with all tokens new and with still tokens reused. Reported per window and
question: the spread and the number of sign flips of the raw probability, and of the declared answer after the
detector.

    .venv/bin/python scripts/film/stabilize_test.py <dir> 201 <out.json>
"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[1] / "python"))
import numpy as np, torch
from PIL import Image
from looks_lib import flow, gauss
from token_levels import levels, GH, GW, PW, PH
WINDOWS = {"201": ((55, 64), (114, 123))}
QS = {"201": ("Is there a grey loudspeaker visible in this image? Answer yes or no.", "Is the woman visible in this image? Answer yes or no.")}
H_THRESH = 4.0


def detector(p, h=H_THRESH, eps=1e-3):
    """Declared answer per frame: starts at the first frame's sign; flips when the summed log-odds against the
    current answer passes h (one-sided CUSUM, reset at a flip)."""
    s = np.log((p + eps) / (1 - p + eps)); state = s[0] > 0; E = 0.0; out = []
    for v in s:
        E = max(0.0, E + (-v if state else v))
        if E > h:
            state = not state; E = 0.0
        out.append(state)
    return np.array(out)


def main():
    d, tag, out = sys.argv[1], sys.argv[2], sys.argv[3]; col = np.load(f"{d}/sam{tag}.npz")["col"]; T, H, W, _ = col.shape
    grey = col.astype(np.float32).mean(3); smooth = np.stack([gauss(x, 1) for x in grey]); flows = np.load(f"{d}/flows{tag}.npy")
    lev, src = levels(grey, smooth, flows, False); frames = [t for a, b in WINDOWS[tag] for t in range(a, b + 1)]
    cache = HERE.parents[1] / "out" / f"_toks{tag}.npy"; toks = np.load(cache); have = set(np.nonzero(np.abs(toks).sum((1, 2)) > 0)[0].tolist())
    need = sorted(set(frames) | {int(f) for t in frames for f in np.unique(src[t, :, 0])}); todo = [t for t in need if t not in have]
    from e2e_vlm import VLM
    vlm = VLM(); m = vlm.model; vis = getattr(m, "visual", None) or m.model.visual; big = lambda a: Image.fromarray(a).resize((PW, PH), Image.BICUBIC)
    def prep(img, q):
        text = vlm.proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}], add_generation_prompt=True)
        return vlm.proc(text=[text], images=[img], return_tensors="pt")
    grab = {}; hk = vis.register_forward_hook(lambda mod, a, o: grab.__setitem__("t", o.pooler_output.detach().float().cpu().numpy().astype(np.float16)))
    print(f"encoding {len(todo)} frames", flush=True)
    with torch.no_grad():
        for t in todo:
            i = prep(big(col[t]), "x"); vis(i["pixel_values"].to(vlm.dev, torch.float16), grid_thw=i["image_grid_thw"].to(vlm.dev)); toks[t] = grab["t"]
    hk.remove(); np.save(cache, toks)
    tk = vlm.proc.tokenizer; words = {k: [i for i in range(len(tk)) if tk.decode([i]).strip().lower() == k] for k in ("yes", "no")}
    state = {}; hk = vis.register_forward_hook(lambda mod, a, o: setattr(o, "pooler_output", torch.from_numpy(state["grid"]).to(o.pooler_output.device, o.pooler_output.dtype)))
    res = {}; t0 = time.time(); n = 0
    for q in QS[tag]:
        res[q] = {"full": {}, "still": {}}
        for t in frames:
            for k, grid in (("full", toks[t]), ("still", toks[src[t, :, 0], src[t, :, 1]])):
                state["grid"] = np.ascontiguousarray(grid); i = prep(big(col[t]), q).to(vlm.dev)
                with torch.no_grad():
                    p = torch.softmax(m(**i).logits[0, -1].float(), -1).cpu().numpy()
                y, no = p[words["yes"]].sum(), p[words["no"]].sum(); res[q][k][t] = float(y / (y + no)); n += 1
            print(f"  frame {t:>3}: all new {res[q]['full'][t]:.2f}  still reused {res[q]['still'][t]:.2f} | {q.split(' Answer')[0]}", flush=True)
    hk.remove(); print(f"{n} forward passes, {(time.time() - t0) / n:.1f} s each")
    flips = lambda b: int((np.diff(b.astype(int)) != 0).sum())
    for q in QS[tag]:
        for a, b in WINDOWS[tag]:
            ts = list(range(a, b + 1)); line = f"  frames {a}-{b} | {q.split(' Answer')[0]}:"
            for k in ("full", "still"):
                p = np.array([res[q][k][t] for t in ts]); dec = detector(p)
                line += f"  {k}: spread {p.max() - p.min():.2f}, raw flips {flips(p > 0.5)}, after detector {flips(dec)};"
            print(line)
    json.dump({q: {k: {str(t): v for t, v in d_.items()} for k, d_ in r.items()} for q, r in res.items()}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
