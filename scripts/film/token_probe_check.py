#!/usr/bin/env python3
"""Sanity checks for token_probe.py on one shot: (a) tokens fed through the hook give the same answer as the untouched
model; (b) how much of the first-token mass the yes/no readout stands on; (c) the jitter between adjacent frames with
all tokens new, as a noise floor for the gaps between feeding schemes.

    .venv/bin/python scripts/film/token_probe_check.py <dir> 201
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[1] / "python"))
import numpy as np, torch
from PIL import Image
from token_levels import GH, GW, PW, PH
from token_probe import PROBES
d, tag = sys.argv[1], sys.argv[2]; col = np.load(f"{d}/sam{tag}.npz")["col"]; toks = np.load(HERE.parents[1] / "out" / f"_toks{tag}.npy"); have = set(np.nonzero(np.abs(toks).sum((1, 2)) > 0)[0].tolist())
from e2e_vlm import VLM
vlm = VLM(); m = vlm.model; vis = getattr(m, "visual", None) or m.model.visual; big = lambda a: Image.fromarray(a).resize((PW, PH), Image.BICUBIC)
tk = vlm.proc.tokenizer; words = {k: [i for i in range(len(tk)) if tk.decode([i]).strip().lower() == k] for k in ("yes", "no")}
def prep(img, q):
    text = vlm.proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}], add_generation_prompt=True)
    return vlm.proc(text=[text], images=[img], return_tensors="pt").to(vlm.dev)
state = {}; hk = vis.register_forward_hook(lambda mod, a, o: setattr(o, "pooler_output", torch.from_numpy(state["grid"]).to(o.pooler_output.device, o.pooler_output.dtype)) if "grid" in state else None)
def first(t, q, grid=None):
    state.clear()
    if grid is not None:
        state["grid"] = np.ascontiguousarray(grid)
    with torch.no_grad():
        p = torch.softmax(m(**prep(big(col[t]), q)).logits[0, -1].float(), -1).cpu().numpy()
    y, no = p[words["yes"]].sum(), p[words["no"]].sum(); top = np.argsort(-p)[:4]
    return y / (y + no), y + no, [(tk.decode([int(i)]), round(float(p[i]), 3)) for i in top]
qs = [q for k, q in PROBES[tag]["q"] if k == "yesno"]
print("(a) hook with the frame's own tokens against the untouched model:")
for q in qs[:2]:
    f0 = PROBES[tag]["frames"][2]; a = first(f0, q); b = first(f0, q, toks[f0]); print(f"   untouched P(yes) {a[0]:.3f}  hooked {b[0]:.3f}  | {q.split(' Answer')[0]}")
print("(b) share of the first-token probability on yes or no, and the four likeliest first tokens:")
for t in (PROBES[tag]["frames"][0], PROBES[tag]["frames"][-1]):
    for q in qs:
        r = first(t, q, toks[t]); print(f"   frame {t:>3}: P(yes) {r[0]:.2f}, yes+no mass {r[1]:.2f}, top {r[2]} | {q.split(' Answer')[0]}")
print("(c) adjacent frames, all tokens new (natural jitter):")
for centre in (PROBES[tag]["frames"][1], PROBES[tag]["frames"][-1]):
    fr = [t for t in range(centre - 3, centre + 4) if t in have]
    for q in qs[:2]:
        vals = [first(t, q, toks[t])[0] for t in fr]; print(f"   frames {fr}: P(yes) {[round(float(v), 2) for v in vals]}, spread {max(vals) - min(vals):.2f} | {q.split(' Answer')[0]}")
hk.remove()
