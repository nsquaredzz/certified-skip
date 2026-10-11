#!/usr/bin/env python3
"""Why the differences in token_levels.py are small: a graded readout on questions whose answer changes in the shot.

Instead of the first word the model says, this reads the probability it puts on "yes" against "no" (or on each
count), under the same four ways of feeding it: full, still, moved, stale. Tokens are cached in out/_toks<tag>.npy.

    .venv/bin/python scripts/film/token_probe.py <dir> <tag> <out.json>
"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[1] / "python"))
import numpy as np, torch
from PIL import Image
from looks_lib import flow, gauss
from token_levels import levels, GH, GW, PW, PH
PROBES = {
    "201": dict(frames=(20, 40, 60, 80, 100, 120), q=(("yesno", "Is there a grey loudspeaker visible in this image? Answer yes or no."),
                                                      ("yesno", "Is the woman visible in this image? Answer yes or no."),
                                                      ("yesno", "Is the man in the red jacket closer to the camera than the grey-haired man? Answer yes or no."),
                                                      ("count", "How many people are visible in this image? Answer with a single number."))),
    "356": dict(frames=(20, 50, 80, 110, 134), q=(("yesno", "Is the young man's mouth open? Answer yes or no."),
                                                  ("yesno", "Is the young man smiling? Answer yes or no."),
                                                  ("yesno", "Is the robot hand touching the young man? Answer yes or no."))),
}


def main():
    d, tag, out = sys.argv[1], sys.argv[2], sys.argv[3]; col = np.load(f"{d}/sam{tag}.npz")["col"]; T, H, W, _ = col.shape
    grey = col.astype(np.float32).mean(3); smooth = np.stack([gauss(x, 1) for x in grey]); fp = Path(f"{d}/flows{tag}.npy")
    flows = np.load(fp) if fp.exists() else np.stack([flow(grey[t], grey[t + 1]) for t in range(T - 1)]); np.save(fp, flows)
    runs = {"still": levels(grey, smooth, flows, False), "moved": levels(grey, smooth, flows, True)}
    from e2e_vlm import VLM
    vlm = VLM(); m = vlm.model; vis = getattr(m, "visual", None) or m.model.visual; big = lambda a: Image.fromarray(a).resize((PW, PH), Image.BICUBIC)
    def prep(img, q):
        text = vlm.proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}], add_generation_prompt=True)
        return vlm.proc(text=[text], images=[img], return_tensors="pt")
    P = PROBES[tag]; cache = Path(HERE.parents[1] / "out" / f"_toks{tag}.npy"); need = sorted({0} | set(P["frames"]) | {int(f) for _, sr in runs.values() for t in P["frames"] for f in np.unique(sr[t, :, 0])})
    toks = np.load(cache) if cache.exists() else np.zeros((T, GH * GW, 1536), np.float16); have = set(np.nonzero(np.abs(toks).sum((1, 2)) > 0)[0].tolist()) if cache.exists() else set()
    grab = {}; h = vis.register_forward_hook(lambda mod, a, o: grab.__setitem__("t", o.pooler_output.detach().float().cpu().numpy().astype(np.float16)))
    todo = [t for t in need if t not in have]; print(f"encoding {len(todo)} frames ({len(have)} cached)", flush=True); t0 = time.time()
    with torch.no_grad():
        for t in todo:
            i = prep(big(col[t]), "x"); vis(i["pixel_values"].to(vlm.dev, torch.float16), grid_thw=i["image_grid_thw"].to(vlm.dev)); toks[t] = grab["t"]
    h.remove(); np.save(cache, toks)
    if todo:
        print(f"encoded in {(time.time() - t0) / len(todo):.1f} s a frame", flush=True)
    tk = vlm.proc.tokenizer; words = {k: [i for i in range(len(tk)) if tk.decode([i]).strip().lower() == k] for k in ("yes", "no")} | {str(n): [i for i in range(len(tk)) if tk.decode([i]).strip() == str(n)] for n in range(0, 6)}
    state = {}; hk = vis.register_forward_hook(lambda mod, a, o: setattr(o, "pooler_output", torch.from_numpy(state["grid"]).to(o.pooler_output.device, o.pooler_output.dtype)))
    rows = []; t0 = time.time(); n = 0
    for t in P["frames"]:
        grids = {"full": toks[t], "stale": toks[0]} | {k: toks[src[t, :, 0], src[t, :, 1]] for k, (lev, src) in runs.items()}
        for kind, q in P["q"]:
            i = prep(big(col[t]), q).to(vlm.dev); rec = dict(frame=t, kind=kind, q=q)
            for k in ("full", "still", "moved", "stale"):
                state["grid"] = np.ascontiguousarray(grids[k])
                with torch.no_grad():
                    p = torch.softmax(m(**i).logits[0, -1].float(), -1).cpu().numpy()
                n += 1
                if kind == "yesno":
                    y, no = p[words["yes"]].sum(), p[words["no"]].sum(); rec[k] = float(y / (y + no + 1e-9))
                else:
                    c = np.array([p[words[str(v)]].sum() for v in range(6)]); rec[k] = (c / c.sum()).round(3).tolist()
            rows.append(rec)
            if kind == "yesno":
                print(f"  frame {t:>3} P(yes): full {rec['full']:.2f}  still {rec['still']:.2f}  moved {rec['moved']:.2f}  stale {rec['stale']:.2f} | {q.split(' Answer')[0]}", flush=True)
            else:
                print(f"  frame {t:>3} P(count): " + "  ".join(f"{k} {np.argmax(rec[k])} ({max(rec[k]):.2f})" for k in ("full", "still", "moved", "stale")) + f" | {q.split(' Answer')[0]}", flush=True)
    hk.remove(); print(f"{n} forward passes, {(time.time() - t0) / n:.1f} s each"); json.dump(rows, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
