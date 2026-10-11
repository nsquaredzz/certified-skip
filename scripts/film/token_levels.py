#!/usr/bin/env python3
"""Three levels of movement, in the token space of a frozen video LLM (Qwen2-VL-2B).

Every token cell of every frame of a film shot is put in one of three levels:
  still   the content under it has not changed and has not moved: the token the model already holds is kept
  moved   the content came from somewhere else and is still certified along its path: the token it had there is used
  new     neither: the token is taken from an encoding of the current frame
Certified: tracked points (looks_lib.py) whose grey level is within DELTA of what it was when their token was made.
A cell is explained by a held token if points carrying that one token cover SHARE of the cell.

The language model is then asked questions about the people in the shot with, in place of the vision tower's output:
  full    the current frame's own tokens
  still   still tokens reused, everything else new
  moved   still and moved tokens reused, the rest new
  stale   the first frame's tokens, never refreshed (shows the questions depend on the frame)
New tokens are cut from an encoding of the whole current frame, so this measures what the reuse decisions do to the
answers, not the cost of encoding part of a frame.

    .venv/bin/python scripts/film/token_levels.py <dir with sam<tag>.npz> <tag> <out.npz>
"""
import os, sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[1] / "python"))
import numpy as np, torch
from PIL import Image
from looks_lib import DELTA, flow, gauss, sample
GH, GW, PW, PH, SHARE, STEP = 16, 39, 1092, 448, 0.6, 2
QUESTIONS = {
    "201": dict(frames=(30, 60, 90, 127), q=("How many people are visible in this image? Answer with a single number.",
                                             "Is the man in the red jacket holding a phone to his ear? Answer yes or no.",
                                             "Is there a woman's face visible in this image? Answer yes or no.",
                                             "Is the man in the red jacket on the left or on the right of the grey-haired man? Answer left or right.",
                                             "Is there a grey loudspeaker visible in this image? Answer yes or no.")),
    "356": dict(frames=(67, 134), q=("How many people are visible in this image? Answer with a single number.",
                                     "Is the young man looking down? Answer yes or no.",
                                     "Is the young man's mouth open? Answer yes or no.")),
}


def levels(grey, smooth, flows, use_moved):
    """Per frame and cell: the level (0 still, 1 moved, 2 new) and the token used, as (frame, cell) of its encoding."""
    T, H, W = grey.shape; ys, xs = np.mgrid[STEP // 2:H:STEP, STEP // 2:W:STEP]; g0 = np.stack([ys.ravel(), xs.ravel()], 1).astype(np.float32)
    cell = lambda p: np.clip((p[:, 0] * GH / H).astype(int), 0, GH - 1) * GW + np.clip((p[:, 1] * GW / W).astype(int), 0, GW - 1)
    home = cell(g0); expected = np.bincount(home, minlength=GH * GW); lev = np.full((T, GH * GW), 2, np.uint8); src = np.zeros((T, GH * GW, 2), np.int32)
    src[0, :, 1] = np.arange(GH * GW); p = g0.copy(); anchor = g0.copy(); tok = home.copy(); g = smooth[0][g0[:, 0].astype(int), g0[:, 1].astype(int)].copy()   # tok: frame * cells + cell
    for t in range(1, T):
        v = flows[t - 1].astype(np.float32); p = p + np.stack([sample(v[0], p[:, 0], p[:, 1]), sample(v[1], p[:, 0], p[:, 1])], 1)
        ok = (p[:, 0] >= 0) & (p[:, 0] <= H - 1) & (p[:, 1] >= 0) & (p[:, 1] <= W - 1); ok[ok] &= np.abs(sample(smooth[t], p[ok, 0], p[ok, 1]) - g[ok]) <= DELTA
        p, anchor, tok, g = p[ok], anchor[ok], tok[ok], g[ok]; c = cell(p); order = np.lexsort((tok, c)); cs, ts = c[order], tok[order]; moved = np.hypot(*(p - anchor).T)[order]
        cut = np.nonzero((np.diff(cs) != 0) | (np.diff(ts) != 0))[0] + 1; a = np.r_[0, cut]; b = np.r_[cut, len(cs)]; best = {}
        for i, j in zip(a.tolist(), b.tolist()):                # runs of points in one cell carrying one token
            k = int(cs[i])
            if k not in best or j - i > best[k][0]:
                best[k] = (j - i, int(ts[i]), float(moved[i:j].mean()))
        fresh = np.ones(GH * GW, bool)
        for k, (n, tk, d) in best.items():
            if n >= SHARE * expected[k]:
                still = tk % (GH * GW) == k and d < 2.0
                if still or use_moved:
                    lev[t, k] = 0 if still else 1; src[t, k] = (tk // (GH * GW), tk % (GH * GW)); fresh[k] = False
        src[t, fresh, 0] = t; src[t, fresh, 1] = np.nonzero(fresh)[0]
        keep = ~fresh[c]; add = fresh[home]                     # a new cell forgets what was in it and starts again from this frame
        p = np.concatenate([p[keep], g0[add]]); anchor = np.concatenate([anchor[keep], g0[add]]); tok = np.concatenate([tok[keep], t * GH * GW + home[add]])
        g = np.concatenate([g[keep], smooth[t][g0[add, 0].astype(int), g0[add, 1].astype(int)]])
    return lev.reshape(T, GH, GW), src


def main():
    d, tag, out = sys.argv[1], sys.argv[2], sys.argv[3]; col = np.load(f"{d}/sam{tag}.npz")["col"]; T, H, W, _ = col.shape
    grey = col.astype(np.float32).mean(3); smooth = np.stack([gauss(x, 1) for x in grey]); fp = Path(f"{d}/flows{tag}.npy")
    flows = np.load(fp) if fp.exists() else np.stack([flow(grey[t], grey[t + 1]) for t in range(T - 1)]); np.save(fp, flows)
    runs = {"still": levels(grey, smooth, flows, False), "moved": levels(grey, smooth, flows, True)}
    for k, (lev, _) in runs.items():
        print(f"{tag} {k}: share of tokens after the first frame: still {100 * (lev[1:] == 0).mean():.1f} %, moved {100 * (lev[1:] == 1).mean():.1f} %, new (encoded) {100 * (lev[1:] == 2).mean():.1f} %", flush=True)
    from e2e_vlm import VLM
    vlm = VLM(); m = vlm.model; vis = getattr(m, "visual", None) or m.model.visual; big = lambda a: Image.fromarray(a).resize((PW, PH), Image.BICUBIC)
    def prep(img, q):
        text = vlm.proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}], add_generation_prompt=True)
        return vlm.proc(text=[text], images=[img], return_tensors="pt")
    grab = {}
    h = vis.register_forward_hook(lambda mod, a, o: grab.__setitem__("t", o.pooler_output.detach().float().cpu().numpy().astype(np.float16)))
    toks = np.zeros((T, GH * GW, 1536), np.float16); t0 = time.time()
    with torch.no_grad():
        need = sorted({0, 1} | set(QUESTIONS[tag]["frames"]) | {int(f) for _, sr in runs.values() for t in QUESTIONS[tag]["frames"] for f in np.unique(sr[t, :, 0])})
        print(f"encoding the {len(need)} frames whose tokens are used at the frames asked about", flush=True)
        for t in need:
            i = prep(big(col[t]), "x"); assert i["image_grid_thw"].tolist() == [[1, 2 * GH, 2 * GW]], i["image_grid_thw"]
            vis(i["pixel_values"].to(vlm.dev, torch.float16), grid_thw=i["image_grid_thw"].to(vlm.dev)); toks[t] = grab["t"]
        # does a token survive being moved? the same picture shifted right by exactly one cell
        sh = np.asarray(big(col[0])); sh = np.concatenate([sh[:, :28], sh[:, :-28]], 1); i = prep(Image.fromarray(sh), "x")
        vis(i["pixel_values"].to(vlm.dev, torch.float16), grid_thw=i["image_grid_thw"].to(vlm.dev)); s = grab["t"].astype(np.float32).reshape(GH, GW, -1)
    h.remove(); print(f"encoded {len(need) + 1} frames, {(time.time() - t0) / (len(need) + 1):.2f} s a frame", flush=True)
    o = toks[0].astype(np.float32).reshape(GH, GW, -1); cos = lambda a, b: float(((a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))).mean())
    probe = dict(moved_vs_fresh=cos(o[:, 1:-1], s[:, 2:]), left_in_place=cos(o[:, 2:], s[:, 2:]), next_frame=cos(o, toks[1].astype(np.float32).reshape(GH, GW, -1)), unrelated=cos(o[:, :GW // 2], o[:, GW - GW // 2:][::-1]))
    print("cosine with the token a fresh encoding gives: a token carried one cell along with its content", round(probe["moved_vs_fresh"], 3), "| a token left where it was", round(probe["left_in_place"], 3),
          "| the same cell one frame later", round(probe["next_frame"], 3), "| an unrelated cell", round(probe["unrelated"], 3), flush=True)
    # how far the tokens the model would hold are from the current frame's own, over the frame and over the people (SAM 2's masks)
    ref = np.load(f"{d}/sam{tag}.npz")["masks"]; unit = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True); close = []
    for t in QUESTIONS[tag]["frames"]:
        full = unit(toks[t].astype(np.float32)); ppl = np.asarray(Image.fromarray(ref[t].any(0)).resize((GW, GH), Image.NEAREST)).ravel()
        row = dict(frame=t, people_cells=int(ppl.sum()))
        for k, g in (("still", toks[runs["still"][1][t, :, 0], runs["still"][1][t, :, 1]]), ("moved", toks[runs["moved"][1][t, :, 0], runs["moved"][1][t, :, 1]]), ("stale", toks[0])):
            c = (unit(g.astype(np.float32)) * full).sum(-1); row[k] = float(c.mean()); row[k + "_people"] = float(c[ppl].mean()) if ppl.any() else float("nan")
            row[k + "_new"] = float((runs[k][0][t] == 2).mean()) if k in runs else 0.0
        close.append(row); print(f"  frame {t:>3}: closeness to the frame's own tokens (1 = identical), whole frame / on the people: " + "   ".join(f"{k} {row[k]:.3f} / {row[k + '_people']:.3f}" for k in ("still", "moved", "stale")), flush=True)
    state = {}
    hk = vis.register_forward_hook(lambda mod, a, o: setattr(o, "pooler_output", torch.from_numpy(state["grid"]).to(o.pooler_output.device, o.pooler_output.dtype)) if "grid" in state else None)
    Q = QUESTIONS[tag]; rows = []; norm = lambda s: next((w for w in ("left", "right") if w in s.lower()), "".join(ch for ch in s.lower().split()[0] if ch.isalnum())) if s.strip() else ""
    for t in (() if os.environ.get("NOASK") else Q["frames"]):
        grids = {"full": toks[t], "stale": toks[0]} | {k: toks[src[t, :, 0], src[t, :, 1]] for k, (lev, src) in runs.items()}
        for q in Q["q"]:
            ans = {}; vlm_tokens = 12 if "left or right" in q else 6
            for k in ("full", "still", "moved", "stale"):
                state["grid"] = np.ascontiguousarray(grids[k]); ans[k] = norm(vlm.ask(np.asarray(big(col[t])), q, vlm_tokens))
            rows.append(dict(frame=t, q=q, **ans)); print(f"  frame {t:>3} | full {ans['full']:<6} still {ans['still']:<6} moved {ans['moved']:<6} stale {ans['stale']:<6} | {q}", flush=True)
    hk.remove()
    for k in ("still", "moved", "stale"):
        print(f"{tag}: answers the same as with full tokens, {k}: {sum(r[k] == r['full'] for r in rows)} of {len(rows)}")
    print(f"{vlm.n} queries, {vlm.t / max(vlm.n, 1):.1f} s each")
    np.savez_compressed(out, lev_still=runs["still"][0], lev_moved=runs["moved"][0], rows=json.dumps(rows), probe=json.dumps(probe), close=json.dumps(close))


if __name__ == "__main__":
    main()
