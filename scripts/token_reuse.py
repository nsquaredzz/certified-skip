#!/usr/bin/env python3
"""Token reuse inside the model: the vision encoder sees only the blocks the rule sends.

Part VII (e2e_vlm.py) showed the model an image. Here nothing is recomposed into an image. Qwen2-VL keeps one
token per 28x28 block of its input (2x2 patches of 14 px). The skip rule runs on those blocks, on the frame resized
exactly as the model's processor resizes it, so a patch of the rule is a token of the model. On each frame the
vision encoder is run on the blocks the rule keeps and on nothing else; every other token is the one computed when
its block was last sent. The language model answers from that mixed set of tokens.

Rule: quotient + multi-scale + sequential (z' = 8) without the per-patch brightness offset, so that what is
certified is the copy the model holds (THEORY.md section 9), patch 28.

On each query frame three sets of tokens go through the same language-model call:
  full    every block encoded from the current frame (the reference; what an unpruned model computes)
  view    every block encoded from the image of held copies (Part VII's protocol: no saving, isolates the pruning)
  reuse   cached tokens, refreshed only where the rule kept (the saving is real: the encoder ran on those blocks only)
A kept block is encoded together with the other kept blocks of its frame (--context none) or, in addition, with
the ring of blocks around them, which are encoded for context and not stored (--context ring).

E2  untouched footage: "How many people are visible?" and "Is there a person visible?", agreement with `full`.
E1  (--trials n) the fade-in square of Part VII, same geometry: does it reach the answer through reused tokens?

    .venv/bin/python scripts/token_reuse.py [--clips hall,lobby,virat] [--frames 15] [--trials 0] [--context ring] [--out out/token_reuse]
"""
import argparse, json, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from certskip import video
from certskip.native import NativeSequentialPruner
from e2e_vlm import VLM, CLIPS, upscale, yes_no, number
from real_bench import schedule

B = 28                                   # one token of the model: 2x2 patches of 14 px


# ------------------------------------------------------------------ model
class TokenVLM(VLM):
    """The model of Part VII with its vision encoder opened: encode any subset of blocks, answer from any tokens."""

    def __init__(self, name="Qwen/Qwen2-VL-2B-Instruct"):
        super().__init__(name)
        from transformers.vision_utils import get_vision_position_ids
        self._positions = get_vision_position_ids
        self.core = self.model.model; self.vis = self.core.visual; self.vdev = self.vis.get_device()
        self.last_enc = 0.0

    def pixels(self, image):
        out = self.proc.image_processor(images=[Image.fromarray(image).convert("RGB")], return_tensors="pt")
        return out["pixel_values"], out["image_grid_thw"]          # (4 * blocks, 3*2*14*14) in block order, (1, 3)

    def size(self, H, W):
        g = self.pixels(np.zeros((H, W), np.uint8))[1][0].tolist()
        return g[1] * 14, g[2] * 14                                 # the size the processor resizes an H x W image to

    def encode(self, image, blocks=None):
        """Tokens of every block of the image, or of the listed blocks (row-major indices) encoded together."""
        torch = self.torch
        pix, grid = self.pixels(image)
        pos = self._positions(grid, self.vis.spatial_merge_size)   # (4 * blocks, 2): row and column of each patch
        if blocks is not None:
            rows = (torch.as_tensor(blocks)[:, None] * 4 + torch.arange(4)).flatten()
            pix, pos = pix[rows], pos[rows]
        t0 = time.perf_counter()
        with torch.no_grad():
            h = self.vis.patch_embed(pix.to(self.vdev))
            pe = self.vis.rotary_pos_emb(h, pos.to(self.vdev))
            cu = torch.tensor([0, h.shape[0]], device=self.vdev, dtype=torch.int32)
            for blk in self.vis.blocks:
                h = blk(h, cu_seqlens=cu, position_embeddings=pe)
            tok = self.vis.merger(h)
        if self.vdev.type == "cuda":
            torch.cuda.synchronize()
        self.last_enc = time.perf_counter() - t0
        return tok

    def ask_tokens(self, tokens, image, question, max_new_tokens=6):
        """VLM.ask with the image tokens supplied by the caller in place of the vision encoder's output."""
        self.core.get_image_features = lambda *a, **k: SimpleNamespace(pooler_output=(tokens,))
        try:
            return self.ask(image, question, max_new_tokens)
        finally:
            del self.core.get_image_features


# ------------------------------------------------------------------ stream
def ring(keep):
    """Kept blocks and their eight neighbours."""
    p = np.pad(keep, 1); h, w = keep.shape
    return np.logical_or.reduce([p[1 + dy: 1 + dy + h, 1 + dx: 1 + dx + w] for dy in (-1, 0, 1) for dx in (-1, 0, 1)])


class Reuse:
    """The token cache of one stream: each block's token as it was computed when the block was last sent."""

    def __init__(self, vlm, first, sched, context="ring", z=8.0):
        H, W = first.shape
        self.vlm, self.context = vlm, context
        self.pr = NativeSequentialPruner(H, W, B, multiscale=sched, z=z, offset="none")
        self.pr.step(first); self.cache = vlm.encode(first)
        self.sent, self.encoded, self.t_enc, self.t_all = [], [], [], []

    def step(self, frame):
        torch = self.vlm.torch; t0 = time.perf_counter(); enc_t = 0.0
        keep = self.pr.step(frame)[0]
        enc = ring(keep) if self.context == "ring" and keep.any() else keep
        if keep.any():
            blocks = np.flatnonzero(enc)
            tok = self.vlm.encode(frame, blocks); enc_t = self.vlm.last_enc
            dst = torch.as_tensor(np.flatnonzero(keep), device=tok.device)
            src = torch.as_tensor(np.flatnonzero(keep.ravel()[blocks]), device=tok.device)
            self.cache[dst] = tok[src].to(self.cache.dtype)
        self.sent.append(float(keep.mean())); self.encoded.append(float(enc.mean()))
        self.t_enc.append(enc_t); self.t_all.append(time.perf_counter() - t0)
        return keep

    def held(self):
        return self.pr.view(False)


def to_model(frame, up, size):
    """A frame as the processor would hand it to the model: Part VII's upscaling, then the processor's own resize."""
    return np.asarray(Image.fromarray(upscale(frame, up)).resize((size[1], size[0]), Image.BICUBIC))


def cosine(vlm, a, b):
    c = vlm.torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=1)
    return float(c.mean()), float(c.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="hall,lobby,virat"); ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--trials", type=int, default=0); ap.add_argument("--max-frames", type=int, default=900)
    ap.add_argument("--context", choices=("none", "ring"), default="ring"); ap.add_argument("--out", default="out/token_reuse")
    ap.add_argument("--fade-s", type=float, default=8.0); ap.add_argument("--contrast", type=int, default=70)
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    sched = schedule(80.0, 0.5)
    vlm = TokenVLM(a.model)
    print(f"model on {vlm.dev}; context: {a.context}")
    results = {}
    for name in a.clips.split(","):
        spec = CLIPS[name]; up = spec["up"]; fps = spec["fps"]
        fr = video.read_gray(spec["path"], fps=fps, scale=spec["scale"])
        T, H, W = fr.shape
        size = vlm.size(H * up, W * up); gh, gw = size[0] // B, size[1] // B
        print(f"\n=== {name}: {T} frames {W}x{H}; model input {size[1]}x{size[0]}, {gh}x{gw} = {gh * gw} tokens")
        res = {"tokens": gh * gw, "E1": [], "E2": []}
        # ---------------- E2: untouched footage, reused tokens against a full encode
        n = min(T, a.max_frames); idx = set(np.linspace(30, n - 1, a.frames).astype(int).tolist()) if a.frames else set()
        if idx:
            ru = Reuse(vlm, to_model(fr[0], up, size), sched, a.context); t_full = []
            for t in range(1, n):
                frame = to_model(fr[t], up, size); ru.step(frame)
                if t not in idx:
                    continue
                full = vlm.encode(frame); t_full.append(vlm.last_enc); view = vlm.encode(ru.held())
                toks = {"full": full, "view": view, "reuse": ru.cache}
                rec = {"t": t, "sent": ru.sent[-1], "cos_view": cosine(vlm, view, full), "cos_reuse": cosine(vlm, ru.cache, full)}
                for key, tok in toks.items():
                    rec[key] = {"count": number(vlm.ask_tokens(tok, frame, "How many people are visible in this image? Answer with a single number.")),
                                "person": yes_no(vlm.ask_tokens(tok, frame, "Is there a person visible in this image? Answer yes or no."))}
                res["E2"].append(rec)
                print(f"  E2 t={t:4d}: count full/view/reuse = {rec['full']['count']}/{rec['view']['count']}/{rec['reuse']['count']}  person = {rec['full']['person']}/{rec['view']['person']}/{rec['reuse']['person']}"
                      f"  | token cosine to full: view {rec['cos_view'][0]:.3f} (min {rec['cos_view'][1]:.2f})  reuse {rec['cos_reuse'][0]:.3f} (min {rec['cos_reuse'][1]:.2f})", flush=True)
            e2 = res["E2"]
            res["cost"] = dict(sent=float(np.mean(ru.sent)), encoded=float(np.mean(ru.encoded)), enc_s=float(np.mean(ru.t_enc)), step_s=float(np.mean(ru.t_all)), full_s=float(np.mean(t_full)))
            c = res["cost"]
            print(f"  cost per frame: blocks sent {100 * c['sent']:.1f}%, encoded {100 * c['encoded']:.1f}% (with context); vision encoder {1000 * c['enc_s']:.0f} ms against {1000 * c['full_s']:.0f} ms for the full frame "
                  f"({c['full_s'] / max(c['enc_s'], 1e-9):.0f}x); rule + preprocessing + encoder {1000 * c['step_s']:.0f} ms")
            for key in ("view", "reuse"):
                print(f"  E2 {key:5s}: count agreement {np.mean([r[key]['count'] == r['full']['count'] for r in e2]):.2f}, person agreement {np.mean([r[key]['person'] == r['full']['person'] for r in e2]):.2f}, "
                      f"token cosine {np.mean([r['cos_' + key][0] for r in e2]):.3f}")
        # ---------------- E1: the fade-in square of Part VII, through reused tokens
        box = spec["box"]; hold = pre = int(2 * fps); nf = int(min(a.fade_s, (T - pre - hold - 2) / fps) * fps); y0f, y1f, x0f, x1f = spec["floor"]
        for trial in range(a.trials):
            t0 = int(rng.integers(pre, max(pre + 1, T - nf - hold)))
            y = int(rng.integers(y0f, max(y0f + 1, y1f - box))); x = int(rng.integers(x0f, max(x0f + 1, x1f - box)))
            sign = -1 if fr[t0, y:y + box, x:x + box].mean() > 110 else 1
            seg = fr[t0 - pre: t0 + nf + hold].astype(np.int32).copy()
            ramp = np.concatenate([np.zeros(pre), np.linspace(0, a.contrast, nf), np.full(hold, a.contrast)])
            for i, c in enumerate(ramp):
                seg[i, y:y + box, x:x + box] += int(round(sign * c))
            seg = np.clip(seg, 0, 255).astype(np.uint8)
            ru = Reuse(vlm, to_model(seg[0], up, size), sched, a.context)
            for f in seg[1:]:
                last = to_model(f, up, size); ru.step(last)
            word = "dark" if sign < 0 else "bright"
            q = f"Is there a {word} square patch on the floor or wall in this image? Answer yes or no."
            toks = {"full": vlm.encode(last), "control": vlm.encode(to_model(fr[t0 + nf + hold - 1], up, size)), "view": vlm.encode(ru.held()), "reuse": ru.cache}
            answers = {k: vlm.ask_tokens(tok, last, q) for k, tok in toks.items()}
            rec = {"trial": trial, "sign": sign, "answers": answers, "yes": {k: yes_no(v) for k, v in answers.items()}, "sent": float(np.mean(ru.sent)), "encoded": float(np.mean(ru.encoded))}
            res["E1"].append(rec)
            print(f"  E1 trial {trial:2d} {word:6s}: full={answers['full']!r:6} control={answers['control']!r:6} view={answers['view']!r:6} reuse={answers['reuse']!r:6}  | blocks sent {100 * rec['sent']:.1f}%, encoded {100 * rec['encoded']:.1f}%", flush=True)
        e1 = res["E1"]
        if e1:
            perc = [r for r in e1 if r["yes"]["full"] == 1 and r["yes"]["control"] != 1]
            print(f"  E1 yes-rate: full {np.mean([r['yes']['full'] == 1 for r in e1]):.2f}  control {np.mean([r['yes']['control'] == 1 for r in e1]):.2f}  view {np.mean([r['yes']['view'] == 1 for r in e1]):.2f}  reuse {np.mean([r['yes']['reuse'] == 1 for r in e1]):.2f}")
            if perc:
                print(f"  E1 conditional on the model perceiving the object in the full frame ({len(perc)} trials): view {np.mean([r['yes']['view'] == 1 for r in perc]):.2f}  reuse {np.mean([r['yes']['reuse'] == 1 for r in perc]):.2f}")
        results[name] = res
        Path(a.out + ".json").write_text(json.dumps(results, indent=1))
    print(f"\n{vlm.n} queries, {vlm.t / max(vlm.n, 1):.1f} s per query on {vlm.dev}; wrote {a.out}.json")


if __name__ == "__main__":
    main()
