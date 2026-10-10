#!/usr/bin/env python3
"""Does the model mind being shown centres? (THEORY.md section 11, RESULTS.md Part X)

Holding the centre of the next L frames sends about half as many patches as holding the current frame, at the same
certificate. What the model holds is then the midrange of a few consecutive frames, an image that was never a
frame. This asks the model of Part VII the questions of its experiment E2 on those held copies:

  "How many people are visible?" and "Is there a person visible?", on the full frame and on the copy held by
  the multi-scale rule without offset (Delta_0 = 80, no sub-pixel shift) at latency L = 0 (send-on-delta: a frame
  is held), 2 and 8 frames. Agreement with the model's own answer on the full frame, and the share of patches sent.

    .venv/bin/python scripts/centre_vlm.py [--clips hall,lobby] [--frames 15] [--look 0,2,8] [--out out/centre_vlm]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from certskip import video, centre as ct
from e2e_vlm import VLM, CLIPS, upscale, yes_no, number
from real_bench import schedule

P = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="hall,lobby"); ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--look", default="0,2,8"); ap.add_argument("--out", default="out/centre_vlm")
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    a = ap.parse_args()
    look = [int(v) for v in a.look.split(",")]; sched = schedule(80.0, 0.5)
    # the held copies first (CPU), then the model
    plans = {}
    for name in a.clips.split(","):
        spec = CLIPS[name]; fr = video.read_gray(spec["path"], fps=spec["fps"], scale=spec["scale"])[:900]
        T, H, W = fr.shape; idx = np.linspace(30, T - 1, a.frames).astype(int).tolist(); held, sent = {}, {}
        for L in look:
            grab = {}
            out = ct.CentrePruner(H, W, P, multiscale=sched, lookahead=L).run(fr, on_frame=lambda s, k, h: grab.__setitem__(s, np.clip(np.rint(h), 0, 255).astype(np.uint8)) if s in idx else None)
            assert out["worst"] < 1.0
            held[L] = grab; sent[L] = float(100 * out["keep"][1:].mean())
        plans[name] = (fr, idx, held, sent)
        print(f"{name}: {T} frames {W}x{H}; patches sent per frame, %: " + ", ".join(f"L={L}: {sent[L]:.2f}" for L in look), flush=True)
    vlm = VLM(a.model)
    print(f"model on {vlm.dev}")
    results = {}
    for name, (fr, idx, held, sent) in plans.items():
        up = CLIPS[name]["up"]; rows = []
        print(f"\n=== {name}")
        for t in idx:
            rec = {"t": int(t)}; H, W = fr[t].shape; gh, gw = H // P, W // P
            for key, img in [("full", fr[t])] + [(f"L{L}", held[L][t]) for L in look]:
                if key != "full":                                   # the pruner works on whole patches; the margin, if any, is shown as it is
                    v = fr[t].copy(); v[: gh * P, : gw * P] = img; img = v
                rec[key] = {"count": number(vlm.ask(upscale(img, up), "How many people are visible in this image? Answer with a single number.")),
                            "person": yes_no(vlm.ask(upscale(img, up), "Is there a person visible in this image? Answer yes or no."))}
            rows.append(rec)
            print(f"  t={t:4d}: count full / " + " / ".join(f"L={L}" for L in look) + " = " + "/".join(str(rec[k]["count"]) for k in ["full"] + [f"L{L}" for L in look])
                  + "   person = " + "/".join(str(rec[k]["person"]) for k in ["full"] + [f"L{L}" for L in look]), flush=True)
        for L in look:
            k = f"L{L}"
            print(f"  latency {L}: patches sent {sent[L]:.2f}%; count agreement {np.mean([r[k]['count'] == r['full']['count'] for r in rows]):.2f}, person agreement {np.mean([r[k]['person'] == r['full']['person'] for r in rows]):.2f}")
        results[name] = {"sent": {str(L): sent[L] for L in look}, "rows": rows}
        Path(a.out + ".json").write_text(json.dumps(results, indent=1))
    print(f"\n{vlm.n} queries, {vlm.t / max(vlm.n, 1):.1f} s per query on {vlm.dev}; wrote {a.out}.json")


if __name__ == "__main__":
    main()
