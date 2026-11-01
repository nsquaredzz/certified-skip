#!/usr/bin/env python3
"""Moving cameras: the warp-group certificate on DAVIS 2017 and on a film with cuts.

DAVIS 2017 (trainval, 480p) has 90 clips shot with moving, often handheld cameras and a pixel mask
of the moving object(s) in every frame.  For each rule we report the share of patches dropped, the
share of *object* patches dropped (patches overlapping the annotated object) and of background
patches, the share of patches kept for lack of a reference (newly revealed area), the number of
frames treated as cuts, and the interpolation term iota.  Tears of Steel (CC-BY) adds real shot cuts.

Rules
  fixed range           core range rule, Delta = 32: the fixed-camera baseline
  fixed quotient+MS     sub-pixel quotient + multi-scale at Delta0 = 80 (the recommended fixed-camera rule)
  warp range            per-frame homography against a world canvas, range certificate at Delta = 32
  warp+refine range     the same plus a per-patch sub-pixel refinement (|delta| <= 1 px)
  warp+refine MS        the same with the multi-scale certificate at Delta0 = 80

Soundness: on a sample of dropped patches the certificate is re-checked with exact persistence
diagrams against the warped reference the rule constructed (Theorem A).

    python3 scripts/warp_bench.py --davis all --frames 40 --workers 6 --out out/warp_davis
    python3 scripts/warp_bench.py --film 120,360,540 --film-seconds 60 --out out/warp_film
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from certskip import core, video, warp, topology as tp
from certskip.warpcert import WarpGroupPruner

ROOT = Path(__file__).resolve().parents[1]
DAVIS = ROOT / "data/davis/DAVIS"
P = 16
PIX_BINS = np.array([-0.5, 0.5, 4, 8, 16, 32, 64, 128, 256, 1e9])   # per-pixel interpolation term, grey levels


def schedule(delta0, gamma=0.5, radii=(0, 1, 2, 3)):
    return {r: delta0 * (2 * r + 1) ** (-gamma) for r in radii}


def read_davis(clip, max_frames):
    from PIL import Image
    jd = DAVIS / "JPEGImages/480p" / clip; ad = DAVIS / "Annotations/480p" / clip
    names = sorted(p.name[:-4] for p in jd.glob("*.jpg"))[:max_frames]
    fr, ms = [], []
    for n in names:
        im = Image.open(jd / f"{n}.jpg").convert("L")
        fr.append(np.asarray(im, np.uint8))
        a = np.asarray(Image.open(ad / f"{n}.png"))
        ms.append(a > 0)
    fr = np.stack(fr); ms = np.stack(ms)
    H, W = fr.shape[1:]
    Hg, Wg = (H // P) * P, (W // P) * P
    return fr[:, :Hg, :Wg], ms[:, :Hg, :Wg]


def rules_for(H, W):
    return {
        "fixed range":        lambda: core.Pruner(H, W, P, 32.0),
        "fixed quotient+MS":  lambda: warp.WarpPruner(H, W, P, 32.0, 1.0, multiscale=schedule(80.0)),
        "warp range":         lambda: WarpGroupPruner(H, W, P, 32.0, model="homography", delta_max=0.0, margin=max(H, W) // 2),
        "warp+refine range":  lambda: WarpGroupPruner(H, W, P, 32.0, model="homography", delta_max=1.0, margin=max(H, W) // 2),
        "warp+refine MS":     lambda: WarpGroupPruner(H, W, P, multiscale=schedule(80.0), model="homography", delta_max=1.0, margin=max(H, W) // 2),
    }


def run_rule(name, factory, frames, masks=None, audit=0, rng=None):
    pr = factory()
    T, H, W = frames.shape; gh, gw = H // P, W // P
    keeps, iotas, cuts, norefs, pix_hist = [], [], [], [], []
    audited = {"checked": 0, "violations": 0, "worst": 0.0}
    t0 = time.perf_counter()
    for t in range(T):
        keep, score, c = pr.step(frames[t])
        keeps.append(keep)
        last = getattr(pr, "last", {})
        iotas.append(last.get("iota", np.zeros((gh, gw), np.float32)))
        if "iota_pix" in last and (~keep).any():
            pix = core.patch_grid(last["iota_pix"], P)[~keep].ravel()
            pix_hist.append(np.histogram(pix, bins=PIX_BINS)[0])
        cuts.append(bool(last.get("cut", False)))
        norefs.append(last.get("noref", np.zeros((gh, gw), bool)))
        if audit and t > 0 and isinstance(pr, WarpGroupPruner) and not last.get("cut", False):
            view = pr._view
            fg = core.patch_grid(frames[t].astype(np.float64), P); vg = core.patch_grid(view, P)
            dropped = list(zip(*np.where(~keep)))
            if dropped:
                for i in rng.choice(len(dropped), min(audit, len(dropped)), replace=False):
                    gy, gx = dropped[i]
                    rep = tp.check_patch(fg[gy, gx], vg[gy, gx] - c[gy, gx], 32.0)
                    audited["checked"] += 1; audited["violations"] += int(not rep["ok"])
                    if rep["half_spread"] > 0:
                        audited["worst"] = max(audited["worst"], max(rep["bottleneck_dark"], rep["bottleneck_bright"]) / rep["half_spread"])
    dt = (time.perf_counter() - t0) / T
    keep = np.stack(keeps); iota = np.stack(iotas); noref = np.stack(norefs)
    out = {"drop": core.drop_rate(keep), "ms_per_frame": 1000 * dt, "cuts": int(sum(cuts)),
           "noref": float(noref[1:].mean()), "iota_median": float(np.median(iota[1:][~keep[1:]])) if (~keep[1:]).any() else 0.0,
           "iota_p90": float(np.percentile(iota[1:][~keep[1:]], 90)) if (~keep[1:]).any() else 0.0}
    if pix_hist:
        h = np.sum(pix_hist, 0).astype(float); cdf = np.cumsum(h) / h.sum()
        out["iota_pix_frac_le"] = {str(b): float(cdf[i]) for i, b in enumerate(PIX_BINS[1:])}
    if masks is not None:
        obj = np.stack([core.patch_grid(m, P).any((2, 3)) for m in masks])
        k = keep[1:]; o = obj[1:]
        out["drop_object"] = float(1 - k[o].mean()) if o.any() else np.nan
        out["drop_background"] = float(1 - k[~o].mean()) if (~o).any() else np.nan
        out["object_patch_frac"] = float(o.mean())
    if audit:
        out["audit"] = audited
    return out


def davis_clip(args):
    clip, max_frames, audit = args
    rng = np.random.default_rng(0)
    frames, masks = read_davis(clip, max_frames)
    T, H, W = frames.shape
    res = {"T": T, "H": H, "W": W}
    for name, factory in rules_for(H, W).items():
        res[name] = run_rule(name, factory, frames, masks, audit=audit, rng=rng)
    return clip, res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--davis", default=None, help="'all' or comma-separated clip names")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--audit", type=int, default=2, help="dropped patches audited per frame (warp rules)")
    ap.add_argument("--film", default=None, help="comma-separated start seconds of excerpts of Tears of Steel")
    ap.add_argument("--film-seconds", type=float, default=60.0)
    ap.add_argument("--film-fps", type=float, default=6.0)
    ap.add_argument("--out", default="out/warp_bench")
    a = ap.parse_args()
    results = {}
    t_all = time.perf_counter()
    if a.davis:
        clips = sorted(p.name for p in (DAVIS / "JPEGImages/480p").iterdir() if p.is_dir()) if a.davis == "all" else a.davis.split(",")
        jobs = [(c, a.frames, a.audit) for c in clips]
        if a.workers > 1:
            from multiprocessing import Pool
            with Pool(a.workers) as pool:
                for clip, res in pool.imap_unordered(davis_clip, jobs):
                    results[clip] = res
                    print(f"{clip:24s} " + "  ".join(f"{k}: {v['drop']:.3f} (obj {v.get('drop_object', float('nan')):.2f}, bg {v.get('drop_background', float('nan')):.3f}, cuts {v['cuts']})" for k, v in res.items() if isinstance(v, dict)), flush=True)
                    Path(a.out + ".json").write_text(json.dumps(results, indent=1, default=float))
        else:
            for job in jobs:
                clip, res = davis_clip(job); results[clip] = res
                print(clip, {k: round(v["drop"], 3) for k, v in res.items() if isinstance(v, dict)}, flush=True)
                Path(a.out + ".json").write_text(json.dumps(results, indent=1, default=float))
        # summary
        names = [k for k in next(iter(results.values())) if isinstance(results[next(iter(results))][k], dict)]
        print("\nDAVIS summary (median over clips): rule | drop | drop object | drop background | noref | cuts/clip | iota median | ms/frame")
        for n in names:
            rows = [r[n] for r in results.values()]
            print(f"  {n:20s} | {np.median([r['drop'] for r in rows]):.3f} | {np.nanmedian([r.get('drop_object', np.nan) for r in rows]):.3f} | "
                  f"{np.nanmedian([r.get('drop_background', np.nan) for r in rows]):.3f} | {np.median([r['noref'] for r in rows]):.3f} | "
                  f"{np.mean([r['cuts'] for r in rows]):.2f} | {np.median([r['iota_median'] for r in rows]):.1f} | {np.median([r['ms_per_frame'] for r in rows]):.0f}")
            aud = [r[n].get("audit") for r in results.values() if r[n].get("audit")]
            if aud:
                print(f"      audit: {sum(x['checked'] for x in aud)} dropped patches checked, {sum(x['violations'] for x in aud)} violations, worst bottleneck/(s/2) {max(x['worst'] for x in aud):.3f}")
    if a.film:
        path = ROOT / "data/tears_of_steel_720p.mov"
        for start in [float(s) for s in a.film.split(",")]:
            frames = video.read_gray(str(path), fps=a.film_fps, scale=(960, 400), start=start, max_frames=int(a.film_seconds * a.film_fps)) \
                if "start" in video.read_gray.__code__.co_varnames else _read_excerpt(path, start, a.film_seconds, a.film_fps)
            T, H, W = frames.shape
            key = f"film@{start:g}s"
            results[key] = {"T": T, "H": H, "W": W}
            rng = np.random.default_rng(0)
            for name, factory in rules_for(H, W).items():
                r = run_rule(name, factory, frames, None, audit=a.audit, rng=rng)
                results[key][name] = r
                print(f"{key:12s} {name:20s} drop {r['drop']:.3f}  noref {r['noref']:.3f}  cuts {r['cuts']}  iota med {r['iota_median']:.1f}  {r['ms_per_frame']:.0f} ms/frame", flush=True)
            Path(a.out + ".json").write_text(json.dumps(results, indent=1, default=float))
    print(f"wrote {a.out}.json in {time.perf_counter()-t_all:.0f}s")


def _read_excerpt(path, start, seconds, fps, size=(960, 400)):
    """Grey frames of an excerpt through ffmpeg (video.read_gray has no start offset)."""
    import subprocess
    W, H = size
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start}", "-t", f"{seconds}", "-i", str(path),
           "-vf", f"fps={fps},scale={W}:{H},format=gray", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    n = len(raw) // (W * H)
    return np.frombuffer(raw[: n * W * H], np.uint8).reshape(n, H, W).copy()


if __name__ == "__main__":
    main()
