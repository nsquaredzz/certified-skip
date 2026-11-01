#!/usr/bin/env python3
"""Moving cameras, first step: how much of a real moving-camera video is content the model already
holds, moved by the camera?

Two rules at the same thresholds (quotient + multi-scale, Delta_0 = 80, gamma = 1/2):

  none     the fixed-camera rule as it is, on the raw frames
  camera   certskip.moving.MovingPruner: the reference is kept in world coordinates and warped by an
           estimated camera pose before the same test
  follow   the camera rule, and a held patch that fails in place is tried again as held content
           moved by up to --search pixels; if that is certified the patch is not sent

and, with --strict, the camera rule with the single-pixel certificate (quotient + range, Delta = 32).

  davis   DAVIS 2017 trainval, 480p: 90 short clips from moving cameras with hand-labelled masks of
          the main objects.  Besides the share of patches sent, it reports where the patches that
          are still sent lie relative to the labels: on an object, next to one, or on background.
  film    Tears of Steel, 12 minutes with cuts and camera motion, no labels: the share of patches
          sent over the whole film, the anchors (cuts and lost alignment) and the frontier.

Patches a frame sends fall into three kinds: *anchor* (alignment failed, the whole frame is sent),
*frontier* (the model holds nothing there, mostly the edge the camera moves into) and *refresh*
(held, and the certificate failed).  A *followed* patch is not sent: the model is told where its
content came from.

    python3 scripts/fetch_data.py movie
    python3 scripts/moving_bench.py film  --out out/moving_film
    python3 scripts/moving_bench.py davis --root data/davis/DAVIS --strict --out out/moving_davis
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one thread per worker; the workers are the parallelism
import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from certskip import moving as mv, warp, video
from certskip.core import patch_grid
from certskip.native import NativeSequentialPruner

P = 16
SCHED = {r: 80.0 * (2 * r + 1) ** -0.5 for r in (0, 1, 2, 3)}
FILM = "data/tears_of_steel_720p.mov"


def fixed_rule(H, W):
    """The fixed-camera quotient + multi-scale rule; C++ if the library is built."""
    try:
        pr = NativeSequentialPruner(H, W, P, multiscale=SCHED, z=1e9)
    except RuntimeError:
        pr = warp.WarpPruner(H, W, P, 32.0, 1.0, multiscale=SCHED)
    return lambda f: pr.step(f)[0]


def dilate(m):
    p = np.pad(m, 1)
    return (p[:-2, :-2] | p[:-2, 1:-1] | p[:-2, 2:] | p[1:-1, :-2] | p[1:-1, 1:-1] | p[1:-1, 2:]
            | p[2:, :-2] | p[2:, 1:-1] | p[2:, 2:])


# ------------------------------------------------------------------ DAVIS
def davis_seq(job):
    name, root, strict, search = job
    from PIL import Image
    jd, ad = Path(root) / "JPEGImages" / "480p" / name, Path(root) / "Annotations" / "480p" / name
    files = sorted(jd.glob("*.jpg"))
    first = np.array(Image.open(files[0]).convert("L"))
    H, W = first.shape
    rules = {"none": fixed_rule(H, W), "camera": mv.MovingPruner(H, W, P, SCHED),
             "follow": mv.MovingPruner(H, W, P, SCHED, search=search)}
    if strict:
        rules["camera, range 32"] = mv.MovingPruner(H, W, P, {0: 32.0})
    cnt = {r: dict(total=0, sent=0, anchor_frames=0, anchor=0, frontier=0, refresh=0, followed=0, new_area=0.0, corr=0.0, aligned=0,
                   held_obj=0, refresh_obj=0, followed_obj=0, held_near=0, refresh_near=0, followed_near=0,
                   held_bg=0, refresh_bg=0, followed_bg=0, obj=0, sent_obj=0, bg=0, sent_bg=0) for r in rules}
    for t, fp in enumerate(files):
        f = first if t == 0 else np.array(Image.open(fp).convert("L"))
        keeps = {}
        for r, pr in rules.items():
            keeps[r] = pr(f) if r == "none" else pr.step(f)
        if t == 0:
            continue
        g = np.array(Image.open(ad / (fp.stem + ".png"))) > 0
        of = patch_grid(g, P).mean((2, 3))
        obj = of >= 0.5; near = dilate(of > 0) & ~obj; bg = ~obj & ~near
        for r, k in keeps.items():
            c = cnt[r]; c["total"] += k.size; c["sent"] += int(k.sum())
            c["obj"] += int(obj.sum()); c["sent_obj"] += int((k & obj).sum()); c["bg"] += int(bg.sum()); c["sent_bg"] += int((k & bg).sum())
            if r == "none":
                held = np.ones_like(k); refresh = k; fol = np.zeros_like(k)
            else:
                L = rules[r].last
                if L["anchor"]:
                    c["anchor_frames"] += 1; c["anchor"] += k.size
                    continue
                held = ~L["reveal"]; refresh = k & held; fol = L["followed"]
                c["frontier"] += int((~held).sum()); c["new_area"] += L["new_area"]; c["corr"] += L["corr"]; c["aligned"] += 1
            c["refresh"] += int(refresh.sum()); c["followed"] += int(fol.sum())
            for cls, m in (("obj", obj), ("near", near), ("bg", bg)):
                c["held_" + cls] += int((held & m).sum()); c["refresh_" + cls] += int((refresh & m).sum())
                c["followed_" + cls] += int((fol & m).sum())
    return name, len(files), (H, W), cnt


def run_davis(a):
    root = Path(a.root)
    names = sorted(p.name for p in (root / "JPEGImages" / "480p").iterdir() if p.is_dir())
    if a.limit:
        names = names[: a.limit]
    t0 = time.perf_counter()
    with Pool(a.workers) as pool:
        res = pool.map(davis_seq, [(n, str(root), a.strict, a.search) for n in names], chunksize=1)
    rules = list(res[0][3])
    lines = [f"DAVIS 2017 trainval 480p: {len(res)} clips, {sum(r[1] for r in res)} frames, patch {P}, "
             f"quotient + multi-scale at Delta_0 = 80 unless named otherwise, search {a.search} px; {time.perf_counter() - t0:.0f} s"]
    out = {"clips": {r[0]: {"frames": r[1], "size": r[2], "rules": r[3]} for r in res}, "summary": {}}
    for rule in rules:
        tot = {k: sum(r[3][rule][k] for r in res) for k in res[0][3][rule]}
        per = np.array([1 - r[3][rule]["sent"] / r[3][rule]["total"] for r in res])
        s = {"skip": 1 - tot["sent"] / tot["total"], "skip_clip_median": float(np.median(per)),
             "skip_clip_quartiles": [float(np.percentile(per, 25)), float(np.percentile(per, 75))],
             "sent_anchor": tot["anchor"] / tot["total"], "sent_frontier": tot["frontier"] / tot["total"],
             "sent_refresh": tot["refresh"] / tot["total"], "followed": tot["followed"] / tot["total"], "anchor_frames": tot["anchor_frames"],
             "new_area": tot["new_area"] / max(1, tot["aligned"]), "corr": tot["corr"] / max(1, tot["aligned"]),
             "followed_rate_obj": tot["followed_obj"] / max(1, tot["held_obj"]),
             "followed_rate_near": tot["followed_near"] / max(1, tot["held_near"]),
             "followed_rate_bg": tot["followed_bg"] / max(1, tot["held_bg"]),
             "refresh_rate_obj": tot["refresh_obj"] / max(1, tot["held_obj"]),
             "refresh_rate_near": tot["refresh_near"] / max(1, tot["held_near"]),
             "refresh_rate_bg": tot["refresh_bg"] / max(1, tot["held_bg"]),
             "refresh_on_obj": tot["refresh_obj"] / max(1, tot["refresh"]), "refresh_on_near": tot["refresh_near"] / max(1, tot["refresh"]),
             "refresh_on_bg": tot["refresh_bg"] / max(1, tot["refresh"]),
             "sent_rate_obj": tot["sent_obj"] / max(1, tot["obj"]), "sent_rate_bg": tot["sent_bg"] / max(1, tot["bg"]),
             "obj_share_of_patches": tot["obj"] / tot["total"]}
        out["summary"][rule] = s
        lines += [f"\n[{rule}]",
                  f"  skipped {s['skip']:.3f} of all patches (per clip: median {s['skip_clip_median']:.3f}, quartiles {s['skip_clip_quartiles'][0]:.3f} to {s['skip_clip_quartiles'][1]:.3f})",
                  f"  sent    anchor {s['sent_anchor']:.3f} ({s['anchor_frames']} frames)   frontier {s['sent_frontier']:.3f} (new area {s['new_area']:.3f} of a frame)   refresh {s['sent_refresh']:.3f}",
                  f"  refresh rate of held patches: on an object {s['refresh_rate_obj']:.3f}, next to one {s['refresh_rate_near']:.3f}, background {s['refresh_rate_bg']:.3f}",
                  f"  followed, not sent: {s['followed']:.3f} of all patches; of held patches on an object {s['followed_rate_obj']:.3f}, next to one {s['followed_rate_near']:.3f}, background {s['followed_rate_bg']:.3f}",
                  f"  where the refreshes are: object {s['refresh_on_obj']:.3f}, next to one {s['refresh_on_near']:.3f}, background {s['refresh_on_bg']:.3f}"
                  f"   (object patches are {s['obj_share_of_patches']:.3f} of all)",
                  f"  sent, all kinds: {s['sent_rate_obj']:.3f} of object patches, {s['sent_rate_bg']:.3f} of background patches"]
    lines.append("\nper clip, skipped (none -> camera -> follow), anchors, mean correlation after alignment:")
    for r in sorted(res, key=lambda r: 1 - r[3]["follow"]["sent"] / r[3]["follow"]["total"]):
        cn, cc, cf = r[3]["none"], r[3]["camera"], r[3]["follow"]
        lines.append(f"  {r[0]:22s} {r[1]:3d} frames  {1 - cn['sent'] / cn['total']:.3f} -> {1 - cc['sent'] / cc['total']:.3f} -> {1 - cf['sent'] / cf['total']:.3f}"
                     f"   anchors {cf['anchor_frames']:2d}   corr {cf['corr'] / max(1, cf['aligned']):.2f}")
    return out, lines


# ------------------------------------------------------------------ film
def film_chunk(job):
    start, n, search = job
    rules = None; rows = []
    for t, f in enumerate(video.iter_gray(FILM, start=start, max_frames=n)):
        if rules is None:
            H, W = f.shape
            rules = (fixed_rule(H, W), mv.MovingPruner(H, W, P, SCHED), mv.MovingPruner(H, W, P, SCHED, search=search))
        k0 = rules[0](f); k1 = rules[1].step(f); L = rules[1].last; k2 = rules[2].step(f); L2 = rules[2].last
        if t == 0:
            continue                      # every chunk starts with a frame sent whole, like every clip
        rows.append((k0.mean(), k1.mean(), float(L["anchor"]), 0.0 if L["anchor"] else L["reveal"].mean(),
                     0.0 if L["anchor"] else L["new_area"], L["corr"], k2.mean(), L2["followed"].mean(), float(L2["anchor"])))
    return start, np.array(rows, np.float32)


def run_film(a):
    info = video.probe(FILM)
    fps, dur = info["fps"], info["duration"]
    jobs = [(float(s), int(round(min(a.chunk, dur - s) * fps)), a.search) for s in np.arange(0, dur, a.chunk)]
    if a.limit:
        jobs = jobs[: a.limit]
    t0 = time.perf_counter()
    with Pool(a.workers) as pool:
        res = sorted(pool.map(film_chunk, jobs, chunksize=1), key=lambda r: r[0])
    R = np.concatenate([r[1] for r in res])
    none, cam, anc, fro, area, corr, fol, fold, anc2 = (R[:, i].astype(np.float64) for i in range(9))
    al = anc == 0
    seg = np.diff(np.flatnonzero(np.concatenate([[1], anc, [1]])))          # frames between anchors
    w = int(30 * fps)
    win = [(1 - none[i:i + w].mean(), 1 - cam[i:i + w].mean(), 1 - fol[i:i + w].mean()) for i in range(0, len(R) - w + 1, w)]
    s = {"frames": int(len(R)), "size": [info["width"], info["height"]], "fps": fps, "chunks": len(jobs),
         "skip_none": float(1 - none.mean()), "skip_camera": float(1 - cam.mean()), "skip_follow": float(1 - fol.mean()),
         "followed": float(fold.mean()), "anchors_follow": int(anc2.sum()),
         "sent_anchor": float(anc.mean()), "anchors": int(anc.sum()), "sent_frontier": float(fro.mean()),
         "sent_refresh": float(cam.mean() - anc.mean() - fro.mean()), "new_area": float(area[al].mean()),
         "corr_median": float(np.median(corr[al])), "frames_between_anchors_median": float(np.median(seg)),
         "frames_between_anchors_mean": float(seg.mean()),
         "window30s_none": [float(min(x[0] for x in win)), float(max(x[0] for x in win))],
         "window30s_camera": [float(min(x[1] for x in win)), float(max(x[1] for x in win))],
         "window30s_follow": [float(min(x[2] for x in win)), float(max(x[2] for x in win))]}
    lines = [f"Tears of Steel {info['width']}x{info['height']} @ {fps:g} fps: {s['frames']} frames in {len(jobs)} chunks of {a.chunk:g} s, "
             f"patch {P}, quotient + multi-scale at Delta_0 = 80, search {a.search} px; {time.perf_counter() - t0:.0f} s",
             f"  skipped: no compensation {s['skip_none']:.3f}, camera {s['skip_camera']:.3f}, follow {s['skip_follow']:.3f} "
             f"(followed, not sent: {s['followed']:.3f} of all patches)",
             f"  sent with compensation: anchor {s['sent_anchor']:.3f} ({s['anchors']} frames), frontier {s['sent_frontier']:.3f} "
             f"(new area {s['new_area']:.3f} of a frame), refresh {s['sent_refresh']:.3f}",
             f"  frames between anchors: median {s['frames_between_anchors_median']:.0f}, mean {s['frames_between_anchors_mean']:.0f}; "
             f"correlation after alignment, median {s['corr_median']:.3f}",
             f"  skipped per 30 s window: no compensation {s['window30s_none'][0]:.2f} to {s['window30s_none'][1]:.2f}, "
             f"camera {s['window30s_camera'][0]:.2f} to {s['window30s_camera'][1]:.2f}, follow {s['window30s_follow'][0]:.2f} to {s['window30s_follow'][1]:.2f}"]
    return {"summary": s, "per_frame": {"none": none.round(4).tolist(), "camera": cam.round(4).tolist(), "follow": fol.round(4).tolist(),
                                        "anchor": anc.astype(int).tolist()}}, lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("davis", "film"))
    ap.add_argument("--root", default="data/davis/DAVIS", help="DAVIS folder holding JPEGImages/ and Annotations/")
    ap.add_argument("--strict", action="store_true", help="davis: also run the camera rule with quotient + range at Delta = 32")
    ap.add_argument("--search", type=int, default=16, help="radius in pixels of the follow rule's search")
    ap.add_argument("--chunk", type=float, default=60.0, help="film: seconds per worker chunk; each chunk starts with an anchor")
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips or chunks")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out, lines = run_davis(a) if a.what == "davis" else run_film(a)
    print("\n".join(lines))
    if a.out:
        Path(a.out + ".json").write_text(json.dumps(out, indent=1))
        Path(a.out + ".txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
