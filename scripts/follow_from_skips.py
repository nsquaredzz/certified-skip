#!/usr/bin/env python3
"""First test: can one person be followed from the skip record alone?

The skip record is the keep mask of the certified rule (quotient + multi-scale, no offset, Delta_0 = 80) at the
native frame rate: one bit per patch per frame, no pixels. A blob tracker on that grid follows every blob of sent
patches. A blob that sends nothing stays where it is. When two blobs touch, the identities in them are no longer
separated by the record: that is a break, and the stretch between two breaks is an unbroken segment.

  skip record   each labelled person is picked up on the first frame where a blob holding only that person lies
                under the labelled box, and followed to the end of the labels; at a break the follower takes the
                blob nearest to where the person's speed before the break would have put them (no pixels)
  baseline      the labelled boxes themselves, without identities, every k-th frame (perfect detections), linked
                by position with constant-velocity prediction: what continuity alone gives at 1, 2, 5 fps

Reported per person: share of the labelled time spent in unbroken segments, breaks, share of time the follower is on
the right person, whether it ends on the right person, and the audit: unbroken segments that sit on two different
labelled people. Labels: CAVIAR ground truth (homepages.inf.ed.ac.uk/rbf/CAVIARDATA1), in data/caviar_gt/.

    .venv/bin/python scripts/follow_from_skips.py [--clips Walk1,LeftBag] [--patch 16] [--out out/follow]
"""
import argparse, itertools, json, sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from certskip import video
from certskip.native import NativeSequentialPruner
from real_bench import schedule

CLIPS = ("Walk1", "LeftBag", "Meet_Crowd", "Fight_Chase", "WalkByShop1cor", "EnterExitCrossingPaths1cor", "OneLeaveShopReenter1cor")
FPS = 25
WINDOW = 2          # frames of sends that make one blob map
GAP = 2             # patches: parts of one blob further apart than this are separate blobs
QUIET = 50          # frames of silence after which a blob at the border, or one that hardly ever sent, is dropped
MIN_ACTIVE = 5


def load_gt(path):
    """{frame: {id: (x0, y0, x1, y1)}}"""
    out = {}
    for fr in ET.parse(path).getroot().iter("frame"):
        d = {}
        for o in fr.find("objectlist").findall("object"):
            b = o.find("box"); xc, yc, w, h = (float(b.get(k)) for k in ("xc", "yc", "w", "h"))
            d[int(o.get("id"))] = (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)
        out[int(fr.get("number"))] = d
    return out


def keep_masks(path, P):
    pr = None; keeps = []
    for f in video.iter_gray(path):
        if pr is None:
            pr = NativeSequentialPruner(f.shape[0], f.shape[1], P, multiscale=schedule(80.0, 0.5), z=1e9, offset="none")
        keeps.append(pr.step(f)[0])
    keeps = np.stack(keeps); keeps[0] = False          # the first frame is sent whole
    return keeps


def chronic(keeps, share):
    """Patches that send on more than `share` of the frames: flicker, not a passer-by."""
    return keeps[1:].mean(0) > share


def dilate(m, r=1):
    out = m.copy()
    for _ in range(r):
        p = np.pad(out, 1); out = np.zeros_like(out)
        for dy, dx in itertools.product((0, 1, 2), repeat=2):
            out |= p[dy:dy + m.shape[0], dx:dx + m.shape[1]]
    return out


def components(m):
    lab = np.zeros(m.shape, np.int32); n = 0; H, W = m.shape
    for y, x in zip(*np.nonzero(m)):
        if lab[y, x]:
            continue
        n += 1; lab[y, x] = n; stack = [(y, x)]
        while stack:
            cy, cx = stack.pop()
            for dy, dx in itertools.product((-1, 0, 1), repeat=2):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < H and 0 <= nx < W and m[ny, nx] and not lab[ny, nx]:
                    lab[ny, nx] = n; stack.append((ny, nx))
    return [lab == i for i in range(1, n + 1)]


def centre(m):
    ys, xs = np.nonzero(m); return np.array([ys.mean(), xs.mean()])


class Blob:
    def __init__(self, region, members, t):
        self.region = region; self.members = set(members); self.born = t; self.last = t; self.active = 1
        self.hist = [(t, centre(region))]

    def velocity(self):
        if len(self.hist) < 2:
            return np.zeros(2)
        (t0, c0), (t1, c1) = self.hist[max(0, len(self.hist) - 13)], self.hist[-1]
        return (c1 - c0) / max(t1 - t0, 1)


def mask_grid(path, cell, init=50):
    """The mask tracker's foreground (THEORY.md section 7), its identities dropped, on a grid of `cell` pixels.
    Unlike the skip record this reads pixels on the tracker's active set."""
    from certskip.native import NativeRtSegmenter
    fr = video.read_gray(path); T, H, W = fr.shape; sg = NativeRtSegmenter(H, W, 16, init_frames=fr[:min(init, T)]); out = []
    for t in range(T):
        m = sg.step(fr[t])[:H // cell * cell, :W // cell * cell]
        out.append(m.reshape(H // cell, cell, W // cell, cell).any((1, 3)))
    return np.stack(out)


def track(keeps, window=WINDOW, forget=None):
    """Follow every blob of the skip record. Returns per frame {identity: (region, solo, segment)} and the breaks."""
    T, gh, gw = keeps.shape; blobs = []; next_id = 0; frames = []; breaks = []
    seg = {}                                            # identity -> current segment number
    memo = {}                                           # identity -> (frame, centre, velocity) when it last was alone
    border = np.ones((gh, gw), bool); border[1:-1, 1:-1] = False
    for t in range(T):
        act = keeps[max(0, t - window + 1):t + 1].any(0)
        comps = components(dilate(act, GAP // 2)) if act.any() else []
        comps = [c & act for c in comps]
        halo = [dilate(b.region) for b in blobs]
        link = [[bool((h & c).any()) for c in comps] for h in halo]
        seen_b, seen_c, new = set(), set(), []
        for b0 in range(len(blobs)):
            if b0 in seen_b:
                continue
            gb, gc, todo = {b0}, set(), [("b", b0)]
            while todo:
                kind, i = todo.pop()
                if kind == "b":
                    for j in range(len(comps)):
                        if link[i][j] and j not in gc:
                            gc.add(j); todo.append(("c", j))
                else:
                    for k in range(len(blobs)):
                        if link[k][i] and k not in gb:
                            gb.add(k); todo.append(("b", k))
            seen_b |= gb; seen_c |= gc
            group = [blobs[i] for i in gb]; parts = [comps[j] for j in gc]
            if not parts:                               # silence: every blob of the group stays where it is
                for b in group:
                    quiet = t - b.last
                    if quiet > QUIET and ((b.region & border).any() or b.active < MIN_ACTIVE):
                        continue
                    if forget is not None and quiet > forget:   # a mask that is gone is gone
                        continue
                    new.append(b)
                continue
            members = set().union(*(b.members for b in group))
            if len(group) > 1:                          # blobs touch: a break for everyone in them
                for b in group:
                    if len(b.members) == 1:
                        (m,) = b.members; memo[m] = (t, b.hist[-1][1], b.velocity())
                breaks.append((t, sorted(members)))
                nb = Blob(np.any(parts, 0), members, t); nb.active = max(b.active for b in group); new.append(nb)
                continue
            b = group[0]
            if len(parts) > 1 and len(b.members) > 1:   # a blob holding several identities comes apart: guess by speed
                pos = {m: memo[m][1] + memo[m][2] * (t - memo[m][0]) for m in b.members}
                cs = [centre(p) for p in parts]; got = {i: set() for i in range(len(parts))}
                for m in sorted(b.members):
                    got[int(np.argmin([np.linalg.norm(pos[m] - c) for c in cs]))].add(m)
                for i, p in enumerate(parts):
                    if not got[i]:
                        got[i] = {next_id}; seg[next_id] = 0; next_id += 1
                    nb = Blob(p, got[i], t); nb.active = b.active; new.append(nb)
                    if len(got[i]) == 1:
                        (m,) = got[i]; seg[m] += 1
                continue
            b.region = np.any(parts, 0); b.last = t; b.active += 1; b.hist.append((t, centre(b.region))); new.append(b)
        for j, c in enumerate(comps):
            if j not in seen_c:
                new.append(Blob(c, {next_id}, t)); seg[next_id] = 0; next_id += 1
        blobs = new
        frames.append({m: (b.region, len(b.members) == 1, seg[m]) for b in blobs for m in b.members})
    return frames, breaks


def track_masks(path, margin, init=50):
    """The same per-frame structure from the mask tracker of THEORY.md section 7 (pixels, its own identities).
    An identity is alone while its mask, grown by `margin` pixels, meets no other mask; a touch, or a frame
    without the mask, ends the segment. Unlike the skip record this reads pixels on the tracker's active set."""
    from certskip.native import NativeRtSegmenter
    fr = video.read_gray(path); T, H, W = fr.shape
    sg = NativeRtSegmenter(H, W, 16, init_frames=fr[:min(init, T)]); frames = []; seg = {}; was = {}; breaks = []
    for t in range(T):
        sg.step(fr[t]); lab = sg.labels; cur = {}; ids = [int(i) for i in np.unique(lab[lab > 0])]
        for i in ids:
            m = lab == i; ys, xs = np.nonzero(m)
            y0, y1, x0, x1 = max(ys.min() - margin, 0), ys.max() + margin + 1, max(xs.min() - margin, 0), xs.max() + margin + 1
            near = lab[y0:y1, x0:x1]; solo = not (dilate(m[y0:y1, x0:x1], margin) & (near > 0) & (near != i)).any()
            if i not in seg:
                seg[i] = 0
            elif solo and not was.get(i, False):
                seg[i] += 1
            if not solo and was.get(i, False):
                breaks.append((t, [i]))
            cur[i] = (m, solo, seg[i])
        was = {i: cur[i][1] for i in cur}
        frames.append(cur)
    return frames, breaks, (T, H, W)


def ceiling_px(gt, margin):
    """Per person: share of labelled frames on which the box is more than `margin` pixels clear of every other box."""
    free = {}; n = {}
    for boxes in gt.values():
        for pid, a in boxes.items():
            n[pid] = n.get(pid, 0) + 1
            free[pid] = free.get(pid, 0) + (not any(a[0] - margin < b[2] and b[0] - margin < a[2] and a[1] - margin < b[3] and b[1] - margin < a[3]
                                                    for q, b in boxes.items() if q != pid))
    return {pid: free[pid] / n[pid] for pid in n}


def under(region, boxes, P):
    """Labelled people whose box (grown by half a patch) meets the region, nearest box centre first."""
    cy, cx = (centre(region) + 0.5) * P; hits = []
    ys, xs = np.nonzero(region)
    for pid, (x0, y0, x1, y1) in boxes.items():
        px0, py0, px1, py1 = xs * P, ys * P, xs * P + P, ys * P + P
        if ((px1 > x0 - P / 2) & (px0 < x1 + P / 2) & (py1 > y0 - P / 2) & (py0 < y1 + P / 2)).any():
            hits.append((np.hypot((x0 + x1) / 2 - cx, (y0 + y1) / 2 - cy), pid))
    return [p for _, p in sorted(hits)]


def ceiling(gt, P, shape):
    """Per person: share of labelled frames on which the box, on the patch grid and grown by one patch, meets
    nobody else's. No rule that reads continuity off this grid can have more unbroken time than that."""
    gh, gw = shape; free = {}; n = {}
    for t, boxes in gt.items():
        foot = {}
        for pid, (x0, y0, x1, y1) in boxes.items():
            m = np.zeros(shape, bool); m[max(0, int(y0 // P)):min(gh, int(y1 // P) + 1), max(0, int(x0 // P)):min(gw, int(x1 // P) + 1)] = True
            foot[pid] = dilate(m)
        for pid in foot:
            n[pid] = n.get(pid, 0) + 1
            free[pid] = free.get(pid, 0) + (not any((foot[pid] & foot[q]).any() for q in foot if q != pid))
    return {pid: free[pid] / n[pid] for pid in n}


def follow(frames, gt, P, strict=False):
    people = sorted({p for d in gt.values() for p in d}); rows = []
    for pid in people:
        life = sorted(t for t, d in gt.items() if pid in d and t < len(frames)); ident = t0 = None
        for t in life:                                  # first frame where a blob holding one identity sits on this person only
            for m, (region, solo, _) in frames[t].items():
                if solo and under(region, gt[t], P) == [pid]:
                    ident, t0 = m, t; break
            if ident is not None:
                break
        if ident is None:
            rows.append(dict(person=pid, frames=len(life), picked_up=False)); continue
        span = [t for t in life if t >= t0]; solo_n = right = lost = 0; segs = {}; last_right = False
        for t in span:
            if ident not in frames[t]:
                lost += 1; last_right = False; continue
            region, solo, s = frames[t][ident]; hits = under(region, gt[t], P)
            ok = pid in hits if not (solo or strict) else (hits[:1] == [pid]); right += ok; last_right = ok
            if solo:
                solo_n += 1
                if hits:
                    segs.setdefault(s, set()).add(hits[0])
        n = len(span)
        rows.append(dict(person=pid, frames=len(life), picked_up=True, wait=t0 - life[0], followed=n, unbroken=solo_n / n, right=right / n,
                         lost=lost / n, ends_right=bool(last_right), segments=len(segs), mixed_segments=sum(len(v) > 1 for v in segs.values())))
    return rows


def assign(cost):
    """Smallest total cost matching of rows to columns, by search (the scenes hold at most seven people)."""
    n, m = cost.shape; best = (np.inf, None)
    if n > m:
        r = assign(cost.T); return [(i, j) for j, i in r]
    for perm in itertools.permutations(range(m), n):
        c = sum(cost[i, perm[i]] for i in range(n))
        if c < best[0]:
            best = (c, perm)
    return list(enumerate(best[1])) if best[1] is not None else []


def baseline(gt, stride, nframes):
    """Perfect boxes without identities every `stride` frames, linked by predicted position. Returns per person
    the share of sampled frames on which the track started on that person is still on them, and whether it ends there."""
    tracks = {}; nxt = 0; on = {}                       # track -> dict(c, v, who); on[frame] = {track: person}
    for t in range(0, nframes, stride):
        dets = [(pid, np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])) for pid, b in sorted(gt.get(t, {}).items())]
        ids = list(tracks); cur = {}
        if ids and dets:
            cost = np.array([[np.linalg.norm(tracks[k]["c"] + tracks[k]["v"] - c) for _, c in dets] for k in ids])
            pairs = assign(cost)
        else:
            pairs = []
        used_t, used_d = set(), set()
        for i, j in pairs:
            k = ids[i]; c = dets[j][1]; tracks[k]["v"] = c - tracks[k]["c"]; tracks[k]["c"] = c; tracks[k]["miss"] = 0
            cur[k] = dets[j][0]; used_t.add(k); used_d.add(j)
        for k in ids:
            if k not in used_t:
                tracks[k]["miss"] += 1
                if tracks[k]["miss"] > 2:
                    del tracks[k]
        for j, (pid, c) in enumerate(dets):
            if j not in used_d:
                tracks[nxt] = dict(c=c, v=np.zeros(2), miss=0, first=pid); cur[nxt] = pid; nxt += 1
        on[t] = cur
    first = {}
    for t in sorted(on):
        for k, pid in on[t].items():
            first.setdefault(k, pid)
    out = {}
    for pid in sorted({p for d in gt.values() for p in d}):
        ts = [t for t in sorted(on) if pid in gt.get(t, {})]
        if not ts:
            continue
        k = next((k for k, p in on[ts[0]].items() if p == pid), None)
        right = sum(on[t].get(k) == pid for t in ts)
        out[pid] = dict(right=right / len(ts), ends_right=on[ts[-1]].get(k) == pid, samples=len(ts))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=",".join(CLIPS)); ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--ignore-chronic", type=float, default=None, help="drop patches that send on more than this share of the frames (found on the whole clip)")
    ap.add_argument("--source", choices=("record", "masks", "ids"), default="record",
                    help="record: the keep mask alone; masks: blobs of the mask tracker's foreground; ids: the mask tracker's own identities")
    ap.add_argument("--margin", type=int, default=4, help="masks, ids: pixels two masks must be apart")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(); P = a.patch; res = {}; lines = []
    a.out = a.out or str(ROOT / "out" / {"record": "follow", "masks": "follow_masks", "ids": "follow_ids"}[a.source])

    def say(s=""):
        print(s, flush=True); lines.append(s)

    say(f"{'patch %d; skip record' % P if a.source == 'record' else 'mask tracker (%s), margin %d px,' % (a.source, a.margin)} at {FPS} fps; baseline: perfect boxes without identities at 1, 2, 5 and 25 fps")
    for name in a.clips.split(","):
        gt = load_gt(ROOT / "data" / "caviar_gt" / f"{name}.xml")
        if a.source == "masks":
            P = a.margin; keeps = mask_grid(ROOT / "data" / f"caviar_{name}.mpg", P); sent = float("nan"); bad = 0
            frames, breaks = track(keeps, window=1, forget=5); rows = follow(frames, gt, P); top = ceiling(gt, P, keeps.shape[1:])
        elif a.source == "ids":
            frames, breaks, shape = track_masks(ROOT / "data" / f"caviar_{name}.mpg", a.margin); rows = follow(frames, gt, 1, strict=True)
            top = ceiling_px(gt, a.margin); keeps = np.zeros((shape[0], 1, 1), bool); sent = float("nan"); bad = 0
        else:
            keeps = keep_masks(ROOT / "data" / f"caviar_{name}.mpg", P)
            sent = 100 * keeps[1:].mean(); bad = 0
            if a.ignore_chronic is not None:
                c = chronic(keeps, a.ignore_chronic); bad = int(c.sum()); keeps = keeps & ~c
            frames, breaks = track(keeps); rows = follow(frames, gt, P); top = ceiling(gt, P, keeps.shape[1:])
        base = {f: baseline(gt, FPS // f, len(keeps)) for f in (1, 2, 5, 25)}
        say(f"\n=== {name}: {len(keeps)} frames, {keeps.shape[1]}x{keeps.shape[2]} patches, sent {sent:.2f} %, flicker patches ignored {bad}, breaks in the record {len(breaks)}")
        say("  person  labelled  ceiling%  wait  unbroken%  segs  mixed  right%  lost%  ends   | baseline right% / ends, at 1, 2, 5, 25 fps")
        for r in rows:
            b = "  ".join(f"{100 * base[f][r['person']]['right']:5.1f}/{'Y' if base[f][r['person']]['ends_right'] else 'n'}" if r["person"] in base[f] else "    -  " for f in (1, 2, 5, 25))
            if not r["picked_up"]:
                say(f"  {r['person']:>6}  {r['frames']:>8}  {100 * top[r['person']]:>8.1f}  never picked up alone{'':33}| {b}"); continue
            r["ceiling"] = top[r["person"]]
            say(f"  {r['person']:>6}  {r['frames']:>8}  {100 * top[r['person']]:>8.1f}  {r['wait']:>4}  {100 * r['unbroken']:>9.1f}  {r['segments']:>4}  {r['mixed_segments']:>5}  {100 * r['right']:>6.1f}  {100 * r['lost']:>5.1f}  {'Y' if r['ends_right'] else 'n':>4}   | {b}")
        res[name] = dict(frames=len(keeps), sent=sent, breaks=len(breaks), people=rows, baseline={str(f): {str(k): v for k, v in base[f].items()} for f in base})
    Path(a.out + ".json").write_text(json.dumps(res, indent=1)); Path(a.out + ".txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
