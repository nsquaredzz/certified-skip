#!/usr/bin/env python3
"""Follow one person through space-time tubes, against perfect boxes linked by position at 1 and 2 fps.

The clip is a block (t, y, x) of one number per pixel, its distance from the empty scene. A person is a ridge of
that block. Ridges are separated by persistence: two of them stay apart while the pass between them is at least
`tau` grey levels below the lower one, and only if they exist at the same time for `coexist` frames (pieces that
follow one another in time are one object). This is persistence-based clustering (Chazal et al. 2013) on the
video block; the tubes are found on the whole clip, so the follower sees the future.

  empty scene   causal: median of the first 50 frames, then a running mean that learns only where the frame is
                close to it;  median: the median of the whole clip, which no online method has
  follower      picked up on the first labelled frame where a tube lies under that person's box and nobody
                else's. It stays on its tube; when the tube ends it moves to the tube that is nearest, within
                `reach` pixels and `wait` frames, to where the person's speed would have put them; with no such
                tube it stays where it was
  baseline      the labelled boxes without identities at 1 and 2 fps, linked by predicted position
                (follow_from_skips.baseline)

  combined      the baseline, with the tubes consulted where position is ambiguous: among the assignments of
                boxes to tracks whose position cost is within `margin` pixels of the best, take the one the tubes
                support most (the share of a track's tube cells at its last box found again under the new box).
                `always` drops the margin and takes the assignment the tubes support most

Reported per person: share of the followed time on the right person, and whether the follower ends there.
Labels: CAVIAR ground truth in data/caviar_gt/.

    .venv/bin/python scripts/follow_tubes.py [--clips Walk1,LeftBag] [--background causal] [--out out/follow_tubes]
"""
import argparse, gc, itertools, json, resource, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from certskip import video
from follow_from_skips import CLIPS, FPS, baseline, load_gt, under

DS = 2              # pixels per cell of the block
GMIN = 12.0         # grey levels: cells closer than this to the empty scene are not in the block
MIN_FRAMES = 10     # a tube shorter than this is not followed
CHUNK = 1 << 18     # cells of the block turned into Python numbers at a time


def change(path, background):
    fr = video.read_gray(path); T, H, W = fr.shape; g = np.empty((T, H // DS, W // DS), np.float32)
    if background == "median":
        b = np.median(fr[::max(1, T // 120)], 0).astype(np.float32)
    else:
        b = np.median(fr[:min(50, T)], 0).astype(np.float32)
    for t in range(T):
        f = fr[t].astype(np.float32); d = np.abs(f - b)
        g[t] = d[:H // DS * DS, :W // DS * DS].reshape(H // DS, DS, W // DS, DS).mean((1, 3))
        if background == "causal":
            b += np.where(d < 15, 0.02, 0.0005) * (f - b)
    del fr
    return g


def tubes(g, tau, coexist):
    """Cluster label per cell of the block above GMIN. Returns (t, y, x, cluster).
    Labels live in one int32 per cell of the block and the cells are walked in chunks: a dict entry and a list
    item per cell above GMIN is what ran the machine out of memory on the long clips."""
    T, h, w = g.shape; pad = np.zeros((T + 2, h + 2, w + 2), np.float32); pad[1:-1, 1:-1, 1:-1] = g; flat = pad.ravel()
    pos = np.nonzero(flat >= GMIN)[0]; order = pos[np.argsort(-flat[pos], kind="stable")]; del pos
    sy, st = w + 2, (h + 2) * (w + 2)
    offs = [dt * st + dy * sy + dx for dt, dy, dx in itertools.product((-1, 0, 1), repeat=3) if (dt, dy, dx) != (0, 0, 0)]
    labels = np.full(flat.size, -1, np.int32); lab = memoryview(labels); val = memoryview(flat)
    parent = []; birth = []; t0 = []; t1 = []

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]; c = parent[c]
        return c

    for a in range(0, len(order), CHUNK):
        for p in order[a:a + CHUNK].tolist():
            v = val[p]; tt = p // st; best = -1; bv = -1.0; roots = set()
            for o in offs:
                c = lab[p + o]
                if c >= 0:
                    r = find(c); roots.add(r); q = val[p + o]
                    if q > bv:
                        bv = q; best = r
            if best < 0:
                lab[p] = len(parent); parent.append(len(parent)); birth.append(v); t0.append(tt); t1.append(tt)
            else:
                lab[p] = best; t0[best] = min(t0[best], tt); t1[best] = max(t1[best], tt)
                for r in roots:
                    a2 = find(best); r = find(r)
                    if r != a2:
                        lo, hi = (r, a2) if birth[r] <= birth[a2] else (a2, r)
                        if birth[lo] - v < tau or min(t1[lo], t1[hi]) - max(t0[lo], t0[hi]) < coexist:
                            parent[lo] = hi; t0[hi] = min(t0[hi], t0[lo]); t1[hi] = max(t1[hi], t1[lo])
    del lab, val
    root = np.array(parent, np.int64)
    while True:                                         # every cluster to its root
        up = root[root]
        if np.array_equal(up, root):
            break
        root = up
    cl = root[labels[order]] if len(order) else np.zeros(0, np.int64)
    return order // st - 1, (order % st) // sy - 1, order % sy - 1, cl


def by_tube(t, y, x, cl):
    """{tube: {frame: (ys, xs)}} for tubes of at least MIN_FRAMES frames."""
    o = np.lexsort((t, cl)); t, y, x, cl = t[o], y[o], x[o], cl[o]; out = {}
    cut = np.nonzero((np.diff(cl) != 0) | (np.diff(t) != 0))[0] + 1
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(t)]):
        out.setdefault(int(cl[a]), {})[int(t[a])] = (y[a:b], x[a:b])
    return {c: f for c, f in out.items() if len(f) >= MIN_FRAMES}


def follow(tb, gt, shape, nframes, reach, wait):
    cen = {c: {t: np.array([ys.mean(), xs.mean()]) for t, (ys, xs) in f.items()} for c, f in tb.items()}
    first = {c: min(f) for c, f in tb.items()}; last = {c: max(f) for c, f in tb.items()}; rows = []

    def region(c, t):
        m = np.zeros(shape, bool); m[tb[c][t]] = True; return m

    for pid in sorted({p for d in gt.values() for p in d}):
        life = sorted(t for t, d in gt.items() if pid in d and t < nframes); cur = t0 = None
        for t in life:                                  # first frame where a tube lies under this person and nobody else
            got = [(len(f[t][0]), c) for c, f in tb.items() if t in f and len(f[t][0]) >= 8 and under(region(c, t), gt[t], DS) == [pid]]
            if got:
                cur, t0 = max(got)[1], t; break
        if cur is None:
            rows.append(dict(person=pid, frames=len(life), picked_up=False)); continue
        span = [t for t in life if t >= t0]; hist = []; reg = None; right = hops = 0; nxt = None; ok = False
        for t in span:
            if nxt is not None and t >= nxt[1]:         # the tube chosen at the last hop has begun
                cur, nxt = nxt[0], None; hops += 1
            if t in tb[cur]:
                reg = region(cur, t); hist.append((t, cen[cur][t])); hist = hist[-13:]
            elif t > last[cur] and nxt is None:         # the tube has ended: where would the person be?
                (ta, ca), (tz, cz) = hist[0], hist[-1]; v = (cz - ca) / max(tz - ta, 1); best = None
                for c in tb:
                    if c == cur or last[c] < t or first[c] > t + wait:
                        continue
                    f = min(u for u in tb[c] if u >= t); d = np.linalg.norm(cen[c][f] - (cz + v * (f - tz))) * DS
                    if d < reach and (best is None or d < best[0]):
                        best = (d, c, f)
                if best:
                    nxt = (best[1], best[2])
                    if best[2] == t:
                        cur, nxt = best[1], None; hops += 1; reg = region(cur, t); hist.append((t, cen[cur][t])); hist = hist[-13:]
            ok = under(reg, gt[t], DS)[:1] == [pid]; right += ok
        rows.append(dict(person=pid, frames=len(life), picked_up=True, wait=t0 - life[0], followed=len(span), right=right / len(span),
                         ends_right=bool(ok), hops=hops))
    return rows


def under_boxes(tz, t, boxes):
    """Per labelled box at frame t, the tubes under it: {tube: share of the box's tube cells}. Cells under two
    boxes go to the nearer centre. The identities of `boxes` are used only as keys."""
    out = {pid: {} for pid in boxes}
    if t not in tz or not boxes:
        return out
    y, x, cl = tz[t]; px, py = (x + 0.5) * DS, (y + 0.5) * DS; best = np.full(len(x), np.inf); who = np.full(len(x), -1)
    for pid, (x0, y0, x1, y1) in boxes.items():
        d = np.hypot(px - (x0 + x1) / 2, py - (y0 + y1) / 2); take = (px >= x0 - 2) & (px <= x1 + 2) & (py >= y0 - 2) & (py <= y1 + 2) & (d < best)
        best[take] = d[take]; who[take] = pid
    for pid in boxes:
        c, n = np.unique(cl[who == pid], return_counts=True)
        out[pid] = dict(zip(c.tolist(), (n / max(n.sum(), 1)).tolist()))
    return out


def linked(gt, stride, nframes, tz=None, margin=0.0, trace=None):
    """follow_from_skips.baseline, with the tubes deciding among assignments within `margin` pixels of the best.
    tz None: position only. `trace`, a dict, is filled with {sampled frame: {track: person it is on}}."""
    tracks = {}; nxt = 0; on = {}
    for t in range(0, nframes, stride):
        boxes = gt.get(t, {}); dets = [(pid, np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])) for pid, b in sorted(boxes.items())]
        sig = under_boxes(tz, t, boxes) if tz is not None else {}; ids = list(tracks); cur = {}; pairs = []
        if ids and dets:
            cost = np.array([[np.linalg.norm(tracks[k]["c"] + tracks[k]["v"] - c) for _, c in dets] for k in ids])
            aff = np.array([[sum(min(w, sig[pid].get(c, 0.0)) for c, w in tracks[k]["sig"].items()) for pid, _ in dets] for k in ids]) if tz is not None else None
            flip = cost.shape[0] > cost.shape[1]; C = cost.T if flip else cost; A = (aff.T if flip else aff) if aff is not None else None
            n, m = C.shape; rows = range(n); C = C.tolist(); A = A.tolist() if A is not None else None
            c1 = np.inf; near = []; top = None      # near: assignments within `margin` of the best so far, never all of them
            for perm in itertools.permutations(range(m), n):
                c = sum(C[i][perm[i]] for i in rows); c1 = min(c1, c)
                if A is None:
                    if top is None or c < top[0]:
                        top = (c, perm)
                elif margin == np.inf:
                    k = (sum(A[i][perm[i]] for i in rows), -c, perm)
                    if top is None or k > top:
                        top = k
                elif c <= c1 + margin:
                    near.append((c, perm))
                    if len(near) > 4096:
                        near = [k for k in near if k[0] <= c1 + margin]
            if A is None:
                perm = top[1]
            elif margin == np.inf:
                perm = top[2]
            else:
                perm = max((sum(A[i][q[i]] for i in rows), -c, q) for c, q in near if c <= c1 + margin)[2]
            pairs = [(perm[i], i) if flip else (i, perm[i]) for i in rows]
        used_t, used_d = set(), set()
        for i, j in pairs:
            k = ids[i]; c = dets[j][1]; tracks[k]["v"] = c - tracks[k]["c"]; tracks[k]["c"] = c; tracks[k]["miss"] = 0
            tracks[k]["sig"] = sig.get(dets[j][0], {}); cur[k] = dets[j][0]; used_t.add(k); used_d.add(j)
        for k in ids:
            if k not in used_t:
                tracks[k]["miss"] += 1
                if tracks[k]["miss"] > 2:
                    del tracks[k]
        for j, (pid, c) in enumerate(dets):
            if j not in used_d:
                tracks[nxt] = dict(c=c, v=np.zeros(2), miss=0, sig=sig.get(pid, {})); cur[nxt] = pid; nxt += 1
        on[t] = cur
    if trace is not None:
        trace.update(on)
    out = {}
    for pid in sorted({p for d in gt.values() for p in d}):
        ts = [t for t in sorted(on) if pid in gt.get(t, {})]
        if not ts:
            continue
        k = next((k for k, p in on[ts[0]].items() if p == pid), None); right = sum(on[t].get(k) == pid for t in ts)
        out[pid] = dict(right=right / len(ts), ends_right=on[ts[-1]].get(k) == pid, samples=len(ts))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=",".join(CLIPS)); ap.add_argument("--background", choices=("causal", "median"), default="causal")
    ap.add_argument("--tau", type=float, default=25.0); ap.add_argument("--coexist", type=int, default=25)
    ap.add_argument("--reach", type=float, default=32.0); ap.add_argument("--wait", type=int, default=25)
    ap.add_argument("--margin", type=float, default=24.0, help="combined: pixels of position cost within which the tubes decide")
    ap.add_argument("--mem-gb", type=float, default=3.0, help="stop with MemoryError past this much address space instead of taking the machine down; 0: no limit")
    ap.add_argument("--out", default=str(ROOT / "out" / "follow_tubes"))
    a = ap.parse_args(); res = {}; lines = []
    if a.mem_gb:
        resource.setrlimit(resource.RLIMIT_AS, (int(a.mem_gb * 2 ** 30),) * 2)

    def say(s=""):
        print(s, flush=True); lines.append(s)

    say(f"tubes: tau {a.tau:g}, coexist {a.coexist} frames, empty scene {a.background}; follower: reach {a.reach:g} px, wait {a.wait} frames; baseline: perfect boxes without identities")
    g = t = y = x = cl = tb = tz = None
    for name in a.clips.split(","):
        del g, t, y, x, cl, tb, tz; gc.collect()         # the last clip's block goes before the next one is read
        gt = load_gt(ROOT / "data" / "caviar_gt" / f"{name}.xml"); g = change(ROOT / "data" / f"caviar_{name}.mpg", a.background)
        t, y, x, cl = tubes(g, a.tau, a.coexist); tb = by_tube(t, y, x, cl); rows = follow(tb, gt, g.shape[1:], len(g), a.reach, a.wait)
        o = np.argsort(t, kind="stable"); ts, ys, xs, cs = t[o], y[o], x[o], cl[o]; cut = np.nonzero(np.diff(ts))[0] + 1
        tz = {int(ts[i]): (ys[i:j], xs[i:j], cs[i:j]) for i, j in zip(np.r_[0, cut], np.r_[cut, len(ts)])} if len(ts) else {}
        base = {f: baseline(gt, FPS // f, len(g)) for f in (1, 2)}
        assert all(linked(gt, FPS // f, len(g)) == base[f] for f in (1, 2))
        comb = {f"{f} fps, {k}": linked(gt, FPS // f, len(g), tz, m) for f in (1, 2) for k, m in (("ambiguous", a.margin), ("always", np.inf))}
        say(f"\n=== {name}: {len(g)} frames, {len(tb)} tubes")
        say("  person  labelled  wait  hops  right%  ends   | baseline right% / ends, at 1 and 2 fps | combined (ambiguous), at 1 and 2 fps")
        for r in rows:
            b = "  ".join(f"{100 * base[f][r['person']]['right']:5.1f}/{'Y' if base[f][r['person']]['ends_right'] else 'n'}" if r["person"] in base[f] else "    -  " for f in (1, 2))
            b += " | " + "  ".join(f"{100 * comb[f'{f} fps, ambiguous'][r['person']]['right']:5.1f}/{'Y' if comb[f'{f} fps, ambiguous'][r['person']]['ends_right'] else 'n'}" if r["person"] in comb[f"{f} fps, ambiguous"] else "    -  " for f in (1, 2))
            if not r["picked_up"]:
                say(f"  {r['person']:>6}  {r['frames']:>8}  never picked up alone  | {b}"); continue
            say(f"  {r['person']:>6}  {r['frames']:>8}  {r['wait']:>4}  {r['hops']:>4}  {100 * r['right']:>6.1f}  {'Y' if r['ends_right'] else 'n':>4}   | {b}")
        res[name] = dict(frames=len(g), tubes=len(tb), people=rows, baseline={str(f): {str(k): v for k, v in base[f].items()} for f in base},
                         combined={k: {str(q): v for q, v in c.items()} for k, c in comb.items()},
                         peak_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        say(f"  peak memory so far {res[name]['peak_mb']:.0f} MB")
        Path(a.out + ".json").write_text(json.dumps(res, indent=1)); Path(a.out + ".txt").write_text("\n".join(lines) + "\n")
    say()
    for label, names in (("all clips", list(res)), ("clips the settings were not chosen on", [n for n in res if n not in ("Walk1", "Meet_Crowd")])):
        R = [r for n in names for r in res[n]["people"]]; pu = [r for r in R if r["picked_up"]]
        if not R:
            continue
        say(f"{label}: people {len(R)}, picked up {len(pu)} | follower: ends right {sum(r['ends_right'] for r in pu)}, right all the way {sum(r['right'] == 1 for r in pu)}, "
            f"right at least 90 % {sum(r['right'] >= 0.9 for r in pu)}, median right {100 * np.median([r['right'] for r in pu]):.1f} %")
        for f in ("1", "2"):
            B = [v for n in names for v in res[n]["baseline"][f].values()]
            say(f"   baseline {f} fps: ends right {sum(v['ends_right'] for v in B)}, right all the way {sum(v['right'] == 1 for v in B)}, "
                f"right at least 90 % {sum(v['right'] >= 0.9 for v in B)}, median right {100 * np.median([v['right'] for v in B]):.1f} %, mean right {100 * np.mean([v['right'] for v in B]):.1f} %")
            for k in ("ambiguous", "always"):
                B = [v for n in names for v in res[n]["combined"][f"{f} fps, {k}"].values()]
                say(f"   combined {f} fps, tubes {k}: ends right {sum(v['ends_right'] for v in B)}, right all the way {sum(v['right'] == 1 for v in B)}, "
                    f"right at least 90 % {sum(v['right'] >= 0.9 for v in B)}, mean right {100 * np.mean([v['right'] for v in B]):.1f} %")
    Path(a.out + ".json").write_text(json.dumps(res, indent=1)); Path(a.out + ".txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
