#!/usr/bin/env python3
"""Build the animated demos shown in README.md from the footage in data/.

    python3 scripts/make_readme_media.py                 # everything
    python3 scripts/make_readme_media.py skip blindspot  # a subset: skip | blindspot | masks

Every demo is written twice into assets/: a GIF, which plays inline on GitHub, and an H.264 MP4 at
full quality. Needs the C++ library (make -C cpp), ffmpeg, Pillow, and the clips fetched by
scripts/fetch_data.py. The blind-spot demo replays trials of scripts/e2e_vlm.py with their exact
geometry and stamps the answers recorded in out/e2e_e1b.json; it does not run the language model.
"""
import argparse, json, shutil, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python")); sys.path.insert(0, str(ROOT / "scripts"))
from certskip import video, baselines as bl                      # noqa: E402
from certskip.core import drop_rate                              # noqa: E402
from certskip.native import NativeSequentialPruner, NativeRtSegmenter   # noqa: E402
from certskip.segment import render_mask                         # noqa: E402
from real_bench import schedule                                  # noqa: E402

P = 16
ASSETS = ROOT / "assets"
BG, FG, MUTED = (13, 17, 23), (240, 246, 252), (150, 160, 172)
GREEN, AMBER, RED, TRACK = (74, 222, 128), (251, 191, 36), (248, 113, 113), (48, 54, 61)
SCHED = schedule(80.0, 0.5)                                       # quotient + multi-scale operating point of RESULTS Part II
KEEP = None                                                       # set by --keep-intermediate


# ----------------------------------------------------------------------------- drawing
def font(px, bold=False):
    names = (["Arial Bold.ttf", "Helvetica.ttc", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"] if bold
             else ["Arial.ttf", "Helvetica.ttc", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"])
    dirs = ["/System/Library/Fonts/Supplemental", "/System/Library/Fonts", "/Library/Fonts",
            "/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation", "/usr/share/fonts"]
    for n in names:
        for d in dirs:
            p = Path(d) / n
            if p.exists():
                try:
                    return ImageFont.truetype(str(p), int(px))
                except Exception:
                    pass
    return ImageFont.load_default()


def grey_panel(img, up=1, size=None):
    """(H, W) uint8 -> RGB PIL image, integer up-scaling (crisp pixels) or resize to `size`."""
    im = Image.fromarray(img).convert("RGB")
    if up > 1:
        im = im.resize((img.shape[1] * up, img.shape[0] * up), Image.NEAREST)
    if size:
        im = im.resize(size, Image.LANCZOS)
    return im


def outline_patches(im, keep, cell, colour=GREEN, width=2):
    d = ImageDraw.Draw(im)
    for gy, gx in zip(*np.where(keep)):
        d.rectangle([gx * cell, gy * cell, (gx + 1) * cell - 1, (gy + 1) * cell - 1], outline=colour, width=width)
    return im


class Layout:
    """A row of equal panels: a title above each, an optional verdict row under each, a footer line and bar."""
    def __init__(self, n, pw, ph, bar=True, verdict=False):
        self.n, self.pw, self.ph = n, pw, ph
        self.gap = max(6, pw // 64) if n > 1 else 0
        self.W = n * pw + (n - 1) * self.gap
        self.W += self.W % 2
        u = self.W / 100.0                                        # all type scales with the canvas width
        self.u = u
        self.f_title, self.f_sub, self.f_foot, self.f_big = font(1.75 * u, True), font(1.38 * u), font(1.6 * u), font(2.2 * u, True)
        self.head = int(5.6 * u); self.verd = int(6.2 * u) if verdict else 0; self.foot = int((6.4 if bar else 3.9) * u)
        self.H = self.head + ph + self.verd + self.foot
        self.H += self.H % 2

    def x(self, i):
        return i * (self.pw + self.gap)

    @property
    def rows(self):
        """Vertical extent of the panels, as fractions of the canvas height."""
        return self.head / self.H, (self.head + self.ph) / self.H

    def compose(self, panels, titles, subs):
        can = Image.new("RGB", (self.W, self.H), BG)
        d = ImageDraw.Draw(can)
        for i, (p, t, s) in enumerate(zip(panels, titles, subs)):
            can.paste(p, (self.x(i), self.head))
            d.text((self.x(i) + 0.6 * self.u, 0.7 * self.u), t, font=self.f_title, fill=FG)
            d.text((self.x(i) + 0.6 * self.u, 3.05 * self.u), s, font=self.f_sub, fill=MUTED)
        return can, d

    def footer(self, d, left, right=None, frac=None, frac_colour=GREEN):
        y = self.head + self.ph + self.verd
        d.text((0.6 * self.u, y + 1.0 * self.u), left, font=self.f_foot, fill=FG)
        if right:
            w = d.textlength(right, font=self.f_foot)
            d.text((self.W - w - 0.6 * self.u, y + 1.0 * self.u), right, font=self.f_foot, fill=FG)
        if frac is not None:
            x0, x1, yb, hb = int(0.6 * self.u), int(self.W - 0.6 * self.u), int(y + 4.0 * self.u), max(4, int(0.9 * self.u))
            d.rectangle([x0, yb, x1, yb + hb], fill=TRACK)
            d.rectangle([x0, yb, x0 + max(2, int((x1 - x0) * min(max(frac, 0.0), 1.0))), yb + hb], fill=frac_colour)

    def verdict(self, d, i, text, colour, sub="", big=True):
        """One line of verdict, and one of explanation, in the row under panel i."""
        x0, y0 = self.x(i), self.head + self.ph
        d.rectangle([x0, y0 + int(0.6 * self.u), x0 + max(3, int(0.35 * self.u)), y0 + self.verd - 1], fill=colour)
        d.text((x0 + 1.1 * self.u, y0 + (0.6 if big else 0.9) * self.u), text, font=self.f_big if big else self.f_foot, fill=colour)
        if sub:
            d.text((x0 + 1.1 * self.u, y0 + 3.7 * self.u), sub, font=self.f_sub, fill=FG if big else MUTED)


class Movie:
    """RGB frames -> lossless intermediate -> assets/<stem>.mp4 and assets/<stem>.gif."""
    def __init__(self, stem, w, h, fps, tmp):
        self.stem, self.w, self.h, self.fps = stem, w, h, fps
        self.tmp = Path(tmp) / f"{stem}.mkv"; self.n = 0
        self.proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                                      "-r", str(fps), "-i", "-", "-c:v", "ffv1", str(self.tmp)], stdin=subprocess.PIPE)

    def add(self, im, repeat=1):
        assert im.size == (self.w, self.h), (im.size, self.w, self.h)
        buf = np.asarray(im.convert("RGB"), np.uint8).tobytes()
        for _ in range(repeat):
            self.proc.stdin.write(buf); self.n += 1

    def close(self, gif_w=880, gif_fps=None, mp4_w=1280, colours=128, crf=23, dead_band=7, rows=(0.0, 1.0)):
        self.proc.stdin.close(); self.proc.wait()
        ASSETS.mkdir(exist_ok=True)
        mp4, gif = ASSETS / f"{self.stem}.mp4", ASSETS / f"{self.stem}.gif"
        mw = min(mp4_w, self.w)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(self.tmp), "-vf", f"scale={mw}:-2:flags=lanczos",
                        "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)], check=True)
        # GIF. A GIF stores every pixel that differs from the previous frame, so sensor noise on a static
        # wall costs as much as a walking person. A per-pixel dead band (a pixel is redrawn only once it has
        # moved more than `dead_band` grey levels away from what is on screen) removes that cost and nothing
        # else. It is applied to the camera panels only (`rows`, as fractions of the height), never to the
        # text. The MP4 next to it is unfiltered.
        gw = min(gif_w, self.w); gw -= gw % 2; gh = int(round(self.h * gw / self.w / 2)) * 2; gf = gif_fps or self.fps
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(self.tmp), "-vf", f"fps={gf},scale={gw}:{gh}:flags=lanczos",
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], check=True, stdout=subprocess.PIPE).stdout
        fr = np.frombuffer(raw, np.uint8).reshape(-1, gh, gw, 3).copy()
        if dead_band > 0:
            r0, r1 = int(np.ceil(rows[0] * gh)) + 1, int(np.floor(rows[1] * gh)) - 1
            shown = fr[0, r0:r1].astype(np.int16)
            for i in range(1, len(fr)):
                cur = fr[i, r0:r1].astype(np.int16)
                moved = np.abs(cur - shown).max(axis=2) > dead_band
                shown[moved] = cur[moved]
                fr[i, r0:r1] = shown.astype(np.uint8)
        vf = f"split[a][b];[a]palettegen=max_colors={colours}:stats_mode=full[p];[b][p]paletteuse=dither=none:diff_mode=rectangle"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{gw}x{gh}", "-r", str(gf), "-i", "-",
                        "-filter_complex", vf, "-loop", "0", str(gif)], input=fr.tobytes(), check=True)
        if KEEP:
            shutil.copy(self.tmp, KEEP / self.tmp.name)
        print(f"  {self.stem}: {self.n} frames at {self.fps} fps  ->  {mp4.name} {mp4.stat().st_size / 1e6:.1f} MB,  "
              f"{gif.name} {gif.stat().st_size / 1e6:.1f} MB ({gw}x{gh}, {gf} fps, {len(fr)} frames)", flush=True)


# ----------------------------------------------------------------------------- 1. the rule at work
SKIP = {
    "hallway": dict(path="data/hall_monitor_cif.y4m", fps=30, scale=None, up=2, seconds=8.0, gif_fps=10, where="office hallway, analog CCTV"),
    "lobby":   dict(path="data/caviar_LeftBag.mpg", fps=25, scale=None, up=2, seconds=7.0, gif_fps=12.5, where="lobby, MPEG-1 CCTV"),
    "parking": dict(path="data/VIRAT_S_000200_00.mp4", fps=15, scale=(960, 544), up=1, seconds=9.0, gif_fps=15, where="parking lot, 720p surveillance"),
}


def skip_demo(name, tmp):
    c = SKIP[name]
    src = lambda: video.iter_gray(ROOT / c["path"], fps=c["fps"], scale=c["scale"])
    first = next(iter(src())); H, W = first.shape; gh, gw = H // P, W // P; N = gh * gw
    kept = []                                                     # pass 1: where is the action?
    pr = NativeSequentialPruner(H, W, P, multiscale=SCHED, z=8.0)
    for f in src():
        kept.append(int(pr.step(f)[0].sum()))
    kept = np.array(kept); T = len(kept); n = min(int(c["seconds"] * c["fps"]), T - 1)
    warm = min(int(2 * c["fps"]), T - n)                          # let the sequential null calibrate first
    csum = np.concatenate([[0], np.cumsum(kept)])
    s0 = warm + int(np.argmax(csum[warm + n: T + 1] - csum[warm: T + 1 - n])) if T - n > warm else warm
    up = c["up"]; lay = Layout(2, W * up, H * up)
    mv = Movie(f"skip_{name}", lay.W, lay.H, c["fps"], tmp)
    pr = NativeSequentialPruner(H, W, P, multiscale=SCHED, z=8.0)
    tot_k = tot = 0
    for t, f in enumerate(src()):                                 # pass 2: render the window
        k = pr.step(f)[0]
        if t > 0:
            tot_k += int(k.sum()); tot += N
        if t < s0 or t >= s0 + n:
            continue
        left = outline_patches(grey_panel(f, up), k, P * up, width=max(2, up * 2 - 1))
        right = grey_panel(pr.view(), up)
        can, d = lay.compose([left, right], ["Camera", "What the model is given"],
                             ["green: the patches sent to the model on this frame", "everything else is reused from the last copy it was sent"])
        lay.footer(d, f"sent on this frame: {int(k.sum())} of {N} patches", f"skipped so far: {100 * (1 - tot_k / max(tot, 1)):.1f} %",
                   frac=k.sum() / N)
        mv.add(can)
    print(f"  {name}: {W}x{H} at {c['fps']} fps, window {s0 / c['fps']:.1f}-{(s0 + n) / c['fps']:.1f} s, clip drop rate {1 - kept[1:].sum() / (N * (T - 1)):.4f}")
    mv.close(gif_fps=c["gif_fps"], rows=lay.rows)


# ----------------------------------------------------------------------------- 2. the blind spot
def blindspot_demo(picks, tmp, trials=16, fade_s=8.0, contrast=70, results="out/e2e_e1b.json"):
    from e2e_vlm import CLIPS                                     # same clips, RNG stream and calibration as the experiment
    from render_e2e import replay_trials, plant, run_views_keeps
    res = json.loads((ROOT / results).read_text())
    rng = np.random.default_rng(0)
    for name in ("hall", "lobby", "virat"):
        spec = CLIPS[name]
        fr = video.read_gray(ROOT / spec["path"], fps=spec["fps"], scale=spec["scale"])
        geoms = replay_trials(rng, fr, spec, trials, fade_s, contrast)
        if name not in picks:
            continue
        T, H, W = fr.shape; fps = spec["fps"]; N = (H // P) * (W // P)
        d_cert = drop_rate(NativeSequentialPruner(H, W, P, multiscale=SCHED, z=1e9).run(fr)["keep"])
        tau, _, _ = bl.calibrate(bl.consecutive_mean, fr[: min(T, 600)], P, d_cert)
        g = geoms[picks[name]]; rec = res[name]["E1"][picks[name]]
        assert rec["sign"] == g["sign"], "the RNG replay does not match the recorded trial"
        seg, ramp = plant(fr, g, contrast)
        views, keeps = run_views_keeps(seg, SCHED, tau)
        up = 2 if W < 600 else 1
        lay = Layout(3, W * up, H * up, verdict=True)
        step = max(1, round(2 * fps / 12.5)); out_fps = fps / step * 2    # play the fade at twice real time
        mv = Movie(f"blindspot_{'parking' if name == 'virat' else name}", lay.W, lay.H, out_fps, tmp)
        b = g["box"] * up; m = 3 * up
        box = [g["x"] * up - m, g["y"] * up - m, g["x"] * up + b + m, g["y"] * up + b + m]
        titles = ["Camera", "Heuristic skip rule", "Certified skip rule"]
        subs = ["a square fades in over %.0f seconds" % g["fade_s"], "frame-to-frame mean change (EVS-style)", "change since the last copy sent, with a proof"]
        tot = {k: [0, 0] for k in ("evs", "certified")}
        last = len(seg) - 1
        a_ = rec["answers"]; seen = rec["present_in_view"]
        q = f"Is there a {'dark' if g['sign'] < 0 else 'bright'} square patch on the floor or wall?"
        say = lambda key: "Qwen2-VL answers “%s”" % a_[key].strip(".")
        good = lambda key: a_[key].lower().startswith("yes")
        def frame(t, card):
            ps = []
            for key, img in (("truth", seg[t]), ("evs", views["evs"][t]), ("certified", views["certified"][t])):
                p = grey_panel(img, up)
                if key != "truth":
                    outline_patches(p, keeps[key][t], P * up, width=max(2, up * 2 - 1))
                ImageDraw.Draw(p).rectangle(box, outline=AMBER, width=max(2, up + 1))
                ps.append(p)
            can, d = lay.compose(ps, titles, subs)
            frac = (t - g["pre"]) / max(g["nf"], 1)
            stage = "before the square appears" if frac < 0 else (f"fading in: contrast {ramp[t]:.0f} of {contrast} grey levels" if frac < 1 else "fade complete")
            dr = {k: 100 * (1 - tot[k][0] / max(tot[k][1], 1)) for k in tot}
            lay.footer(d, stage, f"patches skipped so far: heuristic {dr['evs']:.1f} %, certified {dr['certified']:.1f} %",
                       frac=min(max(frac, 0), 1), frac_colour=AMBER)
            if card:
                lay.verdict(d, 0, say("truth"), GREEN if good("truth") else RED, "shown the full frame")
                lay.verdict(d, 1, say("evs"), GREEN if good("evs") else RED, "the square reached the model" if seen["evs"] else "the square never reached the model")
                lay.verdict(d, 2, say("certified"), GREEN if good("certified") else RED, "the square reached the model" if seen["certified"] else "the square never reached the model")
            else:
                lay.verdict(d, 0, "When the fade ends, Qwen2-VL is asked:", MUTED, "“" + q + "”", big=False)
                lay.verdict(d, 1, "green: patches sent on this frame", MUTED, "yellow: where the square is", big=False)
                lay.verdict(d, 2, "both rules skip the same share of patches", MUTED, "on the untouched clip (%.1f %%)" % (100 * d_cert), big=False)
            return can
        for t in range(len(seg)):
            if t > 0:
                for k in tot:
                    tot[k][0] += int(keeps[k][t].sum()); tot[k][1] += N
            if t > 0 and (t % step == 0 or t == last):          # frame 0 is all green: every rule sends its first frame whole
                mv.add(frame(t, False))
        mv.add(frame(last, True), repeat=int(4.0 * out_fps))
        print(f"  {name} trial {picks[name]}: answers {rec['answers']}, object in view {rec['present_in_view']}, matched drop rate {d_cert:.3f}")
        mv.close(gif_w=960, rows=lay.rows)


# ----------------------------------------------------------------------------- 3. object masks
MASKS = {
    "hallway": dict(path="data/hall_monitor_cif.y4m", fps=30, up=2, size=None, start=1.5, seconds=8.0, gif_fps=10,
                    seg=dict(), speed="120+ fps on a laptop CPU"),
    "street":  dict(path="data/VIRAT_S_050000_05_1080p.mp4", fps=30, up=1, size=(960, 540), start=16.0, seconds=9.0, gif_fps=10,
                    seg=dict(min_area=400, close_r=1, snap_eps=16.0), speed="19 fps on a laptop CPU"),
    "forecourt": dict(path="data/VIRAT_S_040000_00b_1080p.mp4", fps=30, up=1, size=(960, 540), start=9.0, seconds=9.0, gif_fps=10,
                      seg=dict(min_area=400, close_r=1, snap_eps=16.0), speed="25 fps on a laptop CPU"),
}


def masks_demo(name, tmp):
    c = MASKS[name]
    it = video.iter_gray(ROOT / c["path"], fps=c["fps"])
    init = [next(it).copy() for _ in range(50)]
    H, W = init[0].shape
    kw = dict(z0=3.0, gamma=0.6, lam=1.2, min_area=40, shadows=True, close_r=2, rho=0.6, snap_r=3, snap_eps=36.0); kw.update(c["seg"])
    seg = NativeRtSegmenter(H, W, P, steps=3, mu=0.5, init_frames=np.stack(init), **kw)
    pw, ph = (c["size"] if c["size"] else (W * c["up"], H * c["up"]))
    lay = Layout(2, pw, ph, bar=False)
    t_seg = 0.0
    mv = Movie(f"masks_{name}", lay.W, lay.H, c["fps"], tmp)
    s0, s1 = int(c["start"] * c["fps"]), int((c["start"] + c["seconds"]) * c["fps"])
    def frames():
        yield from init
        yield from it
    for t, f in enumerate(frames()):
        if t >= s1:
            break
        t0 = time.perf_counter(); mask = seg.step(f); t_seg += time.perf_counter() - t0
        if t < s0:
            continue
        rgb = np.repeat(f[..., None], 3, -1); flat = render_mask(seg.labels)
        over = rgb.astype(np.int32); over[mask] = 0.45 * over[mask] + 0.55 * flat[mask]; over = over.astype(np.uint8)
        lw = max(1, W // 640)
        for pid in np.unique(seg.labels[seg.labels > 0]):
            ys, xs = np.where(seg.labels == pid); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max(); col = flat[ys[0], xs[0]]
            over[y0:y0 + lw, x0:x1 + 1] = col; over[y1 - lw + 1:y1 + 1, x0:x1 + 1] = col
            over[y0:y1 + 1, x0:x0 + lw] = col; over[y0:y1 + 1, x1 - lw + 1:x1 + 1] = col
        mk = lambda a: Image.fromarray(a).resize((pw, ph), Image.NEAREST if c["size"] is None else Image.LANCZOS)
        can, d = lay.compose([mk(over), mk(flat)], ["Camera, with the tracked objects", "Object mask"],
                             ["one colour per object, no class labels, nothing learned", "pixel-accurate, tracked from frame to frame"])
        lay.footer(d, f"{W}×{H}   ·   {c['speed']}   ·   objects in view: {len(np.unique(seg.labels[seg.labels > 0]))}")
        mv.add(can)
    print(f"  {name}: {W}x{H}, segmentation {s1 / t_seg:.1f} fps measured in this run (rendering excluded)")
    mv.close(gif_fps=c["gif_fps"], colours=160, rows=lay.rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="*", default=["skip", "blindspot", "masks"])
    ap.add_argument("--only", default=None, help="comma-separated demo names within the chosen groups")
    ap.add_argument("--keep-intermediate", default=None, help="directory in which to keep the lossless intermediates")
    a = ap.parse_args()
    global KEEP
    if a.keep_intermediate:
        KEEP = Path(a.keep_intermediate); KEEP.mkdir(parents=True, exist_ok=True)
    only = set(a.only.split(",")) if a.only else None
    want = lambda n: only is None or n in only
    tmp = tempfile.mkdtemp(prefix="certskip_media_")
    try:
        if "skip" in a.what:
            for n in SKIP:
                if want(n):
                    skip_demo(n, tmp)
        if "blindspot" in a.what:
            picks = {k: v for k, v in {"lobby": 0, "virat": 4}.items() if want(k) or (k == "virat" and want("parking"))}
            if picks:
                blindspot_demo(picks, tmp)
        if "masks" in a.what:
            for n in MASKS:
                if want(n):
                    masks_demo(n, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
