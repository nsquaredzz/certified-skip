"""Tests for the moving-camera certificate, the clairvoyant optimum and the reducer re-implementations
(paper/: Theorems A-D, Proposition E)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from certskip import core, topology as tp, optimal as op, reducers as rd, baselines as bl  # noqa: E402
from certskip.warpcert import WarpGroupPruner, apply_h, translation, sample_bilinear  # noqa: E402


def _scene(rng, H, W):
    yy, xx = np.mgrid[0:H, 0:W]
    img = 120 + 40 * np.sin(xx / 17.0) * np.cos(yy / 11.0)
    for _ in range(30):
        cy, cx, r, a = rng.uniform(0, H), rng.uniform(0, W), rng.uniform(4, 30), rng.uniform(-60, 60)
        img += a * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))
    # fine texture (lightly smoothed noise), so that a pan is not a small change
    tex = rng.normal(0, 18, (H, W))
    tex = 0.25 * (tex + np.roll(tex, 1, 0) + np.roll(tex, 1, 1) + np.roll(tex, (1, 1), (0, 1)))
    img += tex
    return np.clip(img, 0, 255)


# ---------------------------------------------------------------- warp certificate
def test_fixed_camera_is_the_range_rule_bit_for_bit():
    rng = np.random.default_rng(0)
    base = _scene(rng, 96, 128)
    frames = np.clip(base[None] + rng.normal(0, 3, (12, 96, 128)), 0, 255).astype(np.uint8)
    frames[5:, 40:48, 60:68] = 20
    a = core.Pruner(96, 128, 16, 32.0).run(frames)
    b = WarpGroupPruner(96, 128, 16, 32.0, model="none", delta_max=0.0, margin=8).run(frames)
    assert np.array_equal(a["keep"], b["keep"])
    assert np.allclose(a["spread"], b["spread"] * 32.0)          # score = spread / delta
    assert b["iota"].max() == 0.0                                 # integer positions: no interpolation term


def test_registration_recovers_a_pan_and_drops_the_background():
    rng = np.random.default_rng(1)
    big = _scene(rng, 200, 260)
    seq = []
    for t in range(12):
        x0, y0 = 40 + 2 * t, 30 + t
        seq.append(np.clip(big[y0:y0 + 96, x0:x0 + 128] + rng.normal(0, 1.0, (96, 128)), 0, 255).astype(np.uint8))
    seq = np.stack(seq)
    pr = WarpGroupPruner(96, 128, 16, 32.0, model="translation", delta_max=1.0, margin=48)
    r = pr.run(seq)
    u, v = apply_h(pr.G, np.array([0.0]), np.array([0.0]))
    assert abs(u[0] - (48 + 22)) < 0.1 and abs(v[0] - (48 + 11)) < 0.1
    have_ref = ~r["revealed"][1:]                                 # patches whose area was already in the canvas
    assert r["keep"][1:][have_ref].mean() < 0.05                  # background with a reference is dropped
    fixed = core.Pruner(96, 128, 16, 32.0).run(seq)["keep"]
    assert core.drop_rate(fixed) < 0.5                            # the fixed-camera rule cannot


def test_theorem_a_holds_on_every_dropped_patch_against_the_warped_reference():
    """The certificate is checked against W + c, the reference the rule actually constructed."""
    rng = np.random.default_rng(2)
    big = _scene(rng, 200, 260)
    seq = []
    for t in range(8):
        x0, y0 = 40 + 1.5 * t, 30 + 0.5 * t
        xi, yi = int(round(x0)), int(round(y0))
        seq.append(np.clip(big[yi:yi + 96, xi:xi + 128] + rng.normal(0, 2.0, (96, 128)), 0, 255).astype(np.uint8))
    pr = WarpGroupPruner(96, 128, 16, 32.0, model="translation", delta_max=1.0, margin=48)
    checked = 0
    for t, f in enumerate(seq):
        pr.step(f)                                                 # replay to get W: recompute from state before write
    # replay properly: run again, inspecting W before the write
    pr = WarpGroupPruner(96, 128, 16, 32.0, model="translation", delta_max=1.0, margin=48)
    pr.step(seq[0])
    for f in seq[1:]:
        # reconstruct W + c from the view (dropped patches show W + c exactly)
        keep, score, c = pr.step(f)
        view = pr._view
        fg = core.patch_grid(f.astype(np.float64), 16); vg = core.patch_grid(view, 16)
        for gy, gx in zip(*np.where(~keep)):
            rep = tp.check_patch(fg[gy, gx], vg[gy, gx] - c[gy, gx], 32.0)   # ref = W (view minus c)
            assert rep["ok"] and rep["spread"] < 32.0
            checked += 1
    assert checked > 50


def test_sample_bilinear_is_exact_on_integers_and_oscillation_bounds_the_interpolation():
    rng = np.random.default_rng(3)
    img = rng.uniform(0, 255, (20, 30))
    valid = np.ones(img.shape, bool)
    py, px = np.mgrid[2:17, 2:27].astype(np.float64)
    val, ok, osc, nn = sample_bilinear(img, valid, px, py)
    assert np.array_equal(val, img[2:17, 2:27]) and osc.max() == 0 and ok.all()
    val, ok, osc, nn = sample_bilinear(img, valid, px + 0.3, py + 0.7)
    assert np.all(np.abs(val - nn) <= osc + 1e-9)                 # Proposition B, per pixel


def test_cut_resets_and_counts():
    rng = np.random.default_rng(4)
    a = np.clip(_scene(rng, 96, 128) + rng.normal(0, 1, (96, 128)), 0, 255).astype(np.uint8)
    b = np.clip(_scene(np.random.default_rng(99), 96, 128) + rng.normal(0, 1, (96, 128)), 0, 255).astype(np.uint8)
    seq = np.stack([a, a, a, b, b, b])
    pr = WarpGroupPruner(96, 128, 16, 32.0, model="translation", delta_max=1.0, margin=16)
    r = pr.run(seq)
    assert r["cut"][3] and pr.n_cuts == 1 and r["keep"][3].all()
    assert r["keep"][4].mean() < 0.1


# ---------------------------------------------------------------- Theorem D
def test_theorem_d_i_and_ii_on_random_walks():
    rng = np.random.default_rng(5)
    for _ in range(40):
        X = np.cumsum(rng.normal(0, rng.uniform(0.5, 6), (80, 8, 8)), 0)
        r = op.competitive_check(X, 20.0)
        assert r["i_holds"] and r["ii_holds"]


def test_opt_is_feasible_and_no_smaller_schedule_exists_by_brute_force():
    rng = np.random.default_rng(6)
    for _ in range(20):
        T = 9
        X = np.cumsum(rng.normal(0, 4, (T, 4, 4)), 0)
        best, sched = op.opt_keeps(X, 10.0)
        # feasibility
        anchor = 0
        for t in range(T):
            if t in sched:
                anchor = t
            else:
                assert op.stat_range(X[t] - X[anchor]) < 10.0
        # brute force over all schedules containing 0
        import itertools
        found = None
        for k in range(1, T + 1):
            for comb in itertools.combinations(range(1, T), k - 1):
                keeps = {0, *comb}; ok = True; anchor = 0
                for t in range(T):
                    if t in keeps:
                        anchor = t
                    elif op.stat_range(X[t] - X[anchor]) >= 10.0:
                        ok = False; break
                if ok:
                    found = k; break
            if found:
                break
        assert found == best


def test_theorem_d_iii_adversary_forces_every_frame():
    greedy = lambda xs, ks: False
    for gam in (1.0, 1.5, 1.9):
        xs, ks, opt = op.adversary_sequence(32.0, gam, 100, greedy)
        assert ks[2:].all() and opt <= 2
    eager = lambda xs, ks: abs(xs[-1] - [x for x, k in zip(xs[:-1], ks) if k][-1]) >= 0.5 * 32
    xs, ks, opt = op.adversary_sequence(32.0, 1.5, 100, eager)
    assert ks[2:].all() and opt <= 2


# ---------------------------------------------------------------- Proposition E
def test_consecutive_rules_never_see_a_slow_ramp_but_anchored_rules_do():
    rng = np.random.default_rng(7)
    base = _scene(rng, 64, 64)
    T = 120
    frames = base[None] + rng.normal(0, 1.0, (T, 64, 64))
    ramp = np.linspace(0, 60, T)                                  # 0.5 grey levels per frame
    frames[:, 20:28, 20:28] += ramp[:, None, None]
    frames = np.clip(frames, 0, 255).astype(np.uint8)
    P = 16
    gy, gx = 20 // P, 20 // P
    # calibrate every consecutive rule to drop 97 % on the static control, then look at the ramp patch
    ctrl = np.clip(base[None] + rng.normal(0, 1.0, (T, 64, 64)), 0, 255).astype(np.uint8)
    for fn in (rd.consecutive, rd.cosine_consecutive, rd.frame_consecutive):
        thr, _, d = bl.calibrate(fn, ctrl, P, 0.97, hi=2.0 if fn is rd.cosine_consecutive else 255.0, iters=25)
        keep = fn(frames, P, thr)
        assert keep[1:, gy, gx].sum() <= 4                        # at most the static false-fire budget
    keep = core.Pruner(64, 64, P, 32.0).run(frames)["keep"]
    assert keep[1:, gy, gx].sum() >= 1 and keep[1:, gy, gx].any()
    first = 1 + int(np.argmax(keep[1:, gy, gx]))
    assert ramp[first] >= 32 - 14 and ramp[first] <= 32 + 16    # sent when the accumulated change (plus noise spread) reaches Delta
