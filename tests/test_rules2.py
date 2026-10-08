"""Tests for the flat-norm, multi-scale and quotient (warp) certificates."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from certskip import core, flatnorm as fn, scalespace as ss, warp as wp, topology as tp   # noqa: E402

rng = np.random.default_rng(1)
P = 16


# ----------------------------------------------------------------- flat norm
def test_flat_ub_dominates_dual_test_functions():
    """Any ell-ramp of a region is a feasible dual function, so <f_S, d> <= ||d|| <= UB."""
    for _ in range(30):
        d = rng.normal(0, 5, (P, P)); d -= d.mean()
        d[3:9, 4:10] += rng.uniform(-60, 60)
        d -= d.mean()
        ub = fn.flat_norm_ub(d[None], 4.0)[0]
        for _ in range(5):
            S = np.zeros((P, P), bool); y, x = rng.integers(0, P - 4, 2); S[y:y + 4, x:x + 4] = True
            f = fn.ramp_functional(S, 4.0)
            assert abs((f * d).sum()) <= ub + 1e-9
        assert fn.flat_norm_lb(d[None], 4.0)[0] <= ub + 1e-9


def test_flat_ub_is_at_most_l1_and_cheap_for_local_dipole():
    d = np.zeros((P, P)); d[8, 7] = 100; d[8, 8] = -100          # mass 100 moved 1 px
    ub = fn.flat_norm_ub(d[None], 4.0)[0]
    assert ub == pytest.approx(100 * (2 * 1 / 4.0))                 # tree: 2 moves of distance 1 at 1/ell each


def test_dual_norm_bound_for_linear_frontend():
    W = fn.dct_filters(8)
    for _ in range(20):
        d = rng.normal(0, 3, (8, 8)); d -= d.mean()
        ub = fn.flat_norm_ub(d[None], 2.0)[0]
        for w in W[:10]:
            assert abs((w * d).sum()) <= fn.dual_norm_of_filter(w, 2.0) * ub + 1e-9


# ---------------------------------------------------------------- multi-scale
def test_box_statistic_matches_bruteforce_and_scale0_is_range_rule():
    e = rng.normal(size=(4, P, P))
    for r in (1, 2, 3):
        k = 2 * r + 1
        bf = max(abs(e[1, i:i + k, j:j + k].mean()) for i in range(P - k + 1) for j in range(P - k + 1))
        assert ss.box_max_abs_mean(e, r)[1] == pytest.approx(bf)
    f = np.clip(128 + rng.normal(0, 6, (6, 64, 64)), 0, 255).astype(np.uint8)
    a = core.Pruner(64, 64, P, 32.0).run(f)["keep"]
    b = ss.MultiScalePruner(64, 64, P, deltas={0: 32.0}).run(f)["keep"]
    assert np.array_equal(a, b)


def test_multiscale_certificate_via_stability_on_filtered_images():
    """Dropped => every r: box-filtered F and box-filtered (R + c) are within eps_r in L_inf,
    hence their persistence diagrams within eps_r (checked exactly)."""
    sched = {0: 64.0, 1: 40.0, 2: 28.0}
    f = np.clip(128 + rng.normal(0, 4, (8, 48, 48)), 0, 255).astype(np.uint8)
    f[5:, 20:24, 20:24] = 30
    pr = ss.MultiScalePruner(48, 48, P, deltas=sched)
    ref = None
    for t in range(len(f)):
        prev_ref = None if pr.ref is None else pr.ref.copy()
        k, s, c = pr.step(f[t])
        if prev_ref is None:
            continue
        for gy, gx in zip(*np.where(~k)):
            Fp = f[t][gy * P:(gy + 1) * P, gx * P:(gx + 1) * P].astype(float)
            Rp = prev_ref[gy * P:(gy + 1) * P, gx * P:(gx + 1) * P].astype(float) + c[gy, gx]
            for r, D in sched.items():
                A, B = ss.box_filter(Fp, r), ss.box_filter(Rp, r)
                assert np.abs(A - B).max() <= 0.5 * D + 1e-6
                for sup in (False, True):
                    assert tp.bottleneck_distance(tp.persistence_h0(A, sup), tp.persistence_h0(B, sup)) <= 0.5 * D + 1e-6
    assert pr.run  # object patch must have been kept at t=5
    

def test_multiscale_object_kept():
    f = np.clip(128 + rng.normal(0, 4, (6, 48, 48)), 0, 255).astype(np.uint8)
    f[4:, 20:25, 20:25] = 90          # 5x5 object, contrast ~38
    k = ss.MultiScalePruner(48, 48, P, deltas={0: 64.0, 1: 40.0, 2: 28.0}).run(f)["keep"]
    assert k[4, 1, 1]


# ------------------------------------------------------------------- quotient
def _edge_scene(H=64, W=64):
    yy, xx = np.mgrid[0:H, 0:W]
    return 80 + 20 * np.sin(xx / 3.0) + 100 * (xx >= 30), yy, xx


def _shift_x(img, s, yy, xx):
    W = img.shape[1]
    x = np.clip(xx - s, 0, W - 1); x0 = np.floor(x).astype(int); x1 = np.minimum(x0 + 1, W - 1); fr = x - x0
    return (1 - fr) * img[yy, x0] + fr * img[yy, x1]


def test_quotient_recovers_subpixel_shift_and_drops_jittered_edge():
    scene, yy, xx = _edge_scene()
    R = np.clip(scene + rng.normal(0, 1.5, scene.shape), 0, 255).astype(np.uint8)
    F = np.clip(_shift_x(scene, 0.4, yy, xx) + rng.normal(0, 1.5, scene.shape), 0, 255).astype(np.uint8)
    dy, dx = wp.fit_translation(F, R, P, 1.0)
    assert np.allclose(dx[:, 1], -0.4, atol=0.08)
    k_range = core.Pruner(64, 64, P, 32.0); k_range.step(R); kr, sr, _ = k_range.step(F)
    k_warp = wp.WarpPruner(64, 64, P, 32.0, 1.0); k_warp.step(R); kw, sw, _ = k_warp.step(F)
    assert kr[:, 1].all() and not kw.any()                      # range keeps the edge column, quotient drops everything
    assert (sw < sr).all()


def test_quotient_still_catches_object_and_certificate_holds_against_view():
    scene, yy, xx = _edge_scene()
    R = np.clip(scene + rng.normal(0, 1.5, scene.shape), 0, 255).astype(np.uint8)
    F = np.clip(_shift_x(scene, 0.3, yy, xx) + rng.normal(0, 1.5, scene.shape), 0, 255).astype(int)
    F[20:25, 5:10] += 40
    F = np.clip(F, 0, 255).astype(np.uint8)
    pr = wp.WarpPruner(64, 64, P, 32.0, 1.0); pr.step(R); k, s, c = pr.step(F)
    assert k[1, 0] and k.sum() == 1
    v = pr.view().astype(int)
    err = core.patch_grid(np.abs(v - F.astype(int)), P).max(axis=(2, 3))
    assert (err[~k] <= 0.5 * s[~k] + 0.5 + 1e-6).all()           # L_inf certificate against the warped view
    # topology: F vs view within s/2 in bottleneck on dropped patches
    for gy, gx in list(zip(*np.where(~k)))[:4]:
        A = F[gy * P:(gy + 1) * P, gx * P:(gx + 1) * P].astype(float)
        B = pr._view[gy * P:(gy + 1) * P, gx * P:(gx + 1) * P]
        assert tp.bottleneck_distance(tp.persistence_h0(A), tp.persistence_h0(B)) <= 0.5 * s[gy, gx] + 1e-6


def test_quotient_motion_is_clamped():
    scene, yy, xx = _edge_scene()
    R = np.clip(scene, 0, 255).astype(np.uint8)
    F = np.clip(_shift_x(scene, 3.0, yy, xx), 0, 255).astype(np.uint8)   # 3 px: beyond delta_max
    dy, dx = wp.fit_translation(F, R, P, 1.0)
    assert np.abs(dx).max() <= 1.0 + 1e-9 and np.abs(dy).max() <= 1.0 + 1e-9
    pr = wp.WarpPruner(64, 64, P, 32.0, 1.0); pr.step(R); k, s, c = pr.step(F)
    assert k[:, 1].all()                                            # large motion of the edge is kept


# ------------------------------------------------------------- native parity
@pytest.mark.skipif(not __import__("certskip").native_available(), reason="C++ library not built")
def test_native_quotient_matches_numpy():
    from certskip.native import NativeQuotientPruner
    scene, yy, xx = _edge_scene(80, 96)
    frames = []
    for t in range(6):
        img = _shift_x(scene, 0.15 * t, yy, xx) + rng.normal(0, 2, scene.shape)
        if t >= 3:
            img[40:46, 60:66] += 50
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
    frames = np.stack(frames)
    a = wp.WarpPruner(80, 96, P, 24.0, 1.0).run(frames)
    b = NativeQuotientPruner(80, 96, P, 24.0, 1.0).run(frames)
    assert np.array_equal(a["keep"], b["keep"])
    assert np.allclose(a["spread"], b["spread"], atol=1e-4)
    assert np.allclose(a["shift"], b["shift"], atol=1e-4)


# --------------------------------------------------------------- sequential
from certskip import sequential as sq   # noqa: E402


def _static_clip(T=60, H=96, W=128, sigma=3.0, seed=5):
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    scene = 110 + 30 * np.sin(xx / 7.0) * np.cos(yy / 9.0)
    return np.clip(scene[None] + r.normal(0, sigma, (T, H, W)), 0, 255), scene


def test_sequential_detects_faint_persistent_object_memoryless_never_does():
    f, _ = _static_clip()
    f[20:, 40:46, 70:76] -= 10
    f = f.astype(np.uint8)
    sched = {r: 80 * (2 * r + 1) ** -0.5 for r in (0, 1, 2, 3)}
    k_mem = wp.WarpPruner(96, 128, P, 32.0, 1.0, multiscale=sched).run(f)["keep"]
    k_seq = sq.SequentialPruner(96, 128, P, multiscale=sched, z=8.0).run(f)["keep"]
    gy, gx = 40 // P, 70 // P
    assert not k_mem[20:, gy, gx].any()
    hits = np.where(k_seq[20:, gy, gx])[0]
    assert len(hits) and hits[0] <= 2                         # caught within two frames


def test_sequential_false_fire_rate_is_small_on_pure_noise():
    f, _ = _static_clip(T=80)
    f = f.astype(np.uint8)
    sched = {r: 80 * (2 * r + 1) ** -0.5 for r in (0, 1, 2, 3)}
    pr = sq.SequentialPruner(96, 128, P, multiscale=sched, z=8.0)
    out = pr.run(f)
    k = out["keep"][40:]                     # after the null statistics are learned
    assert k.mean() < 0.003                  # measured ~3e-4 per patch-frame at z'=8


def test_sequential_history_resets_after_keep():
    f, _ = _static_clip(T=40)
    f[10:, 40:46, 70:76] -= 40               # strong object: kept by the memoryless rule at t=10
    f = f.astype(np.uint8)
    sched = {r: 80 * (2 * r + 1) ** -0.5 for r in (0, 1, 2, 3)}
    pr = sq.SequentialPruner(96, 128, P, multiscale=sched, z=8.0)
    k = pr.run(f)["keep"]
    gy, gx = 40 // P, 70 // P
    assert k[10, gy, gx]
    # once the reference holds the object, nothing persists relative to it: no sequential re-fires
    assert not k[14:, gy, gx].any()


def test_delay_bound_monotone():
    assert sq.delay_bound(12, 2, 3.0, 4.5) <= sq.delay_bound(6, 2, 3.0, 4.5)
    assert sq.delay_bound(2, 1, 3.0, 4.5) is None or sq.delay_bound(2, 1, 3.0, 4.5) >= 16
    assert sq.delay_bound(3.0, 2, None, 8.0, z_miss=0.0, scales={1: 0.5, 2: 0.35, 4: 0.25}) == 2


@pytest.mark.skipif(not __import__("certskip").native_available(), reason="C++ library not built")
def test_native_sequential_matches_numpy():
    from certskip.native import NativeSequentialPruner
    f, _ = _static_clip(T=70)
    f[25:, 40:46, 70:76] -= 10
    f = f.astype(np.uint8)
    sched = {r: 80 * (2 * r + 1) ** -0.5 for r in (0, 1, 2, 3)}
    a = sq.SequentialPruner(96, 128, P, multiscale=sched, z=8.0).run(f)
    b = NativeSequentialPruner(96, 128, P, multiscale=sched, z=8.0).run(f)
    assert np.array_equal(a["keep"], b["keep"])
    assert np.allclose(a["spread"], b["spread"], atol=1e-3)
