"""Run with: python3 -m pytest tests -q   (from certified-skip/)"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import certskip as cs                      # noqa: E402
from certskip import topology as tp        # noqa: E402
from certskip import baselines as bl       # noqa: E402
from certskip import noise as nz           # noqa: E402

rng = np.random.default_rng(0)


def noisy_static(T=30, H=96, W=128, sigma=3.0, base=None, lo=0, hi=256):
    base = rng.integers(lo, hi, (H, W)).astype(np.int32) if base is None else base
    return np.clip(base[None] + rng.normal(0, sigma, (T, H, W)), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- rule semantics
def test_first_frame_fully_kept():
    f = noisy_static(T=3)
    pr = cs.Pruner(*f.shape[1:], 16, 32.0)
    k, s, c = pr.step(f[0])
    assert k.all() and (s == 0).all() and (c == 0).all()


def test_drop_iff_spread_below_delta_and_reference_updates():
    f = noisy_static(T=2, sigma=0, lo=40, hi=200)      # headroom so +shift never clips
    pr = cs.Pruner(*f.shape[1:], 16, 10.0)
    pr.step(f[0])
    nxt = f[1].astype(np.int32)
    nxt[0:16, 0:16] += 5              # pure shift, spread 0 -> dropped
    nxt[0:4, 16:20] += 9              # spread 9 < 10 -> dropped
    nxt[16:20, 0:4] += 10             # spread 10 == delta -> kept
    nxt = np.clip(nxt, 0, 255).astype(np.uint8)
    k, s, c = pr.step(nxt)
    assert not k[0, 0] and s[0, 0] == 0 and c[0, 0] == 5
    assert not k[0, 1] and s[0, 1] == 9
    assert k[1, 0] and s[1, 0] == 10
    ref = pr.reference()
    assert np.array_equal(ref[16:32, 0:16], nxt[16:32, 0:16])          # kept patch updated
    assert np.array_equal(ref[0:16, 0:16], f[1][0:16, 0:16])           # dropped patch not updated
    # compensated view removes the shift on the dropped patch
    v = pr.view()
    assert np.array_equal(v[0:16, 0:16], nxt[0:16, 0:16])


def test_max_shift_cap():
    f = noisy_static(T=2, sigma=0, lo=40, hi=200)
    pr = cs.Pruner(*f.shape[1:], 16, 32.0, max_shift=4.0)
    pr.step(f[0])
    nxt = np.clip(f[1].astype(np.int32) + 20, 0, 255).astype(np.uint8)   # uniform +20 (edge clipping aside)
    k, s, c = pr.step(nxt)
    assert k.all() and (s == 0).all()   # spread 0 everywhere but shift 20 > 4 -> kept


def test_uint8_does_not_wrap():
    f = np.zeros((2, 16, 16), np.uint8)
    f[1, 0, 0] = 255
    pr = cs.Pruner(16, 16, 16, 300.0)
    pr.step(f[0]); k, s, c = pr.step(f[1])
    assert s[0, 0] == 255 and c[0, 0] == 127.5 and not k[0, 0]


def test_view_error_bound_holds_pointwise():
    f = noisy_static(T=20, sigma=4.0)
    pr = cs.Pruner(*f.shape[1:], 16, 40.0)
    for t in range(len(f)):
        k, s, c = pr.step(f[t])
        v = pr.view().astype(np.int32)
        err = cs.patch_grid(np.abs(v - f[t].astype(np.int32)), 16).max(axis=(2, 3))
        # +0.5 for integer rounding of the compensated view
        assert (err[~k] <= 0.5 * s[~k] + 0.5 + 1e-6).all()


# ---------------------------------------------------------------- native parity
@pytest.mark.skipif(not cs.native_available(), reason="C++ library not built (run make in cpp/)")
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_native_matches_numpy_bitwise(dtype):
    f = noisy_static(T=25, H=80, W=112, sigma=5.0)
    if dtype is np.uint16:
        f = (f.astype(np.uint16) * 4)
    elif dtype is np.float32:
        f = f.astype(np.float32) / 3.0
    delta = 32.0 if dtype is np.uint8 else (128.0 if dtype is np.uint16 else 10.0)
    a = cs.Pruner(*f.shape[1:], 16, delta, max_shift=20.0)
    b = cs.NativePruner(*f.shape[1:], 16, delta, max_shift=20.0, dtype=dtype)
    ra, rb = a.run(f), b.run(f)
    assert np.array_equal(ra["keep"], rb["keep"])
    assert np.array_equal(ra["spread"], rb["spread"])
    assert np.array_equal(ra["shift"], rb["shift"])
    assert np.array_equal(a.reference(), b.reference())
    assert np.array_equal(a.view(), b.view())


@pytest.mark.skipif(not cs.native_available(), reason="C++ library not built")
def test_native_step_and_run_agree_and_reset():
    f = noisy_static(T=10)
    a = cs.NativePruner(*f.shape[1:], 16, 32.0)
    r1 = a.run(f)
    a.reset()
    r2 = {"keep": np.stack([a.step(x)[0] for x in f])}
    assert np.array_equal(r1["keep"], r2["keep"])


def test_non_multiple_frame_size_ignores_remainder():
    f = noisy_static(T=4, H=100, W=130)
    pr = cs.make_pruner(100, 130, 16, 32.0)
    out = pr.run(f)
    assert out["keep"].shape == (4, 6, 8)


# ---------------------------------------------------------------- topology
def test_persistence_simple_blob():
    p = np.full((16, 16), 100.0); p[5:8, 5:8] = 50
    D = tp.persistence_h0(p)
    assert D.shape == (1, 2) and D[0, 0] == 50 and D[0, 1] == 100
    assert tp.features_at_least(D, 50) == 1 and tp.features_at_least(D, 51) == 0
    assert tp.persistence_h0(np.full((8, 8), 7.0)).shape == (0, 2)
    Db = tp.persistence_h0(p, superlevel=True)        # bright: the plateau, contrast 50
    assert tp.max_contrast(Db) == 50


def test_persistence_two_blobs_elder_rule():
    p = np.full((16, 16), 100.0); p[2:4, 2:4] = 10; p[10:12, 10:12] = 30
    D = tp.persistence_h0(p)
    pers = sorted(tp.persistence(D))
    assert pers == [70.0, 90.0]                        # (30->100) and (10->100)


def test_bottleneck_basic():
    A = np.array([[0.0, 10.0]]); B = np.array([[1.0, 12.0]])
    assert tp.bottleneck_distance(A, B) == 2.0
    assert tp.bottleneck_distance(A, np.zeros((0, 2))) == 5.0     # to diagonal
    assert tp.bottleneck_distance(A, A) == 0.0
    C = np.array([[0.0, 10.0], [0.0, 1.0]])
    assert tp.bottleneck_distance(A, C) == 0.5                     # extra tiny point -> diagonal


def test_stability_bound_random_patches():
    """||f-g||_inf = eps  =>  bottleneck <= eps (the theorem the rule rests on)."""
    for _ in range(50):
        f = rng.integers(0, 256, (12, 12)).astype(float)
        eps = rng.uniform(0, 40)
        g = f + rng.uniform(-eps, eps, f.shape)
        e = np.abs(f - g).max()
        for sup in (False, True):
            bd = tp.bottleneck_distance(tp.persistence_h0(f, sup), tp.persistence_h0(g, sup))
            assert bd <= e + 1e-9


def test_soundness_audit_zero_violations():
    f = noisy_static(T=12, H=64, W=64, sigma=2.0, lo=100, hi=200)   # sigma 2 -> most patches dropped
    f[6:, 20:24, 20:24] = 0                          # object of contrast >= 100 appears -> must be kept
    pr = cs.Pruner(64, 64, 16, 32.0)
    out = pr.run(f)
    assert out["keep"][6, 1, 1]
    rep = tp.soundness_audit(f, out["keep"], 16, 32.0, max_patches=None)
    assert rep["checked"] > 0 and rep["violations"] == 0
    assert rep["worst_bottleneck_over_half_spread"] <= 1.0 + 1e-9


# ---------------------------------------------------------------- optimality (tightness)
def test_rule_is_tight_any_spread_delta_change_can_create_delta_object():
    """For a random change pattern d with spread exactly delta, there is a
    scene (reference) where applying d creates a dark feature of contrast
    delta out of a flat patch.  So no change-only rule can safely drop at a
    spread >= delta: the certified threshold is the optimal one."""
    delta = 32.0
    for _ in range(200):
        d = rng.uniform(0, 1, (8, 8))
        d = (d - d.min()) / (d.max() - d.min()) * delta       # spread exactly delta
        ref = 128.0 - d + rng.uniform(0, 0.0, d.shape)         # scene = flat 128 minus d ...
        # ... is itself not flat, so instead take the flat scene and show the
        # feature set changes by a full delta-contrast object:
        flat = np.full((8, 8), 128.0)
        new = flat + d
        D0, D1 = tp.persistence_h0(flat), tp.persistence_h0(new)
        assert tp.features_at_least(D0, delta) == 0
        assert tp.features_at_least(D1, delta) == 1              # object of contrast = delta appeared
        assert tp.bottleneck_distance(D0, D1) == pytest.approx(delta / 2)


# ---------------------------------------------------------------- baselines + noise
def test_calibration_hits_target():
    f = noisy_static(T=20, sigma=3.0)
    tau, keep, d = bl.calibrate(bl.consecutive_mean, f, 16, 0.9)
    assert abs(d - 0.9) < 0.03
    _, keep2, d2 = bl.calibrate(bl.certified, f, 16, 0.9)
    assert abs(d2 - 0.9) < 0.05


def test_noise_estimate_and_prediction():
    f = noisy_static(T=30, H=128, W=128, sigma=3.0)
    est = nz.estimate_temporal_noise(f, 16)
    assert abs(est["sigma_pixel"] - 3.0) < 0.6
    s = nz.simulate_spread(3.0, 16)
    assert 20 < s.mean() < 28            # ~7.5 sigma for 256 samples
    pred = nz.predicted_drop_rate(3.0, 32.0, 16)
    meas = nz.sweep_delta(f, [32.0], 16)[0]["drop_rate"]
    assert abs(pred - meas) < 0.1
