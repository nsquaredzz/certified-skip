"""Tests for THEORY.md section 11: the fewest sends a certificate allows, and holding a centre."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from certskip import centre as ct, warp as wp   # noqa: E402

P = 16
SCHED = {r: 80 * (2 * r + 1) ** -0.5 for r in (0, 1, 2, 3)}
SUP = {0: 80.0}                                                     # sup norm, eps = 40 grey levels


def _clip(T=48, H=64, W=96, seed=11):
    """A noisy textured scene with a slow lighting drift, a flickering region and an object crossing it."""
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    scene = 110 + 30 * np.sin(xx / 7.0) * np.cos(yy / 9.0)
    f = scene[None] + r.normal(0, 3.0, (T, H, W)) + np.linspace(0, 25, T)[:, None, None]
    f[:, 16:32, 48:64] += 30 * (np.arange(T) % 2)[:, None, None]    # flicker of 30 grey levels on one patch
    for t in range(10, 40):
        x0 = 2 * (t - 10); f[t, 36:52, x0:x0 + 14] -= 70            # a dark object moving 2 px per frame
    return np.clip(f, 0, 255).astype(np.uint8)


def _sends(frames, sched, L):
    out = ct.CentrePruner(frames.shape[1], frames.shape[2], P, multiscale=sched, lookahead=L).run(frames)
    return out["keep"].sum(0), out["worst"]


def test_no_lookahead_is_send_on_delta():
    f = _clip()
    a = ct.CentrePruner(64, 96, P, multiscale=SCHED, lookahead=0).run(f)["keep"]
    b = wp.WarpPruner(64, 96, P, delta_max=0.0, multiscale=SCHED, offset="none").run(f)["keep"]
    assert np.array_equal(a, b)                                      # the rule of section 9 without the sub-pixel shift


@pytest.mark.parametrize("sched", [SUP, SCHED])
def test_certificate_holds_on_every_frame_at_every_lookahead(sched):
    f = _clip()
    for L in (0, 1, 3, 6):
        n, worst = _sends(f, sched, L)
        assert worst < 1.0                                          # the held copy is within the certificate of every frame it stood for


@pytest.mark.parametrize("sched", [SUP, SCHED])
def test_the_sandwich_of_theorems_13_14_and_16(sched):
    f = _clip(); T = len(f)
    low = ct.fewest_sends(f, P, multiscale=sched)                    # no policy sends fewer
    half = ct.fewest_sends(f, P, multiscale=sched, tolerance=0.5)    # the same bound at half the tolerance
    for L in (0, 1, 3, 6):
        n, _ = _sends(f, sched, L)
        assert (low <= n).all()                                      # Theorem 13
        if sched is SUP:
            assert (n <= low + -(-T // (L + 1))).all()               # Theorem 16: at most one extra send per L + 1 frames
    n0, _ = _sends(f, sched, 0)
    assert (n0 <= half).all()                                        # Theorem 14: send-on-delta is optimal for half the tolerance
    assert low.sum() < n0.sum()                                      # and on this clip it does pay for holding a frame


def test_sup_norm_bound_is_attained_with_enough_lookahead():
    f = _clip()
    n, worst = _sends(f, SUP, len(f))                                # the whole future in view: the offline optimum
    assert np.array_equal(n, ct.fewest_sends(f, P, multiscale=SUP)) and worst < 1.0


def test_flicker_costs_a_frame_holder_every_frame_and_a_centre_holder_one_send():
    T = 40; f = np.full((T, 16, 16), 100.0); f[::2] += 25; f[1::2] -= 25      # range 50: below 2 eps = 80, above eps = 40
    f = f.astype(np.uint8)
    assert ct.fewest_sends(f, P, multiscale=SUP)[0, 0] == 1
    assert _sends(f, SUP, 0)[0][0, 0] == T                           # send-on-delta holds an extreme and is wrong every frame
    assert _sends(f, SUP, 1)[0][0, 0] == 1                           # one frame of lookahead: hold the midpoint, never send again
    assert ct.fewest_sends(f, P, multiscale=SUP, tolerance=0.5)[0, 0] == T    # Theorem 14's upper bound is attained
