"""Tests for the moving-camera rule: pose recovery, the certificate on every dropped patch, anchors."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from certskip import moving as mv, warp as wp, scalespace as ss   # noqa: E402
from certskip.core import patch_grid                              # noqa: E402

P = 16
H, W = 160, 224
SCHED = {r: 80.0 * (2 * r + 1) ** -0.5 for r in (0, 1, 2, 3)}
YY, XX = np.mgrid[0:H, 0:W].astype(float)


def texture(rng, h, w, amp=45.0, passes=6):
    t = rng.normal(0, 1, (h, w))
    for _ in range(passes):
        t = (t + np.roll(t, 1, 0) + np.roll(t, -1, 0) + np.roll(t, 1, 1) + np.roll(t, -1, 1)) / 5
    return np.clip(128 + amp * t / t.std(), 0, 255)


def camera(t, moving=True):
    if not moving:
        return np.array([[1.0, 0.0, 120.0], [0.0, 1.0, 90.0]])
    s = 1 + 0.06 * np.sin(t / 25.0); th = 0.015 * np.sin(t / 18.0)
    a, b = s * np.cos(th), s * np.sin(th)
    return np.array([[a, -b, 120 + 1.6 * t + 2 * np.sin(t / 5.0)], [b, a, 90 + 0.5 * t + 1.5 * np.cos(t / 7.0)]])


def shoot(world, A, rng, obj=None, at=None):
    X, Y = mv.apply(A, XX, YY)
    f, _ = mv.bilinear(world, X, Y); f = f.copy()
    inside = np.zeros(f.shape, bool)
    if obj is not None:
        o, inside = mv.bilinear(obj, XX - at[0], YY - at[1])
        f[inside] = o[inside]
    return np.clip(np.rint(f + rng.normal(0, 1.0, f.shape)), 0, 255).astype(np.uint8), inside


def certificate_violations(frame, view, keep, deltas):
    """Theorem 3 on the patches that were not sent: every box mean of frame - view below eps_r."""
    e = patch_grid(frame.astype(float) - view, P)
    return sum(int((ss.box_max_abs_mean(e, r)[~keep] >= 0.5 * D).sum()) for r, D in deltas.items())


def test_a_camera_that_does_not_move_reveals_nothing_and_matches_the_fixed_rule():
    rng = np.random.default_rng(0)
    world = texture(rng, 360, 480)
    pr = mv.MovingPruner(H, W, P, SCHED); fixed = wp.WarpPruner(H, W, P, 32.0, 1.0, multiscale=SCHED)
    agree = total = 0
    for t in range(25):
        f, _ = shoot(world, camera(t, moving=False), rng)
        f[60:72, 20 + 3 * t: 32 + 3 * t] = 230                      # something walks through
        k = pr.step(f); kf = fixed.step(f)[0]
        if t:
            assert not pr.last["anchor"] and not pr.last["reveal"].any()
            agree += int((k == kf).sum()); total += k.size
    assert agree / total > 0.995


def test_pose_is_recovered_and_camera_motion_is_skipped():
    rng = np.random.default_rng(1)
    world = texture(rng, 360, 480)
    pr = mv.MovingPruner(H, W, P, SCHED); fixed = wp.WarpPruner(H, W, P, 32.0, 1.0, multiscale=SCHED)
    A0 = np.vstack([camera(0), [0, 0, 1]])
    E0 = np.array([[1, 0, (pr.canvas.W - W) // 2], [0, 1, (pr.canvas.H - H) // 2], [0, 0, 1.0]])
    corners = np.array([[0, W - 1, 0, W - 1], [0, 0, H - 1, H - 1], [1, 1, 1, 1.0]])
    errs = []; sent = sent_fixed = refresh = total = 0
    for t in range(40):
        A = camera(t)
        f, _ = shoot(world, A, rng)
        k = pr.step(f); kf = fixed.step(f)[0]
        if not t:
            continue
        assert not pr.last["anchor"]
        true = np.linalg.inv(A0) @ np.vstack([A, [0, 0, 1]])        # frame t -> frame 0
        est = np.linalg.inv(E0) @ np.vstack([pr.last["pose"], [0, 0, 1]])
        errs.append(np.abs(((true - est) @ corners)[:2]).max())
        sent += int(k.sum()); sent_fixed += int(kf.sum()); refresh += int((k & ~pr.last["reveal"]).sum()); total += k.size
        assert certificate_violations(f, pr.last["view"], k, SCHED) == 0
    assert np.median(errs) < 0.1 and max(errs) < 0.5               # pixels, at the frame corners
    assert refresh / total < 0.02                                   # held patches almost never fail
    assert sent / total < 0.25 < 0.7 < sent_fixed / total           # the rest of what is sent is the frontier


def test_an_object_that_moves_on_its_own_is_followed_and_the_certificate_holds():
    rng = np.random.default_rng(2)
    world = texture(rng, 360, 480); obj = texture(rng, 64, 80, amp=55.0)
    plain = mv.MovingPruner(H, W, P, SCHED); follow = mv.MovingPruner(H, W, P, SCHED, search=12)
    sent = {"plain": 0, "follow": 0}; followed = n = 0
    for t in range(40):
        f, inside = shoot(world, camera(t), rng, obj, (20.3 + 2.2 * t, 50 + 0.4 * t))
        kp = plain.step(f); kf = follow.step(f)
        if not t:
            continue
        on = patch_grid(inside, P).all((2, 3))                      # patches wholly on the object
        sent["plain"] += int((kp & on).sum()); sent["follow"] += int((kf & on).sum())
        followed += int((follow.last["followed"] & on).sum()); n += int(on.sum())
        assert certificate_violations(f, plain.last["view"], kp, SCHED) == 0
        assert certificate_violations(f, follow.last["view"], kf, SCHED) == 0
        assert not (follow.last["followed"] & kf).any()             # a followed patch is not sent
    assert sent["plain"] / n > 0.9                                  # without the search the object is re-sent
    assert followed / n > 0.3 and sent["follow"] < sent["plain"]


def test_a_cut_anchors_and_sends_the_whole_frame():
    rng = np.random.default_rng(3)
    a, b = texture(rng, 360, 480), texture(rng, 360, 480)
    pr = mv.MovingPruner(H, W, P, SCHED)
    for t in range(8):
        k = pr.step(shoot(a, camera(t), rng)[0])
    assert not pr.last["anchor"] and k.mean() < 0.5
    k = pr.step(shoot(b, camera(8), rng)[0])
    assert pr.last["anchor"] and k.all()
    k = pr.step(shoot(b, camera(9), rng)[0])
    assert not pr.last["anchor"] and k.mean() < 0.5


def test_the_canvas_keeps_its_content_when_it_scrolls():
    rng = np.random.default_rng(4)
    c = mv.Canvas(120, 160)
    c.img[:] = rng.uniform(0, 255, c.img.shape); c.seen[:] = True
    before = c.img[40:60, 70:90].copy()
    assert c.follow(100.0, 50.0, 200.0, 110.0)                      # a box that sticks out to the right
    assert (c.ox, c.oy) != (0, 0)
    assert np.array_equal(c.img[40 - c.oy: 60 - c.oy, 70 - c.ox: 90 - c.ox], before)
    assert not c.seen[:, 160 - c.ox:].any()                         # what scrolled in has not been seen
    assert not c.follow(0.0, 0.0, 400.0, 50.0)                      # larger than the window
