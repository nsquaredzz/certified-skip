"""Segmentation layer: exactness of the TV relaxation on a toy problem, clean-up utilities,
and C++ / numpy agreement of the real-time tracker."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import certskip as cs                                                     # noqa: E402
from certskip import segment as sg, native                                # noqa: E402

rng = np.random.default_rng(3)


def test_tv_segment_recovers_a_blob_and_rejects_speckle():
    H, W = 48, 64
    f = np.full((H, W), 0.8)                                               # background evidence
    f[14:34, 20:44] = -1.5                                                 # a strong rectangle of foreground evidence
    f[5, 5] = -3.0                                                         # a one-pixel speckle
    w = np.ones((H, W))
    u = sg.tv_segment(f, w, lam=1.0, iters=300)
    m = u > 0.5
    assert m[14:34, 20:44].mean() > 0.97 and not m[5, 5]                   # blob kept, speckle smoothed away
    assert m.sum() < 20 * 24 * 1.15


def test_fill_holes_and_closing_bridge_gaps():
    m = np.zeros((40, 40), bool); m[10:30, 10:30] = True; m[18:22, 10:30] = False   # object cut in two by a 4-px gap
    closed = sg._erode(sg._dilate(m, 2), 2)
    assert closed[18:22, 12:28].all()
    holed = m.copy(); holed[15:17, 15:17] = False
    assert native.fill_holes(holed.astype(np.uint8))[15:17, 15:17].all()


def _clip(T=40, H=96, W=128, sigma=2.5):
    yy, xx = np.mgrid[0:H, 0:W]
    scene = 120 + 25 * np.sin(xx / 6.0) * np.cos(yy / 8.0)
    fr = np.clip(scene[None] + rng.normal(0, sigma, (T, H, W)), 0, 255)
    for t in range(12, T):                                                 # a 14x10 object moving 2 px/frame
        x = 20 + 2 * (t - 12)
        fr[t, 40:54, x:x + 10] = 40
    return fr.astype(np.uint8)


def test_realtime_tracker_segments_moving_object():
    fr = _clip(); H, W = fr.shape[1:]
    seg = sg.RealtimeSegmenter(H, W, 16, init_frames=fr[:10], steps=3, min_area=20)
    for t in range(fr.shape[0]):
        m = seg.step(fr[t])
    x = 20 + 2 * (39 - 12)
    truth = np.zeros((H, W), bool); truth[40:54, x:x + 10] = True
    inter = (m & truth).sum(); union = (m | truth).sum()
    assert inter / union > 0.7
    assert len(seg.ids) == 1


@pytest.mark.skipif(not cs.native_available(), reason="C++ library not built")
def test_native_segmenter_matches_numpy():
    fr = _clip(); H, W = fr.shape[1:]
    a = sg.RealtimeSegmenter(H, W, 16, init_frames=fr[:10], steps=3, min_area=20)
    b = native.NativeRtSegmenter(H, W, 16, init_frames=fr[:10], steps=3, min_area=20)
    ious = []
    for t in range(fr.shape[0]):
        ma = a.step(fr[t]); mb = b.step(fr[t])
        u = (ma | mb).sum(); ious.append((ma & mb).sum() / u if u else 1.0)
    assert np.mean(ious[15:]) > 0.9
