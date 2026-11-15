"""CertViT: streaming with the halo is exactly the encoder on the held image, locality is what it says, logits are bounded."""
import sys
from pathlib import Path
import numpy as np
import pytest
torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from certskip.certvit import CertViT, StreamingCertViT

GH, GW, P = 6, 7, 8


def model(seed=0, depth=4, radius=1):
    torch.manual_seed(seed)
    return CertViT((GH, GW), patch=P, dim=32, depth=depth, heads=2, radius=radius).eval()


def frames(n, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 256, (GH * P, GW * P), generator=g).float() for _ in range(n)]


def hold(V, frame, keep):
    m = keep.view(GH, 1, GW, 1).expand(GH, P, GW, P).reshape(GH * P, GW * P)
    return torch.where(m, frame, V)


def test_keep_all_equals_fresh_pass():
    m = model(); s = StreamingCertViT(m); f = frames(3)
    s.step(f[0], torch.ones(GH, GW, dtype=torch.bool))
    out = s.step(f[1], torch.ones(GH, GW, dtype=torch.bool))
    assert torch.allclose(out, m.encode(f[1]), atol=1e-5)


@pytest.mark.parametrize("radius,depth", [(1, 4), (2, 3)])
def test_halo_is_exact_on_the_held_image(radius, depth):
    m = model(depth=depth, radius=radius); s = StreamingCertViT(m); f = frames(6); g = torch.Generator().manual_seed(5)
    V = f[0].clone(); s.step(f[0], torch.ones(GH, GW, dtype=torch.bool))
    for t in range(1, 6):
        keep = torch.rand(GH, GW, generator=g) < 0.15
        V = hold(V, f[t], keep); out = s.step(f[t], keep, "full")
        assert torch.allclose(out, m.encode(V), atol=1e-5), t
        for a, b in zip(s.h, m.encode(V, all_layers=True)):
            assert torch.allclose(a, b, atol=1e-5)


def test_no_halo_is_stale_next_to_a_sent_patch():
    m = model(depth=2); s = StreamingCertViT(m); f = frames(2)
    V = f[0].clone(); s.step(f[0], torch.ones(GH, GW, dtype=torch.bool))
    keep = torch.zeros(GH, GW, dtype=torch.bool); keep[3, 3] = True
    V = hold(V, f[1], keep); out = s.step(f[1], keep, "none"); ref = m.encode(V)
    err = (out - ref).abs().amax(-1).view(GH, GW)
    ys, xs = torch.meshgrid(torch.arange(GH), torch.arange(GW), indexing="ij")
    dist = torch.maximum((ys - 3).abs(), (xs - 3).abs())
    assert err[(dist == 1)].max() > 1e-4                                # the neighbours are stale
    assert err[(dist > 2)].max() < 1e-5                                 # and nothing beyond depth * r is


def test_locality_radius_grows_by_r_per_layer():
    m = model(depth=4, radius=1); a, b = frames(2); b = a.clone(); b[3 * P:4 * P, 3 * P:4 * P] += 40.0
    ha, hb = m.encode(a, True), m.encode(b, True)
    for l in range(5):
        diff = (ha[l] - hb[l]).abs().amax(-1).view(GH, GW) > 1e-6
        ys, xs = diff.nonzero(as_tuple=True)
        reach = max(int((ys - 3).abs().max()), int((xs - 3).abs().max())) if len(ys) else 0
        assert reach <= l                                               # r = 1: at layer l, at most l patches away
        if l: assert diff[3, 3]


def test_logits_are_bounded():
    m = model(); torch.manual_seed(9)
    for b in m.blocks:
        q, k, v = b.kv(torch.randn(GH * GW, 32) * 50)
        s = torch.einsum("mhd,nhd->hmn", q, k) * b.scale()
        assert s.abs().max() <= float(b.scale().detach().max()) + 1e-4
