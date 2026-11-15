"""The KV-cache path is exactly the training path; eviction and re-indexing do what they say."""
import sys
from pathlib import Path
import numpy as np
import pytest
torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from longvid.model import LongVidDiT, KVCache
from longvid import data


def small(seed=0):
    torch.manual_seed(seed); m = LongVidDiT(frame=(16, 16), patch=4, dim=32, depth=3, heads=2).eval()
    for p in m.parameters():                                   # zero-init heads would make every comparison trivial
        if p.dim() > 1 and p.abs().sum() == 0: torch.nn.init.normal_(p, std=0.05)
    for b in m.blocks: torch.nn.init.normal_(b.ada.weight, std=0.1)
    torch.nn.init.normal_(m.fada.weight, std=0.1); return m


def test_cache_equals_training_forward():
    m = small(); B, Fn = 2, 4; g = torch.Generator().manual_seed(1)
    clean = torch.randn(B, Fn, m.N, m.pd, generator=g); noisy = torch.randn(B, Fn, m.N, m.pd, generator=g); sig = torch.rand(B, Fn, generator=g)
    with torch.no_grad():
        ref = m.forward_train(noisy, clean, sig)
        for j in range(Fn):
            cache = KVCache(m.depth)
            for i in range(j):
                m.forward_chunk(clean[:, i], torch.zeros(B), i, cache, write=True)
            out = m.forward_chunk(noisy[:, j], sig[:, j], j, cache, write=False)
            assert torch.allclose(out, ref[:, j], atol=1e-4), j


def test_a_noisy_frame_does_not_see_the_future():
    m = small(); B, Fn = 1, 4; g = torch.Generator().manual_seed(2)
    clean = torch.randn(B, Fn, m.N, m.pd, generator=g); noisy = torch.randn(B, Fn, m.N, m.pd, generator=g); sig = torch.rand(B, Fn, generator=g)
    c2, n2 = clean.clone(), noisy.clone(); c2[:, 3] += 5; n2[:, 3] += 5
    with torch.no_grad():
        a, b = m.forward_train(noisy, clean, sig), m.forward_train(n2, c2, sig)
    assert torch.allclose(a[:, :3], b[:, :3], atol=1e-5) and not torch.allclose(a[:, 3], b[:, 3])


def test_window_and_sink_keep_the_right_frames():
    m = small(); cache = KVCache(m.depth, sink=2, window=3); B = 1
    for t in range(10):
        m.forward_chunk(torch.randn(B, m.N, m.pd), torch.zeros(B), t, cache, write=True)
    assert cache.ts == [0, 1, 7, 8, 9]
    cache2 = KVCache(m.depth, sink=2, window=3, reindex=True)
    for t in range(10):
        m.forward_chunk(torch.randn(B, m.N, m.pd), torch.zeros(B), t, cache2, write=True)
    assert cache2.times() == [0, 1, 2, 3, 4] and cache2.query_t(10) == 5


def test_window_equals_full_when_nothing_is_evicted():
    m = small(); B = 1; g = torch.Generator().manual_seed(3); xs = [torch.randn(B, m.N, m.pd, generator=g) for _ in range(4)]
    full, win = KVCache(m.depth), KVCache(m.depth, sink=1, window=8)
    with torch.no_grad():
        for t in range(3):
            m.forward_chunk(xs[t], torch.zeros(B), t, full, True); m.forward_chunk(xs[t], torch.zeros(B), t, win, True)
        a = m.forward_chunk(xs[3], torch.full((B,), 0.5), 3, full); b = m.forward_chunk(xs[3], torch.full((B,), 0.5), 3, win)
    assert torch.allclose(a, b, atol=1e-6)


def test_world_is_a_coherent_video():
    ep, bg, cols = data.episode(np.random.default_rng(0), 40)
    assert ep.shape == (40, 3, 32, 32) and ep.min() >= -1 and ep.max() <= 1
    diff = np.abs(ep - bg[None]).max(1) > 0.3
    assert 0 < diff.mean() < 0.35 and (diff.sum((1, 2)) > 0).all()
