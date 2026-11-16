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


from longvid.model import MergeCache, Block
from longvid.sample import rollout


def _commit_all(m, cache, xs):
    with torch.no_grad():
        for t, x in enumerate(xs):
            m.forward_chunk(x, torch.zeros(x.shape[0]), t, cache, write=True)


def test_merge_with_zero_tolerance_is_the_full_cache():
    m = small(); g = torch.Generator().manual_seed(4); xs = [torch.randn(2, m.N, m.pd, generator=g) for _ in range(4)]
    full, merge = KVCache(m.depth), MergeCache(m.depth, eps=0.0)
    _commit_all(m, full, xs); _commit_all(m, merge, xs)
    q = torch.randn(2, m.N, m.pd, generator=g)
    with torch.no_grad():
        a = m.forward_chunk(q, torch.full((2,), 0.5), 4, full); b = m.forward_chunk(q, torch.full((2,), 0.5), 4, merge)
    assert torch.allclose(a, b, atol=1e-4) and merge.entries() == 4 * m.N


def test_identical_content_is_one_entry_with_a_count():
    m = small(); x = torch.randn(2, m.N, m.pd); merge = MergeCache(m.depth, eps=0.01)
    _commit_all(m, merge, [x] * 5)
    assert merge.entries() == m.N                                      # one entry per location, however many frames
    assert int(merge.count[:, 0].min()) == 5 and merge.count[:, 1:].max() == 1 and float(merge.tpos[:, 0].min()) == 4.0


def test_one_entry_with_log_count_equals_its_copies():
    torch.manual_seed(0); blk = Block(32, 2, (4, 6, 6)).eval(); torch.nn.init.normal_(blk.ada.weight, std=0.1)
    B, n, E = 1, 5, 6; x = torch.randn(B, n, 32); cond = torch.randn(B, n, 32)
    pos = torch.cat([torch.full((n, 1), 3), torch.arange(n)[:, None], torch.zeros(n, 1, dtype=torch.long)], 1)
    k = torch.nn.functional.normalize(torch.randn(B, E, 2, 16), dim=-1); v = torch.randn(B, E, 2, 16)
    pe = torch.cat([torch.zeros(E, 1, dtype=torch.long), torch.arange(E)[:, None], torch.zeros(E, 1, dtype=torch.long)], 1)
    with torch.no_grad():
        copies = torch.cat([k, k[:, :1], k[:, :1]], 1)[:, :E + 2]; vc = torch.cat([v, v[:, :1], v[:, :1]], 1)
        pc = torch.cat([pe, pe[:1], pe[:1]], 0)                          # entry 0 appears three times, at the same position
        full, _ = blk(x, cond, pos, past=(copies, vc, pc))
        bias = torch.zeros(B, E); bias[:, 0] = torch.log(torch.tensor(3.0))
        merged, _ = blk(x, cond, pos, past=(k, v, pe, bias))
    assert torch.allclose(full, merged, atol=1e-5)


def test_window_drops_what_changed_and_keeps_what_stayed():
    m = small(); B = 1; static = torch.randn(B, m.N, m.pd); merge = MergeCache(m.depth, eps=0.01, window=3); g = torch.Generator().manual_seed(6)
    with torch.no_grad():
        for t in range(12):
            x = static.clone(); x[:, : m.N // 2] = torch.randn(B, m.N // 2, m.pd, generator=g)         # half the locations change every frame
            m.forward_chunk(x, torch.zeros(B), t, merge, write=True)
    assert merge.entries() <= m.N // 2 * 4 + m.N // 2                   # changing half: at most window + 1 frames of entries; static half: one each
    assert merge.alive[0, 0, m.N // 2:].all() or merge.alive[0].sum(0)[m.N // 2:].min() == 1


def test_rollout_runs_with_every_cache():
    m = small(); prompt = torch.rand(2, 1, 3, 16, 16) * 2 - 1
    for mk in (None, lambda d: MergeCache(d, 0.2), lambda d: MergeCache(d, 0.2, window=4)):
        out = rollout(m, prompt, 5, steps=2, make_cache=mk, gen=torch.Generator().manual_seed(0))
        assert out.shape == (2, 5, 3, 16, 16) and torch.isfinite(out).all()


from longvid.model import MotionMergeCache


def test_motion_with_radius_zero_is_merge():
    m = small(); g = torch.Generator().manual_seed(8); base = torch.randn(2, m.N, m.pd, generator=g)
    xs = [base.clone() for _ in range(4)]; xs[2][:, :3] += 1.0                         # a few locations change on one frame
    a, b = MergeCache(m.depth, 0.1), MotionMergeCache(m.depth, 0.1, (m.gh, m.gw), radius=0)
    _commit_all(m, a, xs); _commit_all(m, b, xs); q = torch.randn(2, m.N, m.pd, generator=g)
    with torch.no_grad():
        oa = m.forward_chunk(q, torch.full((2,), 0.5), 4, a); ob = m.forward_chunk(q, torch.full((2,), 0.5), 4, b)
    assert a.entries() == b.entries() and torch.allclose(oa, ob, atol=1e-5)


def test_a_pan_by_one_token_adds_only_the_new_column():
    m = small(); gh, gw = m.gh, m.gw; B, T = 1, 6; g = torch.Generator().manual_seed(9)
    world = torch.randn(B, gh, gw + T, m.pd, generator=g)                               # content wider than the frame
    xs = [world[:, :, t:t + gw].reshape(B, gh * gw, m.pd) for t in range(T)]           # the view slides right by one token a frame
    c = MotionMergeCache(m.depth, 0.01, (gh, gw), radius=1); _commit_all(m, c, xs)
    assert c.entries() == gh * gw + (T - 1) * gh                                        # every frame: only the column that entered
    plain = MergeCache(m.depth, 0.01); _commit_all(m, plain, xs)
    assert plain.entries() == T * gh * gw                                               # a merge that cannot move keeps nothing
    alive0 = c.alive[0, 0].view(gh, gw)                                                  # first frame's entries: column 0 left the view
    assert alive0.all() and float(c.xpos[0, 0].view(gh, gw)[0, 1]) == 0.0              # the entry made at column 1 now sits at column 0


def test_an_entry_is_claimed_once():
    m = small(); gh, gw = m.gh, m.gw; B = 1; g = torch.Generator().manual_seed(10)
    x0 = torch.randn(B, gh * gw, m.pd, generator=g); x1 = torch.randn(B, gh * gw, m.pd, generator=g)
    x1.view(B, gh, gw, -1)[:, 1, 1] = x0.view(B, gh, gw, -1)[:, 1, 1]; x1.view(B, gh, gw, -1)[:, 1, 2] = x0.view(B, gh, gw, -1)[:, 1, 1]
    c = MotionMergeCache(m.depth, 0.01, (gh, gw), radius=1); _commit_all(m, c, [x0, x1])
    assert c.entries() == 2 * gh * gw - 1 and int(c.count.max()) == 2                  # two copies of one patch: one merges, one is new


def test_pan_world_slides():
    fr, bgs, cols = data.episode_pan(np.random.default_rng(3), 5, speeds=(8,))
    assert fr.shape == (5, 3, 32, 32) and np.allclose(bgs[1][:, :, :-8], bgs[0][:, :, 8:])
