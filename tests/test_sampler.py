"""The one-sided miss guarantee of the adaptive look scheduler must hold against an adversary,
and the scheduler must recover the right uniform rate for memoryless onsets."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from certskip.sampler import AdaptiveSampler, UniformSampler   # noqa: E402


class _Adversary:
    """Fake per-patch rule: plants an onset (3 new patches, moving around) whenever frames were skipped."""
    def __init__(self, gh=8, gw=8):
        self.gh, self.gw, self.last_t, self.i = gh, gw, None, 0

    def step(self, frame):
        t = int(frame)
        k = np.zeros((self.gh, self.gw), bool)
        if self.last_t is not None and t - self.last_t >= 2:
            r = (self.i * 3) % (self.gh - 1); c = (self.i * 5) % (self.gw - 3)
            k[r, c:c + 3] = True; self.i += 1
        self.last_t = t
        return k, None, None


class _Poisson:
    """Fake rule with memoryless onsets at rate lam per frame, each a fresh 3-patch blob away from the last."""
    def __init__(self, lam, seed=0, gh=12, gw=12):
        self.lam, self.rng, self.gh, self.gw, self.last_t, self.i = lam, np.random.default_rng(seed), gh, gw, None, 0

    def step(self, frame):
        t = int(frame)
        k = np.zeros((self.gh, self.gw), bool)
        n_frames = 1 if self.last_t is None else t - self.last_t
        if self.rng.random() < 1 - (1 - self.lam) ** n_frames:
            r = (self.i * 4) % (self.gh - 1); c = (self.i * 7) % (self.gw - 3); k[r, c:c + 3] = True; self.i += 1
        self.last_t = t
        return k, None, None


def test_guarantee_holds_against_adversary():
    for alpha in (0.05, 0.2):
        s = AdaptiveSampler(_Adversary(), alpha=alpha, gamma=0.3, H=30, decay=0.5)
        s.run(np.arange(3000))
        err = np.array(s.log["err"])
        assert err.mean() <= alpha + s.bound() + 1e-12
        assert np.mean(s.log["h"][-500:]) < 1.5          # it learned to look (almost) every frame


def test_recovers_calibrated_uniform_rate_for_memoryless_onsets():
    lam = 0.01                                           # one onset per 100 frames
    s = AdaptiveSampler(_Poisson(lam, seed=1), alpha=0.1, gamma=0.3, H=30, decay=0.5)
    s.run(np.arange(20000))
    err = np.array(s.log["err"]); h = np.array(s.log["h"])
    assert err.mean() <= 0.1 + s.bound() + 1e-12
    # risk equalisation: alpha / lambda = 10 frames; the learned horizon should be in that neighbourhood
    assert 6 <= np.median(h[-1000:]) <= 16


def test_uniform_sampler_contract():
    s = UniformSampler(_Poisson(0.0, gh=4, gw=4), 7); keeps = s.run(np.arange(100))
    assert sorted(keeps) == list(range(0, 100, 7))
