"""Predictive look scheduling with a distribution-free guarantee.

A video LLM looks at a stream at some rate. Between looks the model's view is
the last kept tokens; at a look the certified per-patch rule (quotient + range)
decides which tokens to refresh with its deterministic certificate. This module
decides WHEN to look next, before seeing the frames it will skip.

Decision.  After a look at frame t with activity state b_t in {active, recent,
quiet} (refreshed >= m patches now / within the last `recent` frames / neither),
choose the horizon by RISK EQUALISATION: with lambda_b the running estimate of
the onset rate (onsets per skipped frame) in state b,

    h_t = clip( round( exp(theta_t) * alpha / lambda_{b_t} ), 1, H ),

so that the predicted onset probability per window is about alpha in every
state; skip h_t - 1 frames and look again at t + h_t. For a constant onset
rate this is a uniform policy at the alpha-calibrated rate; for bursty scenes
the horizons differ by state. The factor exp(theta_t) is the distribution-free
safety net below: it corrects whatever the rate estimates get wrong.

Feedback.  At the next look, err_t = 1 if the certified rule refreshes at least
m patches OUTSIDE the one-patch dilation of the set that was active at the
previous look (a Delta-change arrived in a region we were not tracking: an
onset) AND h_t >= 2 (frames were actually skipped); err_t = 0 otherwise. If no
frame was skipped nothing could have been missed, so err_t = 0 by definition.
Ongoing activity (a walker already being tracked) is not a miss; its staleness
is governed by the shorter active-bin horizon and reported separately.

Update (online integrator on the miss rate, Gibbs & Candes 2021 in the
log-horizon domain, step gamma_n = gamma / n^decay):
        theta_{n+1} = min( log H,  theta_n + gamma_n (alpha - err_n) ).

Theorem (distribution-free, deterministic, one-sided).  For ANY frame sequence
and any number of look decisions N, with a constant step gamma,
        (1/N) sum_n err_n  <=  alpha + theta_1 / (gamma N).
Proof: theta_{n+1} <= theta_n + gamma (alpha - err_n) whenever the floor does
not bind (the ceiling can only lower theta), so theta_{N+1} <= theta_1 +
gamma sum_n (alpha - err_n) + (floor corrections); each floor correction is at
most gamma and happens only when err_n = 1 at theta_n < -log H + gamma, i.e.
when the horizon is already 1 and no frame was skipped, where err = 0 by
definition: a contradiction, so the floor never binds. Hence
sum_n err_n <= N alpha + (theta_1 - theta_{N+1})/gamma <= N alpha + (theta_1 + log H)/gamma. []  With a
decaying step the same argument gives alpha + theta_1 / (sum_n gamma_n), still
vanishing. The lower bound (miss rate >= alpha - ...) is NOT claimed: in a
quiet scene the horizon saturates at H and the miss rate stays below alpha,
which is the desired behaviour.  Deterministically, every persistent
Delta-change is seen within H frames.

What is NOT claimed: transient changes that start and end inside a skipped
window are invisible to any sampler, and the long-run bound says nothing about
any single window. Efficiency (how long the horizons get for a given alpha)
depends on how informative the activity bin is and is measured, not proved
(the 2024 critique "Nothing conformal about adaptive conformal inference"
is right that the guarantee is a property of the feedback loop).
"""
from __future__ import annotations

import numpy as np


def _dilate_patches(k, r=1):
    m = k.astype(bool)
    for _ in range(r):
        g = m.copy()
        g[1:] |= m[:-1]; g[:-1] |= m[1:]; g[:, 1:] |= m[:, :-1]; g[:, :-1] |= m[:, 1:]
        m = g
    return m


def new_activity(k_now, k_prev, m):
    """Number of patches refreshed now outside the dilated previously-refreshed set, and whether >= m."""
    if k_prev is None:
        return int(k_now.sum()), bool(k_now.sum() >= m)
    outside = k_now & ~_dilate_patches(k_prev, 1)
    return int(outside.sum()), bool(outside.sum() >= m)


class AdaptiveSampler:
    def __init__(self, pruner, alpha=0.1, gamma=0.3, H=30, m_refresh=3, g_active=0.25, theta0=None, decay=0.0,
                 recent=25, prior_onsets=1.0, prior_frames=10.0):
        self.pr = pruner
        self.alpha, self.gamma, self.H, self.m, self.g_active = float(alpha), float(gamma), int(H), int(m_refresh), float(g_active)
        self.decay = float(decay); self.n = 0; self.gsum = 0.0
        self.theta = {b: float(0.0 if theta0 is None else theta0) for b in ("active", "recent", "quiet")}   # one integrator per state
        self.theta1 = float(0.0 if theta0 is None else theta0)
        self.gsum = {b: 0.0 for b in ("active", "recent", "quiet")}; self.nerr = {b: 0 for b in ("active", "recent", "quiet")}
        self.recent = int(recent)
        # per-state onset-rate estimates with a conservative prior (prior_onsets per prior_frames skipped frames)
        self.onsets = {b: float(prior_onsets) for b in ("active", "recent", "quiet")}
        self.skipped = {b: float(prior_frames) for b in ("active", "recent", "quiet")}
        self.last_active_t = -10**9
        self.log = {"look_t": [], "h": [], "err": [], "theta": [], "refreshed": [], "bin": [], "new": []}

    def rate_summary(self):
        return {b: (round(self.rate(b), 4), self.horizon(b)) for b in ("active", "recent", "quiet")}

    def bound(self):
        """Deterministic slack of the one-sided guarantee on the overall miss rate: the per-state
        bounds sum, sum_b (theta_1 + log H)/gamma_b, divided by the number of decisions."""
        N = max(sum(self.nerr.values()), 1)
        return sum((self.theta1 + np.log(self.H)) / max(self.gsum[b], 1e-9) * (self.nerr[b] / N) for b in self.nerr if self.nerr[b] > 0)

    def rate(self, b):
        return self.onsets[b] / self.skipped[b]

    def horizon(self, b):
        return int(np.clip(round(np.exp(self.theta[b]) * self.alpha / self.rate(b)), 1, self.H))

    def run(self, frames):
        T = len(frames)
        t = 0
        pending = None                         # (h, bin) of the decision that led to this look
        keeps = {}; k_prev = None
        while t < T:
            k, _, _ = self.pr.step(frames[t])
            if t == 0:
                k = np.zeros_like(k)           # the first frame initialises the reference; it is not activity
            n_ref = int(k.sum())
            n_new, onset = new_activity(k, k_prev, self.m)
            keeps[t] = k
            if pending is not None:
                h_prev, b_prev = pending
                err = 1 if (h_prev >= 2 and onset) else 0
                self.nerr[b_prev] += 1; g = self.gamma / (self.nerr[b_prev] ** self.decay); self.gsum[b_prev] += g
                self.theta[b_prev] = float(min(np.log(self.H), max(-np.log(self.H), self.theta[b_prev] + g * (self.alpha - err))))
                self.log["err"].append(err)
                # learn the per-frame onset rate of the state we were in, from every look (no cold start)
                self.onsets[b_prev] += int(onset); self.skipped[b_prev] += h_prev
            active = n_ref >= self.m
            if active:
                self.last_active_t = t
            b = "active" if active else ("recent" if t - self.last_active_t <= self.recent else "quiet")
            h = self.horizon(b)
            self.log["look_t"].append(t); self.log["h"].append(h); self.log["theta"].append(dict(self.theta))
            self.log["refreshed"].append(n_ref); self.log["bin"].append(b); self.log["new"].append(n_new)
            pending = (h, b)
            k_prev = k
            t += h
        return keeps


class UniformSampler:
    def __init__(self, pruner, h):
        self.pr, self.h = pruner, int(h)
        self.log = {"look_t": [], "refreshed": []}

    def run(self, frames):
        keeps = {}; k_prev = None; self.log["onset"] = []
        for t in range(0, len(frames), self.h):
            k, _, _ = self.pr.step(frames[t])
            if t == 0:
                k = np.zeros_like(k)
            keeps[t] = k
            n_new, onset = new_activity(k, k_prev, 3)
            self.log["look_t"].append(t); self.log["refreshed"].append(int(k.sum())); self.log["onset"].append(int(onset and t > 0))
            k_prev = k
        return keeps
