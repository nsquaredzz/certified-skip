"""The clairvoyant optimum, and how close the causal rule gets to it.

Setting.  Fix one patch and a statistic sigma that is a seminorm on patch
differences (range, L1 mean, the multi-scale statistic with the brightness
shift optimised out, the flat norm: all of them).  A *schedule* is a set of
keep times containing 0.  Its anchor at time t is the last keep <= t.  The
schedule is theta-sound if sigma(F_t - F_anchor(t)) < theta at every skipped
t: the patch the model holds is within theta of the truth, in the currency
of sigma, which is what every certificate in this project promises.

The anchored rule (core.Pruner with sigma = range) keeps exactly when
sigma(F_t - F_anchor) >= Delta.  Write m_Delta for its number of keeps and
OPT_theta for the fewest keeps of any theta-sound schedule, chosen with full
knowledge of the future.

Theorem D.
  (i)   OPT_Delta <= m_Delta                   (the rule is Delta-sound);
  (ii)  m_Delta <= 2 OPT_{Delta/2} + 1         (two keeps per clairvoyant keep at twice the guarantee);
  (iii) for every gamma < 2 and every deterministic causal Delta-sound rule A there are sequences
        on which A keeps every frame while OPT_{Delta/gamma} <= 2: no causal rule is competitive at
        any guarantee tighter than Delta/2.
Proof of (ii).  Let g_1 < ... < g_m be the rule's keeps.  A Delta/2-sound schedule S that keeps
nowhere in the closed interval [g_i, g_{i+1}] has one anchor a < g_i for both endpoints, so
sigma(F_{g_{i+1}} - F_{g_i}) <= sigma(F_{g_{i+1}} - F_a) + sigma(F_a - F_{g_i}) < Delta, contradicting
the rule's keep at g_{i+1}.  Hence S keeps in each of the m - 1 closed intervals; a keep at a shared
endpoint can serve two of them, so S keeps at least ceil((m - 1)/2) times.
Proof of (iii).  Scalar multiples x_t of one pattern p with sigma(p) = 1, so sigma(F_t - F_a) =
|x_t - x_a|.  Put u = (Delta/gamma)(1 - eps).  Send x_1 = u.  If A keeps at 1 its anchor is u, and
alternating +-u forces a keep at every later frame (|2u| >= Delta for eps small) while the anchor 0
covers every value (|u| < Delta/gamma): OPT = 1.  If A skips at 1, send x_2 = Delta (forced keep;
the clairvoyant schedule anchors at 1 instead), then alternate between -eps' and 2u - 2 eps', both
within Delta/gamma of u and each at distance >= Delta from A's previous anchor: A keeps every frame,
OPT = 2.  Voluntary keeps by A only move its anchor to a value the adversary punishes the same way.

The 2 in (ii) is therefore the exact price of causality, for any seminorm.  This file computes
OPT_theta by dynamic programming so that (i) and (ii) can be checked on footage and the measured
ratio m_Delta / OPT_Delta reported.

Semantics.  Frame 0 is always kept.  Soundness is strict (< theta), matching the rules.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- statistics (seminorms of the difference)
def stat_range(D: np.ndarray) -> np.ndarray:
    """range over the last two axes of a stack of differences."""
    return D.max(axis=(-2, -1)) - D.min(axis=(-2, -1))


def stat_mean_abs(D: np.ndarray) -> np.ndarray:
    return np.abs(D).mean(axis=(-2, -1))


def first_violation(X: np.ndarray, a: int, theta: float, stat=stat_range) -> int:
    """First t > a with stat(X_t - X_a) >= theta, or T if none.  X: (T, P, P) one patch over time."""
    T = X.shape[0]
    if a >= T - 1:
        return T
    s = stat(X[a + 1:].astype(np.float64) - X[a].astype(np.float64))
    hit = np.nonzero(s >= theta)[0]
    return int(a + 1 + hit[0]) if hit.size else T


def greedy_keeps(X: np.ndarray, theta: float, stat=stat_range):
    """Keep times of the anchored (causal) rule at threshold theta."""
    T = X.shape[0]
    keeps = [0]
    a = 0
    while True:
        b = first_violation(X, a, theta, stat)
        if b >= T:
            break
        keeps.append(b); a = b
    return keeps


def opt_keeps(X: np.ndarray, theta: float, stat=stat_range):
    """Fewest keeps of any theta-sound schedule (clairvoyant), and one such schedule.

    f(a) = fewest keeps of a sound schedule for frames a.. whose first keep is a.  The next keep b
    must come before or at tau(a), the first violation from anchor a, if tau(a) < T; otherwise none
    is needed.  O(T^2) evaluations of stat, vectorised per anchor."""
    T = X.shape[0]
    tau = np.array([first_violation(X, a, theta, stat) for a in range(T)])
    f = np.zeros(T + 1, np.int64); nxt = np.full(T + 1, -1, np.int64)
    f[T] = 0
    for a in range(T - 1, -1, -1):
        if tau[a] >= T:
            f[a] = 1; nxt[a] = -1
        else:
            cand = np.arange(a + 1, tau[a] + 1)
            j = cand[np.argmin(f[cand])]
            f[a] = 1 + f[j]; nxt[a] = j
    sched = [0]
    a = 0
    while nxt[a] >= 0:
        a = int(nxt[a]); sched.append(a)
    return int(f[0]), sched


def competitive_check(X: np.ndarray, delta: float, stat=stat_range) -> dict:
    """Numbers behind Theorem D (i)-(ii) for one patch sequence."""
    m = len(greedy_keeps(X, delta, stat))
    opt_d, _ = opt_keeps(X, delta, stat)
    opt_h, _ = opt_keeps(X, delta / 2.0, stat)
    return {"m": m, "opt_delta": opt_d, "opt_half": opt_h,
            "i_holds": opt_d <= m, "ii_holds": m <= 2 * opt_h + 1,
            "ratio": m / max(opt_d, 1)}


def adversary_sequence(delta: float, gamma: float, T: int, rule, eps: float = 0.02):
    """The construction of Theorem D (iii) against a causal rule, as scalar multiples of a pattern.

    `rule` is a callable taking the history of scalar values and the rule's own keep history and
    returning True to keep the current frame.  Returns (values, rule_keeps, opt_keeps_at_delta/gamma)."""
    u = (delta / gamma) * (1 - eps)
    xs = [0.0]; keeps = [True]; anchor = 0.0
    # frame 1
    xs.append(u)
    k = rule(xs, keeps) or abs(u - anchor) >= delta
    keeps.append(k); anchor = u if k else anchor
    if k:
        sign = -1.0
        while len(xs) < T:
            xs.append(sign * u); sign = -sign
            kk = rule(xs, keeps) or abs(xs[-1] - anchor) >= delta
            keeps.append(kk); anchor = xs[-1] if kk else anchor
        opt_anchor = 0.0
        opt = sum(1 for x in xs if abs(x - opt_anchor) >= delta / gamma) + 1
        return np.array(xs), np.array(keeps), opt
    # rule skipped frame 1: its anchor is 0; the clairvoyant schedule anchors at u
    xs.append(delta); keeps.append(True); anchor = delta
    e2 = eps * delta / (2 * gamma)
    lo, hi = -e2, 2 * u - 2 * e2
    nxt = lo
    while len(xs) < T:
        xs.append(nxt); nxt = hi if nxt == lo else lo
        kk = rule(xs, keeps) or abs(xs[-1] - anchor) >= delta
        keeps.append(kk); anchor = xs[-1] if kk else anchor
    # clairvoyant: anchors at 0 and 1 (value u) cover everything within delta/gamma
    opt = 2 + sum(1 for x in xs[2:] if abs(x - u) >= delta / gamma)
    return np.array(xs), np.array(keeps), opt
