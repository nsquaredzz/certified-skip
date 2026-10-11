import numpy as np
def box(a, r):
    """Box mean of radius r over the last two axes (edges replicated)."""
    if r <= 0:
        return a
    for ax in (-2, -1):
        p = np.concatenate([np.repeat(np.take(a, [0], ax), r + 1, ax), a, np.repeat(np.take(a, [-1], ax), r, ax)], ax).cumsum(ax)
        n = a.shape[ax]; hi = np.take(p, range(2 * r + 1, 2 * r + 1 + n), ax); lo = np.take(p, range(0, n), ax); a = (hi - lo) / (2 * r + 1)
    return a
def gauss(a, r):
    return box(box(box(a, r), r), r)
def focus(fr, r_fill=6):
    """Per pixel: energy left in the finest scale, spread over a neighbourhood. Out-of-focus regions have none,
    whatever their contrast."""
    f = fr.astype(np.float32); fine = (f - gauss(f, 1)) ** 2
    return np.sqrt(gauss(fine, r_fill))
