#!/usr/bin/env python3
"""Numerical audit of the warp certificate (Theorem A) and of the interpolation term (Proposition B).

Scenes are rendered from a continuous model (smooth blobs, sharp-edged squares, fine texture) so that
the ground truth of "the same scene, re-sampled under a warp" exists.  For random homographies near
the identity, and for sub-pixel translations:

  A. the certificate against the constructed W: on every dropped patch
     d_B(Dgm F, Dgm(W + c)) <= s/2  (must hold exactly: 0 violations);
  B. the per-pixel bound |W - W_NN| <= iota (must hold: 0 violations), and the stability consequence
     d_B(Dgm W, Dgm W_NN) <= iota(patch) (must hold);
  C. what is measured, not proved: how often a feature of contrast >= kappa in the kept pixels survives
     in W with contrast >= kappa - 2 iota, and the actual bottleneck distance between Dgm(W_NN) and the
     diagram of the source pixels re-indexed by the rounded warp (zero for integer translations).

    python3 scripts/warp_audit.py --trials 300 --out out/warp_audit
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from certskip import core, topology as tp
from certskip.warpcert import apply_h, sample_bilinear

P = 16


def scene_fn(rng, kind):
    """Returns a function (y, x) -> intensity on continuous coordinates."""
    blobs = [(rng.uniform(0, 64), rng.uniform(0, 64), rng.uniform(2, 12), rng.uniform(-70, 70)) for _ in range(10)]
    squares = [(rng.uniform(8, 56), rng.uniform(8, 56), rng.uniform(3, 12), rng.uniform(-90, 90)) for _ in range(3)]
    freq = rng.uniform(0.3, 1.2, 2); phase = rng.uniform(0, 6.28, 2); amp = rng.uniform(0, 25)

    def f(y, x):
        v = 128 + 30 * np.sin(x / 9.0 + 0.3) * np.cos(y / 7.0)
        for cy, cx, r, a in blobs:
            v = v + a * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * r * r))
        if kind in ("edges", "mixed"):
            for cy, cx, h, a in squares:
                v = v + a * ((np.abs(y - cy) < h) & (np.abs(x - cx) < h))
        if kind in ("texture", "mixed"):
            v = v + amp * np.sin(freq[0] * x + phase[0]) * np.sin(freq[1] * y + phase[1])
        return v
    return f


def random_warp(rng, kind):
    if kind == "subpixel":
        return np.array([[1, 0, rng.uniform(-1, 1)], [0, 1, rng.uniform(-1, 1)], [0, 0, 1.0]])
    if kind == "integer":
        return np.array([[1, 0, rng.integers(-3, 4)], [0, 1, rng.integers(-3, 4)], [0, 0, 1.0]])
    th = rng.uniform(-0.06, 0.06); sc = rng.uniform(0.95, 1.05)
    G = np.array([[sc * np.cos(th), -sc * np.sin(th), rng.uniform(-3, 3)],
                  [sc * np.sin(th), sc * np.cos(th), rng.uniform(-3, 3)],
                  [rng.uniform(-2e-4, 2e-4), rng.uniform(-2e-4, 2e-4), 1.0]])
    return G


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--delta", type=float, default=32.0)
    ap.add_argument("--out", default="out/warp_audit")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float64)
    out = {}
    t0 = time.perf_counter()
    for wkind in ("integer", "subpixel", "homography"):
        for skind in ("smooth", "edges", "texture", "mixed"):
            stats = {"patches": 0, "dropped": 0, "A_viol": 0, "A_worst": 0.0, "B_pix_viol": 0, "B_dgm_viol": 0,
                     "iota_med": [], "C_reindex_db": [], "C_reindex_over_iota": [], "C_feat_total": 0, "C_feat_survive": 0}
            for trial in range(a.trials // 12):
                f = scene_fn(rng, skind)
                G = random_warp(rng, wkind)
                R = f(yy, xx)                                  # reference frame: scene sampled on the grid
                u, v = apply_h(G, xx, yy)                      # frame pixel x sees scene point G x
                F = f(v, u) + rng.normal(0, 1.0, R.shape)      # new frame: the scene re-sampled under the warp (+ noise)
                W, ok, osc, W_nn = sample_bilinear(R, np.ones(R.shape, bool), u, v)
                Rg = core.patch_grid(R, P); Fg = core.patch_grid(F, P); Wg = core.patch_grid(W, P)
                Og = core.patch_grid(osc, P); Ng = core.patch_grid(W_nn, P); okg = core.patch_grid(ok, P).all((2, 3))
                ur = np.rint(u).astype(int); vr = np.rint(v).astype(int)
                for gy in range(4):
                    for gx in range(4):
                        if not okg[gy, gx]:
                            continue
                        stats["patches"] += 1
                        Fp, Wp, Np, iota = Fg[gy, gx], Wg[gy, gx], Ng[gy, gx], float(Og[gy, gx].max())
                        d = Fp - Wp; s = d.max() - d.min(); c = 0.5 * (d.max() + d.min())
                        if s >= a.delta:
                            continue
                        stats["dropped"] += 1
                        # A. certificate against the constructed W
                        rep = tp.check_patch(Fp, Wp, a.delta)
                        stats["A_viol"] += int(not rep["ok"])
                        if rep["half_spread"] > 0:
                            stats["A_worst"] = max(stats["A_worst"], max(rep["bottleneck_dark"], rep["bottleneck_bright"]) / rep["half_spread"])
                        # B. per-pixel bound and its diagram consequence
                        stats["B_pix_viol"] += int(np.any(np.abs(Wp - Np) > Og[gy, gx] + 1e-9))
                        for sup in (False, True):
                            db = tp.bottleneck_distance(tp.persistence_h0(Wp, sup), tp.persistence_h0(Np, sup))
                            stats["B_dgm_viol"] += int(db > iota + 1e-9)
                        stats["iota_med"].append(iota)
                        # C. measured: W_NN against the source pixels re-indexed by the rounded warp
                        ys, xs = vr[gy * P:(gy + 1) * P, gx * P:(gx + 1) * P], ur[gy * P:(gy + 1) * P, gx * P:(gx + 1) * P]
                        src = R[np.clip(ys, 0, 63), np.clip(xs, 0, 63)]
                        db = max(tp.bottleneck_distance(tp.persistence_h0(Np, sp), tp.persistence_h0(src, sp)) for sp in (False, True))
                        stats["C_reindex_db"].append(db)
                        # features of the source footprint of contrast >= delta that survive in W with contrast >= delta - 2 iota
                        for sp in (False, True):
                            Ds = tp.persistence_h0(src, sp); Dw = tp.persistence_h0(Wp, sp)
                            n_src = tp.features_at_least(Ds, a.delta)
                            n_w = tp.features_at_least(Dw, max(0.0, a.delta - 2 * iota))
                            stats["C_feat_total"] += n_src; stats["C_feat_survive"] += min(n_src, n_w)
            rec = {"patches": stats["patches"], "dropped": stats["dropped"], "A_violations": stats["A_viol"], "A_worst_ratio": stats["A_worst"],
                   "B_pixel_violations": stats["B_pix_viol"], "B_diagram_violations": stats["B_dgm_viol"],
                   "iota_median": float(np.median(stats["iota_med"])) if stats["iota_med"] else 0.0,
                   "iota_p90": float(np.percentile(stats["iota_med"], 90)) if stats["iota_med"] else 0.0,
                   "C_reindex_bottleneck_median": float(np.median(stats["C_reindex_db"])) if stats["C_reindex_db"] else 0.0,
                   "C_reindex_bottleneck_max": float(np.max(stats["C_reindex_db"])) if stats["C_reindex_db"] else 0.0,
                   "C_feature_survival": (stats["C_feat_survive"] / stats["C_feat_total"]) if stats["C_feat_total"] else float("nan"),
                   "C_features": stats["C_feat_total"]}
            out[f"{wkind}/{skind}"] = rec
            print(f"{wkind:10s} {skind:8s}: {rec['dropped']:4d}/{rec['patches']:4d} dropped | A viol {rec['A_violations']} worst {rec['A_worst_ratio']:.3f} | "
                  f"B viol pix {rec['B_pixel_violations']} dgm {rec['B_diagram_violations']} | iota med {rec['iota_median']:5.1f} p90 {rec['iota_p90']:5.1f} | "
                  f"C re-index d_B med {rec['C_reindex_bottleneck_median']:5.2f} max {rec['C_reindex_bottleneck_max']:5.1f} | "
                  f"Delta-features surviving {rec['C_feature_survival']:.2f} of {rec['C_features']}", flush=True)
    Path(a.out + ".json").write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}.json in {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
