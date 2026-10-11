import sys, time, json
sys.path.insert(0, sys.argv[1] + "/keep")
import numpy as np
from looks_lib import *
tag = sys.argv[2]; z = np.load(f"{sys.argv[1]}/sam{tag}.npz"); ref, col = z["masks"], z["col"]; T, K, H, W = ref.shape
grey = col.astype(np.float32).mean(3); smooth = np.stack([gauss(g, 1) for g in grey])
t0 = time.time(); flows = np.stack([flow(grey[t], grey[t + 1]) for t in range(T - 1)]); dt = time.time() - t0
print(f"shot {tag}: {T} frames, {K} people; flow {1000 * dt / (T - 1):.0f} ms a frame (CPU)"); np.save(f"{sys.argv[1]}/flows{tag}.npy", flows)
out, looked, sig = run(grey, smooth, flows, ref); a1 = agreement(out, ref)
r = lambda v: np.argsort(np.argsort(v)); print(f"one look: mean agreement {100 * a1.mean():.1f} %, lowest {100 * a1.min():.1f} %, frames under 90 %: {(a1 < 0.9).sum()} of {T}; "
      f"rank correlation of the break signal with the loss of agreement: {np.corrcoef(r(sig.max(1)), r(1 - a1))[0, 1]:.2f}")
print("  frame:      " + " ".join(f"{t:>4}" for t in range(0, T, 12))); print("  agreement:  " + " ".join(f"{100 * a1[t]:>4.0f}" for t in range(0, T, 12))); print("  signal max: " + " ".join(f"{100 * sig[t].max():>4.0f}" for t in range(0, T, 12)))
print("looks when the signal says so, against the same number of looks evenly spaced:")
rows = []
for th in (0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1):
    o, lk_, s = run(grey, smooth, flows, ref, theta=th); a = agreement(o, ref); n = len(lk_)
    ev = sorted(set(int(round(i * T / n)) for i in range(n))); o2, l2, _ = run(grey, smooth, flows, ref, looks=set(ev)); b = agreement(o2, ref)
    rows.append(dict(theta=th, looks=n, at=lk_, mean=a.mean(), low=a.min(), u_mean=b.mean(), u_low=b.min(), under=int((a < 0.9).sum()), u_under=int((b < 0.9).sum())))
    print(f"  signal over {100 * th:>3.0f} %: {n:>2} looks ({100 * n / T:>4.1f} % of frames) | by signal: mean {100 * a.mean():.1f}, lowest {100 * a.min():.1f}, under 90: {int((a < 0.9).sum()):>3} frames | evenly: mean {100 * b.mean():.1f}, lowest {100 * b.min():.1f}, under 90: {int((b < 0.9).sum()):>3}")
json.dump(rows, open(f"{sys.argv[1]}/looks{tag}.json", "w"))
