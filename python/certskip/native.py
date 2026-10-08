"""ctypes binding to the C++ core (cpp/libcertskip.{dylib,so}).

Same interface as core.Pruner.  If the library has not been built,
``make_pruner`` returns the numpy reference instead, so everything works
without a compiler; the native path is only faster.
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np

from .core import Pruner

_LIB = None
_DTYPE_CODE = {np.dtype(np.uint8): 0, np.dtype(np.uint16): 1, np.dtype(np.float32): 2}


def _candidates():
    here = Path(__file__).resolve()
    cpp = here.parents[2] / "cpp"
    names = ["libcertskip.dylib", "libcertskip.so"]
    env = os.environ.get("CERTSKIP_LIB")
    if env:
        yield Path(env)
    for n in names:
        yield cpp / n
        yield here.parent / n


def _load():
    global _LIB
    if _LIB is not None:
        return _LIB
    for p in _candidates():
        if p.exists():
            lib = ctypes.CDLL(str(p))
            lib.cs_create.restype = ctypes.c_void_p
            lib.cs_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_int]
            lib.cs_destroy.argtypes = [ctypes.c_void_p]
            lib.cs_grid_h.argtypes = [ctypes.c_void_p]; lib.cs_grid_h.restype = ctypes.c_int
            lib.cs_grid_w.argtypes = [ctypes.c_void_p]; lib.cs_grid_w.restype = ctypes.c_int
            lib.cs_reset.argtypes = [ctypes.c_void_p]
            lib.cs_step.restype = ctypes.c_long
            lib.cs_step.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            lib.cs_run.restype = ctypes.c_long
            lib.cs_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            lib.cs_reference.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
            lib.cs_version.restype = ctypes.c_char_p
            _LIB = lib
            return lib
    return None


def native_available() -> bool:
    return _load() is not None


class NativePruner:
    """Drop-in replacement for core.Pruner backed by the C++ library."""

    def __init__(self, height: int, width: int, patch: int = 16,
                 delta: float = 32.0, max_shift: float | None = None,
                 dtype=np.uint8):
        lib = _load()
        if lib is None:
            raise RuntimeError("libcertskip not built; run `make` in cpp/ or use core.Pruner")
        self.dtype = np.dtype(dtype)
        if self.dtype not in _DTYPE_CODE:
            raise TypeError("native pruner supports uint8, uint16, float32")
        self.H, self.W, self.patch = int(height), int(width), int(patch)
        self.delta, self.max_shift = float(delta), max_shift
        self._lib = lib
        self._h = lib.cs_create(self.H, self.W, self.patch, self.delta,
                                -1.0 if max_shift is None else float(max_shift),
                                _DTYPE_CODE[self.dtype])
        if not self._h:
            raise ValueError("cs_create failed (bad size/patch)")
        self.gh, self.gw = lib.cs_grid_h(self._h), lib.cs_grid_w(self._h)
        self._n = 0
        self.last_shift = np.zeros((self.gh, self.gw), np.float32)

    def __del__(self):
        h = getattr(self, "_h", None)
        if h:
            self._lib.cs_destroy(h)
            self._h = None

    @property
    def initialised(self) -> bool:
        return self._n > 0

    def reset(self) -> None:
        self._lib.cs_reset(self._h)
        self._n = 0
        self.last_shift[:] = 0

    def _check(self, frame):
        frame = np.ascontiguousarray(frame, dtype=self.dtype)
        if frame.shape[-2:] != (self.H, self.W):
            raise ValueError(f"frame shape {frame.shape} != {(self.H, self.W)}")
        return frame

    def step(self, frame: np.ndarray):
        frame = self._check(frame)
        keep = np.empty((self.gh, self.gw), np.uint8)
        spread = np.empty((self.gh, self.gw), np.float32)
        shift = np.empty((self.gh, self.gw), np.float32)
        self._lib.cs_step(self._h, frame.ctypes.data, keep.ctypes.data, spread.ctypes.data, shift.ctypes.data)
        self._n += 1
        k = keep.astype(bool)
        self.last_shift = np.where(k, 0.0, shift).astype(np.float32)
        return k, spread, shift

    def run(self, frames):
        """Vectorised over a (T,H,W) array; falls back to step() for iterables."""
        if not isinstance(frames, np.ndarray) or frames.ndim != 3:
            keeps, spreads, shifts = [], [], []
            for f in frames:
                k, s, c = self.step(f)
                keeps.append(k); spreads.append(s); shifts.append(c)
            return {"keep": np.stack(keeps), "spread": np.stack(spreads), "shift": np.stack(shifts)}
        frames = self._check(frames)
        T = frames.shape[0]
        keep = np.empty((T, self.gh, self.gw), np.uint8)
        spread = np.empty((T, self.gh, self.gw), np.float32)
        shift = np.empty((T, self.gh, self.gw), np.float32)
        self._lib.cs_run(self._h, frames.ctypes.data, T, self.H * self.W,
                         keep.ctypes.data, spread.ctypes.data, shift.ctypes.data)
        self._n += T
        k = keep.astype(bool)
        self.last_shift = np.where(k[-1], 0.0, shift[-1]).astype(np.float32)
        return {"keep": k, "spread": spread, "shift": shift}

    def reference(self) -> np.ndarray:
        if self._n == 0:
            raise RuntimeError("no frame processed yet")
        out = np.empty((self.H, self.W), self.dtype)
        self._lib.cs_reference(self._h, out.ctypes.data, out.size)
        return out

    def view(self, compensate_shift: bool = True) -> np.ndarray:
        ref = self.reference()
        if not compensate_shift:
            return ref
        v = ref.astype(np.float32)
        P = self.patch
        v[: self.gh * P, : self.gw * P] += np.kron(self.last_shift, np.ones((P, P), np.float32))
        if ref.dtype.kind in "ui":
            info = np.iinfo(ref.dtype)
            v = np.clip(np.rint(v), info.min, info.max)
        return v.astype(ref.dtype)


def make_pruner(height, width, patch=16, delta=32.0, max_shift=None, dtype=np.uint8, prefer_native=True):
    """NativePruner when the C++ library is available (and dtype supported), else Pruner."""
    if prefer_native and native_available() and np.dtype(dtype) in _DTYPE_CODE:
        return NativePruner(height, width, patch, delta, max_shift, dtype)
    return Pruner(height, width, patch, delta, max_shift)


# ------------------------------------------------------------------ quotient
def _load_q():
    lib = _load()
    if lib is None or getattr(lib, "_q_ready", False):
        return lib
    try:
        lib.csq_create.restype = ctypes.c_void_p
        lib.csq_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_int]
        lib.csq_destroy.argtypes = [ctypes.c_void_p]
        lib.csq_grid_h.argtypes = [ctypes.c_void_p]; lib.csq_grid_h.restype = ctypes.c_int
        lib.csq_grid_w.argtypes = [ctypes.c_void_p]; lib.csq_grid_w.restype = ctypes.c_int
        lib.csq_reset.argtypes = [ctypes.c_void_p]
        lib.csq_step.restype = ctypes.c_long
        lib.csq_step.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 6
        lib.csq_reference.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
        lib.csq_view.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
        lib._q_ready = True
    except AttributeError:
        return None
    return lib


class NativeQuotientPruner:
    """C++ quotient rule (uint8 frames). Interface as warp.WarpPruner with certify='range'."""

    def __init__(self, height, width, patch=16, delta=32.0, delta_max=1.0, iters=2):
        lib = _load_q()
        if lib is None:
            raise RuntimeError("libcertskip (with quotient rule) not built; run `make` in cpp/")
        self.H, self.W, self.patch = int(height), int(width), int(patch)
        self.delta, self.delta_max, self.iters = float(delta), float(delta_max), int(iters)
        self._lib = lib
        self._h = lib.csq_create(self.H, self.W, self.patch, self.delta, self.delta_max, self.iters)
        if not self._h:
            raise ValueError("csq_create failed")
        self.gh, self.gw = lib.csq_grid_h(self._h), lib.csq_grid_w(self._h)
        self._n = 0
        self.last_motion = None

    def __del__(self):
        h = getattr(self, "_h", None)
        if h:
            self._lib.csq_destroy(h); self._h = None

    @property
    def initialised(self):
        return self._n > 0

    def reset(self):
        self._lib.csq_reset(self._h); self._n = 0

    def step(self, frame):
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if frame.shape != (self.H, self.W):
            raise ValueError("bad frame shape")
        keep = np.empty((self.gh, self.gw), np.uint8)
        spread = np.empty((self.gh, self.gw), np.float32); shift = np.empty_like(spread)
        mdy = np.empty_like(spread); mdx = np.empty_like(spread)
        self._lib.csq_step(self._h, frame.ctypes.data, keep.ctypes.data, spread.ctypes.data, shift.ctypes.data, mdy.ctypes.data, mdx.ctypes.data)
        self._n += 1
        self.last_motion = (mdy, mdx)
        return keep.astype(bool), spread, shift

    def run(self, frames):
        ks, ss, cs_ = [], [], []
        for f in frames:
            k, s, c = self.step(f); ks.append(k); ss.append(s); cs_.append(c)
        return {"keep": np.stack(ks), "spread": np.stack(ss), "shift": np.stack(cs_)}

    def reference(self):
        out = np.empty((self.H, self.W), np.uint8); self._lib.csq_reference(self._h, out.ctypes.data, out.size); return out

    def view(self, compensate_shift=True):
        if not compensate_shift:
            return self.reference()
        v = np.empty((self.H, self.W), np.float32); self._lib.csq_view(self._h, v.ctypes.data, v.size)
        return np.clip(np.rint(v), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------- sequential
def _load_s():
    lib = _load()
    if lib is None or getattr(lib, "_s_ready", False):
        return lib
    try:
        lib.css_create.restype = ctypes.c_void_p
        lib.css_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_double,
                                   ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_double, ctypes.c_int,
                                   ctypes.c_double, ctypes.c_double]
        lib.css_destroy.argtypes = [ctypes.c_void_p]
        lib.css_grid_h.argtypes = [ctypes.c_void_p]; lib.css_grid_h.restype = ctypes.c_int
        lib.css_grid_w.argtypes = [ctypes.c_void_p]; lib.css_grid_w.restype = ctypes.c_int
        lib.css_reset.argtypes = [ctypes.c_void_p]
        lib.css_step.restype = ctypes.c_long
        lib.css_step.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 5
        lib.css_reference.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
        lib.css_view.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
        lib._s_ready = True
    except AttributeError:
        return None
    return lib


class NativeSequentialPruner:
    """C++ sequential rule (uint8 frames). Interface as sequential.SequentialPruner."""

    def __init__(self, height, width, patch=16, multiscale=None, z=8.0, windows=(1, 2, 4, 8, 16), radii=(1, 2, 3),
                 delta_max=1.0, iters=2, sigma_alpha=0.1, min_calib=8, scale_floor=0.5, veto_frac=0.1):
        lib = _load_s()
        if lib is None:
            raise RuntimeError("libcertskip (with sequential rule) not built; run `make` in cpp/")
        self.H, self.W, self.patch = int(height), int(width), int(patch)
        ms = dict(multiscale) if multiscale else {}
        ms_r = np.array(list(ms.keys()), np.int32); ms_d = np.array(list(ms.values()), np.float64)
        win = np.array(windows, np.int32); rad = np.array(radii, np.int32)
        self._lib = lib
        self._h = lib.css_create(self.H, self.W, self.patch, float(delta_max), int(iters), float(z),
                                 len(ms_r), ms_r.ctypes.data if len(ms_r) else None, ms_d.ctypes.data if len(ms_d) else None,
                                 len(win), win.ctypes.data, len(rad), rad.ctypes.data, float(sigma_alpha), int(min_calib),
                                 float(scale_floor), float(veto_frac))
        if not self._h:
            raise ValueError("css_create failed")
        self.gh, self.gw = lib.css_grid_h(self._h), lib.css_grid_w(self._h)
        self.z = float(z)
        self._n = 0
        self.last_seq = None

    def __del__(self):
        h = getattr(self, "_h", None)
        if h:
            self._lib.css_destroy(h); self._h = None

    @property
    def initialised(self):
        return self._n > 0

    def reset(self):
        self._lib.css_reset(self._h); self._n = 0

    def step(self, frame):
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if frame.shape != (self.H, self.W):
            raise ValueError("bad frame shape")
        keep = np.empty((self.gh, self.gw), np.uint8)
        score = np.empty((self.gh, self.gw), np.float32); shift = np.empty_like(score); seqz = np.empty_like(score)
        self._lib.css_step(self._h, frame.ctypes.data, keep.ctypes.data, score.ctypes.data, shift.ctypes.data, seqz.ctypes.data)
        self._n += 1
        self.last_seq = seqz
        return keep.astype(bool), score, shift

    def run(self, frames):
        ks, ss, cs_ = [], [], []
        for f in frames:
            k, s, c = self.step(f); ks.append(k); ss.append(s); cs_.append(c)
        return {"keep": np.stack(ks), "spread": np.stack(ss), "shift": np.stack(cs_)}

    def reference(self):
        out = np.empty((self.H, self.W), np.uint8); self._lib.css_reference(self._h, out.ctypes.data, out.size); return out

    def view(self, compensate_shift=True):
        if not compensate_shift:
            return self.reference()
        v = np.empty((self.H, self.W), np.float32); self._lib.css_view(self._h, v.ctypes.data, v.size)
        return np.clip(np.rint(v), 0, 255).astype(np.uint8)


# ------------------------------------------------------------- mask utilities
def _load_m():
    lib = _load()
    if lib is None or getattr(lib, "_m_ready", False):
        return lib
    try:
        lib.cs_label_components.restype = ctypes.c_int
        lib.cs_label_components.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        lib.cs_fill_holes.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.cs_remove_small.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib._m_ready = True
    except AttributeError:
        return None
    return lib


def label_components(mask):
    """8-connected component labels (0 = background) and the number of components."""
    lib = _load_m()
    m = np.ascontiguousarray(mask, dtype=np.uint8)
    if lib is not None:
        lab = np.zeros(m.shape, np.int32)
        n = lib.cs_label_components(m.ctypes.data, m.shape[0], m.shape[1], lab.ctypes.data)
        return lab, int(n)
    # numpy fallback: iterative min-label propagation (slow but dependency-free)
    H, W = m.shape
    lab = np.where(m > 0, np.arange(1, H * W + 1).reshape(H, W), 0).astype(np.int64)
    while True:
        new = lab.copy()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            sh = np.roll(np.roll(lab, dy, 0), dx, 1)
            if dy == 1: sh[0] = 0
            if dy == -1: sh[-1] = 0
            if dx == 1: sh[:, 0] = 0
            if dx == -1: sh[:, -1] = 0
            cand = np.where((sh > 0) & (new > 0), np.minimum(new, sh), new)
            new = cand
        if np.array_equal(new, lab):
            break
        lab = new
    ids = np.unique(lab[lab > 0])
    remap = np.zeros(lab.max() + 1, np.int32); remap[ids] = np.arange(1, len(ids) + 1)
    return remap[lab], len(ids)


def fill_holes(mask):
    lib = _load_m()
    m = np.ascontiguousarray(mask, dtype=np.uint8).copy()
    if lib is not None:
        lib.cs_fill_holes(m.ctypes.data, m.shape[0], m.shape[1]); return m
    bg = (m == 0)
    reach = np.zeros_like(bg); reach[0, :] = bg[0, :]; reach[-1, :] = bg[-1, :]; reach[:, 0] = bg[:, 0]; reach[:, -1] = bg[:, -1]
    while True:
        grow = reach.copy()
        grow[1:] |= reach[:-1]; grow[:-1] |= reach[1:]; grow[:, 1:] |= reach[:, :-1]; grow[:, :-1] |= reach[:, 1:]
        grow &= bg
        if np.array_equal(grow, reach):
            break
        reach = grow
    m[bg & ~reach] = 1
    return m


def remove_small(mask, min_area):
    lib = _load_m()
    m = np.ascontiguousarray(mask, dtype=np.uint8).copy()
    if lib is not None:
        lib.cs_remove_small(m.ctypes.data, m.shape[0], m.shape[1], int(min_area)); return m
    lab, n = label_components(m)
    if n == 0:
        return m
    areas = np.bincount(lab.ravel(), minlength=n + 1)
    m[(lab > 0) & (areas[lab] < min_area)] = 0
    return m


# ------------------------------------------------------------ real-time segmenter
def _load_g():
    lib = _load()
    if lib is None or getattr(lib, "_g_ready", False):
        return lib
    try:
        lib.csg_create.restype = ctypes.c_void_p
        lib.csg_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p] + [ctypes.c_int] * 6
        lib.csg_destroy.argtypes = [ctypes.c_void_p]
        lib.csg_init_background.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        lib.csg_step.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 4
        lib.csg_stage_ms.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.csg_active_why.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib._g_ready = True
    except AttributeError:
        return None
    return lib


class NativeRtSegmenter:
    """C++ real-time segmenter; interface compatible with segment.RealtimeSegmenter for rendering."""

    def __init__(self, height, width, patch=16, init_frames=None, z0=3.0, gamma=0.6, lam=1.2, tau_edge=6.0, mu=0.5,
                 steps=3, min_area=40, halo=0, fit_iters=2, alpha_bg=0.02, alpha_fg=5e-5, sigma_floor=1.5,
                 shadows=True, certify=None, kkt_tol=0.5, act_delta0=40.0, delta_max=1.0, iters=None, compare_full=False, close_r=2, min_side=8, max_aspect=5.0, rho=0.6, tau_img=12.0, snap_r=3, snap_eps=36.0, act_z=3.0, threads=0, ghost_contrast=6.0, alpha_ghost=0.25):
        lib = _load_g()
        if lib is None:
            raise RuntimeError("libcertskip (with rt segmenter) not built; run `make` in cpp/")
        self.H, self.W, self.P = int(height), int(width), int(patch)
        p = np.array([z0, gamma, lam, tau_edge, mu, alpha_bg, alpha_fg, sigma_floor, kkt_tol, act_delta0, delta_max, close_r, min_side, max_aspect, rho, tau_img, snap_r, snap_eps, act_z, threads, ghost_contrast, alpha_ghost], np.float64)
        self._lib = lib
        self._h = lib.csg_create(self.H, self.W, self.P, p.ctypes.data, int(steps), int(min_area), int(halo), int(fit_iters),
                                 1 if shadows else 0, 1 if certify else 0)
        if not self._h:
            raise ValueError("csg_create failed")
        self.mask = np.zeros((self.H, self.W), bool); self.labels = np.zeros((self.H, self.W), np.int32)
        self.ids = {}; self.vel = {}; self.motion = (0.0, 0.0); self.t = -1
        self.stats = {"active_frac": [], "kkt_frozen_max": [], "kkt_active_max": [], "iou_vs_full": [], "pred_shift": []}
        if init_frames is not None:
            self.initialise(init_frames)

    def __del__(self):
        h = getattr(self, "_h", None)
        if h:
            self._lib.csg_destroy(h); self._h = None

    def initialise(self, frames):
        fr = np.ascontiguousarray(np.asarray(frames, np.uint8))
        self._lib.csg_init_background(self._h, fr.ctypes.data, fr.shape[0])

    STAGES = ["jitter+warp", "illumination", "predict+accumulate", "active set", "evidence", "correction", "kkt", "clean-up", "snap", "background", "tracking"]

    def active_why(self):
        out = np.zeros(4, np.int32); self._lib.csg_active_why(self._h, out.ctypes.data)
        return {"score": int(out[0]), "near": int(out[1]), "kkt": int(out[2]), "patches": int(out[3])}

    def stage_ms(self):
        out = np.zeros(12, np.float64); self._lib.csg_stage_ms(self._h, out.ctypes.data)
        return {k: out[i] / max(self.t, 1) for i, k in enumerate(self.STAGES)}

    def step(self, frame):
        self.t += 1
        fr = np.ascontiguousarray(frame, dtype=np.uint8)
        m = np.empty((self.H, self.W), np.uint8); lab = np.empty((self.H, self.W), np.int32); st = np.zeros(7, np.float64)
        self._lib.csg_step(self._h, fr.ctypes.data, m.ctypes.data, lab.ctypes.data, st.ctypes.data)
        self.mask = m.astype(bool); self.labels = lab
        self.ids = {int(i): int(i) for i in np.unique(lab[lab > 0])}
        self.motion = (float(st[3]), float(st[4]))
        self.stats["active_frac"].append(float(st[0])); self.stats["kkt_frozen_max"].append(float(st[1]))
        self.stats["kkt_active_max"].append(float(st[2])); self.stats["pred_shift"].append(int(st[6]))
        return self.mask
