"""certskip — certified patch skipping for fixed-camera video.

Rule: drop a patch while the spread (max - min) of its change since the last
kept copy is strictly below a contrast level ``delta``.  While dropped, every
topological feature (dark or bright blob, via sublevel/superlevel persistence)
changes contrast by at most the spread, and no feature of contrast >= delta can
appear from or vanish into nothing (bottleneck stability, 2007).

Public API
----------
Pruner            numpy reference implementation (any dtype)
NativePruner      same API backed by the C++ shared library (falls back to Pruner)
make_pruner       picks NativePruner if the library is built, else Pruner
baselines         consecutive-mean, mean-vs-kept, uniform sampling + calibration
topology          0-dim persistence + exact bottleneck distance (soundness checks)
noise             temporal-noise estimate and spread-vs-delta prediction
video             ffmpeg-backed grey frame reader / H.264 writer
"""
from .core import Pruner, patch_grid, spread_and_shift
from .native import NativePruner, native_available, make_pruner
from . import baselines, topology, noise, video

__all__ = [
    "Pruner", "NativePruner", "make_pruner", "native_available",
    "patch_grid", "spread_and_shift",
    "baselines", "topology", "noise", "video",
]
__version__ = "0.1.0"
