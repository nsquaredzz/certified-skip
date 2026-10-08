"""Grey-frame I/O through ffmpeg (no OpenCV / PyAV dependency)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def _ffmpeg():
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found on PATH")
    return exe


def probe(path) -> dict:
    exe = shutil.which("ffprobe")
    if not exe:
        raise RuntimeError("ffprobe not found on PATH")
    out = subprocess.run([exe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,codec_name",
                          "-of", "json", str(path)], check=True, capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s.get("avg_frame_rate", "0/1").split("/")
    fps = float(num) / float(den) if float(den) else None
    return {"width": int(s["width"]), "height": int(s["height"]), "fps": fps,
            "nb_frames": int(s["nb_frames"]) if s.get("nb_frames", "N/A").isdigit() else None,
            "duration": float(s["duration"]) if s.get("duration") else None,
            "codec": s.get("codec_name")}


def _vf(fps, scale):
    parts = []
    if fps:
        parts.append(f"fps={fps}")
    if scale:
        parts.append(f"scale={scale[0]}:{scale[1]}:flags=area")
    parts.append("format=gray")
    return ",".join(parts)


def iter_gray(path, fps: float | None = None, scale=None, start: float | None = None,
              max_frames: int | None = None):
    """Yield (H, W) uint8 grey frames decoded by ffmpeg.
    scale=(W, H) resizes; fps resamples; start seeks (seconds)."""
    info = probe(path)
    W, H = (scale if scale else (info["width"], info["height"]))
    cmd = [_ffmpeg(), "-v", "error", "-nostdin"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(path), "-vf", _vf(fps, scale), "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=W * H * 8)
    n = 0
    try:
        while True:
            buf = proc.stdout.read(W * H)
            if len(buf) < W * H:
                break
            yield np.frombuffer(buf, np.uint8).reshape(H, W)
            n += 1
            if max_frames and n >= max_frames:
                break
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()


def read_gray(path, fps=None, scale=None, start=None, max_frames=None) -> np.ndarray:
    """(T, H, W) uint8 array of grey frames."""
    frames = list(iter_gray(path, fps, scale, start, max_frames))
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.stack(frames)


def write_h264(frames: np.ndarray, path, fps: float = 30.0, crf: int = 23, gop: int | None = None,
               preset: str = "medium", codec: str = "libx264") -> Path:
    """Encode (T, H, W) uint8 grey frames to an H.264 (yuv420p) file."""
    frames = np.ascontiguousarray(frames, np.uint8)
    T, H, W = frames.shape
    cmd = [_ffmpeg(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}",
           "-r", str(fps), "-i", "-", "-an", "-c:v", codec, "-preset", preset, "-crf", str(crf),
           "-pix_fmt", "yuv420p"]
    if gop:
        cmd += ["-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0"]
    cmd += [str(path)]
    subprocess.run(cmd, input=frames.tobytes(), check=True)
    return Path(path)


def encode_roundtrip(frames: np.ndarray, crf: int = 23, gop: int | None = None, fps: float = 30.0,
                     keep_file: str | None = None) -> np.ndarray:
    """Encode with H.264 and decode back, returning (T, H, W) uint8.  This is
    how synthetic scenes acquire realistic codec noise."""
    tmpdir = None
    if keep_file:
        out = Path(keep_file)
    else:
        tmpdir = tempfile.mkdtemp(prefix="certskip_")
        out = Path(tmpdir) / "clip.mp4"
    try:
        write_h264(frames, out, fps=fps, crf=crf, gop=gop)
        dec = read_gray(out)
        return dec[: len(frames)]
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
