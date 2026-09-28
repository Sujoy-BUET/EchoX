"""Shared helpers used by every DSP module."""
from __future__ import annotations

import os

import numpy as np
from scipy import signal as sps

try:
    from numba import njit

    HAVE_NUMBA = True
except ImportError:  # numba is optional; loops still run, just slower
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f


EPS = 1e-10


def as_float(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float64)


def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        x = x.mean(axis=1)
    return as_float(x)


def db(x, floor=-120.0):
    return np.maximum(20.0 * np.log10(np.abs(x) + EPS), floor)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + EPS))


def odd_kernel(size: int, limit: int) -> int:
    """A median/min filter kernel that actually fits the array.

    scipy's filters return NaN when the kernel is wider than the axis with
    mode="reflect", so every kernel is clamped to the axis length and forced
    odd (an even kernel has no centre sample).
    """
    size = int(min(max(int(size), 1), max(int(limit), 1)))
    return size if size % 2 else max(1, size - 1)


def pow2_for(sr: int, seconds: float) -> int:
    """Nearest power-of-two FFT size covering roughly `seconds` of audio."""
    return int(2 ** round(np.log2(max(sr * seconds, 32))))


# ---------------------------------------------------------------------------
# FFT engine. "own" = echox.dsp.transforms (hand-written radix-2, verified to
# 1e-13 against numpy), "numpy" = numpy / scipy. Every transform in the app goes
# through the four functions below, so this one switch covers all of it.
# ---------------------------------------------------------------------------

FFT_ENGINES = ("own", "numpy")
_engine = os.environ.get("ECHOX_FFT", "own").lower()
if _engine not in FFT_ENGINES:
    _engine = "own"


def fft_engine() -> str:
    return _engine


def set_fft_engine(name: str):
    global _engine
    if name not in FFT_ENGINES:
        raise ValueError(f"FFT engine must be one of {FFT_ENGINES}")
    _engine = name


def _own():
    from . import transforms   # late import: transforms needs njit from here
    return transforms


def rfft(frames: np.ndarray) -> np.ndarray:
    """Real FFT along the last axis."""
    return _own().rfft(frames) if _engine == "own" else np.fft.rfft(frames, axis=-1)


def rfftfreq(n: int, d: float = 1.0) -> np.ndarray:
    return np.arange(n // 2 + 1) / (n * d)


def stft(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """(n_fft/2 + 1, frames), Hann window, n_fft/2 zero padding at both ends."""
    if _engine == "own":
        return _own().stft(x, n_fft, hop)
    _, _, X = sps.stft(x, nperseg=n_fft, noverlap=n_fft - hop, window="hann",
                       boundary="zeros", padded=True)
    return X


def istft(X: np.ndarray, n_fft: int, hop: int, length: int) -> np.ndarray:
    if _engine == "own":
        return _own().istft(X, n_fft, hop, length)
    _, y = sps.istft(X, nperseg=n_fft, noverlap=n_fft - hop, window="hann",
                     boundary=True)
    if len(y) < length:
        y = np.pad(y, (0, length - len(y)))
    return y[:length]


def match_length(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))


def report(progress, value: float, text: str = ""):
    if progress is not None:
        progress(float(np.clip(value, 0.0, 1.0)), text)
