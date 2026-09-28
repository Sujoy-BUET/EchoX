"""Controlled experiments: add known noise to a clean signal, keep the noise
as a known reference, and score any denoiser with SNR."""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

from .common import EPS, as_float, rms

NOISE_TYPES = ["White", "Pink", "Hum 50 Hz", "Hum 60 Hz", "Fan (brown)"]


def make_noise(kind: str, n: int, sr: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    if kind.startswith("Hum"):
        f = 60.0 if "60" in kind else 50.0
        return sum((1.0 / k) * np.sin(2 * np.pi * f * k * t + rng.uniform(0, 2 * np.pi))
                   for k in (1, 2, 3, 5))
    w = rng.standard_normal(n)
    if kind == "Pink":   # 3-pole/3-zero 1/f approximation (J. O. Smith)
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        return sps.lfilter(b, a, w)
    if kind == "Fan (brown)":
        return sps.lfilter([1.0], [1.0, -0.995], w)
    return w


def acoustic_path(sr: int, seed: int = 1) -> np.ndarray:
    """A short random decaying impulse response: the noise reaches the main
    microphone through this path, the reference microphone hears it directly."""
    rng = np.random.default_rng(seed)
    n = max(8, int(0.002 * sr))
    h = rng.standard_normal(n) * np.exp(-np.arange(n) / (n / 4))
    h[0] = 1.0
    return h / np.sqrt(np.sum(h ** 2))


def add_noise(clean: np.ndarray, sr: int, kind: str, snr_db: float):
    """Returns (noisy, reference_noise). noisy = clean + g*(h*v), reference = v."""
    clean = as_float(clean)
    v = make_noise(kind, len(clean), sr)
    at_mic = sps.lfilter(acoustic_path(sr), 1.0, v)
    g = rms(clean) / (rms(at_mic) * 10 ** (snr_db / 20))
    return clean + g * at_mic, g * v


def snr_db(clean: np.ndarray, estimate: np.ndarray) -> float:
    n = min(len(clean), len(estimate))
    c, e = clean[:n], estimate[:n]
    return float(10 * np.log10((np.sum(c ** 2) + EPS) / (np.sum((c - e) ** 2) + EPS)))
