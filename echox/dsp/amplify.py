"""Loudness tools: gain, normalisation, dynamic range compression, limiting."""
from __future__ import annotations

import numpy as np

from .common import EPS, as_float, njit, rms


def gain_db(x, db):
    return x * 10 ** (db / 20)


def normalize_peak(x, target_db=-1.0):
    peak = np.max(np.abs(x)) + EPS
    return x * (10 ** (target_db / 20) / peak)


def normalize_rms(x, target_db=-18.0):
    return x * (10 ** (target_db / 20) / rms(x))


@njit(cache=True)
def _compress(x, thr_db, ratio, knee_db, att, rel):
    """Feed-forward compressor with a soft knee.
    Static curve (dB): below the knee no change; above, slope 1/ratio.
    Gain reduction is smoothed with separate attack/release one-pole filters."""
    N = x.shape[0]
    y = np.empty(N)
    g_smooth = 0.0
    half = knee_db / 2.0
    slope = 1.0 - 1.0 / ratio
    for n in range(N):
        lvl = 20.0 * np.log10(abs(x[n]) + 1e-10)
        over = lvl - thr_db
        if over <= -half:
            gr = 0.0
        elif over >= half or knee_db <= 0.0:
            gr = slope * over
        else:
            gr = slope * (over + half) ** 2 / (2.0 * knee_db)
        coef = att if gr > g_smooth else rel
        g_smooth = coef * g_smooth + (1.0 - coef) * gr
        y[n] = x[n] * 10.0 ** (-g_smooth / 20.0)
    return y


def compress(x, sr, threshold_db=-24.0, ratio=4.0, knee_db=6.0, attack_ms=5.0,
             release_ms=120.0, makeup_db=0.0):
    att = np.exp(-1.0 / (sr * attack_ms / 1000))
    rel = np.exp(-1.0 / (sr * release_ms / 1000))
    y = _compress(as_float(x), threshold_db, max(ratio, 1.0), knee_db, att, rel)
    return gain_db(y, makeup_db)


def soft_limit(x, ceiling_db=-1.0, knee=0.8):
    """Linear below knee*ceiling, then a tanh curve that approaches the ceiling
    smoothly, so peaks never clip."""
    c = 10 ** (ceiling_db / 20)
    k = knee * c
    a = np.abs(x)
    over = a > k
    y = x.copy()
    y[over] = np.sign(x[over]) * (k + (c - k) * np.tanh((a[over] - k) / (c - k)))
    return y


def amplify(x, sr, mode="gain", amount_db=6.0, use_compressor=False, comp=None,
            use_limiter=True, ceiling_db=-1.0):
    """mode: 'gain' adds amount_db; 'peak' normalises peak to amount_db dBFS;
    'rms' normalises RMS to amount_db dBFS."""
    y = as_float(x)
    if use_compressor:
        y = compress(y, sr, **(comp or {}))
    if mode == "peak":
        y = normalize_peak(y, amount_db)
    elif mode == "rms":
        y = normalize_rms(y, amount_db)
    else:
        y = gain_db(y, amount_db)
    if use_limiter:
        y = soft_limit(y, ceiling_db)
    return y
