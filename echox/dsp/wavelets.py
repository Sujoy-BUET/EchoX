"""Hand-written discrete wavelet transform (orthogonal, periodized).

Only the published scaling-filter coefficients are stored here; the transform,
its inverse and the thresholding are implemented below.

One level, signal length N (even), scaling filter g (length L):
    a[k] = sum_m g[m]  x[(2k + m) mod N]        approximation (low-pass)
    d[k] = sum_m hh[m] x[(2k + m) mod N]        detail (high-pass)
    hh[m] = (-1)^m g[L-1-m]                     quadrature mirror
Rows of that analysis matrix are orthonormal, so the inverse is its transpose:
    x[(2k + m) mod N] += a[k] g[m] + d[k] hh[m]

Edges: the signal is mirrored ([x, reversed x]) before the periodic transform,
so the wrap-around is continuous and no click appears at the start or end.
"""
from __future__ import annotations

import numpy as np

from .common import njit

# Scaling (reconstruction low-pass) filters, sum = sqrt(2), sum of squares = 1.
FILTERS = {
    "haar": (0.7071067811865476, 0.7071067811865476),
    "db4": (0.2303778133088965, 0.7148465705529157, 0.6308807679298589, -0.027983769416859854, -0.18703481171909309, 0.030841381835560764, 0.0328830116668852, -0.010597401785069032),
    "db8": (0.05441584224310401, 0.31287159091429995, 0.6756307362972898, 0.5853546836542067, -0.015829105256349306, -0.2840155429615469, 0.0004724845739132828, 0.12874742662047847, -0.017369301001807547, -0.044088253930794755, 0.013981027917398282, 0.008746094047405777, -0.004870352993451574, -0.00039174037337694705, 0.0006754494064505693, -0.00011747678412476953),
    "sym8": (0.0018899503327594609, -0.0003029205147213668, -0.01495225833704823, 0.003808752013890615, 0.049137179673607506, -0.027219029917056003, -0.05194583810770904, 0.3644418948353314, 0.7771857517005235, 0.4813596512583722, -0.061273359067658524, -0.1432942383508097, 0.007607487324917605, 0.03169508781149298, -0.0005421323317911481, -0.0033824159510061256),
    "coif5": (-0.000212081862067494, 0.0003585777411617577, 0.0021782943778456947, -0.00415931262757864, -0.010131584846900276, 0.023408322118927783, 0.028169744270532353, -0.09192158806008609, -0.052046670253554764, 0.42157126673075435, 0.7742936228603274, 0.4379823066591634, -0.06203775157498196, -0.10556315130733723, 0.041287530472117834, 0.032674799467057355, -0.019758391600965465, -0.009159507338676163, 0.006761520220620417, 0.0024315754425382886, -0.0016616273039298788, -0.0006375589261258812, 0.0003018579416682448, 0.00014035632812373243, -4.12198619242655e-05, -2.1270221672515614e-05, 3.7007277113394796e-06, 2.0612203985788783e-06, -1.6237995172048338e-07, -9.604010112767894e-08),
}

WAVELETS = tuple(FILTERS)


def filters(name: str):
    g = np.asarray(FILTERS[name], dtype=np.float64)
    L = len(g)
    hh = ((-1.0) ** np.arange(L)) * g[::-1]
    return g, hh


def max_level(n: int, name: str) -> int:
    L = len(FILTERS[name])
    return max(0, int(np.floor(np.log2(n / (L - 1))))) if L > 1 else 0


@njit(cache=True)
def _analysis(x, g, hh, a, d):
    n = x.shape[0]
    L = g.shape[0]
    for k in range(n // 2):
        sa = 0.0
        sd = 0.0
        base = 2 * k
        for m in range(L):
            v = x[(base + m) % n]
            sa += g[m] * v
            sd += hh[m] * v
        a[k] = sa
        d[k] = sd


@njit(cache=True)
def _synthesis(a, d, g, hh, x):
    n = x.shape[0]
    L = g.shape[0]
    for k in range(a.shape[0]):
        base = 2 * k
        for m in range(L):
            x[(base + m) % n] += a[k] * g[m] + d[k] * hh[m]


def dwt(x: np.ndarray, name: str):
    g, hh = filters(name)
    x = np.ascontiguousarray(x, dtype=np.float64)
    a = np.empty(len(x) // 2)
    d = np.empty(len(x) // 2)
    _analysis(x, g, hh, a, d)
    return a, d


def idwt(a: np.ndarray, d: np.ndarray, name: str) -> np.ndarray:
    g, hh = filters(name)
    x = np.zeros(2 * len(a))
    _synthesis(np.ascontiguousarray(a), np.ascontiguousarray(d), g, hh, x)
    return x


def wavedec(x: np.ndarray, name: str, level: int):
    """Returns ([aJ, dJ, ..., d1], original length). Input is mirrored and
    padded so that every level has an even length."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    block = 2 ** max(level - 1, 0)
    padded = int(np.ceil(n / block)) * block
    xp = np.pad(x, (0, padded - n), mode="symmetric") if padded > n else x
    a = np.concatenate([xp, xp[::-1]])
    details = []
    for _ in range(level):
        a, d = dwt(a, name)
        details.append(d)
    return [a] + details[::-1], n


def waverec(coeffs, name: str, n: int) -> np.ndarray:
    a = coeffs[0]
    for d in coeffs[1:]:
        a = idwt(a, d, name)
    return a[:n]


def threshold(d: np.ndarray, T: float, mode: str = "soft") -> np.ndarray:
    if mode == "hard":
        return d * (np.abs(d) > T)
    return np.sign(d) * np.maximum(np.abs(d) - T, 0.0)
