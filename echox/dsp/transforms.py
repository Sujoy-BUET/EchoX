"""Hand-written transforms: iterative radix-2 FFT, real FFT, Bluestein, STFT.

Nothing here calls numpy.fft or scipy.fft. numba compiles the butterfly loops.

    fft(x)          complex DFT along the last axis, any length
    ifft(X)         inverse, via  ifft(X) = conj(fft(conj(X))) / N
    rfft(x)         real input, returns N/2 + 1 bins, costs one N/2-point FFT
    irfft(X, n)     inverse of rfft
    stft / istft    Hann analysis + weighted overlap-add, same framing and
                    scaling as scipy.signal.stft(boundary="zeros", padded=True)
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .common import njit

# ---------------------------------------------------------------------------
# Tables (cached per size)
# ---------------------------------------------------------------------------


def is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


@lru_cache(maxsize=32)
def _bitrev(n: int) -> np.ndarray:
    """rev[i] = i with its log2(n) bits reversed. n = 8: [0 4 2 6 1 5 3 7]."""
    bits = n.bit_length() - 1
    rev = np.zeros(n, dtype=np.int64)
    for b in range(bits):
        rev |= ((np.arange(n) >> b) & 1) << (bits - 1 - b)
    return rev


@lru_cache(maxsize=32)
def _twiddle(n: int) -> np.ndarray:
    """W_n^k = exp(-2j*pi*k/n) for k < n/2."""
    k = np.arange(n // 2)
    return np.cos(2 * np.pi * k / n) - 1j * np.sin(2 * np.pi * k / n)


def hann(n: int) -> np.ndarray:
    """Periodic Hann window (sums to a constant under 75 % and 50 % overlap)."""
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


# ---------------------------------------------------------------------------
# Radix-2 decimation-in-time, in place, one row per signal
# ---------------------------------------------------------------------------


@njit(cache=True)
def _fft_rows(a, rev, w):
    rows, n = a.shape
    for r in range(rows):
        # 1. bit-reversal permutation (swap each pair once)
        for i in range(n):
            j = rev[i]
            if j > i:
                t = a[r, i]
                a[r, i] = a[r, j]
                a[r, j] = t
        # 2. log2(n) stages of butterflies; size = 2, 4, 8, ... n
        size = 2
        while size <= n:
            half = size // 2
            step = n // size                  # twiddle stride for this stage
            for start in range(0, n, size):
                for k in range(half):
                    tw = w[k * step] * a[r, start + k + half]
                    u = a[r, start + k]
                    a[r, start + k] = u + tw
                    a[r, start + k + half] = u - tw
            size *= 2


def _fft_pow2(x: np.ndarray) -> np.ndarray:
    n = x.shape[-1]
    a = np.array(x, dtype=np.complex128).reshape(-1, n)   # copy, rows contiguous
    if n > 1:
        _fft_rows(a, _bitrev(n), _twiddle(n))
    return a.reshape(x.shape)


def _fft_bluestein(x: np.ndarray) -> np.ndarray:
    """Any length N through a power-of-two circular convolution:
    X[k] = conj(c[k]) * sum_n (x[n] conj(c[n])) c[k-n],  c[n] = exp(j*pi*n^2/N)."""
    n = x.shape[-1]
    m = 1 << int(np.ceil(np.log2(2 * n - 1)))
    k = np.arange(n)
    chirp = np.exp(1j * np.pi * (k * k % (2 * n)) / n)
    a = np.zeros(x.shape[:-1] + (m,), dtype=np.complex128)
    a[..., :n] = x * np.conj(chirp)
    b = np.zeros(m, dtype=np.complex128)
    b[:n] = chirp
    b[m - n + 1:] = chirp[1:][::-1]
    conv = ifft(_fft_pow2(a) * _fft_pow2(b))
    return conv[..., :n] * np.conj(chirp)


def fft(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return _fft_pow2(x) if is_pow2(x.shape[-1]) else _fft_bluestein(x)


def ifft(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    return np.conj(fft(np.conj(X))) / X.shape[-1]


# ---------------------------------------------------------------------------
# Real FFT: pack even/odd samples into one complex signal of half length
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _half_twiddle(n: int) -> np.ndarray:
    k = np.arange(n // 2 + 1)
    return np.exp(-2j * np.pi * k / n)


@njit(cache=True)
def _pack_even_odd(x, out):
    rows, h = out.shape
    for r in range(rows):
        for m in range(h):
            out[r, m] = complex(x[r, 2 * m], x[r, 2 * m + 1])


@njit(cache=True)
def _rfft_post(Z, tw, out):
    rows, h = Z.shape
    for r in range(rows):
        for k in range(h + 1):
            zk = Z[r, k % h]
            zn = Z[r, (h - k) % h].conjugate()
            e = 0.5 * (zk + zn)
            o = -0.5j * (zk - zn)
            out[r, k] = e + tw[k] * o


@njit(cache=True)
def _irfft_pre(X, tw, out):
    rows, h = out.shape
    for r in range(rows):
        for k in range(h):
            xk = X[r, k]
            xr = X[r, h - k].conjugate()
            e = 0.5 * (xk + xr)
            o = 0.5 * (xk - xr) * tw[k].conjugate()
            out[r, k] = e + 1j * o


@njit(cache=True)
def _unpack_even_odd(z, out):
    rows, h = z.shape
    for r in range(rows):
        for m in range(h):
            out[r, 2 * m] = z[r, m].real
            out[r, 2 * m + 1] = z[r, m].imag


def rfft(x: np.ndarray) -> np.ndarray:
    """z[m] = x[2m] + j x[2m+1];  Z = FFT_{N/2}(z)
    E[k] = (Z[k] + Z*[-k]) / 2,  O[k] = (Z[k] - Z*[-k]) / 2j
    X[k] = E[k] + W_N^k O[k],  k = 0 .. N/2   (E, O periodic in N/2)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[-1]
    if n % 2 or n < 4 or not is_pow2(n):
        return fft(x)[..., : n // 2 + 1]
    h = n // 2
    rows = np.ascontiguousarray(x.reshape(-1, n))
    z = np.empty((rows.shape[0], h), dtype=np.complex128)
    _pack_even_odd(rows, z)
    _fft_rows(z, _bitrev(h), _twiddle(h))
    out = np.empty((rows.shape[0], h + 1), dtype=np.complex128)
    _rfft_post(z, _half_twiddle(n), out)
    return out.reshape(x.shape[:-1] + (h + 1,))


def irfft(X: np.ndarray, n: int) -> np.ndarray:
    """Undo rfft: E[k] = (X[k] + X*[h-k]) / 2,  O[k] = (X[k] - X*[h-k]) W^-k / 2,
    z = IFFT_{N/2}(E + jO), then x[2m] = Re z, x[2m+1] = Im z."""
    X = np.asarray(X, dtype=np.complex128)
    if n % 2 or n < 4 or not is_pow2(n):
        full = np.concatenate([X, np.conj(X[..., 1:(n + 1) // 2][..., ::-1])], axis=-1)
        return ifft(full).real
    h = n // 2
    rows = np.ascontiguousarray(X.reshape(-1, X.shape[-1]))
    Z = np.empty((rows.shape[0], h), dtype=np.complex128)
    _irfft_pre(rows, _half_twiddle(n), Z)
    # inverse FFT as conj(FFT(conj(Z))) / h, done in place
    np.conjugate(Z, out=Z)
    _fft_rows(Z, _bitrev(h), _twiddle(h))
    np.conjugate(Z, out=Z)
    Z /= h
    out = np.empty((rows.shape[0], n))
    _unpack_even_odd(Z, out)
    return out.reshape(X.shape[:-1] + (n,))


def rfftfreq(n: int, d: float = 1.0) -> np.ndarray:
    return np.arange(n // 2 + 1) / (n * d)


# ---------------------------------------------------------------------------
# STFT / ISTFT
# ---------------------------------------------------------------------------


def frame_signal(x: np.ndarray, n: int, hop: int) -> np.ndarray:
    """(frames, n) view-free copy of overlapping frames."""
    count = 1 + (len(x) - n) // hop
    idx = np.arange(count)[:, None] * hop + np.arange(n)[None, :]
    return x[idx]


def stft(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Returns (n_fft/2 + 1, frames). Zero-pads n_fft/2 on each side, then pads
    the end so frames tile exactly; each frame is scaled by 1/sum(window)."""
    win = hann(n_fft)
    x = np.pad(np.asarray(x, dtype=np.float64), (n_fft // 2, n_fft // 2))
    extra = (-(len(x) - n_fft) % hop) % n_fft
    x = np.pad(x, (0, extra))
    frames = frame_signal(x, n_fft, hop) * win
    return (rfft(frames) / win.sum()).T


def istft(X: np.ndarray, n_fft: int, hop: int, length: int) -> np.ndarray:
    """Weighted overlap-add:  y = sum_t w * frame_t / sum_t w^2, then drop the
    n_fft/2 samples of padding at both ends."""
    win = hann(n_fft)
    frames = irfft(X.T, n_fft) * win.sum() * win
    count = frames.shape[0]
    total = n_fft + (count - 1) * hop
    y = np.zeros(total)
    norm = np.zeros(total)
    w2 = win * win
    for t in range(count):
        s = t * hop
        y[s:s + n_fft] += frames[t]
        norm[s:s + n_fft] += w2
    ok = norm > 1e-10
    y[ok] /= norm[ok]
    y = y[n_fft // 2: total - n_fft // 2]
    if len(y) < length:
        y = np.pad(y, (0, length - len(y)))
    return y[:length]
