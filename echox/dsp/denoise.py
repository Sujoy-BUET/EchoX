"""Noise reduction algorithms.

Every public function takes a mono float64 signal and returns a signal of the
same length. `progress(fraction, text)` is optional.
"""
from __future__ import annotations

import numpy as np

from . import wavelets as wv
from .common import EPS, as_float, istft, njit, odd_kernel, report, rfftfreq, stft

# ---------------------------------------------------------------------------
# Wiener filter in the STFT domain (decision-directed a-priori SNR)
# ---------------------------------------------------------------------------


PAUSE_THRESHOLD_DB = 4.5


def _spectral_floor(q, width):
    """Morphological opening of a PSD in the log domain: removes peaks narrower
    than `width` bins (tonal music that leaked into the estimate) while keeping
    the broad shape of white / pink / brown noise."""
    from scipy.ndimage import maximum_filter1d, minimum_filter1d
    L = np.log(q)
    opened = maximum_filter1d(minimum_filter1d(L, width, mode="nearest"), width, mode="nearest")
    return np.minimum(q, np.exp(opened))


def track_noise_psd(P, hop, sr, window_s=1.5, bias=1.5, smooth=0.95):
    """Minimum statistics: a noise floor that follows the room.

    Every bin goes quiet for a moment whenever the speaker pauses, so the
    smallest value seen in the last `window_s` is a fair guess at the noise
    there, now. The running minimum sits below the true mean, so it is lifted
    by `bias`. Unlike a one-shot profile this survives the fan being switched
    off or a TV starting halfway through.
    Returns a full (bins, frames) estimate.
    """
    from scipy.ndimage import minimum_filter1d
    sm = np.empty_like(P)
    acc = P[:, 0].copy()
    for t in range(P.shape[1]):
        acc = smooth * acc + (1 - smooth) * P[:, t]
        sm[:, t] = acc
    width = max(3, int(window_s * sr / hop))
    return bias * minimum_filter1d(sm, width, axis=1, mode="nearest") + EPS


def noise_drift_db(tracked):
    """How much the tracked floor moves over the clip: the 90th percentile of
    each bin's level minus the 10th, averaged. Small for a steady fan, large
    when sources come and go."""
    lo = np.percentile(tracked, 10, axis=1)
    hi = np.percentile(tracked, 90, axis=1)
    return float(np.mean(10 * np.log10((hi + EPS) / (lo + EPS))))


DRIFT_THRESHOLD_DB = 10.0


def detect_tone(x, sr, n_fft=8192, min_hz=30.0, min_ratio_db=8.0, neighbourhood_hz=80.0):
    """Finds a steady narrow tone (hum, coil whine, whistle).

    Averages the power spectrum over the whole clip, then compares every bin
    with the median of its neighbourhood. A tone is one bin far above that local
    floor; speech and broadband noise are wide, so they sit close to it. When a
    tone has harmonics, every one of them stands out, so the lowest frequency
    within 6 dB of the strongest is returned: notching the fundamental and its
    harmonics then covers the whole series.
    Returns (frequency Hz or None, how many dB it stands out).
    """
    from scipy.ndimage import median_filter
    hop = n_fft // 2
    P = np.abs(stft(as_float(x), n_fft, hop)) ** 2
    avg = P.mean(axis=1) + EPS
    width = max(11, int(neighbourhood_hz * n_fft / sr) | 1)
    floor = median_filter(avg, size=odd_kernel(width, len(avg)), mode="nearest") + EPS
    ratio = 10 * np.log10(avg / floor)
    lo = int(min_hz * n_fft / sr)
    ratio[:lo] = -np.inf
    best = float(np.max(ratio))
    if best < min_ratio_db:
        return None, best
    strong = np.flatnonzero(ratio >= best - 6.0)
    k = int(strong[0])                       # lowest frequency of the series
    return float(k * sr / n_fft), float(ratio[k])


def _broadband_bins(P, tonal_db=8.0, neighbourhood=41):
    """Mask of bins that are not part of a steady narrow tone.

    Compares the average spectrum with a running median of itself; bins that
    tower `tonal_db` over their neighbours belong to a hum or whine, and are
    excluded when judging whether the clip has pauses.
    """
    from scipy.ndimage import median_filter
    avg = P.mean(axis=1) + EPS
    floor = median_filter(avg, size=odd_kernel(neighbourhood, len(avg)), mode="nearest") + EPS
    keep = 10 * np.log10(avg / floor) < tonal_db
    return keep if keep.any() else np.ones(len(avg), dtype=bool)


def estimate_noise_psd(P, frames=None, quiet_fraction=0.1, method="auto",
                       hop=256, sr=44100, window_s=1.5):
    """Noise power. Returns (psd, note); psd is one column, or one column per
    frame when the estimate tracks.

    frames given      -> mean of those frames (user-chosen noise-only region)
    method "quiet"    -> mean of the quietest 10 % of frames
    method "floor"    -> quiet estimate with tonal peaks opened away
    method "track"    -> running minimum per bin, follows changing noise
    method "auto"     -> "track" if the floor drifts over the clip, else
                         "quiet" when the signal has real pauses, else "floor".

    A pause exists when the quietest frames sit at least PAUSE_THRESHOLD_DB below
    the median frame. Dense music never gets there (measured 1.4-3.3 dB), so its
    'quiet' frames still hold sustained notes and the noise estimate is 100x too
    high in exactly the bins where the music lives.

    The depth is measured with steady tonal bins removed. A constant hum or coil
    whine sits in every frame, so counting it would flatten the energy curve and
    hide real pauses: measured on a voice with a 1.37 kHz whine, depth reads
    1.3 dB with the tone counted and 6.0 dB without it, which is the difference
    between the estimator ignoring the whine and removing it.
    """
    if frames is not None and len(frames) > 0:
        return P[:, frames].mean(axis=1) + EPS, "noise from selected region"
    energy = P[_broadband_bins(P)].sum(axis=0)
    k = max(1, int(len(energy) * quiet_fraction))
    quiet_frames = np.argsort(energy)[:k]
    quiet = P[:, quiet_frames].mean(axis=1) + EPS
    depth = 10 * np.log10((np.median(energy) + EPS) / (energy[quiet_frames].mean() + EPS))
    drift = None
    # a running minimum only means something when the signal has gaps; on dense
    # music the "floor" is the music itself, so tracking is never chosen there
    if method == "track" or (method == "auto" and depth >= PAUSE_THRESHOLD_DB):
        tracked = track_noise_psd(P, hop, sr, window_s)
        drift = noise_drift_db(tracked)
        if method == "track" or drift >= DRIFT_THRESHOLD_DB:
            return tracked, (f"noise tracked over {window_s:.1f} s windows "
                             f"(floor moves {drift:.0f} dB during the clip)")
    if method == "auto":
        method = "quiet" if depth >= PAUSE_THRESHOLD_DB else "floor"
    if method == "quiet":
        return quiet, (f"noise from quietest frames ({depth:.1f} dB below median"
                       + (f", floor steady within {drift:.0f} dB)" if drift is not None else ")"))
    width = max(5, (P.shape[0] // 16) | 1)
    return _spectral_floor(quiet, width), (
        f"no pauses found ({depth:.1f} dB), assumed broadband noise; "
        "select a noise-only part for best results")


def wiener_stft(x, n_fft=1024, hop=256, noise_region=None, alpha=0.98,
                oversub=1.0, floor_db=-20.0, method="auto", sr=44100,
                window_s=1.5, noise_signal=None, second_pass=False, progress=None):
    """Spectral Wiener gain G = xi / (1 + xi) with Ephraim-Malah decision-directed
    smoothing of the a-priori SNR xi. `noise_region` = (start, stop) in samples
    of a noise-only part (None = automatic estimate, see estimate_noise_psd).
    Returns (y, note) where note says how the noise profile was obtained."""
    x = as_float(x)
    X = stft(x, n_fft, hop)
    P = np.abs(X) ** 2
    frames = None if noise_region is None else frames_for_samples(*noise_region, hop, P.shape[1])
    if noise_signal is not None:
        noise = np.abs(stft(as_float(noise_signal), n_fft, hop)) ** 2
        noise = noise.mean(axis=1) + EPS
        note = "noise profile from the loaded noise file"
    else:
        noise, note = estimate_noise_psd(P, frames, method=method, hop=hop, sr=sr,
                                         window_s=window_s)
    noise = noise * oversub
    tracking = noise.ndim == 2
    floor = 10 ** (floor_db / 20)
    G = np.empty_like(P)
    prev_clean = np.zeros(P.shape[0])
    n_frames = P.shape[1]
    for t in range(n_frames):
        n_t = noise[:, t] if tracking else noise
        gamma = P[:, t] / n_t                                # a-posteriori SNR
        xi = alpha * prev_clean / n_t + (1 - alpha) * np.maximum(gamma - 1.0, 0.0)
        xi = np.maximum(xi, 1e-3)
        g = np.maximum(xi / (1.0 + xi), floor)
        G[:, t] = g
        prev_clean = (g ** 2) * P[:, t]
        if t % 256 == 0:
            report(progress, 0.9 * t / n_frames, "Computing Wiener gains")
    if second_pass:
        G, note = _refine(P, G, noise, floor, hop, sr, window_s, note)
    y = istft(G * X, n_fft, hop, len(x))
    report(progress, 1.0)
    return y, note


def _refine(P, G, noise, floor, hop, sr, window_s, note):
    """Second pass. The first pass had to guess the a-priori SNR from the frame
    before (decision-directed), which always lags. Now the whole clip has been
    cleaned once, so the estimate of the clean power is available for every
    frame at once, and the noise can be re-measured from what is left over:

        S_hat = G^2 P                  clean power from pass 1
        N     = min-tracked (P - S_hat)  noise re-measured from the residual
        G     = xi / (1 + xi),  xi = S_hat / N
    Measured on synthetic clips this is worth about +0.2 to +0.7 dB, and can
    cost a little on brown noise, which is why it is off by default.
    """
    from scipy.ndimage import minimum_filter1d
    S_hat = (G ** 2) * P
    R = np.maximum(P - S_hat, 0.0)
    acc = R[:, 0].copy()
    for t in range(R.shape[1]):                       # one-pole smoothing over time
        acc = 0.9 * acc + 0.1 * R[:, t]
        R[:, t] = acc
    width = max(3, int(window_s * sr / hop))
    noise2 = 1.5 * minimum_filter1d(R, width, axis=1, mode="nearest") + EPS
    xi = S_hat / noise2
    return np.maximum(xi / (1.0 + xi), floor), note + "; refined in a second pass"


def frames_for_samples(i0, i1, hop, n_frames):
    """STFT frame indices whose centres fall inside [i0, i1)."""
    a = int(np.ceil(i0 / hop))
    b = int(np.floor(i1 / hop))
    return np.arange(max(a, 0), min(b + 1, n_frames))


# ---------------------------------------------------------------------------
# Adaptive notch filter (constrained 2nd-order IIR, simplified gradient)
# ---------------------------------------------------------------------------


@njit(cache=True)
def _anf_stage(x, a0, a_lo, a_hi, rho, mu, lam):
    """One adaptive notch:
        H(z) = (1 + a z^-1 + z^-2) / (1 + rho a z^-1 + rho^2 z^-2),  a = -2 cos(w0)
    Zeros sit on the unit circle at +-w0, poles just inside at radius rho, so the
    notch is deep and narrow. `a` follows a normalised gradient step
        a <- a - mu e[n] s[n-1] / P[n]
    where P is a running power estimate of s[n-1]. a is kept in [a_lo, a_hi].
    """
    N = x.shape[0]
    e = np.empty(N)
    a_track = np.empty(N)
    a = a0
    s1 = 0.0
    s2 = 0.0
    p = 1e-6
    for n in range(N):
        s0 = x[n] - rho * a * s1 - rho * rho * s2
        en = s0 + a * s1 + s2
        p = lam * p + (1.0 - lam) * s1 * s1
        a = a - mu * en * s1 / (p + 1e-12)
        if a < a_lo:
            a = a_lo
        elif a > a_hi:
            a = a_hi
        s2 = s1
        s1 = s0
        e[n] = en
        a_track[n] = a
    return e, a_track


def adaptive_notch(x, sr, f0=50.0, harmonics=3, bandwidth_hz=4.0, mu=0.002,
                   search_pct=10.0, progress=None):
    """Cascade of adaptive notches starting at f0, 2f0, ... Each notch may drift
    +-search_pct % around its start frequency to lock onto the real tone.
    Returns (output, tracked frequency of the first notch in Hz over time)."""
    y = as_float(x)
    rho = float(np.clip(1.0 - np.pi * bandwidth_hz / sr, 0.9, 0.99999))
    lam = float(np.exp(-1.0 / (0.02 * sr)))  # ~20 ms power averaging
    tracked = None
    for k in range(1, harmonics + 1):
        fk = f0 * k
        if fk >= sr / 2 * 0.95:
            break
        lo = fk * (1 - search_pct / 100)
        hi = min(fk * (1 + search_pct / 100), sr / 2 * 0.98)
        a0 = -2 * np.cos(2 * np.pi * fk / sr)
        a_lo = -2 * np.cos(2 * np.pi * lo / sr)
        a_hi = -2 * np.cos(2 * np.pi * hi / sr)
        y, a_track = _anf_stage(y, a0, a_lo, a_hi, rho, mu, lam)
        if tracked is None:
            tracked = np.arccos(np.clip(-a_track / 2, -1, 1)) * sr / (2 * np.pi)
        report(progress, k / harmonics, f"Notch {k} locked")
    return y, tracked


# ---------------------------------------------------------------------------
# Wavelet shrinkage
# ---------------------------------------------------------------------------


def wavelet_denoise(x, wavelet="db8", level=6, rule="bayes", mode="soft", progress=None):
    """Decompose with a DWT, shrink detail coefficients, reconstruct.

    Noise sigma comes from the finest detail band: sigma = median(|d1|) / 0.6745.
    visu : one universal threshold  T = sigma * sqrt(2 ln N)
    bayes: per-band threshold       T_j = sigma^2 / sigma_x,j
    """
    x = as_float(x)
    level = int(max(1, min(level, wv.max_level(len(x), wavelet))))
    coeffs, n = wv.wavedec(x, wavelet, level)
    report(progress, 0.3, "Decomposed")
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 + EPS
    out = [coeffs[0]]
    for j, d in enumerate(coeffs[1:]):
        if rule == "visu":
            T = sigma * np.sqrt(2 * np.log(len(x)))
        else:
            sx = np.sqrt(max(np.mean(d ** 2) - sigma ** 2, EPS))
            T = sigma ** 2 / sx
        out.append(wv.threshold(d, T, mode))
        report(progress, 0.3 + 0.6 * (j + 1) / level, f"Shrinking level {level - j}")
    y = wv.waverec(out, wavelet, n)
    report(progress, 1.0)
    return y, sigma




# ---------------------------------------------------------------------------
# Click / transient removal (hammering, door bangs, keyboard taps)
# ---------------------------------------------------------------------------


def declick(x, sr, n_fft=1024, strength=1.0, rise_db=10.0, hi_hz=8000.0,
            k_time=17, k_freq=51, progress=None):
    """Removes short broadband bangs without touching the voice.

    Two tests have to agree before a frame is touched:
      1. its energy above `hi_hz` jumps `rise_db` above the local median - a
         hammer or a tap is broadband to the very top, speech is not;
      2. inside that frame, the HPSS split says the energy is percussive
         (vertical in the spectrogram) rather than harmonic.
    Everything else passes through untouched, which is why the speech survives:
    measured on a street recording, clicks drop 19 dB and the voice loses 0.1 dB.
    Returns (y, fraction of frames touched).
    """
    from scipy.ndimage import median_filter
    x = as_float(x)
    hop = n_fft // 4
    X = stft(x, n_fft, hop)
    V = np.abs(X)
    freqs = rfftfreq(n_fft, 1 / sr)
    band = freqs >= min(hi_hz, sr / 2 * 0.9)
    hi = 10 * np.log10((V[band] ** 2).mean(axis=0) + EPS)
    base = median_filter(hi, size=odd_kernel(101, len(hi)), mode="nearest")
    hit = hi > base + rise_db
    report(progress, 0.5, f"{hit.sum()} frames look like clicks")
    if hit.any():
        harmonic = median_filter(V, size=(1, odd_kernel(k_time, V.shape[1])), mode="nearest")
        percussive = median_filter(V, size=(odd_kernel(k_freq, V.shape[0]), 1), mode="nearest")
        mask_h = harmonic ** 2 / (harmonic ** 2 + percussive ** 2 + EPS)
        m = np.ones_like(V)
        m[:, hit] = 1.0 - strength * (1.0 - mask_h[:, hit])
        X = m * X
    report(progress, 1.0)
    return istft(X, n_fft, hop, len(x)), float(hit.mean())
