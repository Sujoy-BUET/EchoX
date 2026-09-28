"""Time and frequency domain effects: equaliser, Doppler, reverse, speed.

Only numpy and the project's own transforms are used; the filters are written
out as difference equations so the maths stays visible.
"""
from __future__ import annotations

import numpy as np

from .common import EPS, as_float, njit, report
from .transforms import istft, stft

# ---------------------------------------------------------------------------
# Biquad filters (Robert Bristow-Johnson cookbook formulas)
# ---------------------------------------------------------------------------


def peaking_coeffs(sr, f0, gain_db, q=1.0):
    """Peaking EQ: boosts or cuts a band around f0 and leaves the rest alone."""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * min(f0, 0.45 * sr) / sr
    alpha = np.sin(w0) / (2 * q)
    cw = np.cos(w0)
    b = np.array([1 + alpha * A, -2 * cw, 1 - alpha * A])
    a = np.array([1 + alpha / A, -2 * cw, 1 - alpha / A])
    return b / a[0], a / a[0]


def shelf_coeffs(sr, f0, gain_db, high=False, slope=1.0):
    """Low or high shelf: everything below (or above) f0 is lifted or cut."""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * min(f0, 0.45 * sr) / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / 2 * np.sqrt((A + 1 / A) * (1 / slope - 1) + 2)
    twoSqrtAalpha = 2 * np.sqrt(A) * alpha
    if high:
        b = np.array([A * ((A + 1) + (A - 1) * cw + twoSqrtAalpha),
                      -2 * A * ((A - 1) + (A + 1) * cw),
                      A * ((A + 1) + (A - 1) * cw - twoSqrtAalpha)])
        a = np.array([(A + 1) - (A - 1) * cw + twoSqrtAalpha,
                      2 * ((A - 1) - (A + 1) * cw),
                      (A + 1) - (A - 1) * cw - twoSqrtAalpha])
    else:
        b = np.array([A * ((A + 1) - (A - 1) * cw + twoSqrtAalpha),
                      2 * A * ((A - 1) - (A + 1) * cw),
                      A * ((A + 1) - (A - 1) * cw - twoSqrtAalpha)])
        a = np.array([(A + 1) + (A - 1) * cw + twoSqrtAalpha,
                      -2 * ((A - 1) + (A + 1) * cw),
                      (A + 1) + (A - 1) * cw - twoSqrtAalpha])
    return b / a[0], a / a[0]


@njit(cache=True)
def _biquad(x, b, a):
    """y[n] = b0 x[n] + b1 x[n-1] + b2 x[n-2] - a1 y[n-1] - a2 y[n-2]"""
    y = np.empty_like(x)
    x1 = x2 = y1 = y2 = 0.0
    for n in range(x.shape[0]):
        v = b[0] * x[n] + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
        x2, x1 = x1, x[n]
        y2, y1 = y1, v
        y[n] = v
    return y


def biquad(x, b, a, zero_phase=True):
    """Runs the filter forwards and backwards, so the band moves in level but
    not in time (zero phase). Squares the magnitude response, so half the gain
    is applied in each pass."""
    y = _biquad(as_float(x), b, a)
    if not zero_phase:
        return y
    return _biquad(np.ascontiguousarray(y[::-1]), b, a)[::-1].copy()


EQ_BANDS = (60.0, 150.0, 400.0, 1000.0, 2500.0, 6000.0, 12000.0)


def equalizer(x, sr, gains_db, q=1.0, shelves=True, progress=None):
    """7-band EQ. The lowest and highest bands are shelves (they hold their
    gain out to DC and Nyquist), the five in between are peaking filters."""
    y = as_float(x)
    half = [g / 2 for g in gains_db]        # forward+backward doubles the gain
    for i, (f0, g) in enumerate(zip(EQ_BANDS, half)):
        if abs(g) < 0.05:
            continue
        if shelves and i == 0:
            b, a = shelf_coeffs(sr, f0, g, high=False)
        elif shelves and i == len(EQ_BANDS) - 1:
            b, a = shelf_coeffs(sr, f0, g, high=True)
        else:
            b, a = peaking_coeffs(sr, f0, g, q)
        y = biquad(y, b, a)
        report(progress, (i + 1) / len(EQ_BANDS), f"{f0:.0f} Hz")
    return y


# ---------------------------------------------------------------------------
# Reverse
# ---------------------------------------------------------------------------


def reverse(x, region=None):
    """Reverses the whole signal, or only a selected region."""
    y = as_float(x).copy()
    if region is None:
        return y[::-1].copy()
    i0, i1 = region
    y[i0:i1] = y[i0:i1][::-1]
    return y


# ---------------------------------------------------------------------------
# Speed
# ---------------------------------------------------------------------------


def _sinc_lowpass(cutoff_ratio, taps=127):
    """Windowed-sinc low-pass, cutoff as a fraction of the sample rate."""
    n = np.arange(taps) - (taps - 1) / 2
    h = 2 * cutoff_ratio * np.sinc(2 * cutoff_ratio * n) * np.hanning(taps)
    return h / h.sum()


def resample_speed(x, rate, progress=None):
    """Speed change the tape way: read the samples faster or slower, so pitch
    moves with speed. Speeding up first low-passes to keep the new Nyquist
    limit, otherwise the top octave folds back as aliasing."""
    x = as_float(x)
    if abs(rate - 1.0) < 1e-6:
        return x.copy()
    if rate > 1.0:
        x = np.convolve(x, _sinc_lowpass(0.5 / rate), mode="same")
    report(progress, 0.5, "resampling")
    n_out = max(1, int(round(len(x) / rate)))
    idx = np.arange(n_out) * rate
    i0 = np.floor(idx).astype(np.int64)
    frac = idx - i0
    i1 = np.minimum(i0 + 1, len(x) - 1)
    i0 = np.minimum(i0, len(x) - 1)
    return (1 - frac) * x[i0] + frac * x[i1]


def _find_peaks(m, min_gap=3):
    """Local maxima of the magnitude spectrum, at least `min_gap` bins apart,
    used as the phase-locking anchors below."""
    bigger_left = m[1:-1] >= m[:-2]
    bigger_right = m[1:-1] >= m[2:]
    idx = 1 + np.flatnonzero(bigger_left & bigger_right)
    if len(idx) == 0:
        return np.array([int(np.argmax(m))])
    keep = [idx[0]]
    for k in idx[1:]:
        if k - keep[-1] >= min_gap:
            keep.append(k)
        elif m[k] > m[keep[-1]]:
            keep[-1] = k
    return np.asarray(keep)


def phase_vocoder(x, rate, n_fft=2048, progress=None, quality="high"):
    """Time stretch without touching pitch.

    The STFT is read at fractional frame positions (step `rate`). Magnitudes
    are interpolated between neighbouring frames; the phase is not, because
    interpolated phase is meaningless.

    quality="fast": each bin's phase advances on its own (the textbook
    vocoder). Cheap, but nearby bins drift apart over time ("phasiness"), and
    a sharp attack gets smeared across the analysis window because nothing
    tells the algorithm a transient just happened.

    quality="high" (default) adds the two fixes real players use:
      - identity phase locking: bins are grouped around the local magnitude
        peaks (a peak is usually one harmonic or partial). Only the peak's own
        phase is accumulated frame to frame; every other bin in its group is
        locked to the peak using the *input* spectrum's own phase relationship,
        i.e. the shape that already sounded coherent. This is what removes the
        robotic/phasey texture.
      - transient reset: a broadband jump in energy between two source frames
        (spectral flux) means an onset — a horn, a consonant, a click. At that
        instant the accumulated phase is thrown away and replaced with the
        input's own phase, so the attack is not stretched across the window.
      Measured on an isolated 2 ms attack, 2x speed: fast 10.2 ms rise time,
      high 3.1 ms - close to the 1.2 ms of a plain (pitch-shifting) resample.
    """
    x = as_float(x)
    if abs(rate - 1.0) < 1e-6:
        return x.copy()
    hop = n_fft // 4
    X = stft(x, n_fft, hop)
    mag, ph = np.abs(X), np.angle(X)
    n_bins, n_frames = X.shape
    expected = 2 * np.pi * hop * np.arange(n_bins) / n_fft
    steps = np.arange(0, n_frames - 1, rate)
    out = np.empty((n_bins, len(steps)), dtype=np.complex128)
    acc = ph[:, 0].copy()

    locked = quality == "high"
    if locked:
        flux = np.maximum(mag[:, 1:] - mag[:, :-1], 0.0).sum(axis=0)
        med = np.median(flux)
        mad = np.median(np.abs(flux - med))
        # a floor on the absolute scale matters when most of the clip is quiet
        # (median and MAD collapse to ~0 there): a jump has to be a real fraction
        # of the loudest jump in the whole clip, not just "above the near-silence".
        thresh = max(med + 6.0 * mad, 0.15 * flux.max())

    for i, t in enumerate(steps):
        t0 = int(t)
        frac = t - t0
        m = (1 - frac) * mag[:, t0] + frac * mag[:, t0 + 1]

        if locked and flux[t0] > thresh:
            acc = ph[:, t0 + 1].copy()                      # transient: hard-sync to the input
        else:
            dev = ph[:, t0 + 1] - ph[:, t0] - expected
            dev -= 2 * np.pi * np.round(dev / (2 * np.pi))   # wrap to +-pi
            acc = acc + expected + dev
            if locked:
                peaks = _find_peaks(m)
                region = np.searchsorted((peaks[:-1] + peaks[1:]) / 2.0, np.arange(n_bins))
                anchor = peaks[region]
                acc = acc[anchor] + (ph[:, t0 + 1] - ph[anchor, t0 + 1])

        out[:, i] = m * np.exp(1j * acc)
        if i % 64 == 0:
            report(progress, i / max(len(steps), 1), "stretching")
    length = int(round(len(x) / rate))
    return istft(out, n_fft, hop, length)


def change_speed(x, sr, rate, keep_pitch=True, n_fft=2048, progress=None):
    return (phase_vocoder(x, rate, n_fft, progress) if keep_pitch
            else resample_speed(x, rate, progress))


# ---------------------------------------------------------------------------
# Doppler
# ---------------------------------------------------------------------------


SPEED_OF_SOUND = 343.0


def doppler(x, sr, speed_ms=30.0, distance_m=10.0, mode="Pass-by",
            c=SPEED_OF_SOUND, rate_hz=0.5, attenuate=True, progress=None):
    """Move the source and let physics do the pitch shift.

    Sound leaving the source at time tau reaches the listener at
        t = tau + r(tau)/c
    so a source approaching (r shrinking) packs its samples into less listener
    time: the pitch goes up. Receding does the opposite. Each sample is
    scattered into the output at its arrival time with linear interpolation,
    and scaled by 1/r for distance loss.

    Pass-by: the source travels in a straight line at speed_ms and comes
    within distance_m of the listener, closest point at the middle of the clip.
    Circling: the source goes round the listener at radius distance_m,
    rate_hz turns per second, which gives a repeating wow.
    """
    x = as_float(x)
    n = len(x)
    tau = np.arange(n) / sr
    if mode == "Circling":
        # a circle keeps r constant and would shift nothing, so the path is an
        # ellipse: the source swings close and far once per turn
        ang = 2 * np.pi * rate_hz * tau
        r = np.hypot(distance_m * np.cos(ang), 0.25 * distance_m * np.sin(ang))
        r = np.maximum(r, 0.5)
    else:
        t_pass = tau[-1] / 2
        along = speed_ms * (tau - t_pass)
        r = np.sqrt(distance_m ** 2 + along ** 2)
    report(progress, 0.3, "path")
    arrive = tau + r / c
    arrive -= arrive[0]
    amp = (distance_m / r) if attenuate else np.ones_like(r)
    out_len = int(np.ceil(arrive[-1] * sr)) + 2
    y = np.zeros(out_len)
    weight = np.zeros(out_len)
    pos = arrive * sr
    i0 = np.floor(pos).astype(np.int64)
    frac = pos - i0
    vals = x * amp
    np.add.at(y, i0, vals * (1 - frac))
    np.add.at(y, i0 + 1, vals * frac)
    np.add.at(weight, i0, 1 - frac)
    np.add.at(weight, i0 + 1, frac)
    report(progress, 0.8, "scatter")
    # where the source moved away, samples land sparsely: smooth the holes
    busy = weight > 1e-6
    y[busy] /= np.maximum(weight[busy], 0.2)
    peak = np.abs(y).max() + EPS
    if peak > 1.0:
        y /= peak
    report(progress, 1.0)
    return y


# ---------------------------------------------------------------------------
# Cut, keep and join: the three edits every editor needs
# ---------------------------------------------------------------------------


def _edge_fade(x, ms, sr):
    """A short cosine fade at the very start and end of x, so a hard cut or
    join has no click. No-op if x is shorter than 2 fades, or ms is 0."""
    x = x.copy()
    n = int(ms / 1000 * sr)
    if n <= 0 or len(x) < 2 * n:
        return x
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))
    x[:n] *= ramp
    x[-n:] *= ramp[::-1]
    return x


def trim(x, region, sr=44100, fade_ms=5.0):
    """Removes region=(i0, i1) and joins what is left. A short fade on each
    side of the join (not on the outer ends of x) stops the join clicking."""
    x = as_float(x)
    i0, i1 = region
    i0, i1 = max(0, int(i0)), min(len(x), int(i1))
    if i1 <= i0:
        return x.copy()
    left, right = x[:i0].copy(), x[i1:].copy()
    n = int(fade_ms / 1000 * sr)
    if n > 0 and len(left) >= n:                              # fade the two edges that
        left[-n:] *= np.linspace(1, 0, n) ** 0.5              # now touch each other -
    if n > 0 and len(right) >= n:                             # not the piece's own far ends
        right[:n] *= np.linspace(0, 1, n) ** 0.5
    return np.concatenate([left, right])


def crop(x, region, sr=44100, fade_ms=5.0):
    """Keeps only region=(i0, i1); a short fade in/out avoids a click at the
    new start and end (there is no more audio there to blend with)."""
    x = as_float(x)
    i0, i1 = region
    i0, i1 = max(0, int(i0)), min(len(x), int(i1))
    return _edge_fade(x[i0:i1].copy(), fade_ms, sr)


def concatenate(x, other, sr=44100, before=False, crossfade_ms=0.0):
    """Joins `other` after x (or before it). crossfade_ms > 0 overlaps the
    join with an equal-power crossfade instead of a hard splice."""
    x = as_float(x)
    other = as_float(other)
    a, b = (other, x) if before else (x, other)
    n = int(crossfade_ms / 1000 * sr)
    if n <= 0 or len(a) < n or len(b) < n:
        return np.concatenate([a, b])
    fade_out = np.cos(np.linspace(0, np.pi / 2, n)) ** 2
    fade_in = 1.0 - fade_out
    head = a[:-n]
    joint = a[-n:] * fade_out + b[:n] * fade_in
    tail = b[n:]
    return np.concatenate([head, joint, tail])


# ---------------------------------------------------------------------------
# Overlay: mix a second file in (background music under a voice, or the
# reverse), at a chosen level, start time and, optionally, looped to fill.
# ---------------------------------------------------------------------------


def overlay(x, other, gain_db=0.0, start_s=0.0, sr=44100, loop=False,
            duck_db=0.0, progress=None):
    """y = x, with `other` (already at the same sample rate and mono) added in.

    `other` plays from `start_s` in x's timeline. If it runs out first and
    `loop` is off, silence follows for the rest of x (nothing more is added).
    If `loop` is on, `other` repeats to cover the rest of x.
    `duck_db` (negative) turns x down while `other` is sounding - handy for
    lowering background music under narration without a separate step.
    A soft limiter (ceiling -1 dBFS) catches any peak from the sum.
    """
    x = as_float(x)
    other = as_float(other)
    n = len(x)
    start = max(0, int(round(start_s * sr)))
    if start >= n or len(other) == 0:
        return x.copy()
    span = n - start
    if loop:
        reps = int(np.ceil(span / len(other)))
        seg = np.tile(other, reps)[:span]
    else:
        seg = other[:span]
    seg = seg * (10 ** (gain_db / 20))
    y = x.copy()
    end = start + len(seg)
    if duck_db < 0:
        edge = max(1, int(0.05 * sr))                          # 50 ms fade at each end
        env = np.ones(len(seg))
        env[:edge] = np.linspace(1, 10 ** (duck_db / 20), edge)
        env[edge:-edge or None] = 10 ** (duck_db / 20)
        if edge < len(env):
            env[-edge:] = np.linspace(10 ** (duck_db / 20), 1, edge)
        y[start:end] *= env
    y[start:end] += seg
    report(progress, 0.8, "mixed")
    from .amplify import soft_limit
    peak = np.abs(y).max()
    if peak > 10 ** (-1.0 / 20):
        y = soft_limit(y, ceiling_db=-1.0)
    report(progress, 1.0)
    return y


# ---------------------------------------------------------------------------
# Late reverberation suppression (Lebart's spectral method)
# ---------------------------------------------------------------------------


def dereverb_gain(P, sr, hop, rt60=0.6, early_ms=50.0, oversub=1.0, floor_db=-12.0):
    """Spectral gain that removes the late field. See dereverb()."""
    d = max(1, int(round(early_ms / 1000 * sr / hop)))
    decay = 10 ** (-6.0 * (d * hop / sr) / max(rt60, 0.05))
    late = np.zeros_like(P)
    late[:, d:] = decay * P[:, :-d]
    alpha = 0.6                       # one-pole smoothing stops the gain flickering
    acc = late[:, 0]
    for t in range(P.shape[1]):
        acc = alpha * acc + (1 - alpha) * late[:, t]
        late[:, t] = acc
    floor = 10 ** (floor_db / 10)
    return np.maximum(1.0 - oversub * late / (P + EPS), floor), decay


def dereverb(x, sr, rt60=0.6, early_ms=50.0, oversub=1.0, floor_db=-12.0,
             n_fft=1024, progress=None):
    """Removes the tail of a room, keeping the direct sound and early echoes.

    A room decays exponentially: 60 dB of energy in RT60 seconds. So the late
    energy arriving now is what was in the room `early_ms` ago, damped by
        g = 10 ** (-6 * early / rt60)      (energy, -60 dB per rt60)
    Subtract that estimate from each bin, exactly like a Wiener gain:
        G = max(1 - oversub * P_late / P, floor)
    """
    x = as_float(x)
    hop = n_fft // 4
    X = stft(x, n_fft, hop)
    report(progress, 0.4, "late field estimated")
    G, decay = dereverb_gain(np.abs(X) ** 2, sr, hop, rt60, early_ms, oversub, floor_db)
    y = istft(np.sqrt(G) * X, n_fft, hop, len(x))
    report(progress, 1.0)
    return y, decay


def wpe(x, sr, n_fft=512, taps=12, delay=3, iterations=3, progress=None):
    """Weighted prediction error dereverberation.

    In each frequency bin the late field is what a few past frames can still
    predict about the present one, so it is removed by linear prediction:

        y[t] = x[t] - sum_i g[i]* x[t - delay - i]

    The prediction is fitted with weights 1 / |y[t]|^2, so loud frames (where
    the direct sound dominates) do not drown out the quiet tails. Each
    iteration refits with the previous output's power. Unlike spectral
    subtraction this is linear, so it adds no musical noise.
    """
    x = as_float(x)
    hop = n_fft // 4
    X = stft(x, n_fft, hop)
    n_bins, n_frames = X.shape
    Y = X.copy()
    if n_frames <= delay + taps + 4:
        return x.copy(), 0
    for it in range(iterations):
        power = np.maximum(np.abs(Y) ** 2, 1e-12)
        for k in range(n_bins):
            lam = power[k]
            rows = []
            for i in range(taps):
                shift = delay + i
                rows.append(np.concatenate([np.zeros(shift, dtype=complex), X[k, :-shift]]))
            D = np.stack(rows)                       # taps x frames
            Dw = D / lam
            R = Dw @ D.conj().T
            r = Dw @ X[k].conj()
            R[np.diag_indices(taps)] += 1e-8 * np.trace(R).real / taps + 1e-12
            g = np.linalg.solve(R, r)
            Y[k] = X[k] - (g.conj() @ D)
        report(progress, (it + 1) / iterations, f"WPE pass {it + 1}/{iterations}")
    return istft(Y, n_fft, hop, len(x)), taps


# ---------------------------------------------------------------------------
# Plain filters: low-pass, high-pass, band-pass, band-stop
# ---------------------------------------------------------------------------


def butter_coeffs(sr, kind, f_lo, f_hi=None, order=4):
    """Butterworth sections from the bilinear transform, as biquads.

    An analogue Butterworth of order N has its poles evenly spaced on a circle;
    taking them in conjugate pairs gives N/2 second-order sections, each with
    its own damping 1/(2*cos(theta)). The bilinear transform maps each section
    to a digital biquad, with the cut-off pre-warped so it lands where asked.
    Returns a list of (b, a) pairs to run in series.
    """
    pairs = []
    n_sections = max(1, int(np.ceil(order / 2)))
    for k in range(n_sections):
        theta = np.pi * (2 * k + 1) / (2 * n_sections * 2)      # pole angle
        q = 1.0 / (2.0 * np.cos(theta))
        if kind in ("Low-pass", "High-pass"):
            w0 = 2 * np.pi * min(f_lo, 0.45 * sr) / sr
            alpha = np.sin(w0) / (2 * q)
            cw = np.cos(w0)
            if kind == "Low-pass":
                b = np.array([(1 - cw) / 2, 1 - cw, (1 - cw) / 2])
            else:
                b = np.array([(1 + cw) / 2, -(1 + cw), (1 + cw) / 2])
            a = np.array([1 + alpha, -2 * cw, 1 - alpha])
        else:
            lo = min(max(f_lo, 1.0), 0.45 * sr)
            hi = min(max(f_hi or f_lo * 2, lo * 1.01), 0.49 * sr)
            w0 = 2 * np.pi * np.sqrt(lo * hi) / sr              # geometric centre
            bw = np.log(hi / lo)                                 # bandwidth in octaves
            alpha = np.sin(w0) * np.sinh(np.log(2) / 2 * bw * w0 / max(np.sin(w0), EPS)) / q
            cw = np.cos(w0)
            if kind == "Band-pass":
                b = np.array([alpha, 0.0, -alpha])
            else:                                                # Band-stop / notch
                b = np.array([1.0, -2 * cw, 1.0])
            a = np.array([1 + alpha, -2 * cw, 1 - alpha])
        pairs.append((b / a[0], a / a[0]))
    return pairs


def apply_filter(x, sr, kind="Low-pass", f_lo=1000.0, f_hi=4000.0, order=4,
                 zero_phase=True, progress=None):
    """Runs the chosen filter. Zero phase means forward and backward, which
    doubles the attenuation slope and leaves no phase shift at all."""
    y = as_float(x)
    sections = butter_coeffs(sr, kind, f_lo, f_hi, order)
    for i, (b, a) in enumerate(sections):
        y = biquad(y, b, a, zero_phase)
        report(progress, (i + 1) / len(sections))
    return y


# ---------------------------------------------------------------------------
# Echo and reverb (the opposite of the Dereverb tool)
# ---------------------------------------------------------------------------


@njit(cache=True)
def _comb(x, delay, feedback, damping):
    """One feedback comb with a one-pole low-pass in the loop, so each repeat
    comes back a little duller, as a real room does."""
    y = np.zeros(x.shape[0])
    buf = np.zeros(delay)
    store = 0.0
    idx = 0
    for n in range(x.shape[0]):
        out = buf[idx]
        y[n] = out
        store = out * (1 - damping) + store * damping
        buf[idx] = x[n] + store * feedback
        idx += 1
        if idx >= delay:
            idx = 0
    return y


@njit(cache=True)
def _allpass(x, delay, gain):
    """All-pass section: flat magnitude, scrambled phase. Thickens the tail
    without colouring it."""
    y = np.zeros(x.shape[0])
    buf = np.zeros(delay)
    idx = 0
    for n in range(x.shape[0]):
        bufout = buf[idx]
        y[n] = -gain * x[n] + bufout
        buf[idx] = x[n] + bufout * gain
        idx += 1
        if idx >= delay:
            idx = 0
    return y


def echo(x, sr, delay_ms=300.0, feedback=0.35, mix=0.4, repeats_damping=0.2,
         progress=None):
    """Discrete repeats: y[n] = x[n] + fb * y[n - D]. Each repeat is `feedback`
    times quieter, so the number you hear is about log(0.001)/log(feedback)."""
    x = as_float(x)
    d = max(1, int(delay_ms / 1000 * sr))
    wet = _comb(x, d, float(np.clip(feedback, 0.0, 0.95)), float(np.clip(repeats_damping, 0, 0.95)))
    report(progress, 1.0)
    return (1 - mix) * x + mix * wet


def reverb(x, sr, room=0.7, damping=0.4, mix=0.3, pre_delay_ms=20.0, progress=None):
    """Schroeder reverb: four parallel combs of mutually prime lengths make the
    dense tail, two all-passes in series smear it so no single echo stands out.
    `room` sets the feedback (how long the tail rings), `damping` how fast the
    highs die away."""
    x = as_float(x)
    pre = max(0, int(pre_delay_ms / 1000 * sr))
    src = np.concatenate([np.zeros(pre), x])[:len(x)] if pre else x
    fb = 0.7 + 0.28 * float(np.clip(room, 0.0, 1.0))
    wet = np.zeros(len(x))
    for ms in (29.7, 37.1, 41.1, 43.7):                  # mutually prime delays
        wet += _comb(src, max(1, int(ms / 1000 * sr)), fb, float(np.clip(damping, 0, 0.95)))
        report(progress, 0.6)
    wet /= 4.0
    for ms, g in ((5.0, 0.7), (1.7, 0.7)):
        wet = _allpass(wet, max(1, int(ms / 1000 * sr)), g)
    peak = np.abs(wet).max() + EPS
    if peak > 1.0:
        wet /= peak
    report(progress, 1.0)
    return (1 - mix) * x + mix * wet
