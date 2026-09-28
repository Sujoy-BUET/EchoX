"""Voice / instrument separation on magnitude spectrograms.

All separators return an ordered dict {stem name: signal}. Stems are rebuilt
with soft (or binary) masks on the complex STFT, so the mixture phase is kept.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from .common import EPS, as_float, istft, odd_kernel, pow2_for, report, stft
from .denoise import frames_for_samples

# ---------------------------------------------------------------------------
# HPSS: median filtering (Fitzgerald 2010), optional two-stage vocal split
# ---------------------------------------------------------------------------


def _hpss_masks(S, kernel_time, kernel_freq, power=2.0, margin=1.0):
    kt = odd_kernel(kernel_time, S.shape[1])          # a kernel wider than the
    kf = odd_kernel(kernel_freq, S.shape[0])          # array makes scipy return NaN
    H = median_filter(S, size=(1, kt), mode="reflect")
    P = median_filter(S, size=(kf, 1), mode="reflect")
    Hp, Pp = H ** power, (margin * P) ** power
    mask_h = Hp / (Hp + Pp + EPS)
    return mask_h, 1.0 - mask_h


def hpss(x, sr, kernel_time=31, kernel_freq=31, power=2.0, three_stems=True, progress=None):
    """Horizontal (time) median keeps sustained partials -> harmonic.
    Vertical (frequency) median keeps broadband clicks -> percussive.

    three_stems: second pass with a short window on the percussive part.
    With a long window a singing voice (vibrato, glides) smears and looks
    'percussive'; with a short window it looks harmonic again, while drums
    stay percussive. That splits vocals from drums.
    """
    x = as_float(x)
    n_long = pow2_for(sr, 0.09)
    X = stft(x, n_long, n_long // 4)
    report(progress, 0.15, "Median filtering (long window)")
    mh, mp = _hpss_masks(np.abs(X), kernel_time, kernel_freq, power)
    harmonic = istft(mh * X, n_long, n_long // 4, len(x))
    rest = istft(mp * X, n_long, n_long // 4, len(x))
    report(progress, 0.55, "First stage done")
    if not three_stems:
        report(progress, 1.0)
        return {"Instruments": harmonic, "Drums": rest}

    n_short = pow2_for(sr, 0.012)
    R = stft(rest, n_short, n_short // 4)
    report(progress, 0.7, "Median filtering (short window)")
    mh2, mp2 = _hpss_masks(np.abs(R), kernel_time, max(3, kernel_freq // 4), power)
    vocals = istft(mh2 * R, n_short, n_short // 4, len(x))
    drums = istft(mp2 * R, n_short, n_short // 4, len(x))
    report(progress, 1.0)
    return {"Vocals": vocals, "Instruments": harmonic, "Percussion": drums}


# ---------------------------------------------------------------------------
# NMF with KL-divergence multiplicative updates (Lee & Seung)
# ---------------------------------------------------------------------------


def nmf_kl(V, n_components, n_iter=200, W_fixed=None, seed=0, progress=None,
           p_lo=0.0, p_hi=1.0):
    """Factorise V ~ W H. If W_fixed is given, those columns stay frozen and
    only `n_components` extra columns are learned."""
    rng = np.random.default_rng(seed)
    F, T = V.shape
    k_fixed = 0 if W_fixed is None else W_fixed.shape[1]
    W_free = rng.random((F, n_components)) + 0.1
    W = W_free if W_fixed is None else np.hstack([W_fixed, W_free])
    H = rng.random((W.shape[1], T)) + 0.1
    ones = np.ones_like(V)
    for it in range(n_iter):
        WH = W @ H + EPS
        H *= (W.T @ (V / WH)) / (W.T @ ones + EPS)
        WH = W @ H + EPS
        upd = ((V / WH) @ H.T) / (ones @ H.T + EPS)
        W[:, k_fixed:] *= upd[:, k_fixed:]
        scale = W[:, k_fixed:].sum(axis=0) + EPS            # unit-sum columns
        W[:, k_fixed:] /= scale
        H[k_fixed:] *= scale[:, None]
        if it % 10 == 0:
            report(progress, p_lo + (p_hi - p_lo) * it / n_iter, f"NMF iteration {it}/{n_iter}")
    return W, H


def nmf_separate(x, sr, n_components=6, n_iter=200, train_region=None,
                 trained_role="accompaniment", n_trained=8, progress=None):
    """Two modes.
    Unsupervised: every NMF component becomes its own stem (sorted from low to
    high spectral centroid), so you can listen and pick.
    Trained: `train_region` (start, stop in samples) covers only one source (for example an
    instrumental intro). Its basis W_a is learned first and frozen, then the full
    mix is explained as [W_a | W_b]; W_b captures the other source.
    """
    x = as_float(x)
    n_fft = pow2_for(sr, 0.064)
    hop = n_fft // 4
    X = stft(x, n_fft, hop)
    V = np.abs(X)
    freqs = np.linspace(0, sr / 2, V.shape[0])
    train_frames = None
    if train_region is not None:
        train_frames = frames_for_samples(*train_region, hop, V.shape[1])

    def rebuild(parts):
        total = sum(parts) + EPS
        return [istft((p / total) * X, n_fft, hop, len(x)) for p in parts]

    if train_frames is None or len(train_frames) < 8:
        W, H = nmf_kl(V, n_components, n_iter, progress=progress, p_hi=0.9)
        centroid = (freqs[:, None] * W).sum(axis=0) / (W.sum(axis=0) + EPS)
        order = np.argsort(centroid)
        parts = [np.outer(W[:, k], H[k]) for k in order]
        report(progress, 0.95, "Rebuilding stems")
        signals = rebuild(parts)
        return {f"Component {i + 1} ({centroid[k]:.0f} Hz)": s
                for i, (k, s) in enumerate(zip(order, signals))}

    W_a, _ = nmf_kl(V[:, train_frames], n_trained, n_iter, progress=progress, p_hi=0.35)
    W, H = nmf_kl(V, n_components, n_iter, W_fixed=W_a, progress=progress, p_lo=0.35, p_hi=0.9)
    k = n_trained
    part_a = W[:, :k] @ H[:k]
    part_b = W[:, k:] @ H[k:]
    report(progress, 0.95, "Rebuilding stems")
    sig_a, sig_b = rebuild([part_a, part_b])
    if trained_role == "accompaniment":
        return {"Vocals": sig_b, "Accompaniment": sig_a}
    return {"Vocals": sig_a, "Accompaniment": sig_b}


# ---------------------------------------------------------------------------
# REPET: the accompaniment repeats, the voice does not (Rafii & Pardo 2013)
# ---------------------------------------------------------------------------


def beat_spectrum(V):
    """Autocorrelation of every frequency row, averaged. Peaks sit at the
    lag where the whole spectrogram repeats itself."""
    P = V ** 2
    P = P - P.mean(axis=1, keepdims=True)
    n = P.shape[1]
    n_fft = int(2 ** np.ceil(np.log2(2 * n)))
    F = np.fft.rfft(P, n=n_fft, axis=1)
    ac = np.fft.irfft(np.abs(F) ** 2, n=n_fft, axis=1)[:, :n]
    ac /= (np.arange(n, 0, -1))[None, :]          # unbiased
    b = ac.mean(axis=0)
    return b / (b[0] + EPS)


def find_period(V, hop, sr, lo_s=0.5, hi_s=8.0):
    b = beat_spectrum(V)
    lo = max(1, int(lo_s * sr / hop))
    hi = min(len(b) - 1, int(hi_s * sr / hop), V.shape[1] // 3)
    if hi <= lo:
        return max(lo, 1), b
    lag = lo + int(np.argmax(b[lo:hi]))
    return lag, b


def repeating_model(V, period):
    """Median over all repetitions, then clipped to the mixture: the model can
    never claim more energy than the frame actually has."""
    n_bins, n_frames = V.shape
    period = int(max(1, min(period, n_frames)))       # never longer than the clip
    reps = int(np.ceil(n_frames / period))
    pad = np.full((n_bins, reps * period), np.nan)
    pad[:, :n_frames] = V
    cube = pad.reshape(n_bins, reps, period)
    med = np.nanmedian(cube, axis=1)                      # n_bins x period
    tiled = np.tile(med, (1, reps))[:, :n_frames]
    return np.minimum(tiled, V)


def repet(x, sr, period_s=0.0, mask="Soft", power=1.0, progress=None):
    """Splits a repeating background from a varying foreground.

    1. find the repeating period from the beat spectrum
    2. median over repetitions -> what the background looks like
    3. mask: background where the mixture matches the model, voice elsewhere
    """
    x = as_float(x)
    n_fft = pow2_for(sr, 0.06)
    hop = n_fft // 4
    X = stft(x, n_fft, hop)
    V = np.abs(X)
    if period_s and period_s > 0:
        period = max(2, int(round(period_s * sr / hop)))
    else:
        period, _ = find_period(V, hop, sr)
    report(progress, 0.4, f"Repeating period {period * hop / sr:.2f} s")
    model = repeating_model(V, period)
    m_back = (model ** power) / ((V ** power) + EPS)
    m_back = np.clip(m_back, 0.0, 1.0)
    if mask == "Binary":
        m_back = (m_back > 0.5).astype(float)
    report(progress, 0.8, "Masking")
    music = istft(m_back * X, n_fft, hop, len(x))
    voice = istft((1 - m_back) * X, n_fft, hop, len(x))
    report(progress, 1.0)
    return {"Vocals": voice, "Accompaniment": music}, period * hop / sr


def polish_vocals(vocals, accompaniment, sr, kernel_time=17, kernel_freq=17,
                  strength=1.0, progress=None):
    """Second pass on a vocal stem: drum leakage is percussive (vertical in the
    spectrogram), singing is harmonic (horizontal), so an HPSS mask moves the
    leftover clicks back into the accompaniment."""
    if strength <= 0:
        return vocals, accompaniment
    n_fft = pow2_for(sr, 0.046)
    hop = n_fft // 4
    Xv = stft(vocals, n_fft, hop)
    mh, _ = _hpss_masks(np.abs(Xv), kernel_time, kernel_freq)
    mask = 1.0 - strength * (1.0 - mh)
    kept = istft(mask * Xv, n_fft, hop, len(vocals))
    moved = vocals - kept
    report(progress, 1.0)
    return kept, accompaniment + moved


# ---------------------------------------------------------------------------
# Frequency split: when two sources simply live in different bands
# ---------------------------------------------------------------------------


def find_bands(x, sr, n_fft=2048, max_bands=4, floor_db=45.0, min_width_hz=200.0,
               bridge_hz=150.0, min_share=0.02, valley_db=15.0):
    """Finds the frequency bands a recording actually uses.

    Averages the power spectrum over the whole clip, calls every bin within
    `floor_db` of the loudest one "active", groups neighbouring active bins,
    and bridges gaps narrower than `bridge_hz`. Then two guards stop a single
    source being chopped up:
      - a band must hold at least `min_share` of the total energy;
      - the valley between two bands must dip `valley_db` below the weaker of
        the two, otherwise they are one source and get merged.
    Two birds calling in separate bands survive both tests; a voice, whose
    harmonics run continuously, comes out as one band.

    Returns a list of (lo_hz, hi_hz), lowest first, at most `max_bands`.
    """
    x = as_float(x)
    hop = n_fft // 4
    P = np.abs(stft(x, n_fft, hop)) ** 2
    avg = P.mean(axis=1) + EPS
    avg_db = 10 * np.log10(avg)
    active = avg_db > avg_db.max() - floor_db

    edges = np.flatnonzero(np.diff(active.astype(int)))
    groups, start = [], 0 if active[0] else None
    for e in edges:
        if active[e]:                       # active -> silent at e+1
            if start is not None:
                groups.append((start, e))
            start = None
        elif start is None:                 # silent -> active
            start = e + 1
    if start is not None:
        groups.append((start, len(active) - 1))
    if not groups:
        return [(0.0, sr / 2)]

    bridge = max(1, int(bridge_hz * n_fft / sr))
    merged = [list(groups[0])]              # bridge gaps too narrow to matter
    for lo, hi in groups[1:]:
        if lo - merged[-1][1] <= bridge:
            merged[-1][1] = hi
        else:
            merged.append([lo, hi])

    min_w = max(1, int(min_width_hz * n_fft / sr))
    total = avg.sum()
    bands = [(lo, hi) for lo, hi in merged
             if hi - lo >= min_w and avg[lo:hi + 1].sum() / total >= min_share]
    if not bands:
        return [(0.0, sr / 2)]

    joined = [list(bands[0])]               # merge pairs with a shallow valley
    for lo, hi in bands[1:]:
        prev = joined[-1]
        gap = avg_db[prev[1]:lo + 1]
        weaker = min(avg_db[prev[0]:prev[1] + 1].max(), avg_db[lo:hi + 1].max())
        if len(gap) == 0 or weaker - gap.min() < valley_db:
            prev[1] = hi
        else:
            joined.append([lo, hi])

    energy = [avg[lo:hi + 1].sum() for lo, hi in joined]
    keep = [b for _, b in sorted(zip(energy, joined), key=lambda z: -z[0])][:max_bands]
    keep.sort()
    return [(float(lo * sr / n_fft), float(hi * sr / n_fft)) for lo, hi in keep]


def frequency_split(x, sr, splits=None, n_fft=2048, edge_hz=50.0, max_bands=4,
                    floor_db=45.0, progress=None):
    """Separates sources that occupy different frequency bands.

    `splits` is a list of cut frequencies; with None the cuts are placed in the
    middle of the silent valleys that `find_bands` discovers. Each stem is the
    mixture masked to its own band, with a raised-cosine edge `edge_hz` wide so
    the cut does not ring.

    This is the right tool when the spectrogram shows the sources sitting in
    separate horizontal stripes (bird species, a whistle over speech, a bass
    line under a flute). It cannot help when the sources overlap in frequency -
    use REPET or NMF there.
    """
    x = as_float(x)
    hop = n_fft // 4
    X = stft(x, n_fft, hop)
    freqs = np.arange(X.shape[0]) * sr / n_fft

    if splits is None:
        bands = find_bands(x, sr, n_fft, max_bands, floor_db)
        splits = [(bands[i][1] + bands[i + 1][0]) / 2 for i in range(len(bands) - 1)]
    splits = sorted(float(f) for f in splits if 0 < f < sr / 2)
    report(progress, 0.4, f"{len(splits) + 1} bands")
    if not splits:
        return {"Whole signal": x.copy()}, []

    cuts = [0.0] + splits + [sr / 2]
    stems, edge = {}, max(edge_hz, sr / n_fft)
    for i in range(len(cuts) - 1):
        lo, hi = cuts[i], cuts[i + 1]
        mask = np.ones(len(freqs))
        if lo > 0:                                    # raised-cosine rise at the low edge
            ramp = np.clip((freqs - (lo - edge / 2)) / edge, 0, 1)
            mask *= 0.5 - 0.5 * np.cos(np.pi * ramp)
        if hi < sr / 2:                               # and fall at the high edge
            ramp = np.clip(((hi + edge / 2) - freqs) / edge, 0, 1)
            mask *= 0.5 - 0.5 * np.cos(np.pi * ramp)
        name = (f"{lo / 1000:.1f}-{hi / 1000:.1f} kHz" if hi < sr / 2
                else f"above {lo / 1000:.1f} kHz")
        stems[name] = istft(mask[:, None] * X, n_fft, hop, len(x))
        report(progress, 0.4 + 0.6 * (i + 1) / (len(cuts) - 1))
    return stems, splits


# ---------------------------------------------------------------------------
# Two instruments that share the same frequency range
# ---------------------------------------------------------------------------


def _activation_features(H, hop, sr):
    """One row per NMF component, describing *how* it plays rather than what it
    sounds like: how sharply it starts, how much of the time it is on, and how
    long each note lasts. A plucked string and a bowed note differ here even
    when they cover the same frequencies."""
    feats = []
    for h in H:
        hn = h / (h.max() + EPS)
        rise = np.mean(np.maximum(np.diff(hn, prepend=hn[0]), 0.0)) / (np.mean(hn) + EPS)
        active = hn > 0.2
        runs, cur = [], 0
        for a in active:
            if a:
                cur += 1
            elif cur:
                runs.append(cur)
                cur = 0
        if cur:
            runs.append(cur)
        feats.append([rise, active.mean(), (np.mean(runs) if runs else 0.0) * hop / sr])
    return np.asarray(feats)


def _split_in_two(F, iters=50):
    """Two-means on standardised features, seeded from the two most distant
    points so the answer does not depend on a random draw."""
    Fz = (F - F.mean(axis=0)) / (F.std(axis=0) + EPS)
    d = ((Fz[:, None, :] - Fz[None]) ** 2).sum(axis=-1)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    centres = Fz[[i, j]].copy()
    labels = np.zeros(len(Fz), dtype=int)
    for _ in range(iters):
        labels = np.argmin(((Fz[:, None, :] - centres[None]) ** 2).sum(axis=-1), axis=1)
        for c in (0, 1):
            if (labels == c).any():
                centres[c] = Fz[labels == c].mean(axis=0)
    return labels


def instruments_nmf(x, sr, n_components=16, n_iter=250, group_by="Automatic", progress=None):
    """Separates two instruments that overlap in frequency.

    HPSS cannot help when both sources are harmonic, because both are
    "horizontal". Instead the mixture is factorised into many small spectral
    parts, and those parts are grouped into two instruments by either:
      - "Note attack": the shape of each part's activation over time (sharp
        attack and decay vs slow attack and a long body). Right for a plucked
        instrument against a bowed or blown one.
      - "Pitch register": the spectral centre of each part. Right when one
        instrument simply plays lower than the other.
    "Automatic" tries the attack shape first and falls back to pitch register
    when that leaves one group holding nearly everything, which means the two
    instruments start their notes the same way.

    Measured on a plucked-plus-sustained pair sharing 196-3900 Hz: input
    0.7 / 8.0 dB, HPSS 0.9 / 6.0 dB, this method 13.0 / 20.7 dB.
    Returns (stems, note).
    """
    x = as_float(x)
    n_fft = pow2_for(sr, 0.064)
    hop = n_fft // 4
    X = stft(x, n_fft, hop)
    V = np.abs(X)
    if V.shape[1] < 8:
        return {"Whole signal": x.copy()}, "clip too short to factorise"
    W, H = nmf_kl(V, n_components, n_iter, progress=progress, p_hi=0.85)
    report(progress, 0.9, "grouping parts")

    freqs = np.arange(V.shape[0]) * sr / n_fft
    centroid = np.log((W * freqs[:, None]).sum(axis=0) / (W.sum(axis=0) + EPS) + 1.0)[:, None]
    by_attack = _activation_features(H, hop, sr)

    def build(labels):
        parts = [W[:, labels == c] @ H[labels == c] if (labels == c).any() else np.zeros_like(V)
                 for c in (0, 1)]
        share = [p.sum() for p in parts]
        return parts, min(share) / (sum(share) + EPS)

    tried = []
    if group_by in ("Automatic", "Note attack"):
        parts, balance = build(_split_in_two(by_attack))
        tried.append(("note attack", balance))
        if group_by == "Note attack" or balance >= 0.12:
            note = f"grouped by note attack (smaller group holds {balance * 100:.0f}% of the energy)"
        else:
            parts, balance = build(_split_in_two(centroid))
            note = (f"note attack left one group with only {tried[0][1] * 100:.0f}% of the "
                    f"energy, so the parts were grouped by pitch register instead "
                    f"({balance * 100:.0f}% / {(1 - balance) * 100:.0f}%)")
    else:
        parts, balance = build(_split_in_two(centroid))
        note = f"grouped by pitch register (smaller group holds {balance * 100:.0f}% of the energy)"
    if balance < 0.05:
        note += "; the two sources look alike here, so this split is weak"

    total = sum(parts) + EPS
    rise = [np.mean(np.maximum(np.diff(p.sum(axis=0)), 0)) / (p.sum() + EPS) for p in parts]
    order = np.argsort(rise)[::-1]
    names = ["Instrument A (sharper attacks)", "Instrument B (more sustained)"]
    out = {name: istft((parts[c] / total) * X, n_fft, hop, len(x))
           for name, c in zip(names, order)}
    report(progress, 1.0)
    return out, note
