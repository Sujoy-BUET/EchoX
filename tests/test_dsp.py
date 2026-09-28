"""Checks for the hand-written transforms and a few DSP claims made in the README.

numpy.fft and scipy.signal appear here only as references to compare against.
Run either way:
    python tests/test_dsp.py
    python -m pytest tests
"""
import os
import sys

import numpy as np
from scipy import signal as sps

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from echox.dsp import denoise as dn               # noqa: E402
from echox.dsp import effects as fx               # noqa: E402
from echox.dsp import separation as sep           # noqa: E402
from echox.dsp import transforms as T             # noqa: E402
from echox.dsp import wavelets as W               # noqa: E402
from echox.dsp.testbench import add_noise, snr_db  # noqa: E402

rng = np.random.default_rng(0)


def test_fft_matches_numpy_all_lengths():
    for n in (1, 2, 4, 8, 64, 1024, 4096, 3, 6, 7, 100, 1000):
        x = rng.standard_normal((2, n)) + 1j * rng.standard_normal((2, n))
        assert np.abs(T.fft(x) - np.fft.fft(x)).max() < 1e-10 * max(n, 1)
        assert np.abs(T.ifft(T.fft(x)) - x).max() < 1e-12


def test_rfft_and_irfft():
    for n in (4, 16, 1024, 4096, 6, 100):
        x = rng.standard_normal((3, n))
        assert np.abs(T.rfft(x) - np.fft.rfft(x)).max() < 1e-10 * n
        assert np.abs(T.irfft(T.rfft(x), n) - x).max() < 1e-12


def test_parseval():
    x = rng.standard_normal(2048)
    X = T.fft(x)
    assert abs(np.sum(x ** 2) - np.sum(np.abs(X) ** 2) / len(x)) < 1e-9


def test_stft_matches_scipy_and_inverts():
    x = rng.standard_normal(22050)
    for n_fft in (512, 1024, 2048):
        hop = n_fft // 4
        _, _, ref = sps.stft(x, nperseg=n_fft, noverlap=n_fft - hop, window="hann",
                             boundary="zeros", padded=True)
        X = T.stft(x, n_fft, hop)
        assert X.shape == ref.shape
        assert np.abs(X - ref).max() < 1e-12
        assert np.abs(T.istft(X, n_fft, hop, len(x)) - x).max() < 1e-12


def test_dwt_perfect_reconstruction_and_orthogonality():
    for name in W.WAVELETS:
        g, hh = W.filters(name)
        assert abs(g.sum() - np.sqrt(2)) < 1e-9 and abs(np.sum(g ** 2) - 1) < 1e-9
        assert abs(hh.sum()) < 1e-9                          # high-pass: no DC
        x = rng.standard_normal(4096)                        # 2*4096 divides by 2^6
        coeffs, n = W.wavedec(x, name, 6)
        assert np.abs(W.waverec(coeffs, name, n) - x).max() < 1e-9
        energy = sum(np.sum(c ** 2) for c in coeffs)         # mirrored signal = 2x energy
        assert abs(energy / (2 * np.sum(x ** 2)) - 1) < 1e-9


def _music(sr=16000, seconds=4.0):
    t = np.arange(int(sr * seconds)) / sr
    chord = sum(np.sin(2 * np.pi * f * t) for f in (220, 277, 330))
    return 0.2 * chord * (1 + 0.1 * np.sin(2 * np.pi * 0.5 * t))


def _speech(sr=16000, seconds=4.0):
    t = np.arange(int(sr * seconds)) / sr
    f0 = 140 + 30 * np.sin(2 * np.pi * 0.5 * t)
    ph = 2 * np.pi * np.cumsum(f0) / sr
    s = sum((0.6 / k) * np.sin(k * ph) for k in range(1, 15))
    env = sps.lfilter([0.01], [1, -0.99], (np.sin(2 * np.pi * 0.7 * t) > -0.2).astype(float))
    return 0.5 * s * env


def test_wiener_auto_picks_estimator_and_helps():
    sr = 16000
    for clean, expect in ((_music(sr), "no pauses"), (_speech(sr), "quietest")):
        noisy, _ = add_noise(clean, sr, "White", 5)
        y, note = dn.wiener_stft(noisy, 1024, 256)
        assert expect in note, note
        assert snr_db(clean, y) > snr_db(clean, noisy) + 3


def test_equalizer_hits_its_targets():
    sr = 44100
    x = rng.standard_normal(sr)
    gains = [0.0, 0.0, -6.0, 0.0, 9.0, 0.0, 0.0]
    y = fx.equalizer(x, sr, gains)
    f = T.rfftfreq(4096, 1 / sr)

    def band_db(sig, f0):
        segs = sig[:len(sig) // 4096 * 4096].reshape(-1, 4096)
        P = (np.abs(T.rfft(segs)) ** 2).mean(axis=0)
        k = (f > f0 / 1.1) & (f < f0 * 1.1)
        return 10 * np.log10(P[k].mean())

    for f0, g in ((400.0, -6.0), (2500.0, 9.0)):
        assert abs(band_db(y, f0) - band_db(x, f0) - g) < 1.5


def test_speed_modes():
    sr = 22050
    t = np.arange(sr * 2) / sr
    tone = np.sin(2 * np.pi * 440 * t)

    def peak_hz(s):
        n = 8192
        mid = len(s) // 2
        seg = s[mid - n // 2: mid + n // 2] * np.hanning(n)
        return T.rfftfreq(n, 1 / sr)[np.argmax(np.abs(T.rfft(seg)))]

    for rate in (0.75, 1.5):
        moved = fx.resample_speed(tone, rate)
        kept = fx.phase_vocoder(tone, rate)
        assert abs(len(moved) - len(tone) / rate) < sr * 0.05
        assert abs(len(kept) - len(tone) / rate) < sr * 0.05
        assert abs(peak_hz(moved) - 440 * rate) < 10          # pitch follows speed
        assert abs(peak_hz(kept) - 440) < 10                  # pitch unchanged


def test_doppler_matches_theory():
    sr = 22050
    tone = np.sin(2 * np.pi * 500 * np.arange(sr * 6) / sr)
    y = fx.doppler(tone, sr, speed_ms=30, distance_m=10, mode="Pass-by")
    c = fx.SPEED_OF_SOUND

    def peak_hz(s, at_s, n=8192):
        i = int(at_s * sr)
        seg = s[i - n // 2:i + n // 2] * np.hanning(n)
        return T.rfftfreq(n, 1 / sr)[np.argmax(np.abs(T.rfft(seg)))]

    assert abs(peak_hz(y, 1.0) - 500 * c / (c - 30)) < 6      # approaching
    assert abs(peak_hz(y, 5.0) - 500 * c / (c + 30)) < 6      # receding


def test_phase_vocoder_quality_sharpens_transients():
    """The default 'high' quality mode should keep a sharp attack closer to
    sharp than the plain per-bin accumulation ('fast') does."""
    sr = 44100
    n = int(0.4 * sr)
    attack = int(0.002 * sr)
    env = np.ones(n)
    env[:attack] = np.linspace(0, 1, attack)
    env[-attack:] *= np.linspace(1, 0, attack)
    tone = np.sin(2 * np.pi * 440 * np.arange(n) / sr) * env
    x = np.zeros(sr)
    x[int(0.3 * sr):int(0.3 * sr) + n] = tone

    def rise_ms(sig, centre_s, halfwin_ms=60):
        i = int(centre_s * sr)
        w = int(halfwin_ms / 1000 * sr)
        seg = np.abs(sig[max(0, i - w):i + w])
        peak = seg.max()
        a10 = np.argmax(seg > 0.1 * peak)
        a90 = np.argmax(seg > 0.9 * peak)
        return (a90 - a10) / sr * 1000

    rate = 1.5
    fast = fx.phase_vocoder(x, rate, quality="fast")
    high = fx.phase_vocoder(x, rate, quality="high")
    assert rise_ms(high, 0.3 / rate) < rise_ms(fast, 0.3 / rate)

    t = np.arange(sr * 2) / sr
    pure = np.sin(2 * np.pi * 440 * t)

    def peak_hz(s):
        n_ = 8192
        mid = len(s) // 2
        seg = s[mid - n_ // 2: mid + n_ // 2] * np.hanning(n_)
        return T.rfftfreq(n_, 1 / sr)[np.argmax(np.abs(T.rfft(seg)))]

    assert abs(peak_hz(fx.phase_vocoder(pure, rate, quality="high")) - 440) < 10


def test_tiny_and_degenerate_inputs_stay_finite():
    """Filters whose kernel is wider than the array used to return NaN, which
    propagated silently into the output. Every algorithm must survive inputs
    far shorter than its own analysis window."""
    for n in (1, 2, 10, 500):
        x = np.linspace(-0.5, 0.5, n)
        outs = []
        outs += list(sep.hpss(x, 44100, three_stems=False).values())
        outs += list(sep.hpss(x, 44100, three_stems=True).values())
        outs += list(sep.repet(x, 44100)[0].values())
        outs.append(dn.declick(x, 44100)[0])
        outs.append(dn.wiener_stft(x, 1024, 256, sr=44100)[0])
        for y in outs:
            assert len(y) > 0 and np.all(np.isfinite(y)), f"non-finite output for n={n}"

    silence = np.zeros(4410)
    assert np.all(np.isfinite(dn.wiener_stft(silence, 1024, 256, sr=44100)[0]))


def test_frequency_split_separates_bands_but_not_a_voice():
    """Two tones in separate bands must split cleanly; one broadband source
    must not be chopped up."""
    sr = 22050
    t = np.arange(sr * 2) / sr
    low = np.sin(2 * np.pi * 1400 * t) * (np.sin(2 * np.pi * 3 * t) > 0)
    high = np.sin(2 * np.pi * 5000 * t) * (np.sin(2 * np.pi * 2.3 * t) > 0)
    mix = 0.5 * low + 0.5 * high

    bands = sep.find_bands(mix, sr)
    assert len(bands) == 2
    stems, splits = sep.frequency_split(mix, sr)
    assert len(stems) == 2 and 1400 < splits[0] < 5000

    def si_sdr(ref, est):
        a = np.sum(ref * est) / (np.sum(est ** 2) + 1e-12)
        return 10 * np.log10(np.sum(ref ** 2) / (np.sum((ref - a * est) ** 2) + 1e-12))

    got = list(stems.values())
    assert max(si_sdr(0.5 * low, g) for g in got) > 20
    assert max(si_sdr(0.5 * high, g) for g in got) > 20

    # a single harmonic-rich source has no valley to cut at
    f0 = 150 + 20 * np.sin(2 * np.pi * 0.5 * t)
    ph = 2 * np.pi * np.cumsum(f0) / sr
    voice = sum((0.6 / k) * np.sin(k * ph) for k in range(1, 25))
    assert len(sep.find_bands(voice, sr)) == 1


def test_filters_shape_the_spectrum():
    sr = 44100
    x = rng.standard_normal(sr)
    f = T.rfftfreq(4096, 1 / sr)

    def band_db(sig, lo, hi):
        segs = sig[:len(sig) // 4096 * 4096].reshape(-1, 4096)
        P = (np.abs(T.rfft(segs)) ** 2).mean(axis=0)
        k = (f > lo) & (f < hi)
        return 10 * np.log10(P[k].mean())

    low = fx.apply_filter(x, sr, "Low-pass", 1000.0, order=4)
    assert band_db(low, 200, 400) - band_db(x, 200, 400) > -1        # passband kept
    assert band_db(low, 4000, 6000) - band_db(x, 4000, 6000) < -30   # stopband gone

    high = fx.apply_filter(x, sr, "High-pass", 1000.0, order=4)
    assert band_db(high, 200, 400) - band_db(x, 200, 400) < -20
    assert band_db(high, 4000, 6000) - band_db(x, 4000, 6000) > -1

    bp = fx.apply_filter(x, sr, "Band-pass", 500.0, 2000.0, order=4)
    assert band_db(bp, 800, 1200) - band_db(x, 800, 1200) > -3
    assert band_db(bp, 4000, 6000) - band_db(x, 4000, 6000) < -20

    bs = fx.apply_filter(x, sr, "Band-stop", 500.0, 2000.0, order=4)
    assert band_db(bs, 800, 1200) - band_db(x, 800, 1200) < -20


def test_echo_repeats_and_reverb_rings():
    sr = 22050
    imp = np.zeros(sr)
    imp[0] = 1.0
    # with damping off, each repeat is exactly `feedback` times the last
    e = fx.echo(imp, sr, delay_ms=200, feedback=0.5, mix=1.0, repeats_damping=0.0)
    heights = [abs(e[int(k * 0.2 * sr)]) for k in range(1, 4)]
    for a, b in zip(heights, heights[1:]):
        assert b < a
    assert abs(heights[1] / heights[0] - 0.5) < 0.05

    # damping makes the repeats fade faster still
    damped = fx.echo(imp, sr, delay_ms=200, feedback=0.5, mix=1.0, repeats_damping=0.2)
    assert abs(damped[int(0.4 * sr)]) < abs(e[int(0.4 * sr)])

    r = fx.reverb(imp, sr, room=0.8, mix=1.0)
    loud = np.flatnonzero(np.abs(r) > np.abs(r).max() * 0.001)
    assert 0.3 < loud[-1] / sr < 5.0                   # a tail of sensible length
    assert np.all(np.isfinite(r))


def test_instrument_clustering_beats_hpss_on_overlapping_instruments():
    """Two harmonic instruments in the same range: HPSS cannot split them
    because both are 'horizontal', but grouping NMF parts by how their notes
    start can. The clip needs enough notes for the grouping to see a pattern."""
    seed = np.random.default_rng(5)
    sr = 22050
    n = int(sr * 6)

    def pluck(f0, at, dur=0.9):
        i = int(at * sr)
        L = min(int(dur * sr), n - i)
        tt = np.arange(L) / sr
        env = np.exp(-tt / 0.35) * np.minimum(1, tt / 0.004)
        tone = sum((1 / k ** 1.3) * np.sin(2 * np.pi * f0 * k * tt + seed.random())
                   for k in range(1, 10))
        out = np.zeros(n)
        out[i:i + L] = tone * env
        return out

    def sustain(f0, at, dur=1.2):
        i = int(at * sr)
        L = min(int(dur * sr), n - i)
        tt = np.arange(L) / sr
        env = np.minimum(1, tt / 0.12) * np.minimum(1, (dur - tt) / 0.15)
        vib = 1 + 0.004 * np.sin(2 * np.pi * 5 * tt)
        tone = sum((1 / k) * np.sin(2 * np.pi * f0 * k * tt * vib) for k in range(1, 8))
        out = np.zeros(n)
        out[i:i + L] = tone * env
        return out

    plucked = sum(pluck(f, at) for at, f in
                  zip(np.arange(0.1, 5.0, 0.5),
                      (196, 220, 247, 196, 262, 220, 247, 196, 220, 262, 196)))
    sustained = sum(sustain(f, at) for at, f in
                    zip(np.arange(0.3, 4.7, 1.3), (392, 440, 349, 392, 466)))
    plucked /= np.abs(plucked).max()
    sustained /= np.abs(sustained).max()
    a, b = 0.5 * plucked, 0.5 * sustained
    mix = a + b

    def si_sdr(c, e):
        k = min(len(c), len(e))
        c, e = c[:k], e[:k]
        g = np.sum(c * e) / (np.sum(e * e) + 1e-12)
        return 10 * np.log10(np.sum(c ** 2) / (np.sum((c - g * e) ** 2) + 1e-12))

    stems, note = sep.instruments_nmf(mix, sr, n_components=16, n_iter=200)
    assert len(stems) == 2 and note
    got = list(stems.values())
    hp = list(sep.hpss(mix, sr, three_stems=False).values())
    assert max(si_sdr(a, g) for g in got) > max(si_sdr(a, h) for h in hp) + 5
    assert max(si_sdr(b, g) for g in got) > max(si_sdr(b, h) for h in hp) + 5


def test_trim_crop_concatenate():
    sr = 16000
    x = np.arange(sr * 4, dtype=float)

    y = fx.trim(x, (sr, 2 * sr), sr=sr, fade_ms=0)
    assert np.array_equal(y, np.concatenate([x[:sr], x[2 * sr:]]))

    y = fx.crop(x, (sr, 2 * sr), sr=sr, fade_ms=0)
    assert np.array_equal(y, x[sr:2 * sr])

    a, b = np.ones(sr), np.full(sr, 2.0)
    y = fx.concatenate(a, b, sr=sr, crossfade_ms=0)
    assert np.array_equal(y, np.concatenate([a, b]))
    assert len(fx.concatenate(a, b, sr=sr, crossfade_ms=20)) == 2 * sr - int(20 / 1000 * sr)
    assert np.array_equal(fx.concatenate(a, b, sr=sr, before=True), np.concatenate([b, a]))


def test_overlay_mixes_adds_and_loops():
    sr = 16000
    x = np.zeros(sr * 4)
    other = 0.5 * np.sin(2 * np.pi * 440 * np.arange(int(1.5 * sr)) / sr)

    y = fx.overlay(x, other, gain_db=-6.0, start_s=1.0, sr=sr)
    assert np.array_equal(y[:sr], x[:sr])                       # untouched before the start
    seg = other * 10 ** (-6 / 20)
    end = sr + len(seg)
    assert np.allclose(y[sr:end], x[sr:end] + seg, atol=1e-9)
    assert np.array_equal(y[end:], x[end:])                     # silent again once it runs out

    looped = fx.overlay(x, other, gain_db=0.0, start_s=0.0, sr=sr, loop=True)
    assert not np.allclose(looped[-100:], x[-100:])             # the loop reaches the very end

    ducked = fx.overlay(x + 1.0, other, gain_db=-100.0, start_s=0.0, sr=sr, duck_db=-12.0)
    assert ducked[sr // 4] < 1.0                                # x itself was turned down


def test_repet_finds_the_loop_and_splits_it():
    sr = 16000
    t = np.arange(sr * 12) / sr
    loop = sum(np.sin(2 * np.pi * f * t) for f in (220, 277, 330))
    loop *= (np.sin(2 * np.pi * t / 2.0) > 0)                 # repeats every 2 s
    voice = np.sin(2 * np.pi * (600 + 80 * np.sin(2 * np.pi * 0.3 * t)) * t) * (t > 4)
    mix = 0.3 * loop + 0.3 * voice
    stems, period = sep.repet(mix, sr)
    assert abs(period - 2.0) < 0.2
    def snr(c, e):
        return 10 * np.log10(np.sum(c ** 2) / np.sum((c - e) ** 2))
    assert snr(0.3 * voice, stems["Vocals"]) > snr(0.3 * voice, mix) + 3


def test_dereverb_keeps_the_direct_sound_and_shortens_the_tail():
    """WPE should leave a dry signal alone and shrink a reverberant tail."""
    sr = 16000
    t = np.arange(sr * 6) / sr
    f0 = 150 + 20 * np.sin(2 * np.pi * 0.4 * t)
    ph = 2 * np.pi * np.cumsum(f0) / sr
    dry = sum((0.5 / k) * np.sin(k * ph) for k in range(1, 12))
    dry *= (np.sin(2 * np.pi * 0.6 * t) > 0.1)                # pauses expose the tail

    kept, _ = fx.wpe(dry, sr, 1024, 12, 2, 2)
    a = np.sum(dry * kept) / np.sum(kept ** 2)
    assert 10 * np.log10(np.sum(dry ** 2) / np.sum((dry - a * kept) ** 2)) > 15

    n = int(0.4 * sr)
    tail = rng.standard_normal(n) * np.exp(-3 * np.log(10) * np.arange(n) / (0.4 * sr))
    tail[:int(0.05 * sr)] = 0
    tail *= np.sqrt(0.5 / np.sum(tail ** 2))                  # DRR about 3 dB
    ir = tail.copy()
    ir[0] += 1.0
    wet = np.convolve(dry, ir)[:len(dry)]

    def si_sdr(c, e):
        a = np.sum(c * e) / (np.sum(e * e) + 1e-12)
        return 10 * np.log10(np.sum(c ** 2) / np.sum((c - a * e) ** 2))

    y, _ = fx.wpe(wet, sr, 4096, 20, 2, 3)
    assert len(y) == len(wet)
    assert si_sdr(dry, y) > si_sdr(dry, wet)


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} passed")
