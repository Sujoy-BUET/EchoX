"""Every processing feature the UI offers, described as data.

The sidebar builds its controls from `Param` lists, so adding an algorithm
means adding one `Algorithm` entry here; no UI code changes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .dsp import amplify as amp
from .dsp import denoise as dn
from .dsp import effects as fx
from .dsp import separation as sep


class UserError(Exception):
    """A problem the user can fix (missing selection, missing reference...)."""


@dataclass
class Param:
    key: str
    label: str
    kind: str                      # "float" | "int" | "choice" | "bool"
    default: Any
    lo: float = 0.0
    hi: float = 1.0
    log: bool = False
    unit: str = ""
    choices: tuple = ()
    tip: str = ""
    show_if: tuple | None = None   # (other key, tuple of values that show this one)


@dataclass
class Context:
    sr: int
    selection: tuple[int, int] | None = None
    testbench_ref: np.ndarray | None = None
    file_ref: np.ndarray | None = None


@dataclass
class Result:
    output: np.ndarray | None = None
    stems: dict[str, np.ndarray] | None = None
    message: str = ""


@dataclass
class Algorithm:
    key: str
    name: str
    summary: str
    params: list[Param]
    run: Callable[..., Result]
    kind: str = "output"           # "output" replaces the signal, "stems" adds stems
    uses_selection: str = ""       # hint shown under the form
    family: str = "Signal processing"   # badge: where the method comes from


def _need_selection(ctx: Context, why: str):
    if ctx.selection is None or ctx.selection[1] - ctx.selection[0] < ctx.sr * 0.2:
        raise UserError(f"Drag across the waveform to select {why} (at least 0.2 s).")
    return ctx.selection


# --------------------------------------------------------------------- denoise


def run_wiener(x, p, ctx, progress):
    n_fft = int(p["n_fft"])
    region, method, noise_sig = None, "auto", None
    if p["noise_from"] == "Noise file":
        noise_sig = ctx.file_ref
        if noise_sig is None:
            raise UserError("Load a noise-only recording first (Load noise file, in the "
                            "panel below), or pick another noise profile.")
    elif p["noise_from"] == "Selection":
        region = _need_selection(ctx, "a noise-only part")
    elif p["noise_from"] == "First 0.5 s":
        region = (0, int(0.5 * ctx.sr))
    elif p["noise_from"] == "Tracking (changing noise)":
        method = "track"
    elif p["noise_from"] == "Quietest frames":
        method = "quiet"
    elif p["noise_from"] == "Smooth floor":
        method = "floor"
    y, note = dn.wiener_stft(x, n_fft, n_fft // 4, region, p["alpha"], p["oversub"],
                             p["floor_db"], method, ctx.sr, p["window_s"], noise_sig,
                             p["second_pass"], progress)
    return Result(y, message=f"Wiener: {note}")


def run_notch(x, p, ctx, progress):
    f0, found = p["f0"], ""
    if p["find"]:
        detected, ratio = dn.detect_tone(x, ctx.sr)
        if detected is None:
            raise UserError("No steady tone stands out from the rest of the spectrum. "
                            "Turn 'Find the tone' off and set the frequency by hand.")
        f0 = detected
        found = f"found a tone at {f0:.1f} Hz, {ratio:.0f} dB above its neighbours; "
    y, tracked = dn.adaptive_notch(x, ctx.sr, f0, p["harmonics"], p["bandwidth"],
                                   p["mu"], p["search"], progress)
    msg = found + "notch finished"
    if tracked is not None:
        msg = found + f"notch locked at {np.median(tracked[len(tracked) // 2:]):.2f} Hz"
    return Result(y, message=msg)


def run_wavelet(x, p, ctx, progress):
    rule = "bayes" if p["rule"] == "BayesShrink" else "visu"
    y, sigma = dn.wavelet_denoise(x, p["wavelet"], p["level"], rule, p["mode"], progress)
    return Result(y, message=f"Wavelet finished, noise sigma {sigma:.4f}")


def run_declick(x, p, ctx, progress):
    y, frac = dn.declick(x, ctx.sr, int(p["n_fft"]), p["strength"], p["rise_db"],
                         p["hi_hz"], progress=progress)
    if frac == 0:
        return Result(y, message="No broadband clicks found; nothing was changed")
    return Result(y, message=f"Clicks removed from {frac * 100:.1f}% of the frames")


def run_dereverb(x, p, ctx, progress):
    if p["method"] == "WPE (linear prediction)":
        y, _ = fx.wpe(x, ctx.sr, int(p["n_fft"]), p["taps"], p["delay"], p["passes"], progress)
        span = p["taps"] * int(p["n_fft"]) / 4 / ctx.sr
        return Result(y, message=f"WPE removed what the last {span * 1000:.0f} ms could predict")
    y, decay = fx.dereverb(x, ctx.sr, p["rt60"], p["early_ms"], p["oversub"],
                           p["floor_db"], progress=progress)
    return Result(y, message=f"Late field subtracted (decay {10 * np.log10(decay):.1f} dB "
                             f"per {p['early_ms']:.0f} ms)")


DEREVERB_METHODS = ("WPE (linear prediction)", "Spectral subtraction")

DENOISE = [
    Algorithm(
        "wiener", "Wiener", "Per-bin gain from estimated speech and noise power in the STFT.",
        [
            Param("noise_from", "Noise profile", "choice", "Auto",
                  choices=("Auto", "Tracking (changing noise)", "Selection", "Noise file",
                           "First 0.5 s", "Quietest frames", "Smooth floor"),
                  tip="Auto tracks the floor when it moves during the clip, uses the pauses "
                      "when it is steady, and falls back to a smooth broadband floor for "
                      "music with no pauses at all."),
            Param("window_s", "Tracking window", "float", 1.5, 0.5, 5.0, unit="s",
                  tip="Running minimum over this much audio. Shorter follows faster but "
                      "starts eating speech; 1.5 s measured best",
                  show_if=("noise_from", ("Auto", "Tracking (changing noise)"))),
            Param("n_fft", "Frame size", "choice", "1024", choices=("512", "1024", "2048", "4096"),
                  unit="samples"),
            Param("alpha", "Smoothing", "float", 0.98, 0.80, 0.999,
                  tip="Decision-directed weight; higher means less musical noise"),
            Param("oversub", "Noise over-estimate", "float", 1.0, 0.5, 4.0, unit="x"),
            Param("floor_db", "Gain floor", "float", -20.0, -40.0, 0.0, unit="dB",
                  tip="Never attenuate a bin more than this"),
            Param("second_pass", "Second pass", "bool", False,
                  tip="Clean once, then re-measure the noise from the residual and redo the "
                      "gains with no decision-directed lag. Measured +0.2 to +0.7 dB on "
                      "steady noise, slightly worse on brown noise"),
        ],
        run_wiener, uses_selection="Noise profile 'Selection' uses the highlighted region.",
    ),
    Algorithm(
        "notch", "Adaptive notch", "Narrow IIR notches that lock onto hum and its harmonics.",
        [
            Param("find", "Find the tone", "bool", False,
                  tip="Picks the strongest narrow peak in the spectrum, so a coil whine or a "
                      "whistle can be removed without knowing its frequency"),
            Param("f0", "Start frequency", "float", 50.0, 20.0, 5000.0, log=True, unit="Hz",
                  show_if=("find", (False,))),
            Param("harmonics", "Harmonics", "int", 4, 1, 10),
            Param("bandwidth", "Notch width", "float", 4.0, 0.5, 50.0, log=True, unit="Hz"),
            Param("mu", "Adaptation rate", "float", 0.001, 1e-5, 1e-2, log=True),
            Param("search", "Tracking range", "float", 10.0, 1.0, 30.0, unit="%"),
        ],
        run_notch,
    ),
    Algorithm(
        "wavelet", "Wavelet", "Shrinks small DWT detail coefficients, where white noise lives.",
        [
            Param("wavelet", "Wavelet", "choice", "db8", choices=("db4", "db8", "sym8", "coif5", "haar")),
            Param("level", "Levels", "int", 6, 1, 12),
            Param("rule", "Threshold rule", "choice", "BayesShrink", choices=("BayesShrink", "VisuShrink")),
            Param("mode", "Shrinkage", "choice", "soft", choices=("soft", "hard")),
        ],
        run_wavelet,
    ),
    Algorithm(
        "declick", "Click remover", "Removes short broadband bangs (hammering, taps, door "
        "slams) and leaves the voice alone.",
        [
            Param("strength", "Strength", "float", 1.0, 0.2, 1.0),
            Param("rise_db", "Detection threshold", "float", 10.0, 4.0, 20.0, unit="dB",
                  tip="How far the top-octave energy must jump above its local median "
                      "before a frame counts as a click. Lower catches more, and risks "
                      "catching consonants"),
            Param("hi_hz", "Listen above", "float", 8000.0, 3000.0, 16000.0, log=True, unit="Hz",
                  tip="Clicks reach the very top of the spectrum; speech does not"),
            Param("n_fft", "Frame size", "choice", "1024", choices=("512", "1024", "2048")),
        ],
        run_declick,
    ),
    Algorithm(
        "dereverb", "Dereverb", "Removes room echo and reverb tails, keeping the direct sound.",
        [
            Param("method", "Method", "choice", DEREVERB_METHODS[0], choices=DEREVERB_METHODS,
                  tip="WPE predicts the tail from past frames and subtracts it (linear, no "
                      "musical noise). Spectral subtraction is faster but rougher."),
            Param("n_fft", "Frame size", "choice", "4096", choices=("1024", "2048", "4096"),
                  show_if=("method", (DEREVERB_METHODS[0],))),
            Param("taps", "Prediction taps", "int", 20, 4, 60,
                  tip="Each tap covers one hop (frame/4). 20 taps at 4096 reach about 460 ms",
                  show_if=("method", (DEREVERB_METHODS[0],))),
            Param("delay", "Skip frames", "int", 2, 1, 8,
                  tip="Frames left untouched so the direct sound and early echoes survive",
                  show_if=("method", (DEREVERB_METHODS[0],))),
            Param("passes", "Passes", "int", 3, 1, 6, show_if=("method", (DEREVERB_METHODS[0],))),
            Param("rt60", "RT60", "float", 0.6, 0.1, 3.0, unit="s",
                  tip="How long the room takes to decay 60 dB",
                  show_if=("method", (DEREVERB_METHODS[1],))),
            Param("early_ms", "Early window", "float", 50.0, 20.0, 150.0, unit="ms",
                  show_if=("method", (DEREVERB_METHODS[1],))),
            Param("oversub", "Over-subtract", "float", 1.0, 0.5, 4.0, unit="x",
                  show_if=("method", (DEREVERB_METHODS[1],))),
            Param("floor_db", "Gain floor", "float", -12.0, -30.0, -3.0, unit="dB",
                  show_if=("method", (DEREVERB_METHODS[1],))),
        ],
        run_dereverb,
    ),
]

# -------------------------------------------------------------------- separate


def _polish(stems, p, ctx):
    """Optional HPSS clean-up pass on the vocal stem (see sep.polish_vocals)."""
    if not p.get("polish") or "Vocals" not in stems:
        return stems
    other = "Accompaniment" if "Accompaniment" in stems else "Instruments"
    if other not in stems:
        return stems
    v, a = sep.polish_vocals(stems["Vocals"], stems[other], ctx.sr,
                             strength=p.get("polish_strength", 1.0))
    out = dict(stems)
    out["Vocals"], out[other] = v, a
    return out


def run_hpss(x, p, ctx, progress):
    if p["mode"] == INSTRUMENT_MODES[1]:
        stems, note = sep.instruments_nmf(x, ctx.sr, p["components"], p["iterations"],
                                          p["group_by"], progress)
        return Result(stems=stems, message=f"Two instruments: {note}")
    three = p["stems"].startswith("3")
    stems = sep.hpss(x, ctx.sr, p["k_time"] | 1, p["k_freq"] | 1, p["power"], three, progress)
    stems = _polish(stems, p, ctx) if three else stems
    return Result(stems=stems, message="Split into " + ", ".join(stems))


def run_repet(x, p, ctx, progress):
    stems, period = sep.repet(x, ctx.sr, p["period"], p["mask"], p["power"], progress)
    stems = _polish(stems, p, ctx)
    return Result(stems=stems, message=f"REPET: background repeats every {period:.2f} s")


def run_nmf(x, p, ctx, progress):
    region, role = None, "accompaniment"
    if p["mode"] == "Trained on selection":
        region = _need_selection(ctx, "a part with only one source")
        role = "accompaniment" if p["selection_has"] == "Instruments only" else "vocals"
    stems = sep.nmf_separate(x, ctx.sr, p["components"], p["iterations"], region, role,
                             progress=progress)
    stems = _polish(stems, p, ctx)
    return Result(stems=stems, message=f"NMF produced {len(stems)} stems")


def run_frequency_split(x, p, ctx, progress):
    splits = None
    if p["mode"] == "Manual":
        splits = [f for f in (p["split1"], p["split2"], p["split3"]) if f > 0]
        if not splits:
            raise UserError("Set at least one split frequency, or switch the mode to Automatic.")
    stems, used = sep.frequency_split(x, ctx.sr, splits, max_bands=p["max_bands"],
                                      floor_db=p["floor_db"], edge_hz=p["edge_hz"],
                                      progress=progress)
    if len(stems) < 2:
        raise UserError("The sources here share the same frequency range, so a frequency split "
                        "cannot separate them. Try REPET or NMF instead.")
    where = ", ".join(f"{f / 1000:.2f} kHz" for f in used)
    return Result(stems=stems, message=f"Split into {len(stems)} bands at {where}")


INSTRUMENT_MODES = ("Sustained vs percussive (HPSS)", "Two overlapping instruments (NMF)")

SEPARATE = [
    Algorithm(
        "repet", "REPET", "The accompaniment repeats, the voice does not: median over repetitions "
        "gives the background.",
        [
            Param("period", "Repeat period", "float", 0.0, 0.0, 12.0, unit="s",
                  tip="0 = find it automatically from the beat spectrum"),
            Param("power", "Mask sharpness", "float", 2.0, 1.0, 4.0),
            Param("mask", "Mask", "choice", "Soft", choices=("Soft", "Binary")),
            Param("polish", "Clean up vocals", "bool", True,
                  tip="Second HPSS pass that pushes leftover drum clicks back into the "
                      "accompaniment; worth about +0.7 dB on the demo song"),
            Param("polish_strength", "Clean-up strength", "float", 1.0, 0.2, 1.5,
                  show_if=("polish", (True,))),
        ],
        run_repet, kind="stems",
    ),
    Algorithm(
        "hpss", "Instruments",
        "Two ways to split instruments: sustained vs percussive (HPSS), or two instruments "
        "that share the same frequency range, grouped by how their notes start.",
        [
            Param("mode", "Method", "choice", INSTRUMENT_MODES[0], choices=INSTRUMENT_MODES,
                  tip="HPSS separates horizontal (sustained) from vertical (hits). Use the "
                      "second one when both instruments are harmonic and overlap in pitch, "
                      "where HPSS cannot help."),
            Param("stems", "Output", "choice", "2 stems: instruments, drums",
                  choices=("2 stems: instruments, drums",
                           "3 stems: vocals, instruments, percussion"),
                  show_if=("mode", (INSTRUMENT_MODES[0],))),
            Param("k_time", "Time kernel", "int", 31, 5, 101, unit="frames",
                  show_if=("mode", (INSTRUMENT_MODES[0],))),
            Param("k_freq", "Frequency kernel", "int", 31, 5, 101, unit="bins",
                  show_if=("mode", (INSTRUMENT_MODES[0],))),
            Param("power", "Mask sharpness", "float", 2.0, 1.0, 4.0,
                  show_if=("mode", (INSTRUMENT_MODES[0],))),
            Param("polish", "Clean up vocals", "bool", True,
                  show_if=("stems", ("3 stems: vocals, instruments, percussion",)),
                  tip="Second HPSS pass that pushes leftover drum clicks back into the "
                      "instruments stem"),
            Param("polish_strength", "Clean-up strength", "float", 1.0, 0.2, 1.5,
                  show_if=("polish", (True,))),
            Param("components", "Parts to find", "int", 16, 4, 40,
                  tip="More parts describe the mixture in finer detail before they are "
                      "grouped into two instruments",
                  show_if=("mode", (INSTRUMENT_MODES[1],))),
            Param("iterations", "Iterations", "int", 250, 50, 600, log=True,
                  show_if=("mode", (INSTRUMENT_MODES[1],))),
            Param("group_by", "Group the parts by", "choice", "Automatic",
                  choices=("Automatic", "Note attack", "Pitch register"),
                  tip="Note attack suits a plucked instrument against a bowed one; pitch "
                      "register suits one instrument playing lower than the other",
                  show_if=("mode", (INSTRUMENT_MODES[1],))),
        ],
        run_hpss, kind="stems",
    ),
    Algorithm(
        "nmf", "NMF", "Factorises the spectrogram into spectral shapes W and their activations H.",
        [
            Param("mode", "Mode", "choice", "Trained on selection",
                  choices=("Trained on selection", "Unsupervised")),
            Param("selection_has", "Selection contains", "choice", "Instruments only",
                  choices=("Instruments only", "Vocals only"),
                  show_if=("mode", ("Trained on selection",))),
            Param("components", "Components", "int", 6, 2, 16),
            Param("iterations", "Iterations", "int", 200, 50, 600, log=True),
            Param("polish", "Clean up vocals", "bool", True,
                  tip="Second HPSS pass that pushes leftover drum clicks back into the "
                      "accompaniment; worth about +0.7 dB on the demo song"),
            Param("polish_strength", "Clean-up strength", "float", 1.0, 0.2, 1.5,
                  show_if=("polish", (True,))),
        ],
        run_nmf, kind="stems", family="Machine learning",
        uses_selection="Trained mode learns one source from the highlighted region, "
                       "for example an instrumental intro.",
    ),
    Algorithm(
        "fsplit", "Frequency split",
        "For sources that sit in different frequency bands - two bird species, a whistle over "
        "speech. Finds the silent valleys between them and cuts there.",
        [
            Param("mode", "Splits", "choice", "Automatic", choices=("Automatic", "Manual")),
            Param("max_bands", "Most bands", "int", 4, 2, 6,
                  show_if=("mode", ("Automatic",))),
            Param("floor_db", "Silence threshold", "float", 45.0, 20.0, 80.0, unit="dB",
                  tip="How far below the loudest bin counts as empty spectrum",
                  show_if=("mode", ("Automatic",))),
            Param("split1", "Split 1", "float", 2000.0, 50.0, 20000.0, log=True, unit="Hz",
                  show_if=("mode", ("Manual",))),
            Param("split2", "Split 2", "float", 0.0, 0.0, 20000.0, unit="Hz",
                  tip="0 = unused", show_if=("mode", ("Manual",))),
            Param("split3", "Split 3", "float", 0.0, 0.0, 20000.0, unit="Hz",
                  tip="0 = unused", show_if=("mode", ("Manual",))),
            Param("edge_hz", "Edge softness", "float", 50.0, 5.0, 500.0, log=True, unit="Hz",
                  tip="Raised-cosine width at each cut, so the split does not ring"),
        ],
        run_frequency_split, kind="stems",
    ),
]

# --------------------------------------------------------------------- amplify

AMP_MODES = ("Gain", "Normalize peak", "Normalize RMS")


def run_amplify(x, p, ctx, progress):
    mode = {"Gain": "gain", "Normalize peak": "peak", "Normalize RMS": "rms"}[p["mode"]]
    amount = {"gain": p["gain_db"], "peak": p["peak_db"], "rms": p["rms_db"]}[mode]
    comp = dict(threshold_db=p["threshold"], ratio=p["ratio"], attack_ms=p["attack"],
                release_ms=p["release"], makeup_db=p["makeup"])

    def process(seg):
        return amp.amplify(seg, ctx.sr, mode, amount, p["compressor"], comp,
                           p["limiter"], p["ceiling"])

    if not p["selection_only"]:
        return Result(process(x), message="Amplified whole signal")
    i0, i1 = _need_selection(ctx, "the part to amplify")
    y = x.copy()
    seg = process(x[i0:i1])
    fade = min(int(0.01 * ctx.sr), (i1 - i0) // 4)
    ramp = np.ones(i1 - i0)
    if fade > 0:
        ramp[:fade] = np.linspace(0, 1, fade)
        ramp[-fade:] = np.linspace(1, 0, fade)
    y[i0:i1] = ramp * seg + (1 - ramp) * x[i0:i1]
    return Result(y, message="Amplified selection")


AMPLIFY = [
    Algorithm(
        "loudness", "Loudness", "Gain or normalisation, with optional compression and a soft limiter.",
        [
            Param("mode", "Method", "choice", "Gain", choices=AMP_MODES),
            Param("gain_db", "Gain", "float", 6.0, -24.0, 36.0, unit="dB", show_if=("mode", ("Gain",))),
            Param("peak_db", "Peak target", "float", -1.0, -12.0, 0.0, unit="dBFS",
                  show_if=("mode", ("Normalize peak",))),
            Param("rms_db", "RMS target", "float", -18.0, -40.0, -6.0, unit="dBFS",
                  show_if=("mode", ("Normalize RMS",))),
            Param("compressor", "Compressor", "bool", False,
                  tip="Lifts quiet parts relative to loud ones before the gain stage"),
            Param("threshold", "Threshold", "float", -24.0, -60.0, 0.0, unit="dB",
                  show_if=("compressor", (True,))),
            Param("ratio", "Ratio", "float", 4.0, 1.0, 20.0, unit=":1", show_if=("compressor", (True,))),
            Param("attack", "Attack", "float", 5.0, 0.1, 100.0, log=True, unit="ms",
                  show_if=("compressor", (True,))),
            Param("release", "Release", "float", 120.0, 10.0, 1000.0, log=True, unit="ms",
                  show_if=("compressor", (True,))),
            Param("makeup", "Make-up gain", "float", 0.0, 0.0, 24.0, unit="dB",
                  show_if=("compressor", (True,))),
            Param("limiter", "Soft limiter", "bool", True, tip="Rounds peaks off below the ceiling"),
            Param("ceiling", "Ceiling", "float", -1.0, -12.0, 0.0, unit="dBFS", show_if=("limiter", (True,))),
            Param("selection_only", "Selection only", "bool", False),
        ],
        run_amplify,
    ),
]

# --------------------------------------------------------------------- effects

DOPPLER_MODES = ("Pass-by", "Circling")


FILTER_KINDS = ("Low-pass", "High-pass", "Band-pass", "Band-stop")


def run_filter(x, p, ctx, progress):
    two = p["kind"] in ("Band-pass", "Band-stop")
    if two and p["f_hi"] <= p["f_lo"]:
        raise UserError("The upper edge must be above the lower edge.")
    y = fx.apply_filter(x, ctx.sr, p["kind"], p["f_lo"], p["f_hi"], p["order"],
                        p["zero_phase"], progress)
    where = (f'{p["f_lo"]:.0f}-{p["f_hi"]:.0f} Hz' if two else f'{p["f_lo"]:.0f} Hz')
    slope = p["order"] * 6 * (2 if p["zero_phase"] else 1)
    return Result(y, message=f'{p["kind"]} at {where}, about {slope:.0f} dB per octave')


def run_echo(x, p, ctx, progress):
    if p["kind"] == "Echo (discrete repeats)":
        y = fx.echo(x, ctx.sr, p["delay_ms"], p["feedback"], p["mix"], p["damping"], progress)
        n = int(np.log(0.001) / np.log(max(p["feedback"], 1e-3))) if p["feedback"] > 0 else 1
        return Result(y, message=f'Echo every {p["delay_ms"]:.0f} ms, about {n} audible repeats')
    y = fx.reverb(x, ctx.sr, p["room"], p["damping"], p["mix"], p["pre_delay_ms"], progress)
    return Result(y, message=f'Reverb added (room {p["room"]:.2f}, mix {p["mix"]:.2f})')


def run_equalizer(x, p, ctx, progress):
    gains = [p[f"g{i}"] for i in range(len(fx.EQ_BANDS))]
    if all(abs(g) < 0.05 for g in gains):
        raise UserError("Every band is at 0 dB, so there is nothing to change.")
    y = fx.equalizer(x, ctx.sr, gains, p["q"], progress=progress)
    moved = ", ".join(f"{f:.0f} Hz {g:+.0f}" for f, g in zip(fx.EQ_BANDS, gains) if abs(g) >= 0.05)
    return Result(y, message=f"EQ applied ({moved} dB)")


def run_doppler(x, p, ctx, progress):
    y = fx.doppler(x, ctx.sr, p["speed"], p["distance"], p["mode"],
                   rate_hz=p["rate"], attenuate=p["attenuate"], progress=progress)
    c = fx.SPEED_OF_SOUND
    up, down = c / (c - p["speed"]), c / (c + p["speed"])
    extra = (f", pitch {up:.2f}x approaching and {down:.2f}x receding"
             if p["mode"] == DOPPLER_MODES[0] else "")
    return Result(y, message=f"Doppler applied{extra}")


def run_reverse(x, p, ctx, progress):
    region = _need_selection(ctx, "the part to reverse") if p["selection_only"] else None
    y = fx.reverse(x, region)
    return Result(y, message="Selection reversed" if region else "Signal reversed")


def run_trim(x, p, ctx, progress):
    region = _need_selection(ctx, "the part to remove")
    y = fx.trim(x, region, ctx.sr, p["fade_ms"])
    cut_s = (region[1] - region[0]) / ctx.sr
    return Result(y, message=f"Removed {cut_s:.2f} s; {len(y) / ctx.sr:.2f} s left")


def run_crop(x, p, ctx, progress):
    region = _need_selection(ctx, "the part to keep")
    y = fx.crop(x, region, ctx.sr, p["fade_ms"])
    return Result(y, message=f"Kept {len(y) / ctx.sr:.2f} s")


def run_concatenate(x, p, ctx, progress):
    if ctx.file_ref is None:
        raise UserError("Load a second file first: the 'Load second file' button, below the "
                        "algorithm list in the Denoise section.")
    y = fx.concatenate(x, ctx.file_ref, ctx.sr, p["position"] == "Before this signal",
                       p["crossfade_ms"])
    where = "before" if p["position"] == "Before this signal" else "after"
    return Result(y, message=f"Second file joined {where}; {len(y) / ctx.sr:.2f} s total")


def run_overlay(x, p, ctx, progress):
    if ctx.file_ref is None:
        raise UserError("Load a second file first: the 'Load second file' button, below the "
                        "algorithm list in the Denoise section.")
    y = fx.overlay(x, ctx.file_ref, p["gain_db"], p["start_s"], ctx.sr, p["loop"],
                   p["duck_db"], progress)
    return Result(y, message=f"Mixed in at {p['gain_db']:+.0f} dB from {p['start_s']:.1f} s"
                             + (", looped" if p["loop"] else ""))


def run_speed(x, p, ctx, progress):
    rate = p["rate"]
    keep = p["mode"] == "Keep pitch (phase vocoder)"
    y = fx.change_speed(x, ctx.sr, rate, keep, progress=progress)
    how = "pitch kept" if keep else f"pitch x{rate:.2f}"
    return Result(y, message=f"Speed x{rate:.2f}, {how}: "
                             f"{len(x) / ctx.sr:.1f} s -> {len(y) / ctx.sr:.1f} s")


EFFECTS = [
    Algorithm(
        "filter", "Filter", "Low-pass, high-pass, band-pass or band-stop, built from "
        "Butterworth biquads through the bilinear transform.",
        [
            Param("kind", "Type", "choice", FILTER_KINDS[0], choices=FILTER_KINDS),
            Param("f_lo", "Cut-off", "float", 1000.0, 20.0, 20000.0, log=True, unit="Hz",
                  tip="For band-pass and band-stop this is the lower edge"),
            Param("f_hi", "Upper edge", "float", 4000.0, 30.0, 20000.0, log=True, unit="Hz",
                  show_if=("kind", ("Band-pass", "Band-stop"))),
            Param("order", "Order", "int", 4, 2, 12,
                  tip="Each order adds about 6 dB per octave of slope"),
            Param("zero_phase", "Zero phase", "bool", True,
                  tip="Runs forwards and backwards: no phase shift, and twice the slope"),
        ],
        run_filter,
    ),
    Algorithm(
        "echo", "Echo / Reverb", "Adds repeats or a room, the opposite of the Dereverb tool.",
        [
            Param("kind", "Type", "choice", "Echo (discrete repeats)",
                  choices=("Echo (discrete repeats)", "Reverb (dense tail)")),
            Param("delay_ms", "Delay", "float", 300.0, 20.0, 2000.0, log=True, unit="ms",
                  show_if=("kind", ("Echo (discrete repeats)",))),
            Param("feedback", "Feedback", "float", 0.35, 0.0, 0.9,
                  tip="How much of each repeat is fed back in; 0.5 gives about 10 repeats",
                  show_if=("kind", ("Echo (discrete repeats)",))),
            Param("room", "Room size", "float", 0.7, 0.0, 1.0,
                  show_if=("kind", ("Reverb (dense tail)",))),
            Param("pre_delay_ms", "Pre-delay", "float", 20.0, 0.0, 120.0, unit="ms",
                  tip="Gap before the tail starts, which keeps the direct sound clear",
                  show_if=("kind", ("Reverb (dense tail)",))),
            Param("damping", "Damping", "float", 0.4, 0.0, 0.9,
                  tip="How quickly the high frequencies die away in the tail"),
            Param("mix", "Wet mix", "float", 0.35, 0.0, 1.0),
        ],
        run_echo,
    ),
    Algorithm(
        "equalizer", "Equalizer", "Seven bands of zero-phase biquads: two shelves and five "
        "peaking filters.",
        [Param("q", "Band width (Q)", "float", 1.0, 0.4, 3.0,
               tip="Higher Q means a narrower band")] +
        [Param(f"g{i}", f"{f:.0f} Hz" if f < 1000 else f"{f / 1000:g} kHz", "float", 0.0,
               -18.0, 18.0, unit="dB") for i, f in enumerate(fx.EQ_BANDS)],
        run_equalizer,
    ),
    Algorithm(
        "doppler", "Doppler", "Moves the source past the listener; the changing delay bends the "
        "pitch up then down.",
        [
            Param("mode", "Path", "choice", DOPPLER_MODES[0], choices=DOPPLER_MODES),
            Param("speed", "Source speed", "float", 30.0, 1.0, 120.0, unit="m/s",
                  tip="120 m/s is about 430 km/h; sound itself travels at 343 m/s",
                  show_if=("mode", (DOPPLER_MODES[0],))),
            Param("rate", "Turns per second", "float", 0.5, 0.05, 4.0, unit="Hz",
                  show_if=("mode", (DOPPLER_MODES[1],))),
            Param("distance", "Closest distance", "float", 10.0, 1.0, 100.0, unit="m",
                  tip="Smaller distance means a faster, sharper swoop"),
            Param("attenuate", "Distance volume", "bool", True,
                  tip="Amplitude follows 1/r, so the source fades in and out"),
        ],
        run_doppler,
    ),
    Algorithm(
        "reverse", "Reverse", "Plays the samples backwards.",
        [Param("selection_only", "Selection only", "bool", False)],
        run_reverse, family="Editing",
        uses_selection="With 'Selection only' on, just the highlighted region is reversed.",
    ),
    Algorithm(
        "speed", "Speed", "Changes the tempo, with or without moving the pitch.",
        [
            Param("rate", "Speed", "float", 1.25, 0.25, 4.0, log=True, unit="x",
                  tip="2.0 plays twice as fast, 0.5 half as fast"),
            Param("mode", "Mode", "choice", "Keep pitch (phase vocoder)",
                  choices=("Keep pitch (phase vocoder)", "Move pitch (resample)"),
                  tip="Resampling is the tape effect: faster also means higher"),
        ],
        run_speed,
    ),
    Algorithm(
        "overlay", "Mix in a file",
        "Adds a second, already-loaded file into this one - background music under a voice, "
        "or the reverse.",
        [
            Param("gain_db", "Level", "float", -6.0, -40.0, 12.0, unit="dB",
                  tip="Level of the added file relative to itself, before it's mixed in"),
            Param("start_s", "Start at", "float", 0.0, 0.0, 600.0, unit="s",
                  tip="Where in this signal's timeline the second file begins"),
            Param("loop", "Loop to fill", "bool", False,
                  tip="Repeat the second file for as long as this one lasts"),
            Param("duck_db", "Duck this signal", "float", 0.0, -24.0, 0.0, unit="dB",
                  tip="Turn this signal down while the second file is playing - lowers "
                      "narration under music, or vice versa"),
        ],
        run_overlay, family="Editing",
    ),
    Algorithm(
        "concatenate", "Concatenate", "Joins the loaded second file onto this one, end to end.",
        [
            Param("position", "Position", "choice", "After this signal",
                  choices=("After this signal", "Before this signal")),
            Param("crossfade_ms", "Crossfade", "float", 20.0, 0.0, 500.0, unit="ms",
                  tip="0 is a hard splice; a short crossfade blends the join instead of "
                      "cutting straight from one file to the other"),
        ],
        run_concatenate, family="Editing",
    ),
    Algorithm(
        "trim", "Trim (cut out)", "Removes the selected part and joins what is left.",
        [Param("fade_ms", "Fade at the join", "float", 5.0, 0.0, 100.0, unit="ms",
               tip="A short fade on each side of the join avoids a click")],
        run_trim, family="Editing", uses_selection="Drag across the waveform to choose what to remove.",
    ),
    Algorithm(
        "crop", "Crop (keep only)", "Keeps only the selected part; everything else is dropped.",
        [Param("fade_ms", "Fade in/out", "float", 5.0, 0.0, 100.0, unit="ms")],
        run_crop, family="Editing", uses_selection="Drag across the waveform to choose what to keep.",
    ),
]

SECTIONS = {"Denoise": DENOISE, "Separate": SEPARATE, "Effects": EFFECTS, "Amplify": AMPLIFY}
