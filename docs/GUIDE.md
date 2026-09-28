# Which tool for which sound

Every number below was measured on the clips in `docs/testsounds/`, each of
which ships with its clean reference so you can reproduce them. Figures are SNR
in dB against that reference (SI-SDR for the reverb clip, where a scale change
is not an error). Higher is better; the input column is what you start from.

Reproduce with:

```bash
python main.py docs/testsounds/01_steady_hiss.wav
```

## The test clips

| Clip | What is on it | Made for |
|------|---------------|----------|
| `01_steady_hiss` | voice + constant broadband hiss, 5 dB SNR | Wiener |
| `02_mains_hum_50hz` | voice + 50 Hz hum and its harmonics | Adaptive notch |
| `03_coil_whine_1370hz` | voice + a 1.37 kHz whine and its octave | Notch, "Find the tone" |
| `04_taps_and_bangs` | voice + sharp broadband taps | Click remover |
| `05_reverberant_room` | voice convolved with a 0.7 s room, DRR 3 dB | Dereverb |
| `06_noise_starts_midway` | voice + fan, with a TV added from halfway | Wiener, Tracking profile |
| `07_two_bands` | two chirping sources in separate bands | Frequency split |
| `08_voice_over_music` | real speech over the repeating demo song | REPET / NMF |
| `09_band` | sustained chord + drum hits | Instruments (HPSS) |
| `10_too_quiet` | the voice at -25.5 dBFS | Amplify |
| `11_two_instruments` | plucked + sustained, both in 196-3900 Hz | Instruments, NMF grouping |

## Denoise: every method on every noise

| Clip | input | Wiener Auto | Wiener Selection | Wiener Tracking | Wiener 2-pass | Notch (manual) | Notch (find) | Wavelet | Click remover | Dereverb WPE |
|------|------:|------------:|-----------------:|----------------:|--------------:|---------------:|-------------:|--------:|--------------:|-------------:|
| 01 steady hiss | 5.00 | 16.18 | 15.96 | 13.77 | **16.54** | 4.73 | 4.73 | 14.28 | 5.00 | 5.25 |
| 02 mains hum | 5.00 | 11.44 | 12.06 | 10.62 | 12.09 | 12.97 | 12.56 | 5.00 | 4.76 | **18.14** |
| 03 coil whine | 5.00 | 22.64 | 22.48 | 17.17 | 22.67 | 4.73 | 24.84 | 5.00 | 4.69 | **28.24** |
| 04 taps and bangs | 5.00 | 5.00 | 5.00 | 4.85 | 5.00 | 4.74 | 4.74 | 5.00 | **15.85** | 5.00 |
| 05 reverberant room | 4.72 | 4.63 | 4.72 | 4.63 | 4.73 | 4.56 | 4.56 | 4.72 | 4.69 | **6.57** |
| 06 noise starts midway | 5.00 | 7.37 | 7.75 | **9.22** | 7.48 | 5.14 | 5.14 | 5.00 | 5.00 | 5.22 |

### What that table says

- **Steady broadband noise (hiss, fan, traffic): Wiener.** Turning on Second
  pass adds about +0.4 dB. Wavelet is the runner-up (14.3) and much faster, but
  only on white-ish noise; it does nothing for hum or brown noise, because it
  estimates the noise level from the finest detail band alone.
- **Noise that changes during the clip: Wiener, Tracking profile** (9.22 vs
  7.37 for Auto here). Auto only switches to tracking when the noise floor
  moves by 10 dB or more, so set it by hand when you know the noise changed.
- **Clicks, taps, hammering: Click remover.** Nothing else touches them
  (5.00 = unchanged) because a 90 ms bang never lasts long enough for a noise
  estimator to believe in it.
- **A steady tone (hum, whine, buzz): Dereverb (WPE) is the surprise winner**
  — 18.1 dB on hum and 28.2 dB on the whine, beating the notch filter built for
  the job. WPE removes whatever the past few frames can predict, and a constant
  sine is the most predictable signal there is. Use the notch when you want to
  remove *only* that tone and leave everything else untouched; use WPE when you
  want the strongest removal and do not mind it also thinning room reverb.
- **Room reverb: Dereverb (WPE)**, +1.85 dB. That is the honest size of the
  effect; reverb is hard to undo.
- Notch with **Find the tone** locates an unknown whine by itself (24.84 dB on
  clip 03, where the manual 50 Hz default does nothing). On hum, setting 50 Hz
  by hand is marginally better than the finder (12.97 vs 12.56).

### A suggested order

```
Click remover  ->  Wiener  ->  Notch (if hum remains)  ->  Amplify
```

Each stage removes what the next one cannot see. Run Amplify last: boosting
first would lift the noise along with the voice.

## Separate: every method on every task

Average SI-SDR of the two stems, dB.

| Task | input | REPET | Instruments | NMF (2 comp) | Frequency split |
|------|------:|------:|------------:|-------------:|----------------:|
| 07 two bands (birds) | 3.04 | 40.26 | 3.04 | 53.93 | **81.92** |
| 08 voice over music | 3.75 | 3.07 | **5.47** | 3.03 | 3.75 |
| 09 chord + drums | 4.64 | 19.75 | **26.27** | 13.98 | 4.64 |
| 11 two instruments, same range | 0.73 / 8.04 | — | 0.86 / 5.97 | — | — |

For clip 11 the two instruments overlap in frequency and both are harmonic, so
HPSS barely moves (0.86 / 5.97 dB against an input of 0.73 / 8.04). The
**Instruments → NMF grouping** mode reaches **13.0 / 20.7 dB** on the same clip
by factorising the mixture into many parts and grouping them by how their notes
start (sharp attack and decay vs slow attack and long body). When both
instruments start their notes the same way it falls back to grouping by pitch
register instead, and says so in the status line.

### What that table says

- **Sources in separate frequency bands: Frequency split**, by a wide margin
  (81.9 dB). It finds the silent valley between them and cuts there. It refuses
  to run when the sources overlap in frequency, which is the honest answer
  rather than a bad split.
- **Instruments versus drums: Instruments (HPSS)**, 26.3 dB. Sustained notes
  are horizontal in the spectrogram and hits are vertical, which is exactly
  what its two median filters measure.
- **Voice over music is the hard one.** The best result here is 5.47 dB, and
  REPET actually scores below the input. REPET needs the background to repeat
  *and* the foreground to be sparse; continuous real speech over a song breaks
  the second condition. On the synthetic demo song, where the "vocal" is sparse
  and tonal, REPET reaches 6.58 dB and leads the field. So:
  - repeating backing track, singing on top → **REPET**
  - you can select a few seconds of accompaniment alone → **NMF, trained**
  - dense continuous speech over music → none of them separate it well; expect
    a partial split at best.

On clip 08 REPET picked a 2.58 s period instead of the true 4.0 s. That is not
a detector bug: the mixture's autocorrelation genuinely peaks there once the
speech is added. Forcing 4.0 s by hand gives 2.75 / 6.05 dB, still a small
gain. Two alternative beat-spectrum designs (per-bin normalised mean and
median) were tried and both collapsed to the 0.5 s lower bound, so the simple
mean was kept.

## Amplify: making a quiet file loud

Clip `10_too_quiet` starts at peak -0.4 dBFS but RMS -25.5 dBFS, a crest factor
of 25 dB: the peaks are already at the ceiling while the average is far below.

| Setting | Peak | RMS | Clipped samples |
|---------|-----:|----:|----------------:|
| plain +18 dB gain | 0.0 | -10.4 | **3.19 %** |
| Peak normalise | -1.4 | -26.0 | 0.00 % |
| RMS -14 dBFS + limiter | -1.0 | -15.5 | 0.00 % |
| **Compressor + RMS -12 dBFS + limiter** | -1.0 | **-12.4** | 0.00 % |

Peak normalising does nothing useful here (the peak was already at the top);
plain gain clips 3 % of the samples. Only the compressor lowers the crest
factor enough to make room for real loudness. For listening rather than
editing, the transport volume slider goes to 200 % with a soft knee that holds
peaks at -1 dBFS.

## Effects

| Want | Use | Note |
|------|-----|------|
| faster or slower, voice still natural | Speed, "Keep pitch" | phase-locked vocoder; a sharp attack softens from 20.5 ms to 6.0 ms at 1.5x |
| faster or slower, exactly as a tape does | Speed, "Move pitch" | mathematically exact, pitch rises with speed |
| tone shaping, rumble or hiss trim | Equalizer | two shelves and five peaking biquads, zero phase |
| something passing by | Doppler | real path geometry, pitch follows 343/(343±v) |
| background music under narration | Mix in a file | with optional loop and ducking |
| join two recordings | Concatenate | optional crossfade at the join |
| cut a part out / keep only a part | Trim / Crop | fades at the join so there is no click |
