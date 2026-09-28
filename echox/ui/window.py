"""Main window: connects audio I/O, the algorithm registry and the views."""
from __future__ import annotations

import os
import time
from math import gcd

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QFileDialog, QMainWindow
from scipy.signal import resample_poly

from ..algorithms import SECTIONS, Context
from ..audio import FILE_FILTER, Player, Recorder, load_audio, save_audio
from .. import theme as th
from ..dsp import effects as fx
from ..dsp.common import FFT_ENGINES, rms, set_fft_engine
from ..dsp.testbench import add_noise, snr_db
from .icons import icon
from .layout import LayoutMixin
from .plots import SpectrumView, band_events, format_hz, format_time, spectrogram_db
from .workers import GrowBuffer, SectionPage, Task, centered, slug

CURRENT = "Current"
ORIGINAL = "Original"
MAX_HISTORY = 15
LIVE_WINDOW_S = 10.0


class MainWindow(LayoutMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EchoX")
        self.resize(1400, 880)
        self.setMinimumSize(1100, 700)
        self.setAcceptDrops(True)

        self.player = Player()
        self.recorder = Recorder()
        self.rec_buf: GrowBuffer | None = None
        self.sr = 44100
        self.current: np.ndarray | None = None
        self.original: np.ndarray | None = None
        self.stems: dict[str, np.ndarray] = {}
        self.history: list[np.ndarray] = []
        self.future: list[np.ndarray] = []
        self.active = CURRENT
        self.selection: tuple[float, float] | None = None
        self.file_name = ""
        self.clean_ref = self.testbench_ref = self.file_ref = None
        self.file_ref_name = ""
        self.play_array = np.zeros(0)
        self.play_speed = 1.0
        self._speed_cache = {}
        self._speed_task = None
        self.task: Task | None = None
        self._spec_cache: dict[int, tuple] = {}
        self._last_pos = -1
        self._shown_playing = False
        self._t0 = 0.0

        self._build()
        self._shortcuts()
        self._set_loaded(False)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)

    # ============================================================== messages
    def show_status(self, text: str):
        self.error_label.hide()
        self.status_label.setText(text)

    def show_metric(self, text: str):
        self.error_label.hide()
        self.metric_label.setText(text)
        self.metric_label.show()

    def show_error(self, text: str):
        self.error_label.setText(text)
        self.error_label.show()

    # =============================================================== session
    def open_dialog(self):
        if self._busy_now():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open audio", "", FILE_FILTER)
        if path:
            self.load_file(path)

    def load_file(self, path: str):
        try:
            x, sr = load_audio(path)
        except Exception as exc:
            self.show_error(f"Could not open {os.path.basename(path)}: {exc}")
            return
        if len(x) < sr * 0.1:
            self.show_error("That file is shorter than 0.1 s.")
            return
        self.file_name = os.path.splitext(os.path.basename(path))[0]
        self._new_session(x, sr)
        self.show_status(f"Opened {os.path.basename(path)}: {len(x) / sr:.1f} s at {sr} Hz, mono")

    def _new_session(self, x: np.ndarray, sr: int):
        self.player.pause()
        self.sr = sr
        self.current = self.original = np.ascontiguousarray(x, dtype=np.float64)
        self.stems = {}
        self.history.clear()
        self.future.clear()
        self.clean_ref = self.testbench_ref = self.file_ref = None
        self._spec_cache.clear()
        self.stem_list.set_stems([])
        self.metric_label.hide()
        self._update_testbench_state()
        self._set_loaded(True)
        self.on_select(None, True)
        self._refresh_sources(CURRENT)
        self.player.seek(0)
        self.fit_view()
        self._update_undo()

    def _set_loaded(self, loaded: bool):
        self.stage.setCurrentIndex(1 if loaded else 0)
        for w in (self.btn_save, self.btn_play, self.btn_stop, self.btn_start, self.btn_loop,
                  self.seek, self.listen_combo):
            w.setEnabled(loaded)

    def sources(self) -> dict[str, np.ndarray]:
        out = {CURRENT: self.current}
        if self.original is not None:
            out[ORIGINAL] = self.original
        out.update(self.stems)
        return out

    def _refresh_sources(self, select: str | None = None):
        names = list(self.sources())
        want = select if select in names else (self.active if self.active in names else CURRENT)
        self.listen_combo.blockSignals(True)
        self.listen_combo.clear()
        self.listen_combo.addItems(names)
        self.listen_combo.setCurrentText(want)
        self.listen_combo.blockSignals(False)
        self._activate(want)

    def _on_listen_changed(self, name: str):
        if name and self.current is not None:
            self._activate(name)

    def _apply_playback(self, arr):
        """Hands the player what it should actually play. At speed 1 that is the
        signal itself; otherwise a phase-vocoded copy (same pitch, new length),
        rendered in the background and cached per speed."""
        self.play_array = arr
        self._speed_task = None
        if abs(self.play_speed - 1.0) < 1e-6:
            self.player.set_source(arr, self.sr, 1.0)
            return
        key = (id(arr), len(arr), round(self.play_speed, 3))
        cached = self._speed_cache.get(key)
        if cached is not None:
            self.player.set_source(cached, self.sr, self.play_speed)
            return
        self.player.set_source(arr, self.sr, 1.0)      # play as-is until it is ready
        rate, source = self.play_speed, arr
        task = Task(lambda progress: fx.phase_vocoder(source, rate, progress=progress))
        self._speed_task = task
        self.show_status(f"Rendering {rate:g}x playback...")

        def ready(result):
            if self._speed_task is not task or self.play_array is not source:
                return                                  # user moved on
            self._speed_cache[key] = result
            if len(self._speed_cache) > 6:
                self._speed_cache.pop(next(iter(self._speed_cache)))
            pos, playing = self.player.pos_source, self.player.playing
            self.player.set_source(result, self.sr, rate)
            self.player.seek_source(pos)
            self._update_loop()
            if playing:
                self.player.play()
            self.show_status(f"Playing at {rate:g}x, pitch unchanged")
        task.done.connect(ready)
        task.failed.connect(self.show_error)
        task.start()

    def set_play_speed(self, rate: float):
        if abs(rate - self.play_speed) < 1e-6 or self.current is None:
            return
        pos, playing = self.player.pos_source, self.player.playing
        self.play_speed = float(rate)
        self.speed_button.setText(f"{rate:g}x")
        self._apply_playback(self.play_array)
        self.player.seek_source(pos)
        self._update_loop()
        if playing:
            self.player.play()

    def set_volume(self, percent: int):
        """Above 100 % the player's soft knee holds the peaks at -1 dBFS, so the
        boost is loudness rather than clipping."""
        self.player.volume = percent / 100
        self.volume_label.setText(f"{percent}%")
        boosted = percent > 100
        self.volume_label.setStyleSheet(f"color: {th.MARIGOLD};" if boosted else "")
        self.volume_label.setToolTip("Boosting; peaks are held at -1 dBFS" if boosted else "")

    def _activate(self, name: str):
        self.active = name
        arr = self.sources()[name]
        self._apply_playback(arr)
        ghost = self.original if (name != ORIGINAL and self.original is not arr) else None
        self.wave.set_duration(len(arr) / self.sr)
        self.spectro.set_duration(len(arr) / self.sr)
        self.wave.set_signal(arr, self.sr, ghost)
        S, hop = self._spectrogram(arr)
        self.spectro.set_audio(S, hop, self.sr)
        self.spectrum.reset()
        self._on_fmax(self.fmax_combo.currentText())
        self._sync_ab_button()
        if self.selection is not None:
            self.on_select(self.selection, True)          # clamp to the new length
        self.wave_sheet.title.setText(f"Waveform: {name}")
        self.spec_sheet.title.setText(f"Spectrogram: {name}")
        self.total.setText(f"/ {format_time(len(arr) / self.sr)}")
        self._last_pos = -1

    def _spectrogram(self, arr: np.ndarray):
        hit = self._spec_cache.get(id(arr))
        if hit is not None and hit[0] is arr:
            return hit[1], hit[2]
        S, hop, _ = spectrogram_db(arr, self.sr)
        self._spec_cache[id(arr)] = (arr, S, hop)
        while len(self._spec_cache) > 12:
            self._spec_cache.pop(next(iter(self._spec_cache)))
        return S, hop

    def _set_current(self, arr: np.ndarray):
        self.history.append(self.current)
        del self.history[:-MAX_HISTORY]
        self.future.clear()
        self.current = np.ascontiguousarray(arr, dtype=np.float64)
        self._refresh_sources(CURRENT)
        self._update_undo()

    def undo(self):
        if self.history and not self._busy_now():
            self.future.append(self.current)
            self.current = self.history.pop()
            self._refresh_sources(CURRENT)
            self._update_undo()
            self.show_status("Undid last change")

    def redo(self):
        if self.future and not self._busy_now():
            self.history.append(self.current)
            self.current = self.future.pop()
            self._refresh_sources(CURRENT)
            self._update_undo()
            self.show_status("Redid change")

    def _update_undo(self):
        busy = self._busy_now()
        self.btn_undo.setEnabled(bool(self.history) and not busy)
        self.btn_redo.setEnabled(bool(self.future) and not busy)

    def save_active(self):
        if self.current is None or self._busy_now():
            return
        self._save(self.active, self.sources()[self.active])

    def _save(self, name: str, arr: np.ndarray):
        default = f"{self.file_name or 'echox'}-{slug(name)}.wav"
        path, _ = QFileDialog.getSaveFileName(self, "Save audio", default,
                                              "WAV (*.wav);;FLAC (*.flac);;OGG Vorbis (*.ogg)")
        if not path:
            return
        try:
            save_audio(path, arr, self.sr)
            self.show_status(f"Saved {os.path.basename(path)}")
        except Exception as exc:
            self.show_error(f"Could not save: {exc}")

    def keep_stem(self, name: str):
        self._set_current(self.stems[name])
        self._retire_testbench(f"Keeping {name}")
        self.show_status(f"{name} is now the working copy")

    def _retire_testbench(self, cause: str):
        """SNR against the stored clean copy only means something while the
        signal is 'clean + noise, then denoised'. A gain change or a stem swap
        breaks that, so scoring stops instead of printing a misleading number."""
        if self.clean_ref is None:
            return
        self.clean_ref = self.testbench_ref = None
        self.metric_label.hide()
        self._update_testbench_state(f"Scoring stopped: {cause} changed the signal, so the "
                                     "clean copy no longer matches. Add noise again to measure.")

    def export_stem(self, name: str):
        self._save(name, self.stems[name])

    # ============================================================ test bench
    def add_test_noise(self):
        if self.current is None:
            self.show_error("Open or record audio first.")
            return
        p = self.tb_form.values()
        clean = self.current.copy()
        noisy, ref = add_noise(clean, self.sr, p["noise"], p["snr"])
        self.clean_ref, self.testbench_ref = clean, ref
        self._set_current(noisy)
        self._update_testbench_state(f"{p['noise']} noise added at {p['snr']:.1f} dB SNR.")
        self.show_metric(f"SNR now {snr_db(clean, noisy):.1f} dB")

    def load_reference(self):
        if self.current is None:
            self.show_error("Open or record audio first.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open reference noise", "", FILE_FILTER)
        if not path:
            return
        try:
            x, sr = load_audio(path)
        except Exception as exc:
            self.show_error(f"Could not open reference: {exc}")
            return
        if sr != self.sr:
            g = gcd(self.sr, sr)
            x = resample_poly(x, self.sr // g, sr // g)
        self.file_ref = x
        self.file_ref_name = os.path.basename(path)
        self._update_testbench_state()
        self._update_second_file_state()
        self.show_status(f"Second file loaded: {self.file_ref_name} ({len(x) / self.sr:.1f} s). "
                         "Use it as Wiener's noise profile, or in Effects -> Mix in a file.")

    def _update_second_file_state(self):
        if self.file_ref is None:
            self.second_state.setText("Nothing loaded yet.")
        else:
            self.second_state.setText(f"{self.file_ref_name} - "
                                      f"{len(self.file_ref) / self.sr:.1f} s, ready to use.")

    def _update_testbench_state(self, first_line: str = ""):
        parts = [first_line] if first_line else []
        if self.clean_ref is not None:
            parts.append("Clean copy stored for scoring.")
        if self.file_ref is not None:
            dur = len(self.file_ref) / self.sr
            parts.append(f"Second file loaded: {getattr(self, 'file_ref_name', '')} "
                         f"({dur:.1f} s).")
        self.tb_state.setText(" ".join(parts) or "Nothing added yet.")

    # ============================================================ processing
    def _select_section(self, i: int):
        for j, page in enumerate(self.pages):
            page.setVisible(j == i)
        self.side_scroll.verticalScrollBar().setValue(0)

    def _selection_samples(self):
        if self.selection is None:
            return None
        a, b = self.selection
        return int(a * self.sr), int(b * self.sr)

    def _busy_now(self) -> bool:
        return (self.task is not None and self.task.isRunning()) or self.recorder.active

    def _set_busy(self, busy: bool):
        for page in self.pages:
            page.run.setEnabled(not busy)
        for w in (self.btn_open, self.btn_add_noise, self.btn_load_ref, self.stem_list,
                  self.fft_combo):
            w.setEnabled(not busy)
        self.progress.setVisible(busy)
        self.progress.setValue(0)
        self._update_undo()

    def run_page(self, page: SectionPage):
        if self.current is None:
            self.show_error("Open or record audio first.")
            return
        if self._busy_now():
            return
        algo, params = page.algorithm, page.values()
        ctx = Context(self.sr, self._selection_samples(), self.testbench_ref, self.file_ref)
        x = self.current.copy()
        scored = page.name == "Denoise" and self.clean_ref is not None \
            and len(self.clean_ref) == len(x)
        before = snr_db(self.clean_ref, x) if scored else None
        self.metric_label.hide()
        self.error_label.setText("")
        self.show_status(f"Running {algo.name}")
        self.task = Task(lambda progress: algo.run(x, params, ctx, progress))
        self.task.progressed.connect(self._on_progress)
        self.task.done.connect(lambda res: self._on_result(algo, res, before))
        self.task.failed.connect(self._on_failed)
        self.task.finished.connect(lambda: self._set_busy(False))
        self._t0 = time.perf_counter()
        self._set_busy(True)
        self.task.start()

    def _on_progress(self, value: float, text: str):
        self.progress.setValue(int(value * 1000))
        if text:
            self.status_label.setText(text)

    def _on_result(self, algo, res, snr_before):
        elapsed = time.perf_counter() - self._t0
        if res.stems:
            self.stems = {f"{k} ({algo.name})": np.ascontiguousarray(v) for k, v in res.stems.items()}
            self.stem_list.set_stems(list(self.stems))
            self.stems_hint.hide()
            self._refresh_sources(next(iter(self.stems)))
        elif res.output is not None:
            self._set_current(res.output)
            if algo not in SECTIONS["Denoise"]:
                self._retire_testbench(algo.name)
        self.show_status(f"{res.message}  ·  {elapsed:.1f} s")
        if snr_before is not None and res.output is not None:
            after = snr_db(self.clean_ref, res.output)
            self.show_metric(f"SNR {snr_before:.1f} dB → {after:.1f} dB ({after - snr_before:+.1f} dB)")

    def _on_failed(self, message: str):
        self.status_label.setText("")
        self.show_error(message)

    # ============================================================= selection
    def on_select(self, sel, finished: bool):
        dur = self._duration()
        if sel is not None:
            a, b = max(0.0, sel[0]), min(dur, sel[1])
            sel = (a, b) if b - a > 0.005 else None
        self.selection = sel
        self.wave.set_selection(sel)
        self.spectro.set_selection(sel)
        if sel:
            self.sel_label.setText(f"Selected {format_time(sel[0], 2)} to {format_time(sel[1], 2)}"
                                   f"  ({sel[1] - sel[0]:.2f} s)")
        else:
            self.sel_label.setText("Click to move, drag to select, scroll to zoom")
        if finished:
            self._update_loop()

    def _update_loop(self):
        s = self._selection_samples() if self.btn_loop.isChecked() else None
        self.player.set_loop(s)

    def _on_fft_engine(self, index: int):
        set_fft_engine(FFT_ENGINES[index])
        self._spec_cache.clear()
        if self.current is not None and not self.recorder.active:
            self._activate(self.active)
        self.show_status("Transforms now use the hand-written FFT" if index == 0
                         else "Transforms now use numpy.fft / scipy.signal")

    def _on_cursor(self, text):
        self.cursor_label.setText(text or "")

    def _on_freq_scale(self, text: str):
        self.spectro.set_freq_scale(text.startswith("Log"))

    def _on_dynamic_range(self, text: str):
        self.spectro.set_dynamic_range(float(text.split()[0]))

    def _on_fmax(self, text: str):
        fmax = {"4 kHz": 4000, "8 kHz": 8000, "11 kHz": 11025}.get(text, self.sr / 2)
        self.spectro.set_max_freq(fmax)

    def fit_view(self):
        self.wave.setXRange(0, max(self._duration(), 0.01), padding=0)

    def _duration(self) -> float:
        return self.player.length_source / self.sr if len(self.player.data) else 0.0

    # ============================================================= transport
    def toggle_play(self):
        if self.current is None or self.recorder.active:
            return
        if self.player.playing:
            self.player.pause()
            return
        try:
            self.player.play()
        except Exception as exc:
            self.show_error(str(exc))

    def stop(self):
        self.player.stop()
        self._last_pos = -1

    def seek_time(self, t: float):
        if self.current is None or self.recorder.active:
            return
        self.player.seek_source(int(t * self.sr))
        self._last_pos = -1

    def jump_to_frequency(self, freq: float):
        """Frequency-domain navigation: move the playhead to the next moment
        where the clicked frequency band becomes loud (wraps around)."""
        if self.current is None or self.recorder.active:
            return
        arr = self.play_array
        S, hop = self._spectrogram(arr)
        times, levels, loudest = band_events(S, hop, self.sr, freq)
        now = self.player.pos_source / self.sr
        later = np.flatnonzero(times > now + 0.05)
        i = int(later[0]) if len(later) else 0
        t = float(times[i])
        self.spectro.mark_frequency(freq)
        if freq > self.spectro.fmax:
            self.fmax_combo.setCurrentText("Full range")
        self.seek_time(t)
        self._follow(t)
        name = f"{format_hz(freq)}Hz band"
        if len(times) == 1:
            self.show_status(f"{name} never drops out; jumped to its loudest moment, "
                             f"{format_time(t)} ({levels[i]:.0f} dB).")
            return
        wrapped = ", wrapped to the start" if not len(later) else ""
        self.show_status(f"{name} rises at {format_time(t)} ({levels[i]:.0f} dB): event "
                         f"{i + 1} of {len(times)}{wrapped}. Click again for the next one.")

    def toggle_original(self, on: bool):
        """A/B: swap between the working copy and the file as it was loaded,
        keeping the playhead and whether it is playing."""
        if self.current is None or self.original is None:
            return
        want = ORIGINAL if on else CURRENT
        if want not in self.sources() or self.listen_combo.currentText() == want:
            return
        pos, playing = self.player.pos_source, self.player.playing
        self.listen_combo.setCurrentText(want)
        self.player.seek_source(min(pos, max(self.player.length_source - 1, 0)))
        if playing:
            self.player.play()

    def _sync_ab_button(self):
        self.btn_ab.blockSignals(True)
        self.btn_ab.setChecked(self.active == ORIGINAL)
        self.btn_ab.setText("Current" if self.active == ORIGINAL else "Original")
        self.btn_ab.blockSignals(False)

    def _sync_play_button(self, playing: bool):
        if playing != self._shown_playing:
            self._shown_playing = playing
            self.btn_play.setIcon(icon("pause" if playing else "play", "#FFFFFF", 22))

    def _follow(self, t: float):
        x0, x1 = self.wave.vb.viewRange()[0]
        w = x1 - x0
        if t > x1 or t < x0:
            start = min(max(0.0, t - 0.02 * w), max(0.0, self._duration() - w))
            self.wave.setXRange(start, start + w, padding=0)

    # ============================================================ live views
    def _tick(self):
        if self.recorder.active:
            self._tick_record()
            return
        if self.current is None:
            return
        playing = self.player.playing
        self._sync_play_button(playing)
        pos = self.player.pos_source
        if playing:
            pos = max(0, pos - self.player.latency_samples())
        if pos == self._last_pos and not playing:
            self.meter.set_db(-60.0)
            return
        self._last_pos = pos
        arr = self.play_array
        t = pos / self.sr
        self.wave.set_playhead(t)
        self.spectro.set_playhead(t)
        if playing:
            self._follow(t)
        frame = centered(arr, pos, SpectrumView.N)
        peak = self.spectrum.update_frame(frame, self.sr)
        self.peak_label.setText(f"Peak {format_hz(peak)}Hz")
        self.scope.update_frame(centered(arr, pos, self.scope.samples_needed(self.sr)), self.sr)
        mid = SpectrumView.N // 2
        self.meter.set_db(20 * np.log10(rms(frame[mid - 1024:mid + 1024])) if playing else -60.0)
        self.clock.setText(format_time(t))
        if not self.seek.isSliderDown() and len(arr):
            self.seek.setValue(int(pos / len(arr) * 10000))

    def toggle_record(self, on: bool):
        if on:
            if self.task is not None and self.task.isRunning():
                self._uncheck_record()
                return
            self.player.pause()
            try:
                self.recorder.start(44100)
            except Exception as exc:
                self._uncheck_record()
                self.show_error(f"Could not start recording: {exc}")
                return
            sr = self.recorder.sr
            self.rec_buf = GrowBuffer(sr * 30)
            self._set_loaded(True)
            self._set_busy(True)
            self.progress.hide()
            self.on_select(None, True)
            self.wave.set_signal(None, sr)
            self.spectro.begin_live(sr, LIVE_WINDOW_S)
            self.spectrum.reset()
            self.scope.reset()
            self.wave_sheet.title.setText("Waveform: recording")
            self.spec_sheet.title.setText("Spectrogram: recording")
            self.btn_record.setText(" Stop recording")
            self.listen_combo.setEnabled(False)
            self.show_status("Recording from the default microphone. Press Ctrl+R or Stop recording to finish.")
            return
        if not self.recorder.active:
            return
        sr = self.recorder.sr
        self.rec_buf.extend(self.recorder.stop())
        data = self.rec_buf.view().astype(np.float64)
        self.rec_buf = None
        self.btn_record.setText(" Record")
        self.listen_combo.setEnabled(True)
        self._set_busy(False)
        if len(data) < sr * 0.2:
            self.show_error("Recording was shorter than 0.2 s, so it was discarded.")
            if self.current is None:
                self._set_loaded(False)
            else:
                self._refresh_sources(CURRENT)
            return
        self.file_name = f"recording-{time.strftime('%H%M%S')}"
        self._new_session(data, sr)
        self.show_status(f"Recorded {len(data) / sr:.1f} s at {sr} Hz")

    def _uncheck_record(self):
        self.btn_record.blockSignals(True)
        self.btn_record.setChecked(False)
        self.btn_record.blockSignals(False)

    def _tick_record(self):
        new = self.recorder.read()
        sr = self.recorder.sr
        if len(new):
            self.rec_buf.extend(new)
            self.spectro.live_append(new)
        y = self.rec_buf.view()
        T = len(y) / sr
        span = max(T, LIVE_WINDOW_S)
        self.wave.set_duration(span)
        self.spectro.set_duration(span)
        x0 = max(0.0, T - LIVE_WINDOW_S)
        self.wave.setXRange(x0, x0 + LIVE_WINDOW_S, padding=0)
        self.wave.set_signal(y, sr)
        self.wave.set_playhead(T)
        self.spectro.set_playhead(T)
        if len(y) >= SpectrumView.N:
            peak = self.spectrum.update_frame(y[-SpectrumView.N:].astype(np.float64), sr)
            self.peak_label.setText(f"Peak {format_hz(peak)}Hz")
            self.scope.update_frame(y[-self.scope.samples_needed(sr):].astype(np.float64), sr)
            self.meter.set_db(20 * np.log10(rms(y[-2048:])))
        self.clock.setText(format_time(T))

    # ============================================================ window glue
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls and not self._busy_now():
            self.load_file(urls[0].toLocalFile())

    def closeEvent(self, e):
        self.timer.stop()
        if self.recorder.active:
            self.recorder.stop()
        self.player.close()
        if self.task is not None and self.task.isRunning():
            self.task.wait()
        super().closeEvent(e)
