"""Time and frequency views.

Interaction model shared by the waveform and the spectrogram (their time axes
are linked, so they always show the same span):
  click            move the playhead there
  drag             select a time range (drag its edges to adjust)
  right click      clear the selection
  wheel            zoom time around the cursor
  shift+drag       pan
  drag playhead    scrub
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, Signal

from .. import theme as th
from ..dsp.common import rfft, rfftfreq
from ..dsp.transforms import hann

# ------------------------------------------------------------------- helpers


def format_time(t: float, decimals: int = 3) -> str:
    t = max(float(t), 0.0)
    m, s = divmod(t, 60.0)
    width = 3 + decimals if decimals else 2
    return f"{int(m)}:{s:0{width}.{decimals}f}"


def format_hz(f: float) -> str:
    if f >= 1000:
        k = f / 1000
        return f"{k:.0f}k" if abs(k - round(k)) < 1e-6 else f"{k:.1f}k"
    return f"{f:.0f}"


class TimeAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        d = 0 if spacing >= 1 else 1 if spacing >= 0.1 else 2 if spacing >= 0.01 else 3
        return [format_time(v, d) for v in values]


class FreqAxis(pg.AxisItem):
    """Ticks in Hz. In log mode the coordinate is log10(Hz), so the label is
    10**value and the ticks sit on decade-friendly frequencies."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.log_hz = False

    def tickStrings(self, values, scale, spacing):
        return [format_hz(10 ** v if self.log_hz else v) for v in values]

    def tickValues(self, lo, hi, size):
        if not self.log_hz:
            return super().tickValues(lo, hi, size)
        nice = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        major = [np.log10(f) for f in nice if lo <= np.log10(f) <= hi]
        return [(1.0, major)]


def auto_levels(S_db: np.ndarray, dynamic_range: float = 70.0):
    """Colour range for one clip: the top is the loudest content actually in
    the file (99.9th percentile, so a single click cannot wash the picture
    out), the bottom is `dynamic_range` dB below it."""
    hi = float(np.percentile(S_db, 99.9))
    return hi - dynamic_range, hi


def to_log_rows(S_db: np.ndarray, sr: int, n_fft: int, f_min: float = 20.0, rows: int = 600):
    """Resample the frequency axis to equal steps in log10(Hz), so the low
    octaves - where music lives - get as much screen as the high ones."""
    f_max = sr / 2
    grid = np.logspace(np.log10(f_min), np.log10(f_max), rows)
    pos = np.clip(grid * n_fft / sr, 0, S_db.shape[0] - 1)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, S_db.shape[0] - 1)
    w = (pos - i0)[:, None]
    return (1 - w) * S_db[i0] + w * S_db[i1], np.log10(f_min), np.log10(f_max)


def spectrogram_db(x: np.ndarray, sr: int, max_cols: int = 4000):
    """Magnitude in dBFS (a full-scale sine reads about 0 dB)."""
    n_fft = 2048 if sr > 24000 else 1024
    hop = int(np.clip(np.ceil(len(x) / max_cols), n_fft // 4, n_fft))
    win = hann(n_fft)
    scale = 2.0 / win.sum()
    pad = np.pad(np.asarray(x, dtype=np.float64), (n_fft // 2, n_fft // 2))
    n_frames = max(1, 1 + (len(pad) - n_fft) // hop)
    out = np.empty((n_fft // 2 + 1, n_frames), dtype=np.float32)
    offsets = np.arange(n_fft)[None, :]
    for c0 in range(0, n_frames, 256):
        c1 = min(n_frames, c0 + 256)
        idx = np.arange(c0, c1)[:, None] * hop + offsets
        spec = np.abs(rfft(pad[idx] * win)) * scale
        out[:, c0:c1] = (20 * np.log10(spec + 1e-9)).T
    return out, hop, n_fft


def band_events(S_db: np.ndarray, hop: int, sr: int, freq: float,
                semitones: float = 1.0, drop_db: float = 10.0):
    """Where does `freq` get loud?  Averages the power in +-semitones around
    freq for every spectrogram column, then returns (times, level_db) for each
    rising edge above (loudest - drop_db): the onsets of that pitch or band.
    A band that is loud the whole time has one event, at its loudest column."""
    n_fft = 2 * (S_db.shape[0] - 1)
    ratio = 2 ** (semitones / 12)
    lo = int(np.floor(freq / ratio * n_fft / sr))
    hi = int(np.ceil(freq * ratio * n_fft / sr))
    lo = int(np.clip(lo, 0, S_db.shape[0] - 1))
    hi = int(np.clip(max(hi, lo + 1), 1, S_db.shape[0]))
    power = np.mean(10 ** (S_db[lo:hi].astype(np.float64) / 10), axis=0)
    level = 10 * np.log10(power + 1e-14)
    if len(level) >= 3:                                   # light 3-column smoothing
        level = np.convolve(level, np.ones(3) / 3, mode="same")
    # the first and last half-window of columns straddle the file edges, where
    # an abrupt start or cut splatters energy into every band; skip them
    edge = int(np.ceil(n_fft / 2 / hop))
    head = edge if len(level) > 3 * edge else 0
    tail = len(level) - edge if len(level) > 3 * edge else len(level)
    level = level[head:tail]
    above = level >= level.max() - drop_db
    rising = np.flatnonzero(above & ~np.concatenate([[False], above[:-1]]))
    if above.all() or len(rising) == 0:          # never quiet: no onsets to find
        rising = np.array([int(np.argmax(level))])
    return (rising + head) * hop / sr, level[rising], float(level.max())


class TimeViewBox(pg.ViewBox):
    sigSeek = Signal(float)
    sigSelect = Signal(float, float, bool)
    sigClear = Signal()

    def __init__(self):
        super().__init__()
        self.setMouseEnabled(x=True, y=False)
        self.setMenuEnabled(False)

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            ev.accept()
            self.sigSeek.emit(self.mapSceneToView(ev.scenePos()).x())
        elif ev.button() == Qt.MouseButton.RightButton:
            ev.accept()
            self.sigClear.emit()

    def mouseDragEvent(self, ev, axis=None):
        shift = ev.modifiers() & Qt.KeyboardModifier.ShiftModifier
        if ev.button() == Qt.MouseButton.LeftButton and not shift:
            ev.accept()
            a = self.mapSceneToView(ev.buttonDownScenePos()).x()
            b = self.mapSceneToView(ev.scenePos()).x()
            self.sigSelect.emit(min(a, b), max(a, b), ev.isFinish())
        else:
            super().mouseDragEvent(ev, axis)


class TimePlot(pg.PlotWidget):
    """Base for views whose x axis is time: playhead, selection, limits, and a
    cursor line that reports what is under the mouse."""

    sigSeek = Signal(float)
    sigSelect = Signal(object, bool)     # (t0, t1) or None, finished
    sigCursor = Signal(object)           # readout string, or None when outside

    def __init__(self, left_axis: pg.AxisItem | None = None):
        self.vb = TimeViewBox()
        axes = {"bottom": TimeAxis("bottom")}
        if left_axis is not None:
            axes["left"] = left_axis
        super().__init__(viewBox=self.vb, axisItems=axes)
        th.style_plot(self.getPlotItem())
        self.setMinimumHeight(120)

        self.region = pg.LinearRegionItem(
            brush=th.alpha(th.MARIGOLD, 46), hoverBrush=th.alpha(th.MARIGOLD, 70),
            pen=pg.mkPen(th.alpha(th.MARIGOLD, 210), width=1.5),
            hoverPen=pg.mkPen(th.MARIGOLD, width=3))
        self.region.setZValue(5)
        self.region.hide()
        self.addItem(self.region, ignoreBounds=True)

        self.playhead = pg.InfiniteLine(0, angle=90, movable=True,
                                        pen=pg.mkPen(th.MARIGOLD, width=2),
                                        hoverPen=pg.mkPen(th.MARIGOLD, width=4))
        self.playhead.setZValue(10)
        self.addItem(self.playhead, ignoreBounds=True)

        self.vb.sigSeek.connect(self.sigSeek)
        self.vb.sigClear.connect(lambda: self.sigSelect.emit(None, True))
        self.vb.sigSelect.connect(lambda a, b, done: self.sigSelect.emit((a, b), done))
        self.playhead.sigDragged.connect(lambda line: self.sigSeek.emit(line.value()))
        self.region.sigRegionChanged.connect(self._region_moved)
        self._updating_region = False

        self.cursor = pg.InfiniteLine(0, angle=90, movable=False,
                                      pen=pg.mkPen(th.alpha(th.GRAPHITE, 120), width=1))
        self.cursor.setZValue(4)
        self.cursor.hide()
        self.addItem(self.cursor, ignoreBounds=True)
        self.scene().sigMouseMoved.connect(self._mouse_moved)

    def _mouse_moved(self, pos):
        if not self.vb.sceneBoundingRect().contains(pos):
            self.cursor.hide()
            self.sigCursor.emit(None)
            return
        pt = self.vb.mapSceneToView(pos)
        self.cursor.setValue(pt.x())
        self.cursor.show()
        self.sigCursor.emit(self.readout(pt.x(), pt.y()))

    def readout(self, t, y):
        return format_time(t, 3)

    def leaveEvent(self, ev):
        super().leaveEvent(ev)
        self.cursor.hide()
        self.sigCursor.emit(None)

    def _region_moved(self):
        if not self._updating_region:
            self.sigSelect.emit(tuple(self.region.getRegion()), False)

    def set_selection(self, sel):
        self._updating_region = True
        if sel is None:
            self.region.hide()
        else:
            self.region.setRegion(sel)
            self.region.show()
        self._updating_region = False

    def set_playhead(self, t: float):
        self.playhead.setValue(t)

    def set_duration(self, seconds: float):
        self.vb.setLimits(xMin=0, xMax=max(seconds, 0.01), minXRange=0.002)


# ------------------------------------------------------------------ waveform


class Envelope:
    """Min/max envelope so a waveform with millions of samples draws with
    about two points per pixel at any zoom level."""

    BASE = 256
    PYRAMID_ABOVE = 1_000_000

    def __init__(self, y: np.ndarray, sr: int):
        self.y, self.sr = y, sr
        self._mn = self._mx = None

    def _pyramid(self):
        if self._mn is None:
            n = len(self.y) // self.BASE * self.BASE
            r = self.y[:n].reshape(-1, self.BASE)
            self._mn, self._mx = r.min(axis=1), r.max(axis=1)
        return self._mn, self._mx

    def window(self, t0: float, t1: float, points: int):
        y, sr = self.y, self.sr
        n = len(y)
        i0 = int(np.clip(np.floor(t0 * sr) - 1, 0, n))
        i1 = int(np.clip(np.ceil(t1 * sr) + 2, 0, n))
        span = i1 - i0
        if span < 2:
            return None
        if span <= points * 2:
            seg = y[i0:i1]
            return np.arange(i0, i1) / sr, seg, seg
        k = span // points
        if span > self.PYRAMID_ABOVE and k >= self.BASE:
            mn, mx = self._pyramid()
            kk = k // self.BASE
            j0 = i0 // self.BASE
            m = (i1 // self.BASE - j0) // kk * kk
            if m <= 0:
                return None
            lo = mn[j0:j0 + m].reshape(-1, kk).min(axis=1)
            hi = mx[j0:j0 + m].reshape(-1, kk).max(axis=1)
            x = (j0 * self.BASE + (np.arange(len(lo)) + 0.5) * kk * self.BASE) / sr
            return x, hi, lo
        m = span // k * k
        seg = y[i0:i0 + m].reshape(-1, k)
        x = (i0 + (np.arange(seg.shape[0]) + 0.5) * k) / sr
        return x, seg.max(axis=1), seg.min(axis=1)


class _Layer:
    def __init__(self, plot: pg.PlotWidget, color, fill, z):
        pen = pg.mkPen(color, width=1)
        self.hi = pg.PlotDataItem(pen=pen)
        self.lo = pg.PlotDataItem(pen=pen)
        self.fill = pg.FillBetweenItem(self.hi, self.lo, brush=fill)
        for item in (self.fill, self.hi, self.lo):
            item.setZValue(z)
            plot.addItem(item)
        self.env: Envelope | None = None

    def render(self, t0, t1, points):
        res = self.env.window(t0, t1, points) if self.env is not None else None
        if res is None:
            self.hi.setData([], [])
            self.lo.setData([], [])
            return
        x, hi, lo = res
        self.hi.setData(x, hi)
        self.lo.setData(x, lo)


class WaveformView(TimePlot):
    def __init__(self):
        super().__init__()
        self.setYRange(-1.05, 1.05, padding=0)
        self.getAxis("left").setTicks([[(-1, "-1"), (-0.5, "-0.5"), (0, "0"), (0.5, "0.5"), (1, "1")]])
        self.ghost = _Layer(self, th.GHOST, th.alpha(th.GHOST, 150), -10)
        self.main = _Layer(self, th.TEAL, th.alpha(th.TEAL, 200), 0)
        self.vb.sigXRangeChanged.connect(self.redraw)

    def set_signal(self, y: np.ndarray | None, sr: int, ghost: np.ndarray | None = None):
        self.main.env = Envelope(y, sr) if y is not None else None
        self.ghost.env = Envelope(ghost, sr) if ghost is not None else None
        self.redraw()

    def readout(self, t, y):
        env = self.main.env
        txt = f"{format_time(t, 3)}   {y:+.3f}"
        if env is not None and env.sr:
            i = int(np.clip(t * env.sr, 0, len(env.y) - 1))
            lo = max(0, i - env.sr // 100)
            level = 20 * np.log10(np.sqrt(np.mean(env.y[lo:i + 1] ** 2)) + 1e-9)
            txt += f"   signal {level:.0f} dBFS"
        return txt

    def redraw(self, *_):
        t0, t1 = self.vb.viewRange()[0]
        points = max(200, int(self.width()))
        self.ghost.render(t0, t1, points)
        self.main.render(t0, t1, points)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if hasattr(self, "main"):      # pyqtgraph resizes once before __init__ finishes
            self.redraw()


# --------------------------------------------------------------- spectrogram


class SpectrogramView(TimePlot):
    """Spectrogram with a linear or log frequency axis, automatic colour range
    and an adjustable dynamic range."""

    def __init__(self):
        super().__init__(left_axis=FreqAxis("left"))
        self.image = pg.ImageItem()
        self.image.setAutoDownsample(True)      # average, don't alias, when zoomed out
        self.image.setColorMap(th.spectro_colormap())
        self.image.setZValue(-20)
        self.addItem(self.image)
        self.levels = (-95.0, -15.0)
        self.dynamic_range = 70.0
        self.log_freq = False
        self.fmax = 8000.0
        self.sr = 44100
        self._live = None
        self._S = None
        self._hop = 256
        self.freq_line = pg.InfiniteLine(0, angle=0, movable=False,
                                         pen=pg.mkPen(th.INK, width=1, style=Qt.PenStyle.DashLine))
        self.freq_line.setZValue(5)
        self.freq_line.hide()
        self.addItem(self.freq_line)

    # ---- coordinates: the y axis is Hz, or log10(Hz) in log mode
    def to_y(self, freq):
        return np.log10(max(freq, 1.0)) if self.log_freq else freq

    def to_freq(self, y):
        return 10 ** y if self.log_freq else y

    def mark_frequency(self, freq):
        if freq is None:
            self.freq_line.hide()
        else:
            self.freq_line.setValue(self.to_y(freq))
            self.freq_line.show()

    def set_audio(self, S_db: np.ndarray, hop: int, sr: int):
        self._live = None
        self.sr = sr
        self._S, self._hop = S_db, hop
        self._draw()

    def _draw(self):
        if self._S is None:
            return
        S, hop, sr = self._S, self._hop, self.sr
        self.levels = auto_levels(S, self.dynamic_range)
        n_fft = 2 * (S.shape[0] - 1)
        width = S.shape[1] * hop / sr
        if self.log_freq:
            img, y0, y1 = to_log_rows(S, sr, n_fft)
        else:
            img, y0, y1 = S, 0.0, sr / 2
        self.getAxis("left").log_hz = self.log_freq
        self.image.setImage(img, levels=self.levels, autoLevels=False)
        self.image.setRect(QRectF(-hop / (2 * sr), y0, width, y1 - y0))
        self.set_max_freq(self.fmax)

    def set_freq_scale(self, log: bool):
        self.log_freq = bool(log)
        self._draw()

    def set_dynamic_range(self, db: float):
        self.dynamic_range = float(db)
        if self._live is not None:
            self.levels = (self.levels[1] - self.dynamic_range, self.levels[1])
            self.image.setLevels(self.levels)
        else:
            self._draw()

    def readout(self, t, y):
        f = self.to_freq(y)
        txt = f"{format_time(t, 3)}   {format_hz(f)}Hz"
        lvl = self.level_at(t, f)
        if lvl is not None:
            txt += f"   {lvl:.0f} dB"
        return txt

    def level_at(self, t: float, freq: float):
        """dB in the spectrogram under the cursor, or None outside the data."""
        if self._S is None:
            return None
        col = int(round(t * self.sr / self._hop))
        row = int(round(freq * 2 * (self._S.shape[0] - 1) / self.sr))
        if not (0 <= col < self._S.shape[1] and 0 <= row < self._S.shape[0]):
            return None
        return float(self._S[row, col])

    def clear(self):
        self.image.clear()
        self._S = None

    def set_max_freq(self, fmax: float):
        self.fmax = fmax
        top = min(fmax, self.sr / 2)
        self.setYRange(self.to_y(20.0) if self.log_freq else 0.0, self.to_y(top), padding=0)

    # live mode: a rolling spectrogram fed with microphone blocks
    def begin_live(self, sr: int, window_s: float = 10.0):
        self.log_freq = False               # the rolling buffer is drawn linearly
        self.getAxis("left").log_hz = False
        self._S = None
        self.levels = (-15.0 - self.dynamic_range, -15.0)
        n_fft = 1024 if sr > 24000 else 512
        hop = n_fft // 2
        cols = int(window_s * sr / hop)
        self.sr = sr
        self._live = dict(n_fft=n_fft, hop=hop, win=hann(n_fft),
                          scale=2.0 / hann(n_fft).sum(), pending=np.zeros(0),
                          buf=np.full((n_fft // 2 + 1, cols), -140.0, dtype=np.float32),
                          frames=0)
        self.set_max_freq(self.fmax)

    def live_append(self, samples: np.ndarray):
        L = self._live
        if L is None:
            return
        L["pending"] = np.concatenate([L["pending"], samples])
        n_fft, hop = L["n_fft"], L["hop"]
        count = (len(L["pending"]) - n_fft) // hop + 1
        if count <= 0:
            return
        idx = np.arange(count)[:, None] * hop + np.arange(n_fft)[None, :]
        spec = np.abs(rfft(L["pending"][idx] * L["win"])) * L["scale"]
        cols = (20 * np.log10(spec + 1e-9)).T.astype(np.float32)
        L["pending"] = L["pending"][count * hop:]
        buf = L["buf"]
        k = min(count, buf.shape[1])
        buf[:, :-k] = buf[:, k:]
        buf[:, -k:] = cols[:, -k:]
        L["frames"] += count
        t_end = L["frames"] * hop / self.sr
        width = buf.shape[1] * hop / self.sr
        self.image.setImage(buf, levels=self.levels, autoLevels=False)
        self.image.setRect(QRectF(t_end - width, 0, width, self.sr / 2))


# ------------------------------------------------------- real-time mini views


class SpectrumView(pg.PlotWidget):
    """Magnitude spectrum of the frame under the playhead, log frequency axis.
    Clicking a frequency emits sigFreqClicked(Hz)."""

    N = 4096
    sigFreqClicked = Signal(float)

    def __init__(self):
        super().__init__()
        plot = self.getPlotItem()
        th.style_plot(plot, left_width=40)
        self.setMouseEnabled(x=False, y=False)
        self.setLogMode(x=True, y=False)
        self.setYRange(-100, 0, padding=0)
        self.setMinimumHeight(120)
        ticks = [(np.log10(f), format_hz(f)) for f in (50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000)]
        self.getAxis("bottom").setTicks([ticks])
        self.getAxis("left").setTicks([[(v, str(v)) for v in (-100, -80, -60, -40, -20, 0)]])
        self.peak = pg.PlotDataItem(pen=pg.mkPen(th.GHOST, width=1))
        self.curve = pg.PlotDataItem(pen=pg.mkPen(th.TEAL, width=1.6),
                                     fillLevel=-140, brush=th.alpha(th.TEAL, 38))
        self.addItem(self.peak)
        self.addItem(self.curve)
        self.marker = pg.InfiniteLine(0, angle=90, movable=False,
                                      pen=pg.mkPen(th.INK, width=1, style=Qt.PenStyle.DashLine))
        self.marker.hide()
        self.addItem(self.marker)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.scene().sigMouseClicked.connect(self._clicked)
        self.win = hann(self.N)
        self._scale = 2.0 / self.win.sum()
        self._sr = None
        self._smooth = None
        self._hold = None

    def _clicked(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton or self._sr is None:
            return
        vb = self.getPlotItem().getViewBox()
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        freq = float(10 ** vb.mapSceneToView(ev.scenePos()).x())
        freq = float(np.clip(freq, 20.0, self._sr / 2))
        self.mark_frequency(freq)
        self.sigFreqClicked.emit(freq)

    def mark_frequency(self, freq):
        if freq is None:
            self.marker.hide()
        else:
            self.marker.setValue(np.log10(freq))
            self.marker.show()

    def update_frame(self, frame: np.ndarray, sr: int) -> float:
        """Draws the spectrum and returns the loudest frequency in Hz."""
        if sr != self._sr:
            self._sr = sr
            self._freqs = rfftfreq(self.N, 1 / sr)[1:]
            self.setXRange(np.log10(20), np.log10(sr / 2), padding=0)
            self._smooth = self._hold = None
        mag = np.abs(rfft(frame * self.win))[1:] * self._scale
        spec = 20 * np.log10(mag + 1e-9)
        self._smooth = spec if self._smooth is None else np.maximum(spec, 0.55 * self._smooth + 0.45 * spec)
        self._hold = self._smooth if self._hold is None else np.maximum(self._smooth, self._hold - 0.6)
        self.curve.setData(self._freqs, self._smooth)
        self.peak.setData(self._freqs, self._hold)
        return float(self._freqs[int(np.argmax(spec))])

    def reset(self):
        self._smooth = self._hold = None
        self.curve.setData([], [])
        self.peak.setData([], [])


class ScopeView(pg.PlotWidget):
    """Oscilloscope: the few milliseconds of waveform around the playhead."""

    def __init__(self, span_ms: float = 60.0):
        super().__init__()
        th.style_plot(self.getPlotItem(), left_width=40)
        self.setMouseEnabled(x=False, y=False)
        self.setMinimumHeight(120)
        self.span_ms = span_ms
        self.setXRange(-span_ms / 2, span_ms / 2, padding=0)
        self.getAxis("bottom").setLabel("ms")
        self.addItem(pg.InfiniteLine(0, angle=90, pen=pg.mkPen(th.MARIGOLD, width=1.5,
                                                             style=Qt.PenStyle.DashLine)))
        self.curve = pg.PlotDataItem(pen=pg.mkPen(th.TEAL, width=1.6))
        self.addItem(self.curve)
        self._ylim = 0.1

    def samples_needed(self, sr: int) -> int:
        return int(sr * self.span_ms / 1000)

    def update_frame(self, seg: np.ndarray, sr: int):
        n = len(seg)
        x = (np.arange(n) - n / 2) / sr * 1000
        self.curve.setData(x, seg)
        target = max(0.02, float(np.max(np.abs(seg))) * 1.15) if n else 0.1
        self._ylim = max(target, self._ylim * 0.92)
        self.setYRange(-self._ylim, self._ylim, padding=0)

    def reset(self):
        self.curve.setData([], [])
