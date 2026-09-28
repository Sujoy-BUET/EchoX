"""Widget construction for MainWindow (header, sidebar, views, transport).
Kept apart from window.py, which holds the behaviour."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QComboBox, QFrame, QHBoxLayout, QMenu, QProgressBar, QPushButton,
                               QScrollArea, QSizePolicy, QSlider, QStackedWidget, QVBoxLayout, QWidget)

from .. import theme as th
from ..algorithms import SECTIONS, Param
from ..dsp.common import fft_engine
from ..dsp.testbench import NOISE_TYPES
from .controls import ButtonGroupBar, LevelMeter, ParamForm, Sheet, StemList, divider, label
from .icons import icon
from .plots import ScopeView, SpectrogramView, SpectrumView, WaveformView
from .workers import SectionPage


class LayoutMixin:
    """Builds every widget and stores it on self. Mixed into MainWindow."""

    def _build(self):
        root = QWidget()
        root.setObjectName("desk")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(18, 14, 18, 14)
        outer.setSpacing(14)
        outer.addLayout(self._build_header())
        body = QHBoxLayout()
        body.setSpacing(18)
        body.addWidget(self._build_sidebar())
        self.stage = QStackedWidget()
        self.stage.addWidget(self._build_empty())
        self.stage.addWidget(self._build_views())
        body.addWidget(self.stage, 1)
        outer.addLayout(body, 1)
        outer.addWidget(self._build_transport())

    def _build_header(self):
        h = QHBoxLayout()
        h.setSpacing(8)
        h.addWidget(label("EchoX", "brand"))
        h.addSpacing(6)
        h.addWidget(label("audio signal workbench", "brandSub"), 0, Qt.AlignmentFlag.AlignBottom)
        h.addStretch(1)
        self.btn_open = QPushButton("Open")
        self.btn_open.clicked.connect(self.open_dialog)
        self.btn_record = QPushButton(" Record")
        self.btn_record.setObjectName("record")
        self.btn_record.setCheckable(True)
        self.btn_record.setIcon(icon("dot", th.RECORD))
        self.btn_record.toggled.connect(self.toggle_record)
        self.btn_save = QPushButton("Save")
        self.btn_save.clicked.connect(self.save_active)
        self.btn_undo = QPushButton("Undo")
        self.btn_undo.clicked.connect(self.undo)
        self.btn_redo = QPushButton("Redo")
        self.btn_redo.clicked.connect(self.redo)
        self.listen_combo = QComboBox()
        self.listen_combo.setPlaceholderText("Nothing loaded")
        self.listen_combo.setMinimumWidth(170)
        self.listen_combo.setToolTip("Which signal you hear and see")
        self.listen_combo.currentTextChanged.connect(self._on_listen_changed)
        for w in (self.btn_open, self.btn_record, self.btn_save):
            h.addWidget(w)
        h.addSpacing(6)
        h.addWidget(divider(vertical=True))
        h.addSpacing(6)
        h.addWidget(self.btn_undo)
        h.addWidget(self.btn_redo)
        h.addSpacing(6)
        h.addWidget(divider(vertical=True))
        h.addSpacing(6)
        h.addWidget(label("Listen to", "muted"))
        h.addWidget(self.listen_combo)
        h.addSpacing(6)
        h.addWidget(divider(vertical=True))
        h.addSpacing(6)
        self.fft_combo = QComboBox()
        self.fft_combo.addItems(["Own FFT", "NumPy FFT"])
        self.fft_combo.setCurrentIndex(0 if fft_engine() == "own" else 1)
        self.fft_combo.setToolTip("Transform engine for every algorithm and view.\n"
                                  "Own = hand-written radix-2 FFT in echox/dsp/transforms.py")
        self.fft_combo.currentIndexChanged.connect(self._on_fft_engine)
        h.addWidget(self.fft_combo)
        return h

    def _build_sidebar(self):
        side = QWidget()
        side.setObjectName("sidebar")
        fm = side.fontMetrics()
        tabs = sum(fm.horizontalAdvance(t) for t in ("Denoise", "Separate", "Effects", "Amplify"))
        side.setFixedWidth(max(330, tabs + 4 * 34 + 24))
        v = QVBoxLayout(side)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)
        self.section_bar = ButtonGroupBar(list(SECTIONS), "segment")
        self.section_bar.changed.connect(self._select_section)
        v.addWidget(self.section_bar)

        content = QWidget()
        cv = QVBoxLayout(content)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)
        self.pages: list[SectionPage] = []
        for name, algos in SECTIONS.items():
            page = SectionPage(name, algos)
            page.run.clicked.connect(lambda _=False, p=page: self.run_page(p))
            self.pages.append(page)
            cv.addWidget(page)
        self.pages[0].lay.addWidget(self._build_testbench())
        self.pages[1].lay.addWidget(self._build_stems())
        # the second file is used by Effects too (Mix in a file, Concatenate),
        # so it gets its own panel there rather than being hidden under Denoise
        self.pages[2].lay.addWidget(self._build_second_file())
        for page in self.pages:
            page.lay.addStretch(1)
        cv.addStretch(1)

        self.side_scroll = QScrollArea()
        self.side_scroll.setWidgetResizable(True)
        self.side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.side_scroll.setWidget(content)
        v.addWidget(self.side_scroll, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.hide()
        self.status_label = label("", "status", wrap=True)
        self.metric_label = label("", "metric", wrap=True)
        self.error_label = label("", "error", wrap=True)
        for w in (self.progress, self.status_label, self.metric_label, self.error_label):
            v.addWidget(w)
        self.metric_label.hide()
        self.error_label.hide()
        self._select_section(0)
        return side

    def _build_testbench(self):
        box = QFrame()
        box.setObjectName("sheet")
        v = QVBoxLayout(box)
        v.setContentsMargins(12, 10, 12, 12)
        v.setSpacing(8)
        v.addWidget(label("Test bench", "sheetTitle"))
        v.addWidget(label("Add known noise to a clean clip, and every denoiser is scored in "
                          "dB SNR against the clean copy.",
                          "hint", wrap=True))
        self.tb_form = ParamForm([
            Param("noise", "Noise", "choice", "White", choices=tuple(NOISE_TYPES)),
            Param("snr", "Starting SNR", "float", 5.0, -10.0, 30.0, unit="dB"),
        ])
        v.addWidget(self.tb_form)
        row = QHBoxLayout()
        self.btn_add_noise = QPushButton("Add noise")
        self.btn_add_noise.clicked.connect(self.add_test_noise)
        self.btn_load_ref = QPushButton("Load second file")
        self.btn_load_ref.setToolTip("Loads once, usable from any section: as Wiener's noise "
                                     "profile (Noise profile = Noise file), or as the track "
                                     "Effects -> Mix in a file adds in.")
        self.btn_load_ref.clicked.connect(self.load_reference)
        row.addWidget(self.btn_add_noise)
        row.addWidget(self.btn_load_ref)
        v.addLayout(row)
        self.tb_state = label("", "hint", wrap=True)
        v.addWidget(self.tb_state)
        self._update_testbench_state()
        return box

    def _build_second_file(self):
        panel = Sheet("Second file")
        v = panel.body
        v.addWidget(label("Needed by 'Mix in a file' and 'Concatenate'. The same file is also "
                          "offered to Wiener as a noise profile.", "hint", wrap=True))
        self.btn_load_ref2 = QPushButton("Load second file")
        self.btn_load_ref2.clicked.connect(self.load_reference)
        v.addWidget(self.btn_load_ref2)
        self.second_state = label("Nothing loaded yet.", "hint", wrap=True)
        v.addWidget(self.second_state)
        return panel

    def _build_stems(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 6, 0, 0)
        v.setSpacing(8)
        v.addWidget(label("Stems", "sheetTitle"))
        self.stems_hint = label("Separated parts appear here. Listen shows a stem in the views, "
                                "Keep makes it the working copy.", "hint", wrap=True)
        v.addWidget(self.stems_hint)
        self.stem_list = StemList()
        self.stem_list.listen.connect(self.listen_combo.setCurrentText)
        self.stem_list.keep.connect(self.keep_stem)
        self.stem_list.export.connect(self.export_stem)
        v.addWidget(self.stem_list)
        return w

    def _build_empty(self):
        w = QWidget()
        outer = QVBoxLayout(w)
        outer.addStretch(2)
        card = QFrame()
        card.setObjectName("sheet")
        card.setFixedWidth(470)
        card.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)
        v = QVBoxLayout(card)
        v.setContentsMargins(28, 26, 28, 26)
        v.setSpacing(12)
        v.setSizeConstraint(QVBoxLayout.SizeConstraint.SetMinimumSize)
        inner = 470 - 2 * 28          # wrapped labels must know their width up front,
        #                               or Qt sizes the card for one long line
        title = label("Start with a sound", "brand")
        v.addWidget(title)
        intro = label("Open a recording or capture one from your microphone. You will see it "
                      "in time and in frequency, and every tool on the left works on it.",
                      "muted", wrap=True)
        intro.setFixedWidth(inner)
        intro.setMinimumHeight(intro.heightForWidth(inner))
        v.addWidget(intro)
        row = QHBoxLayout()
        b_open = QPushButton("Open audio file")
        b_open.setObjectName("primary")
        b_open.clicked.connect(self.open_dialog)
        b_rec = QPushButton(" Record")
        b_rec.setObjectName("record")
        b_rec.setIcon(icon("dot", th.RECORD))
        b_rec.clicked.connect(lambda: self.btn_record.setChecked(True))
        for b in (b_open, b_rec):
            b.setMinimumHeight(b.sizeHint().height())
        row.addWidget(b_open)
        row.addWidget(b_rec)
        row.addStretch(1)
        v.addLayout(row)
        formats = label("WAV, FLAC, OGG, MP3 and (with ffmpeg) M4A work. You can also drop a "
                        "file on this window.", "hint", wrap=True)
        formats.setFixedWidth(inner)
        formats.setMinimumHeight(formats.heightForWidth(inner))
        v.addWidget(formats)
        outer.addWidget(card, 0, Qt.AlignmentFlag.AlignHCenter)
        outer.addStretch(3)
        return w

    def _build_views(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(12)

        self.wave_sheet = Sheet("Waveform")
        self.sel_label = label("Click to move, drag to select, scroll to zoom", "muted")
        self.wave_sheet.add_right(self.sel_label)
        fit = QPushButton("Show all")
        fit.setObjectName("flat")
        fit.clicked.connect(self.fit_view)
        self.wave_sheet.add_right(fit)
        self.wave = WaveformView()
        self.wave_sheet.body.addWidget(self.wave)

        self.spec_sheet = Sheet("Spectrogram")
        self.cursor_label = label("", "muted")
        self.wave_sheet.add_right(self.cursor_label)
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["Linear Hz", "Log Hz"])
        self.scale_combo.setCurrentText("Log Hz")
        self.scale_combo.setToolTip("Log spreads the low octaves out, where most music sits")
        self.spec_sheet.add_right(self.scale_combo)
        self.range_combo = QComboBox()
        self.range_combo.addItems(["50 dB", "70 dB", "90 dB", "110 dB"])
        self.range_combo.setCurrentText("70 dB")
        self.range_combo.setToolTip("Contrast: how far below the loudest content stays visible")
        self.spec_sheet.add_right(self.range_combo)
        self.fmax_combo = QComboBox()
        self.fmax_combo.addItems(["4 kHz", "8 kHz", "11 kHz", "Full range"])
        self.fmax_combo.setCurrentText("Full range")
        self.spec_sheet.add_right(self.fmax_combo)
        self.spectro = SpectrogramView()
        self.spectro.set_freq_scale(self.scale_combo.currentText().startswith("Log"))
        self.spectro.set_dynamic_range(float(self.range_combo.currentText().split()[0]))
        self.scale_combo.currentTextChanged.connect(self._on_freq_scale)
        self.range_combo.currentTextChanged.connect(self._on_dynamic_range)
        self.fmax_combo.currentTextChanged.connect(self._on_fmax)
        self.spectro.setXLink(self.wave)
        self.spec_sheet.body.addWidget(self.spectro)

        row = QHBoxLayout()
        row.setSpacing(12)
        spec_sheet = Sheet("Spectrum at playhead")
        self.peak_label = label("", "muted")
        spec_sheet.add_right(label("Click to jump  ·", "muted"))
        spec_sheet.add_right(self.peak_label)
        self.spectrum = SpectrumView()
        self.spectrum.setToolTip("Click a frequency: the playhead jumps to the next moment that band gets loud.\n"
                                 "Click again for the one after.")
        spec_sheet.body.addWidget(self.spectrum)
        scope_sheet = Sheet("Scope")
        scope_sheet.add_right(label("60 ms around the playhead", "muted"))
        self.scope = ScopeView(60.0)
        scope_sheet.body.addWidget(self.scope)
        row.addWidget(spec_sheet, 3)
        row.addWidget(scope_sheet, 2)

        v.addWidget(self.wave_sheet, 3)
        v.addWidget(self.spec_sheet, 5)
        v.addLayout(row, 3)

        for view in (self.wave, self.spectro):
            view.sigSeek.connect(self.seek_time)
            view.sigSelect.connect(self.on_select)
            view.sigCursor.connect(self._on_cursor)
        self.spectrum.sigFreqClicked.connect(self.jump_to_frequency)
        return w

    def _build_transport(self):
        bar = QFrame()
        bar.setObjectName("sheet")
        h = QHBoxLayout(bar)
        h.setContentsMargins(14, 8, 18, 8)
        h.setSpacing(10)
        self.btn_start = QPushButton()
        self.btn_start.setObjectName("transport")
        self.btn_start.setIcon(icon("start", th.INK))
        self.btn_start.setToolTip("Back to start (Home)")
        self.btn_start.clicked.connect(lambda: self.seek_time(self.selection[0] if self.selection else 0.0))
        self.btn_play = QPushButton()
        self.btn_play.setObjectName("play")
        self.btn_play.setIcon(icon("play", "#FFFFFF", 22))
        self.btn_play.setToolTip("Play or pause (Space)")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_stop = QPushButton()
        self.btn_stop.setObjectName("transport")
        self.btn_stop.setIcon(icon("stop", th.INK))
        self.btn_stop.setToolTip("Stop")
        self.btn_stop.clicked.connect(self.stop)
        self.btn_loop = QPushButton("Loop")
        self.btn_loop.setObjectName("chip")
        self.btn_loop.setCheckable(True)
        self.btn_loop.setToolTip("Loop the selected region")
        self.btn_loop.toggled.connect(lambda _: self._update_loop())
        self.btn_ab = QPushButton("Original")
        self.btn_ab.setObjectName("chip")
        self.btn_ab.setCheckable(True)
        self.btn_ab.setToolTip("A/B against the file you loaded, without losing your place (Tab)")
        self.btn_ab.toggled.connect(self.toggle_original)
        for b in (self.btn_start, self.btn_play, self.btn_stop, self.btn_loop, self.btn_ab):
            b.setMinimumWidth(b.sizeHint().width())      # never clip the label
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            h.addWidget(b)
        self.speed_button = QPushButton("1x")
        self.speed_button.setObjectName("chip")
        self.speed_button.setToolTip("Playback speed. The pitch stays where it is.")
        self.speed_button.setFixedWidth(70)
        menu = QMenu(self.speed_button)
        for rate in (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0):
            menu.addAction(f"{rate:g}x", lambda r=rate: self.set_play_speed(r))
        self.speed_button.setMenu(menu)
        self.speed_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.speed_button.setCursor(Qt.CursorShape.PointingHandCursor)
        h.addWidget(self.speed_button)
        h.addSpacing(10)
        self.clock = label("0:00.000")
        self.clock.setFont(th.clock_font(16))
        self.total = label("/ 0:00.000", "muted")
        self.total.setFont(th.clock_font(11))
        h.addWidget(self.clock)
        h.addWidget(self.total)
        h.addSpacing(10)
        self.seek = QSlider(Qt.Orientation.Horizontal)
        self.seek.setObjectName("seek")
        self.seek.setRange(0, 10000)
        self.seek.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.seek.setMinimumWidth(120)
        self.seek.sliderMoved.connect(lambda v: self.seek_time(v / 10000 * self._duration()))
        h.addWidget(self.seek, 1)
        h.addSpacing(10)
        h.addWidget(label("Vol", "muted"))
        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setFixedWidth(96)
        self.volume.setRange(0, 200)
        self.volume.setValue(90)
        self.volume.setToolTip("Up to 200 %. Above 100 % a soft limiter catches the peaks, "
                               "so a quiet recording gets louder without crackling.")
        self.volume.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.volume.valueChanged.connect(self.set_volume)
        self.player.volume = 0.9
        h.addWidget(self.volume)
        self.volume_label = label("90%", "muted")
        self.volume_label.setFixedWidth(46)
        h.addWidget(self.volume_label)
        h.addSpacing(6)
        self.meter = LevelMeter()
        self.meter.setToolTip("Output level, -60 to 0 dBFS")
        h.addWidget(self.meter)
        return bar

    def _shortcuts(self):
        keys = {
            "Space": self.toggle_play, "Ctrl+O": self.open_dialog, "Ctrl+S": self.save_active,
            "Ctrl+Z": self.undo, "Ctrl+Shift+Z": self.redo, "Ctrl+Y": self.redo,
            "Ctrl+R": lambda: self.btn_record.toggle(), "Escape": lambda: self.on_select(None, True),
            "Home": lambda: self.seek_time(0.0),
            "Tab": lambda: self.btn_ab.toggle(),
        }
        for seq, fn in keys.items():
            QShortcut(QKeySequence(seq), self).activated.connect(fn)
