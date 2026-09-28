"""Widgets for the sidebar and panel chrome."""
from __future__ import annotations

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QFrame, QGridLayout,
                               QHBoxLayout, QLabel, QProgressBar, QPushButton, QSlider,
                               QVBoxLayout, QWidget)

from ..algorithms import Param


def label(text="", name="", wrap=False) -> QLabel:
    w = QLabel(text)
    if name:
        w.setObjectName(name)
    w.setWordWrap(wrap)
    return w


def divider(vertical=False) -> QFrame:
    f = QFrame()
    f.setObjectName("vdivider" if vertical else "divider")
    return f


class Sheet(QFrame):
    """White panel with a title row; `right` holds header widgets."""

    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("sheet")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(6)
        self.header = QHBoxLayout()
        self.header.setSpacing(8)
        self.title = label(title, "sheetTitle")
        self.header.addWidget(self.title)
        self.header.addStretch(1)
        outer.addLayout(self.header)
        self.body = QVBoxLayout()
        self.body.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(self.body, 1)

    def add_right(self, widget: QWidget):
        self.header.addWidget(widget)


class ButtonGroupBar(QWidget):
    """Exclusive checkable buttons. `style` is 'segment' or 'chip'."""

    changed = Signal(int)

    def __init__(self, names: list[str], style="segment", columns: int | None = None):
        super().__init__()
        if style == "segment":
            self.setObjectName("segmentBar")
            self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        grid = QGridLayout(self)
        pad = 3 if style == "segment" else 0
        grid.setContentsMargins(pad, pad, pad, pad)
        grid.setSpacing(2 if style == "segment" else 6)
        cols = columns or len(names)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        for i, name in enumerate(names):
            b = QPushButton(name)
            b.setObjectName(style)
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            self.group.addButton(b, i)
            grid.addWidget(b, i // cols, i % cols)
        self.group.button(0).setChecked(True)
        self.group.idClicked.connect(self.changed)


class ParamForm(QWidget):
    """Builds labelled controls from a list of Param specs."""

    STEPS = 1000

    def __init__(self, params: list[Param]):
        super().__init__()
        self.params = {p.key: p for p in params}
        self.getters = {}
        self.rows = {}
        self.setters = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        for p in params:
            row = QWidget()
            self.rows[p.key] = row
            lay.addWidget(row)
            builder = {"float": self._numeric, "int": self._numeric,
                       "choice": self._choice, "bool": self._bool}[p.kind]
            builder(p, row)
            if p.tip:
                row.setToolTip(p.tip)
        self._refresh_visibility()

    # -- numeric: label, live value, slider (linear or logarithmic)
    @staticmethod
    def _log_safe(p: Param) -> bool:
        """A logarithmic slider needs a positive range; fall back to linear
        rather than raising when a parameter is declared with lo <= 0."""
        return p.log and p.lo > 0 and p.hi > 0

    def _to_slider(self, p: Param, v: float) -> int:
        if self._log_safe(p):
            v = max(v, p.lo)
            u = (math.log(v) - math.log(p.lo)) / (math.log(p.hi) - math.log(p.lo))
        else:
            u = (v - p.lo) / (p.hi - p.lo)
        return round(min(max(u, 0.0), 1.0) * self.STEPS)

    def _from_slider(self, p: Param, i: int):
        u = i / self.STEPS
        v = math.exp(math.log(p.lo) + u * (math.log(p.hi) - math.log(p.lo))) \
            if self._log_safe(p) else p.lo + u * (p.hi - p.lo)
        return int(round(v)) if p.kind == "int" else v

    @staticmethod
    def _fmt(p: Param, v) -> str:
        if p.kind == "int":
            s = f"{v}"
        elif abs(v) < 0.01 and v != 0:
            s = f"{v:.1e}"
        elif abs(v) < 10:
            s = f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) < 1 else f"{v:.2f}"
        else:
            s = f"{v:.1f}"
        return f"{s} {p.unit}".strip()

    def _numeric(self, p: Param, row: QWidget):
        v = QVBoxLayout(row)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label(p.label))
        top.addStretch(1)
        value = label(self._fmt(p, p.default), "paramValue")
        top.addWidget(value)
        v.addLayout(top)
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(0, self.STEPS)
        s.setValue(self._to_slider(p, p.default))
        s.valueChanged.connect(lambda i: value.setText(self._fmt(p, self._from_slider(p, i))))
        v.addWidget(s)
        self.getters[p.key] = lambda: self._from_slider(p, s.value())
        self.setters[p.key] = lambda v, p=p, s=s: s.setValue(self._to_slider(p, v))

    def _choice(self, p: Param, row: QWidget):
        v = QVBoxLayout(row)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        v.addWidget(label(p.label))
        c = QComboBox()
        c.addItems([str(x) for x in p.choices])
        c.setCurrentText(str(p.default))
        c.currentTextChanged.connect(lambda _: self._refresh_visibility())
        v.addWidget(c)
        self.getters[p.key] = c.currentText
        self.setters[p.key] = c.setCurrentText

    def _bool(self, p: Param, row: QWidget):
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(label(p.label))
        h.addStretch(1)
        c = QCheckBox()
        c.setChecked(bool(p.default))
        c.toggled.connect(lambda _: self._refresh_visibility())
        h.addWidget(c)
        self.getters[p.key] = c.isChecked
        self.setters[p.key] = c.setChecked

    def _refresh_visibility(self):
        for key, p in self.params.items():
            if p.show_if and p.show_if[0] in self.getters:
                other, allowed = p.show_if
                self.rows[key].setVisible(self.getters[other]() in allowed)

    def values(self) -> dict:
        return {k: g() for k, g in self.getters.items()}

    def set_value(self, key, value):
        """Used by tests and presets; ignores keys this form does not have."""
        if key in self.setters:
            self.setters[key](value)
            self._refresh_visibility()


class StemList(QWidget):
    """One row per separated stem: listen, keep as working copy, export."""

    listen = Signal(str)
    keep = Signal(str)
    export = Signal(str)

    def __init__(self):
        super().__init__()
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(6)

    def set_stems(self, names: list[str]):
        while self.lay.count():
            item = self.lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for name in names:
            row = QFrame()
            row.setObjectName("sheet")
            h = QHBoxLayout(row)
            h.setContentsMargins(10, 6, 6, 6)
            h.setSpacing(4)
            title = label(name)
            title.setWordWrap(True)
            h.addWidget(title, 1)
            for text, sig, tip in (("Listen", self.listen, "Hear and view this stem"),
                                   ("Keep", self.keep, "Make this stem the working copy"),
                                   ("Export", self.export, "Save this stem as an audio file")):
                b = QPushButton(text)
                b.setObjectName("flat")
                b.setToolTip(tip)
                b.setCursor(Qt.CursorShape.PointingHandCursor)
                b.clicked.connect(lambda _=False, n=name, s=sig: s.emit(n))
                h.addWidget(b)
            self.lay.addWidget(row)


class LevelMeter(QProgressBar):
    """Horizontal level meter in dBFS (-60 to 0)."""

    def __init__(self):
        super().__init__()
        self.setObjectName("meter")
        self.setRange(0, 600)
        self.setTextVisible(False)
        self.setMinimumWidth(50)
        self.setMaximumWidth(110)
        self._level = -60.0

    def set_db(self, value_db: float):
        self._level = max(value_db, self._level - 1.5)     # fall back smoothly
        self.setValue(int((min(max(self._level, -60.0), 0.0) + 60) * 10))
        self.setToolTip(f"{self._level:.1f} dBFS")
