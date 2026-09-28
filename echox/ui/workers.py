"""Small building blocks of the main window: background task, growable
recording buffer, and one sidebar page per section."""
from __future__ import annotations

import re
import traceback

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget

from ..algorithms import UserError
from .controls import ButtonGroupBar, ParamForm, label


class Task(QThread):
    """Runs one algorithm off the GUI thread."""

    progressed = Signal(float, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn(lambda v, text="": self.progressed.emit(v, text)))
        except UserError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class GrowBuffer:
    """Append-only float buffer for live recording (amortised O(1) appends)."""

    def __init__(self, capacity: int):
        self.data = np.zeros(capacity, dtype=np.float32)
        self.n = 0

    def extend(self, x: np.ndarray):
        need = self.n + len(x)
        if need > len(self.data):
            grown = np.zeros(max(need, 2 * len(self.data)), dtype=np.float32)
            grown[: self.n] = self.data[: self.n]
            self.data = grown
        self.data[self.n:need] = x
        self.n = need

    def view(self) -> np.ndarray:
        return self.data[: self.n]


def centered(arr: np.ndarray, center: int, n: int) -> np.ndarray:
    """n samples around `center`, zero-padded at the edges."""
    a = center - n // 2
    out = np.zeros(n, dtype=np.float64)
    lo, hi = max(a, 0), min(a + n, len(arr))
    if hi > lo:
        out[lo - a:hi - a] = arr[lo:hi]
    return out


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


class SectionPage(QWidget):
    """Sidebar page for one section: algorithm chips, form, run button."""

    def __init__(self, name, algorithms):
        super().__init__()
        self.name = name
        self.algorithms = algorithms
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(2, 14, 10, 8)
        self.lay.setSpacing(12)
        self.chips = None
        if len(algorithms) > 1:
            cols = 3 if len(algorithms) == 3 else 2
            self.chips = ButtonGroupBar([a.name for a in algorithms], "chip", columns=cols)
            self.chips.changed.connect(self.select)
            self.lay.addWidget(self.chips)
        self.family = label("", "badge")
        self.family.setToolTip("Where the method comes from. Everything here is classical "
                               "signal processing or unsupervised matrix factorisation - no "
                               "pretrained neural networks are used.")
        self.lay.addWidget(self.family)
        self.summary = label("", "algoSummary", wrap=True)
        self.lay.addWidget(self.summary)
        self.forms = [ParamForm(a.params) for a in algorithms]
        for f in self.forms:
            self.lay.addWidget(f)
        self.hint = label("", "hint", wrap=True)
        self.lay.addWidget(self.hint)
        self.run = QPushButton()
        self.run.setObjectName("primary")
        self.run.setCursor(Qt.CursorShape.PointingHandCursor)
        self.lay.addWidget(self.run)
        self.index = 0
        self.select(0)

    def select(self, i: int):
        self.index = i
        a = self.algorithms[i]
        for j, f in enumerate(self.forms):
            f.setVisible(j == i)
        self.family.setText(a.family.upper())
        self.summary.setText(a.summary)
        self.hint.setText(a.uses_selection)
        self.hint.setVisible(bool(a.uses_selection))
        self.run.setText(f"Run {a.name}")

    @property
    def algorithm(self):
        return self.algorithms[self.index]

    def values(self) -> dict:
        return self.forms[self.index].values()
