"""Look and feel: ink on lab paper, with a highlighter for time.

Signals are drawn in teal ink on white sheets laid over a pale paper desk.
Marigold is reserved for 'where you are' (playhead) and 'what you picked'
(selection), like a highlighter in a lab notebook."""
from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtGui import QColor, QFont

PAPER = "#EEF2F0"      # desk / window background
SHEET = "#FFFFFF"      # plot panels
HAIRLINE = "#D9E1DE"
INK = "#16262B"        # text
GRAPHITE = "#5D6D71"   # secondary text, axes
TEAL = "#1C7C74"       # the signal ink, primary actions
TEAL_SOFT = "#D7ECE7"
MARIGOLD = "#E3902A"   # playhead + selection
GHOST = "#C4CFCC"      # original signal behind the processed one
RECORD = "#CF4152"

UI_FONTS = ["IBM Plex Sans", "Inter", "Segoe UI", "SF Pro Text", "Noto Sans", "Ubuntu", "Arial"]
CLOCK_FONTS = ["IBM Plex Mono", "JetBrains Mono", "Cascadia Mono", "Consolas", "Menlo", "DejaVu Sans Mono"]

# Luminance falls monotonically: quiet bins are paper, loud bins are dark ink.
SPECTRO_STOPS = ["#FFFFFF", "#E3F3EF", "#9FD6CA", "#3FA193", "#1D6468", "#0F2F3B"]


def ui_font(size=10, weight=QFont.Weight.Normal) -> QFont:
    f = QFont()
    f.setFamilies(UI_FONTS)
    f.setPointSizeF(size)
    f.setWeight(weight)
    return f


def clock_font(size=15) -> QFont:
    f = QFont()
    f.setFamilies(CLOCK_FONTS)
    f.setPointSizeF(size)
    f.setWeight(QFont.Weight.Medium)
    return f


def spectro_colormap() -> pg.ColorMap:
    pos = [i / (len(SPECTRO_STOPS) - 1) for i in range(len(SPECTRO_STOPS))]
    return pg.ColorMap(pos, [QColor(c) for c in SPECTRO_STOPS])


def alpha(hex_color: str, a: int) -> QColor:
    c = QColor(hex_color)
    c.setAlpha(a)
    return c


def configure_pyqtgraph():
    pg.setConfigOptions(antialias=True, imageAxisOrder="row-major",
                        background=SHEET, foreground=GRAPHITE)


def style_plot(plot: pg.PlotItem, left_width=46):
    plot.showGrid(x=True, y=True, alpha=0.12)
    for name in ("left", "bottom"):
        ax = plot.getAxis(name)
        ax.setPen(pg.mkPen(HAIRLINE))
        ax.setTextPen(pg.mkPen(GRAPHITE))
        ax.setTickFont(ui_font(8))
    plot.getAxis("left").setWidth(left_width)
    plot.hideButtons()
    plot.setMenuEnabled(False)


STYLESHEET = f"""
* {{ color: {INK}; }}
QMainWindow, QWidget#desk {{ background: {PAPER}; }}
QWidget#sidebar {{ background: {PAPER}; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}

QFrame#sheet {{
    background: {SHEET};
    border: 1px solid {HAIRLINE};
    border-radius: 10px;
}}
QLabel#sheetTitle {{ color: {INK}; font-weight: 600; }}
QLabel#muted {{ color: {GRAPHITE}; }}
QLabel#badge {{
    color: {TEAL}; background: {TEAL_SOFT}; border-radius: 4px;
    padding: 2px 7px; font-size: 9px; font-weight: 700; letter-spacing: 0.08em;
}}
QLabel#hint {{ color: {GRAPHITE}; }}
QLabel#brand {{ font-size: 17pt; font-weight: 600; color: {INK}; }}
QLabel#brandSub {{ color: {GRAPHITE}; }}
QLabel#algoSummary {{ color: {GRAPHITE}; }}
QLabel#paramValue {{ color: {TEAL}; font-weight: 600; }}
QLabel#status {{ color: {GRAPHITE}; }}
QLabel#metric {{
    background: {TEAL_SOFT}; color: {TEAL}; border-radius: 6px; padding: 6px 10px; font-weight: 600;
}}
QLabel#error {{
    background: #F8E1E4; color: {RECORD}; border-radius: 6px; padding: 6px 10px;
}}

QMenu {{
    background: {SHEET}; color: {INK}; border: 1px solid {HAIRLINE}; border-radius: 8px;
    padding: 4px;
}}
QMenu::item {{ padding: 6px 22px 6px 14px; border-radius: 6px; color: {INK}; }}
QMenu::item:selected {{ background: {TEAL_SOFT}; color: {INK}; }}
QComboBox QAbstractItemView {{
    background: {SHEET}; color: {INK}; border: 1px solid {HAIRLINE};
    selection-background-color: {TEAL_SOFT}; selection-color: {INK}; outline: 0;
}}

QPushButton {{
    background: {SHEET}; border: 1px solid {HAIRLINE}; border-radius: 8px;
    padding: 6px 12px; color: {INK};
}}
QPushButton:hover {{ border-color: {TEAL}; }}
QPushButton:pressed {{ background: {TEAL_SOFT}; }}
QPushButton:disabled {{ color: #A9B4B1; border-color: #E6ECEA; }}
QPushButton:focus {{ border: 1px solid {TEAL}; }}

QPushButton#primary {{
    background: {TEAL}; color: white; border: none; font-weight: 600; padding: 9px 14px;
}}
QPushButton#primary:hover {{ background: #176A63; }}
QPushButton#primary:disabled {{ background: #9CC3BE; color: #F1F6F5; }}

QPushButton#record {{ color: {RECORD}; font-weight: 600; }}
QPushButton#record:checked {{ background: {RECORD}; color: white; border-color: {RECORD}; }}

QPushButton#transport {{
    border-radius: 18px; min-width: 36px; max-width: 36px; min-height: 36px; max-height: 36px;
    padding: 0; font-size: 13pt;
}}
QPushButton#play {{
    background: {INK}; color: white; border: none; border-radius: 22px;
    min-width: 44px; max-width: 44px; min-height: 44px; max-height: 44px; padding: 0; font-size: 15pt;
}}
QPushButton#play:hover {{ background: {TEAL}; }}

QPushButton#segment {{
    border: none; border-radius: 7px; background: transparent; padding: 7px 10px; color: {GRAPHITE};
}}
QPushButton#segment:checked {{ background: {SHEET}; color: {INK}; font-weight: 600; }}
QWidget#segmentBar {{ background: #E1E8E5; border-radius: 9px; }}

QPushButton#chip {{
    border-radius: 14px; padding: 5px 12px; background: transparent; color: {GRAPHITE};
    border: 1px solid {HAIRLINE};
}}
QPushButton#chip:checked {{ background: {TEAL}; color: white; border-color: {TEAL}; }}

QPushButton#flat {{ border: none; background: transparent; color: {TEAL}; padding: 4px 6px; }}
QPushButton#flat:hover {{ text-decoration: underline; }}

QComboBox {{
    background: {SHEET}; border: 1px solid {HAIRLINE}; border-radius: 7px; padding: 5px 9px;
}}
QComboBox:hover {{ border-color: {TEAL}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background: {SHEET}; border: 1px solid {HAIRLINE}; selection-background-color: {TEAL_SOFT};
    selection-color: {INK}; outline: none;
}}

QSlider::groove:horizontal {{ height: 4px; background: {HAIRLINE}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {TEAL}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {SHEET}; border: 2px solid {TEAL}; width: 12px; height: 12px;
    margin: -6px 0; border-radius: 8px;
}}
QSlider::handle:horizontal:hover {{ background: {TEAL_SOFT}; }}
QSlider#seek::sub-page:horizontal {{ background: {MARIGOLD}; }}
QSlider#seek::handle:horizontal {{ border-color: {MARIGOLD}; }}

QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{
    width: 30px; height: 16px; border-radius: 8px; background: {HAIRLINE};
}}
QCheckBox::indicator:checked {{ background: {TEAL}; }}

QProgressBar {{
    background: {HAIRLINE}; border: none; border-radius: 3px; max-height: 6px; min-height: 6px;
}}
QProgressBar::chunk {{ background: {TEAL}; border-radius: 3px; }}
QProgressBar#meter::chunk {{ background: {TEAL}; border-radius: 3px; }}

QToolTip {{ background: {INK}; color: white; border: none; padding: 5px 8px; border-radius: 4px; }}
QScrollBar:vertical {{ width: 8px; background: transparent; }}
QScrollBar::handle:vertical {{ background: {HAIRLINE}; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QFrame#divider {{ background: {HAIRLINE}; max-height: 1px; min-height: 1px; }}
QFrame#vdivider {{ background: {HAIRLINE}; max-width: 1px; min-width: 1px; }}
"""


def apply(app):
    """Same light look on every OS.

    Windows 11 in dark mode hands Qt a dark palette, and anything the style
    sheet does not paint itself (menus, combo pop-ups, scroll corners) comes out
    black with dark text. Fusion plus an explicit light palette pins it down."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QPalette
    try:
        app.styleHints().setColorScheme(Qt.ColorScheme.Light)     # Qt 6.8+
    except AttributeError:
        pass
    app.setStyle("Fusion")
    pal = QPalette()
    roles = {
        QPalette.ColorRole.Window: PAPER, QPalette.ColorRole.WindowText: INK,
        QPalette.ColorRole.Base: SHEET, QPalette.ColorRole.AlternateBase: PAPER,
        QPalette.ColorRole.Text: INK, QPalette.ColorRole.Button: SHEET,
        QPalette.ColorRole.ButtonText: INK, QPalette.ColorRole.Highlight: TEAL_SOFT,
        QPalette.ColorRole.HighlightedText: INK, QPalette.ColorRole.ToolTipBase: INK,
        QPalette.ColorRole.ToolTipText: SHEET, QPalette.ColorRole.PlaceholderText: GRAPHITE,
        QPalette.ColorRole.Link: TEAL,
    }
    for role, colour in roles.items():
        pal.setColor(role, QColor(colour))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                 QPalette.ColorRole.ButtonText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, QColor("#A9B4B1"))
    app.setPalette(pal)
    app.setFont(ui_font(10))
    app.setStyleSheet(STYLESHEET)
