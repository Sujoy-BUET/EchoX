"""Small vector icons painted with QPainter (no icon font or image files)."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPixmap


def _canvas(size: int):
    pm = QPixmap(size * 2, size * 2)          # 2x for crisp HiDPI rendering
    pm.fill(Qt.GlobalColor.transparent)
    pm.setDevicePixelRatio(2.0)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    return pm, p


def icon(name: str, color: str, size: int = 18) -> QIcon:
    pm, p = _canvas(size)
    p.setBrush(QColor(color))
    s = float(size)
    if name == "play":
        path = QPainterPath(QPointF(s * 0.32, s * 0.2))
        path.lineTo(QPointF(s * 0.82, s * 0.5))
        path.lineTo(QPointF(s * 0.32, s * 0.8))
        path.closeSubpath()
        p.drawPath(path)
    elif name == "pause":
        p.drawRoundedRect(QRectF(s * 0.26, s * 0.2, s * 0.17, s * 0.6), 1.5, 1.5)
        p.drawRoundedRect(QRectF(s * 0.57, s * 0.2, s * 0.17, s * 0.6), 1.5, 1.5)
    elif name == "stop":
        p.drawRoundedRect(QRectF(s * 0.27, s * 0.27, s * 0.46, s * 0.46), 2, 2)
    elif name == "start":
        p.drawRoundedRect(QRectF(s * 0.22, s * 0.24, s * 0.12, s * 0.52), 1, 1)
        path = QPainterPath(QPointF(s * 0.78, s * 0.24))
        path.lineTo(QPointF(s * 0.38, s * 0.5))
        path.lineTo(QPointF(s * 0.78, s * 0.76))
        path.closeSubpath()
        p.drawPath(path)
    elif name == "dot":
        p.drawEllipse(QRectF(s * 0.3, s * 0.3, s * 0.4, s * 0.4))
    p.end()
    return QIcon(pm)
