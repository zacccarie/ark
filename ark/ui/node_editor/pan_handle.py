"""Pan handle flottant — overlay bottom-right pour translater canvas via drag souris."""
from __future__ import annotations
from PySide6.QtCore import Qt, Signal, QPoint, QPointF, QRectF
from PySide6.QtGui import QPainter, QBrush, QPen, QColor, QPainterPath
from PySide6.QtWidgets import QWidget

from ark.ui import design_tokens as T


class PanHandle(QWidget):
    """Petit carré rouge subtil en bas-droite du viewport.

    Drag → émet pan_delta(dx, dy) en coordonnées viewport (pixels).
    Hover → opacité augmente, indique qu'on peut grab.
    """

    SIZE = 44
    MARGIN = 16

    pan_delta = Signal(float, float)

    def __init__(self, parent):
        super().__init__(parent)
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self._dragging = False
        self._anchor = QPoint()
        self._hover = False
        self.setMouseTracking(True)
        self.reposition()
        self.raise_()

    def reposition(self):
        parent = self.parent()
        if parent is None:
            return
        w = parent.width()
        h = parent.height()
        self.move(w - self.SIZE - self.MARGIN, h - self.SIZE - self.MARGIN)

    # ----- Paint : croix 4-directions + cercle subtil -----
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(0, 0, self.SIZE, self.SIZE)
        # Background pill
        alpha = 220 if self._hover or self._dragging else 160
        bg = QColor(T.BG_OVERLAY)
        bg.setAlpha(alpha)
        p.setBrush(QBrush(bg))
        border_col = QColor(T.ACCENT if self._dragging else
                              (T.BORDER_STRONG if self._hover else T.BORDER_DEFAULT))
        p.setPen(QPen(border_col, 1.2))
        p.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 8, 8)
        # Cross 4 directions
        center = QPointF(self.SIZE / 2, self.SIZE / 2)
        arm = 9
        col = QColor(T.ACCENT if (self._hover or self._dragging) else T.FG_SECONDARY)
        p.setPen(QPen(col, 1.6))
        p.drawLine(center + QPointF(-arm, 0), center + QPointF(arm, 0))
        p.drawLine(center + QPointF(0, -arm), center + QPointF(0, arm))
        # Petits triangles aux extrémités
        for dx, dy in [(arm, 0), (-arm, 0), (0, arm), (0, -arm)]:
            tip = center + QPointF(dx, dy)
            path = QPainterPath()
            if dx != 0:
                sign = 1 if dx > 0 else -1
                path.moveTo(tip)
                path.lineTo(tip + QPointF(-3 * sign, -3))
                path.lineTo(tip + QPointF(-3 * sign, 3))
            else:
                sign = 1 if dy > 0 else -1
                path.moveTo(tip)
                path.lineTo(tip + QPointF(-3, -3 * sign))
                path.lineTo(tip + QPointF(3, -3 * sign))
            path.closeSubpath()
            p.fillPath(path, QBrush(col))

    def enterEvent(self, event):
        self._hover = True
        self.update()

    def leaveEvent(self, event):
        self._hover = False
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._anchor = event.globalPosition().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.update()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._dragging:
            pos = event.globalPosition().toPoint()
            delta = pos - self._anchor
            self._anchor = pos
            self.pan_delta.emit(float(delta.x()), float(delta.y()))
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self.update()
            event.accept()
