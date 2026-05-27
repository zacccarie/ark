"""Socket item — point d'attache pour connexions."""
from __future__ import annotations
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QBrush, QPen, QColor
from PySide6.QtWidgets import QGraphicsItem

from ark.ui import design_tokens as T


SOCKET_COLOR_MAP = {
    "tensor": T.SOCKET_TENSOR,
    "number": T.SOCKET_NUMBER,
    "string": T.SOCKET_STRING,
    "image": T.SOCKET_IMAGE,
    "model": T.SOCKET_MODEL,
    "exec": T.SOCKET_EXEC,
    "any": T.FG_SECONDARY,
}


class SocketItem(QGraphicsItem):
    RADIUS = 5

    def __init__(self, parent_node, label: str, socket_type: str,
                 is_input: bool, index: int = 0):
        super().__init__(parent=parent_node)
        self.parent_node = parent_node
        self.label = label
        self.socket_type = socket_type
        self.is_input = is_input
        self.index = index
        self.connections: list = []
        self.setAcceptHoverEvents(True)
        self._hover = False
        self.setZValue(T.Z_NODE + 1)

    def boundingRect(self) -> QRectF:
        r = self.RADIUS + 2
        return QRectF(-r, -r, 2 * r, 2 * r)

    def paint(self, painter, option, widget=None):
        col = QColor(SOCKET_COLOR_MAP.get(self.socket_type, T.FG_SECONDARY))
        if self._hover:
            painter.setPen(QPen(QColor(T.ACCENT), 1.5))
        else:
            painter.setPen(QPen(QColor(T.BG_DEEP), 1))
        painter.setBrush(QBrush(col))
        painter.drawEllipse(QPointF(0, 0), self.RADIUS, self.RADIUS)

    def hoverEnterEvent(self, event):
        self._hover = True
        self.update()
        self.setCursor(Qt.CursorShape.CrossCursor)

    def hoverLeaveEvent(self, event):
        self._hover = False
        self.update()
        self.unsetCursor()

    def disconnect(self, conn):
        if conn in self.connections:
            self.connections.remove(conn)

    def scene_pos(self) -> QPointF:
        return self.mapToScene(QPointF(0, 0))
