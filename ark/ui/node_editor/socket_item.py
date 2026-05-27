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


def types_compatible(src_type: str, dst_type: str) -> bool:
    """Validation compatibilité types socket pour connexion."""
    if src_type == dst_type:
        return True
    if "any" in (src_type, dst_type):
        return True
    # number ↔ tensor compatibles (scalaire upcast)
    if {src_type, dst_type} == {"number", "tensor"}:
        return True
    return False


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
        self._highlight = False  # drag survol valid target
        self._reject = False     # drag survol invalid target
        self.setZValue(T.Z_NODE + 1)

    def boundingRect(self) -> QRectF:
        r = self.RADIUS + 4
        return QRectF(-r, -r, 2 * r, 2 * r)

    def paint(self, painter, option, widget=None):
        col = QColor(SOCKET_COLOR_MAP.get(self.socket_type, T.FG_SECONDARY))
        if self._reject:
            ring_col = QColor("#E11D2A")
            painter.setPen(QPen(ring_col, 2))
            painter.setBrush(QBrush(QColor(T.BG_DEEP)))
        elif self._highlight:
            painter.setPen(QPen(QColor(T.FG_PRIMARY), 2))
            painter.setBrush(QBrush(col))
        elif self._hover:
            painter.setPen(QPen(QColor(T.ACCENT), 1.5))
            painter.setBrush(QBrush(col))
        else:
            painter.setPen(QPen(QColor(T.BG_DEEP), 1))
            painter.setBrush(QBrush(col))
        r = self.RADIUS + (1 if (self._highlight or self._reject) else 0)
        painter.drawEllipse(QPointF(0, 0), r, r)

    def hoverEnterEvent(self, event):
        self._hover = True
        self.update()
        self.setCursor(Qt.CursorShape.CrossCursor)

    def hoverLeaveEvent(self, event):
        self._hover = False
        self.update()
        self.unsetCursor()

    def set_highlight(self, on: bool, reject: bool = False):
        self._highlight = on and not reject
        self._reject = on and reject
        self.update()

    def disconnect(self, conn):
        if conn in self.connections:
            self.connections.remove(conn)

    def add_connection(self, conn):
        if conn not in self.connections:
            self.connections.append(conn)

    def scene_pos(self) -> QPointF:
        return self.mapToScene(QPointF(0, 0))
