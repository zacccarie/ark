"""Connection bezier item entre 2 sockets."""
from __future__ import annotations
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QPen, QColor, QPainterPath, QBrush
from PySide6.QtWidgets import QGraphicsPathItem

from ark.ui import design_tokens as T
from ark.ui.node_editor.socket_item import SOCKET_COLOR_MAP


class ConnectionItem(QGraphicsPathItem):
    def __init__(self, start_socket=None, end_socket=None):
        super().__init__()
        self.start_socket = start_socket
        self.end_socket = end_socket
        self._end_pos: QPointF | None = None  # pour drag preview
        self.setZValue(T.Z_CONNECTION)
        self.setFlag(self.GraphicsItemFlag.ItemIsSelectable, True)
        self.update_path()

    def set_dragging_end(self, scene_pos: QPointF):
        self._end_pos = scene_pos
        self.update_path()

    def update_path(self):
        if self.start_socket is None:
            return
        p1 = self.start_socket.scene_pos()
        if self.end_socket is not None:
            p2 = self.end_socket.scene_pos()
        elif self._end_pos is not None:
            p2 = self._end_pos
        else:
            return

        dx = abs(p2.x() - p1.x()) * 0.5 + 30
        c1 = QPointF(p1.x() + dx, p1.y())
        c2 = QPointF(p2.x() - dx, p2.y())
        path = QPainterPath(p1)
        path.cubicTo(c1, c2, p2)
        self.setPath(path)

        # Couleur depuis socket source
        col = SOCKET_COLOR_MAP.get(self.start_socket.socket_type, T.FG_SECONDARY)
        pen = QPen(QColor(col))
        pen.setWidthF(1.8 if not self.isSelected() else 2.6)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        if self.isSelected():
            pen.setColor(QColor(T.ACCENT))
        self.setPen(pen)
        self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
