"""NodeItem — node visuel + sockets + handles redim + bypass."""
from __future__ import annotations
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QBrush, QPen, QColor, QFont, QPainterPath, QCursor
from PySide6.QtWidgets import QGraphicsItem, QGraphicsTextItem

from ark.ui import design_tokens as T
from ark.ui.node_editor.socket_item import SocketItem
from ark.nodes.registry import NodeSpec, SocketSpec
import uuid


class NodeItem(QGraphicsItem):
    DEFAULT_WIDTH = 200
    HEADER_H = 24
    ROW_H = 18
    PARAM_H = 16
    SOCKET_MARGIN = 14
    PADDING_BOTTOM = 8
    HANDLE_SIZE = 8

    def __init__(self, spec: NodeSpec):
        super().__init__()
        self.id = f"{spec.name}_{uuid.uuid4().hex[:6]}"
        self.spec = spec
        self.params: dict = {p.name: p.default for p in spec.params}
        self.bypass = False
        self.error: str | None = None
        self.width = self.DEFAULT_WIDTH

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setZValue(T.Z_NODE)
        self.setAcceptHoverEvents(True)
        self._hover = False

        # Sockets
        self.input_sockets: list[SocketItem] = []
        self.output_sockets: list[SocketItem] = []
        for i, s in enumerate(spec.inputs):
            sock = SocketItem(self, s.label, s.socket_type, is_input=True, index=i)
            self.input_sockets.append(sock)
        for i, s in enumerate(spec.outputs):
            sock = SocketItem(self, s.label, s.socket_type, is_input=False, index=i)
            self.output_sockets.append(sock)
        self._layout_sockets()

        # Resize state
        self._resizing = False
        self._resize_start_w = 0
        self._resize_start_pos = QPointF()

    # ------------------------------------------------------------
    # Size / layout
    # ------------------------------------------------------------
    @property
    def height(self) -> int:
        n_sockets = max(len(self.input_sockets), len(self.output_sockets))
        socket_h = max(n_sockets, 1) * self.ROW_H
        param_h = len(self.spec.params) * self.PARAM_H
        thumb_h = 60 if self.spec.has_thumbnail else 0
        return self.HEADER_H + socket_h + param_h + thumb_h + self.PADDING_BOTTOM

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.width, self.height + self.HANDLE_SIZE)

    def _layout_sockets(self):
        for i, sock in enumerate(self.input_sockets):
            sock.setPos(0, self.HEADER_H + self.ROW_H * i + self.ROW_H / 2)
        for i, sock in enumerate(self.output_sockets):
            sock.setPos(self.width, self.HEADER_H + self.ROW_H * i + self.ROW_H / 2)

    # ------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------
    def paint(self, painter, option, widget=None):
        rect = QRectF(0, 0, self.width, self.height)
        path = QPainterPath()
        path.addRoundedRect(rect, T.R_SMALL, T.R_SMALL)

        # Body
        painter.setBrush(QBrush(QColor(T.BG_PANEL_RAISED)))
        border_col = T.ACCENT if self.error else (T.BORDER_STRONG if self.isSelected()
                                                     else (T.BORDER_DEFAULT if not self._hover else T.BORDER_STRONG))
        pen = QPen(QColor(border_col))
        pen.setWidth(2 if (self.isSelected() or self.error) else 1)
        painter.setPen(pen)
        painter.drawPath(path)

        # Header band
        header_rect = QRectF(0, 0, self.width, self.HEADER_H)
        header_path = QPainterPath()
        header_path.addRoundedRect(header_rect, T.R_SMALL, T.R_SMALL)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(T.BG_OVERLAY)))
        painter.drawPath(header_path)
        # Cut rounded bottom
        painter.fillRect(QRectF(0, self.HEADER_H - 4, self.width, 4), QColor(T.BG_OVERLAY))

        # Title
        title_pen = QPen(QColor(T.FG_PRIMARY if not self.bypass else T.FG_DISABLED))
        painter.setPen(title_pen)
        font = QFont("Arial", T.FS_BODY)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(QRectF(8, 0, self.width - 16, self.HEADER_H),
                         Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                         self.spec.title)

        # Bypass indicator (rouge si actif)
        if self.bypass:
            painter.fillRect(QRectF(self.width - 8, 4, 4, self.HEADER_H - 8),
                              QColor(T.ACCENT))

        # Category microlabel
        font_micro = QFont("Arial", T.FS_MICRO)
        painter.setFont(font_micro)
        painter.setPen(QPen(QColor(T.FG_TERTIARY)))
        painter.drawText(QRectF(8, self.HEADER_H + 2, self.width - 16, 12),
                         Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight,
                         self.spec.category.upper())

        # Socket labels
        font_label = QFont("Arial", T.FS_LABEL)
        painter.setFont(font_label)
        painter.setPen(QPen(QColor(T.FG_SECONDARY)))
        for i, s in enumerate(self.spec.inputs):
            y = self.HEADER_H + self.ROW_H * i + self.ROW_H / 2 + 4
            painter.drawText(QRectF(10, y - 8, self.width / 2, 12),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                             s.label)
        for i, s in enumerate(self.spec.outputs):
            y = self.HEADER_H + self.ROW_H * i + self.ROW_H / 2 + 4
            painter.drawText(QRectF(self.width / 2, y - 8, self.width / 2 - 10, 12),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                             s.label)

        # Params summary (mono)
        n_sockets = max(len(self.input_sockets), len(self.output_sockets))
        param_y0 = self.HEADER_H + n_sockets * self.ROW_H + 2
        font_mono = QFont("SF Mono", T.FS_MICRO)
        painter.setFont(font_mono)
        painter.setPen(QPen(QColor(T.FG_TERTIARY)))
        for i, p in enumerate(self.spec.params):
            v = self.params.get(p.name, p.default)
            txt = f"{p.name} = {v}"
            painter.drawText(QRectF(8, param_y0 + i * self.PARAM_H, self.width - 16, self.PARAM_H),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                             txt)

        # Error message bar
        if self.error:
            err_y = self.height - 14
            painter.fillRect(QRectF(0, err_y, self.width, 14), QColor(T.ACCENT_DIM))
            painter.setPen(QPen(QColor(T.ACCENT)))
            painter.drawText(QRectF(8, err_y, self.width - 16, 14),
                             Qt.AlignmentFlag.AlignVCenter,
                             f"⚠ {self.error}")

        # Resize handle (bottom-right corner)
        h_rect = QRectF(self.width - self.HANDLE_SIZE, self.height,
                          self.HANDLE_SIZE, self.HANDLE_SIZE)
        painter.fillRect(h_rect, QColor(T.BORDER_DEFAULT))

    # ------------------------------------------------------------
    # Interaction : hover, double-click toggle bypass, resize
    # ------------------------------------------------------------
    def hoverEnterEvent(self, event):
        self._hover = True
        self.update()

    def hoverLeaveEvent(self, event):
        self._hover = False
        self.update()

    def mouseDoubleClickEvent(self, event):
        # Bypass toggle au double-clic header band
        if event.pos().y() < self.HEADER_H:
            self.bypass = not self.bypass
            self.update()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            # Repaint connections liées
            for sock in self.input_sockets + self.output_sockets:
                for conn in sock.connections:
                    conn.update_path()
        return super().itemChange(change, value)

    def set_param(self, name: str, value):
        if name in self.params:
            self.params[name] = value
            self.update()

    def set_error(self, msg: str | None):
        self.error = msg
        self.update()

    # ------------------------------------------------------------
    # Multi-input dynamique (vague residual_add, concat)
    # ------------------------------------------------------------
    def add_multi_input_slot(self):
        """Ajoute un nouveau slot d'entrée si tous les existants sont connectés.

        Type pris depuis le 1er socket (ou 'any' si vide).
        """
        if not self.spec.is_multi_input:
            return
        # Ne crée un nouveau que si tous les existants sont connectés
        unconnected = [s for s in self.input_sockets if not s.connections]
        if unconnected:
            return
        # Type ref
        sock_type = self.input_sockets[0].socket_type if self.input_sockets else "any"
        idx = len(self.input_sockets)
        label = chr(ord('a') + idx) if idx < 26 else f"in{idx}"
        new_sock = SocketItem(self, label, sock_type, is_input=True, index=idx)
        self.input_sockets.append(new_sock)
        # Update spec.inputs aussi (pour rendu)
        self.spec.inputs.append(SocketSpec(label, sock_type))
        self._layout_sockets()
        self.prepareGeometryChange()
        self.update()

    # ------------------------------------------------------------
    # Resize handle (corner BR)
    # ------------------------------------------------------------
    def _in_resize_handle(self, local_pos: QPointF) -> bool:
        return (local_pos.x() >= self.width - self.HANDLE_SIZE
                and local_pos.y() >= self.height - 2
                and local_pos.y() <= self.height + self.HANDLE_SIZE)

    def mousePressEvent(self, event):
        if self._in_resize_handle(event.pos()):
            self._resizing = True
            self._resize_start_w = self.width
            self._resize_start_pos = event.scenePos()
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._resizing:
            dx = event.scenePos().x() - self._resize_start_pos.x()
            new_w = max(140, int(self._resize_start_w + dx))
            if new_w != self.width:
                self.prepareGeometryChange()
                self.width = new_w
                self._layout_sockets()
                # Update connections liées (output socket position changes)
                for sock in self.output_sockets:
                    for conn in sock.connections:
                        conn.update_path()
                self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._resizing:
            self._resizing = False
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def hoverMoveEvent(self, event):
        if self._in_resize_handle(event.pos()):
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        else:
            self.unsetCursor()
        super().hoverMoveEvent(event)
