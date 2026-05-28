"""Phase 50 — Minimap overview QGraphicsView miniature."""
from __future__ import annotations
from PySide6.QtCore import Qt, QRectF, QPointF, QTimer
from PySide6.QtGui import QPainter, QColor, QBrush, QPen
from PySide6.QtWidgets import QWidget, QGraphicsDropShadowEffect

from ark.ui import design_tokens as T


class Minimap(QWidget):
    """Vue minimap : projection downscale du scene + viewport overlay rouge."""

    def __init__(self, scene=None, view=None, parent=None):
        super().__init__(parent)
        self._scene = scene
        self._view = view
        self.setFixedSize(220, 160)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"Minimap {{ "
            f"  background-color: {T.BG_PANEL}; "
            f"  border: 1px solid {T.BORDER_DEFAULT}; "
            f"  border-radius: {T.R_MEDIUM}px; "
            f"}}"
        )
        sh = QGraphicsDropShadowEffect(self)
        sh.setBlurRadius(20)
        sh.setColor(QColor(0, 0, 0, 140))
        sh.setOffset(0, 4)
        self.setGraphicsEffect(sh)
        # Auto-refresh
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(500)

    def set_scene_view(self, scene, view):
        self._scene = scene
        self._view = view
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(8, 8, -8, -8)

        # Phase 50 fix : guard contre scene supprimée Qt (RuntimeError C++)
        try:
            if self._scene is None:
                raise RuntimeError("no scene")
            nodes = list(self._scene.nodes())
        except (RuntimeError, AttributeError):
            p.setPen(QPen(QColor(T.FG_TERTIARY)))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, "—")
            p.end()
            return

        if not nodes:
            p.setPen(QPen(QColor(T.FG_TERTIARY)))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Scene vide")
            p.end()
            return
        min_x = min(n.pos().x() for n in nodes)
        max_x = max(n.pos().x() + n.width for n in nodes)
        min_y = min(n.pos().y() for n in nodes)
        max_y = max(n.pos().y() + n.height for n in nodes)
        w = max(max_x - min_x, 1)
        h = max(max_y - min_y, 1)

        sx = rect.width() / w
        sy = rect.height() / h
        scale = min(sx, sy) * 0.95
        off_x = rect.x() + (rect.width() - w * scale) / 2
        off_y = rect.y() + (rect.height() - h * scale) / 2

        # Draw nodes
        for n in nodes:
            nx = off_x + (n.pos().x() - min_x) * scale
            ny = off_y + (n.pos().y() - min_y) * scale
            nw = max(n.width * scale, 2)
            nh = max(n.height * scale, 2)
            # Color par status
            cmap = {
                "ok": T.STATE_OK,
                "compiling": T.STATE_COMPILING,
                "training": T.STATE_TRAINING,
                "error": T.STATE_ERROR,
                "warn": T.STATE_WARN,
            }
            col = cmap.get(getattr(n, "status", "idle"), T.FG_TERTIARY)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(col)))
            p.drawRoundedRect(int(nx), int(ny), int(nw), int(nh), 2, 2)

        # Draw connections (lines minces)
        try:
            p.setPen(QPen(QColor(T.BORDER_DEFAULT), 1))
            for c in self._scene.connections():
                if c.start_socket is None or c.end_socket is None:
                    continue
                sn = c.start_socket.parent_node
                en = c.end_socket.parent_node
                sx_ = off_x + (sn.pos().x() + sn.width - min_x) * scale
                sy_ = off_y + (sn.pos().y() + sn.height / 2 - min_y) * scale
                ex_ = off_x + (en.pos().x() - min_x) * scale
                ey_ = off_y + (en.pos().y() + en.height / 2 - min_y) * scale
                p.drawLine(int(sx_), int(sy_), int(ex_), int(ey_))
        except (RuntimeError, AttributeError):
            pass

        # Draw viewport rect (red accent)
        try:
            if self._view is not None:
                view_rect = self._view.mapToScene(
                    self._view.viewport().rect()).boundingRect()
                vx = off_x + (view_rect.x() - min_x) * scale
                vy = off_y + (view_rect.y() - min_y) * scale
                vw = max(view_rect.width() * scale, 4)
                vh = max(view_rect.height() * scale, 4)
                p.setPen(QPen(QColor(T.PRIMARY), 1.5))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(int(vx), int(vy), int(vw), int(vh))
        except (RuntimeError, AttributeError):
            pass

        p.end()

    def mousePressEvent(self, ev):
        """Click center viewport sur position cliquée."""
        if self._view is None or self._scene is None:
            return
        try:
            nodes = list(self._scene.nodes())
        except (RuntimeError, AttributeError):
            return
        if not nodes:
            return
        min_x = min(n.pos().x() for n in nodes)
        max_x = max(n.pos().x() + n.width for n in nodes)
        min_y = min(n.pos().y() for n in nodes)
        max_y = max(n.pos().y() + n.height for n in nodes)
        w, h = max(max_x - min_x, 1), max(max_y - min_y, 1)
        rect = self.rect().adjusted(8, 8, -8, -8)
        scale = min(rect.width() / w, rect.height() / h) * 0.95
        off_x = rect.x() + (rect.width() - w * scale) / 2
        off_y = rect.y() + (rect.height() - h * scale) / 2
        scene_x = (ev.pos().x() - off_x) / scale + min_x
        scene_y = (ev.pos().y() - off_y) / scale + min_y
        self._view.centerOn(QPointF(scene_x, scene_y))
