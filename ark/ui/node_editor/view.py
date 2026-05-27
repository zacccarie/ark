"""ARK QGraphicsView : zoom souris, pan middle/space-drag, double-clic search."""
from __future__ import annotations
from PySide6.QtCore import Qt, QPointF, QPoint
from PySide6.QtGui import QPainter, QKeyEvent, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QGraphicsView

from ark.ui.node_editor.search_overlay import SearchOverlay


class NodeEditorView(QGraphicsView):
    MIN_ZOOM = 0.25
    MAX_ZOOM = 3.0
    ZOOM_STEP = 1.15

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMouseTracking(True)
        self._panning = False
        self._pan_anchor = QPoint()
        self._search: SearchOverlay | None = None

    # ----- Zoom -----
    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = self.ZOOM_STEP if delta > 0 else 1 / self.ZOOM_STEP
        current = self.transform().m11()
        new = current * factor
        if self.MIN_ZOOM <= new <= self.MAX_ZOOM:
            self.scale(factor, factor)

    # ----- Pan : middle button OR space-drag -----
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._start_pan(event.position().toPoint())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._panning:
            delta = event.position().toPoint() - self._pan_anchor
            self._pan_anchor = event.position().toPoint()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton and self._panning:
            self._end_pan()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _start_pan(self, pos: QPoint):
        self._panning = True
        self._pan_anchor = pos
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def _end_pan(self):
        self._panning = False
        self.setCursor(Qt.CursorShape.ArrowCursor)

    # ----- Double-clic empty space → search overlay -----
    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.position().toPoint())
            if item is None:
                self.open_search_overlay(event.position().toPoint())
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def open_search_overlay(self, pos: QPoint | None = None):
        scene = self.scene()
        if scene is None:
            return
        if self._search is not None and self._search.isVisible():
            self._search.close()
        from ark.nodes.registry import NodeRegistry
        # Recupere registry via scene
        registry: NodeRegistry = scene.registry
        self._search = SearchOverlay(registry=registry, parent=self)
        self._search.node_selected.connect(self._on_search_pick)
        if pos is None:
            pos = self.viewport().rect().center()
        global_pos = self.viewport().mapToGlobal(pos)
        self._search.show_at(global_pos)

    def _on_search_pick(self, type_name: str):
        scene = self.scene()
        if scene is None:
            return
        # Spawn au centre vue actuelle
        center = self.mapToScene(self.viewport().rect().center())
        scene.create_node(type_name, x=center.x(), y=center.y())

    # ----- Keyboard shortcuts -----
    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space:
            self._panning = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            return
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self._delete_selected()
            return
        if event.key() == Qt.Key.Key_F and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.open_search_overlay()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        super().keyReleaseEvent(event)

    def _delete_selected(self):
        scene = self.scene()
        if scene is None:
            return
        from ark.ui.node_editor.node_item import NodeItem
        for it in scene.selectedItems():
            if isinstance(it, NodeItem):
                scene.remove_node(it)
