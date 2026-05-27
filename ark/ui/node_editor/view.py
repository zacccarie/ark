"""ARK QGraphicsView : zoom, pan multi-modal (middle/space/3-finger trackpad/handle), search."""
from __future__ import annotations
from PySide6.QtCore import Qt, QPointF, QPoint, QEvent
from PySide6.QtGui import QPainter, QKeyEvent, QMouseEvent, QWheelEvent, QAction
from PySide6.QtWidgets import QGraphicsView, QMenu

from ark.ui.node_editor.search_overlay import SearchOverlay
from ark.ui.node_editor.socket_item import SocketItem, types_compatible
from ark.ui.node_editor.connection_item import ConnectionItem
from ark.ui.node_editor.pan_handle import PanHandle


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
        # Drag connection state
        self._dragging_conn: ConnectionItem | None = None
        self._drag_start_socket: SocketItem | None = None
        self._drag_hover_socket: SocketItem | None = None
        # Trackpad gesture (macOS 3-finger swipe + pinch zoom)
        self.viewport().grabGesture(Qt.GestureType.PanGesture)
        self.viewport().grabGesture(Qt.GestureType.PinchGesture)
        self._gesture_zoom_accum = 1.0
        # Pan handle flottant (overlay bas-droite)
        self.pan_handle = PanHandle(self)
        self.pan_handle.pan_delta.connect(self._on_handle_pan)
        # Tactile : pinch trackpad émet NativeGesture
        self.viewport().setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, True)

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
        if event.button() == Qt.MouseButton.RightButton:
            # Right-click context menu géré dans contextMenuEvent
            super().mousePressEvent(event)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            # Détecte si on a cliqué sur un socket → démarre drag connection
            item = self.itemAt(event.position().toPoint())
            if isinstance(item, SocketItem):
                self._start_drag_connection(item, event)
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
        if self._dragging_conn is not None:
            scene_pos = self.mapToScene(event.position().toPoint())
            self._dragging_conn.set_dragging_end(scene_pos)
            # Highlight survol socket cible
            target = self.itemAt(event.position().toPoint())
            if self._drag_hover_socket and self._drag_hover_socket is not target:
                self._drag_hover_socket.set_highlight(False)
                self._drag_hover_socket = None
            if isinstance(target, SocketItem) and target is not self._drag_start_socket:
                compatible = self._can_connect(self._drag_start_socket, target)
                target.set_highlight(True, reject=not compatible)
                self._drag_hover_socket = target
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton and self._panning:
            self._end_pan()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._dragging_conn is not None:
            self._finish_drag_connection(event)
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

    def _on_handle_pan(self, dx: float, dy: float):
        """Slot pour pan handle widget — translate scrollbars."""
        self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - dx))
        self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - dy))

    # ----- Trackpad gestures (macOS 3-finger swipe = PanGesture, pinch = PinchGesture) -----
    def viewportEvent(self, event):
        if event.type() == QEvent.Type.Gesture:
            return self._handle_gesture(event)
        if event.type() == QEvent.Type.NativeGesture:
            return self._handle_native_gesture(event)
        return super().viewportEvent(event)

    def _handle_gesture(self, event):
        pan = event.gesture(Qt.GestureType.PanGesture)
        if pan is not None:
            delta = pan.delta()
            self.horizontalScrollBar().setValue(
                int(self.horizontalScrollBar().value() - delta.x())
            )
            self.verticalScrollBar().setValue(
                int(self.verticalScrollBar().value() - delta.y())
            )
            return True
        pinch = event.gesture(Qt.GestureType.PinchGesture)
        if pinch is not None:
            sf = pinch.scaleFactor()
            current = self.transform().m11()
            new = current * sf
            if self.MIN_ZOOM <= new <= self.MAX_ZOOM:
                self.scale(sf, sf)
            return True
        return False

    def _handle_native_gesture(self, event):
        gtype = event.gestureType()
        if gtype == Qt.NativeGestureType.ZoomNativeGesture:
            sf = 1.0 + event.value()
            current = self.transform().m11()
            new = current * sf
            if self.MIN_ZOOM <= new <= self.MAX_ZOOM:
                self.scale(sf, sf)
            return True
        if gtype == Qt.NativeGestureType.SwipeNativeGesture:
            return True
        return False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Replace handle bottom-right
        if hasattr(self, "pan_handle"):
            self.pan_handle.reposition()

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
            elif isinstance(it, ConnectionItem):
                scene.remove_connection(it)

    # ----- Drag connection -----
    def _start_drag_connection(self, socket: SocketItem, event: QMouseEvent):
        scene = self.scene()
        if scene is None:
            return
        conn = ConnectionItem(start_socket=socket)
        scene.addItem(conn)
        scene_pos = self.mapToScene(event.position().toPoint())
        conn.set_dragging_end(scene_pos)
        self._dragging_conn = conn
        self._drag_start_socket = socket

    def _can_connect(self, src: SocketItem | None, dst: SocketItem | None) -> bool:
        if src is None or dst is None or src is dst:
            return False
        if src.parent_node is dst.parent_node:
            return False
        # Doit relier output → input (peu importe ordre du drag)
        if src.is_input == dst.is_input:
            return False
        # Type-check (orientation arbitraire, on prend src→dst)
        out_sock = src if not src.is_input else dst
        in_sock = dst if dst.is_input else src
        if not types_compatible(out_sock.socket_type, in_sock.socket_type):
            return False
        # Input ne peut avoir qu'1 connexion (sauf si multi_input du node)
        if in_sock.connections and not in_sock.parent_node.spec.is_multi_input:
            return False
        return True

    def _finish_drag_connection(self, event: QMouseEvent):
        scene = self.scene()
        conn = self._dragging_conn
        start = self._drag_start_socket
        target = self.itemAt(event.position().toPoint())
        # Reset highlight
        if self._drag_hover_socket:
            self._drag_hover_socket.set_highlight(False)
            self._drag_hover_socket = None
        if isinstance(target, SocketItem) and self._can_connect(start, target):
            # Établit la connexion : start_socket = output, end_socket = input
            if not start.is_input:
                conn.start_socket = start
                conn.end_socket = target
            else:
                conn.start_socket = target
                conn.end_socket = start
            conn.start_socket.add_connection(conn)
            conn.end_socket.add_connection(conn)
            scene._connections.append(conn)
            conn.update_path()
            # Multi-input : ajoute un nouveau slot si node le supporte
            if conn.end_socket.parent_node.spec.is_multi_input:
                conn.end_socket.parent_node.add_multi_input_slot()
        else:
            scene.removeItem(conn)
        self._dragging_conn = None
        self._drag_start_socket = None

    # ----- Context menu -----
    def contextMenuEvent(self, event):
        scene = self.scene()
        if scene is None:
            return
        item = self.itemAt(event.pos())
        menu = QMenu(self)
        from ark.ui.node_editor.node_item import NodeItem
        if isinstance(item, NodeItem):
            menu.addAction("Bypass on/off", lambda n=item: self._toggle_bypass(n))
            menu.addAction("Renommer...", lambda: None)
            menu.addSeparator()
            menu.addAction("Dupliquer", lambda n=item: self._duplicate_node(n))
            menu.addAction("Supprimer", lambda n=item: scene.remove_node(n))
        elif isinstance(item, ConnectionItem):
            menu.addAction("Supprimer connexion", lambda c=item: scene.remove_connection(c))
        else:
            menu.addAction("Rechercher node...", lambda p=event.pos(): self.open_search_overlay(p))
            menu.addSeparator()
            menu.addAction("Tout effacer", scene.clear_graph)
        menu.exec(event.globalPos())

    def _toggle_bypass(self, node):
        node.bypass = not node.bypass
        node.update()

    def _duplicate_node(self, node):
        scene = self.scene()
        if scene is None:
            return
        new = scene.create_node(node.spec.name,
                                  x=node.pos().x() + 30,
                                  y=node.pos().y() + 30)
        if new:
            for k, v in node.params.items():
                new.set_param(k, v)
