"""ARK QGraphicsScene — gère nodes + connections."""
from __future__ import annotations
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QPainter, QBrush, QPen, QColor
from PySide6.QtWidgets import QGraphicsScene

from ark.ui import design_tokens as T
from ark.ui.node_editor.node_item import NodeItem
from ark.ui.node_editor.connection_item import ConnectionItem
from ark.nodes.registry import NodeRegistry


class NodeScene(QGraphicsScene):
    GRID_SIZE = 24

    def __init__(self, registry: NodeRegistry, parent=None):
        super().__init__(parent)
        self.registry = registry
        self.setBackgroundBrush(QBrush(QColor(T.BG_DEEP)))
        self.setSceneRect(-5000, -5000, 10000, 10000)
        self._nodes: list[NodeItem] = []
        self._connections: list[ConnectionItem] = []

    # ------------------------------------------------------------
    # Grille de fond (subtile)
    # ------------------------------------------------------------
    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        super().drawBackground(painter, rect)
        pen = QPen(QColor(T.BORDER_SUBTLE))
        pen.setWidthF(0.4)
        painter.setPen(pen)
        left = int(rect.left()) - (int(rect.left()) % self.GRID_SIZE)
        top = int(rect.top()) - (int(rect.top()) % self.GRID_SIZE)
        # Lignes verticales
        x = left
        while x < rect.right():
            painter.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))
            x += self.GRID_SIZE
        # Lignes horizontales
        y = top
        while y < rect.bottom():
            painter.drawLine(int(rect.left()), int(y), int(rect.right()), int(y))
            y += self.GRID_SIZE

    # ------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------
    def nodes(self) -> list[NodeItem]:
        return list(self._nodes)

    def connections(self) -> list[ConnectionItem]:
        return list(self._connections)

    def create_node(self, type_name: str, x: float = 0.0, y: float = 0.0) -> NodeItem | None:
        spec = self.registry.get(type_name)
        if spec is None:
            return None
        node = NodeItem(spec)
        node.setPos(QPointF(x, y))
        self.addItem(node)
        self._nodes.append(node)
        return node

    def remove_node(self, node: NodeItem) -> None:
        # Retire connexions liées
        for c in list(self._connections):
            if c.start_socket and c.start_socket.parent_node is node:
                self.remove_connection(c)
            elif c.end_socket and c.end_socket.parent_node is node:
                self.remove_connection(c)
        self.removeItem(node)
        if node in self._nodes:
            self._nodes.remove(node)

    def add_connection(self, conn: ConnectionItem) -> None:
        self.addItem(conn)
        self._connections.append(conn)

    def remove_connection(self, conn: ConnectionItem) -> None:
        if conn.start_socket:
            conn.start_socket.disconnect(conn)
        if conn.end_socket:
            conn.end_socket.disconnect(conn)
        self.removeItem(conn)
        if conn in self._connections:
            self._connections.remove(conn)

    def clear_graph(self) -> None:
        for c in list(self._connections):
            self.remove_connection(c)
        for n in list(self._nodes):
            self.remove_node(n)
