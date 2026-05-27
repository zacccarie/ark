"""QUndoCommand subclasses pour undo/redo sur scene.

Chaque commande encode un delta réversible. Le QUndoStack appelle redo()
au push initial + à chaque redo, et undo() inverse.
"""
from __future__ import annotations
from PySide6.QtCore import QPointF
from PySide6.QtGui import QUndoCommand


class AddNodeCommand(QUndoCommand):
    def __init__(self, scene, type_name: str, x: float, y: float, parent=None):
        super().__init__(f"Add {type_name}", parent)
        self.scene = scene
        self.type_name = type_name
        self.x = x
        self.y = y
        self.created_node = None

    def redo(self):
        # 1ère exécution : crée. Suivants : restore depuis stocké.
        if self.created_node is None:
            self.created_node = self.scene._raw_create_node(self.type_name, self.x, self.y)
        else:
            self.scene._raw_add_node(self.created_node)

    def undo(self):
        if self.created_node is not None:
            self.scene._raw_remove_node(self.created_node, keep_obj=True)


class RemoveNodeCommand(QUndoCommand):
    def __init__(self, scene, node, parent=None):
        super().__init__(f"Remove {node.spec.title}", parent)
        self.scene = scene
        self.node = node
        # Snapshot connections affectées pour restauration
        self.related_conns = []
        for s in node.input_sockets + node.output_sockets:
            for c in s.connections:
                if c not in self.related_conns:
                    self.related_conns.append(c)

    def redo(self):
        for c in self.related_conns:
            if c in self.scene._connections:
                self.scene._raw_remove_connection(c, keep_obj=True)
        self.scene._raw_remove_node(self.node, keep_obj=True)

    def undo(self):
        self.scene._raw_add_node(self.node)
        for c in self.related_conns:
            self.scene._raw_add_connection(c)


class AddConnectionCommand(QUndoCommand):
    def __init__(self, scene, conn, parent=None):
        super().__init__("Add connection", parent)
        self.scene = scene
        self.conn = conn

    def redo(self):
        self.scene._raw_add_connection(self.conn)

    def undo(self):
        self.scene._raw_remove_connection(self.conn, keep_obj=True)


class RemoveConnectionCommand(QUndoCommand):
    def __init__(self, scene, conn, parent=None):
        super().__init__("Remove connection", parent)
        self.scene = scene
        self.conn = conn

    def redo(self):
        self.scene._raw_remove_connection(self.conn, keep_obj=True)

    def undo(self):
        self.scene._raw_add_connection(self.conn)


class SetParamCommand(QUndoCommand):
    def __init__(self, node, name: str, new_value, old_value, parent=None):
        super().__init__(f"Set {name}", parent)
        self.node = node
        self.name = name
        self.new_value = new_value
        self.old_value = old_value

    def redo(self):
        self.node.params[self.name] = self.new_value
        self.node.update()

    def undo(self):
        self.node.params[self.name] = self.old_value
        self.node.update()


class MoveNodeCommand(QUndoCommand):
    def __init__(self, node, old_pos: QPointF, new_pos: QPointF, parent=None):
        super().__init__("Move node", parent)
        self.node = node
        self.old_pos = QPointF(old_pos)
        self.new_pos = QPointF(new_pos)

    def redo(self):
        self.node.setPos(self.new_pos)

    def undo(self):
        self.node.setPos(self.old_pos)
