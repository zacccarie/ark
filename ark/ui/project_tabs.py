"""Workspace multi-projets : QTabWidget contenant N (NodeScene + NodeEditorView).

Chaque tab = un projet indépendant. Switch propage active scene au reste de
l'app (palette spawn cible la scene active, inspector écoute sa sélection).
"""
from __future__ import annotations
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QUndoStack
from PySide6.QtWidgets import QTabWidget, QWidget

from ark.ui.node_editor.scene import NodeScene
from ark.ui.node_editor.view import NodeEditorView


class ProjectTabs(QTabWidget):
    """Tab widget de scènes ARK. Émet active_changed avec (scene, view, undo_stack)."""

    active_changed = Signal(object, object, object)
    project_closed = Signal(int)

    def __init__(self, registry, parent=None):
        super().__init__(parent)
        self.registry = registry
        self.setTabsClosable(True)
        self.setMovable(True)
        self.setDocumentMode(True)
        self.tabCloseRequested.connect(self._on_close)
        self.currentChanged.connect(self._on_current_changed)
        self._undo_stacks: dict[int, QUndoStack] = {}
        self._scenes: dict[int, NodeScene] = {}  # keep Python refs (sinon GC)

    def new_project(self, name: str = "Sans titre", payload: dict | None = None) -> int:
        """Crée nouveau tab. Si payload fourni, charge directement."""
        undo = QUndoStack()
        undo.setUndoLimit(200)
        scene = NodeScene(registry=self.registry, undo_stack=undo)
        view = NodeEditorView(scene)
        idx = self.addTab(view, name)
        # Préserver refs Python (sinon NodeScene est GC et view.scene() retourne None)
        self._undo_stacks[id(view)] = undo
        self._scenes[id(view)] = scene
        if payload is not None:
            from ark.core.format import payload_to_scene
            payload_to_scene(scene, payload)
            undo.clear()
        self.setCurrentIndex(idx)
        return idx

    def active(self):
        idx = self.currentIndex()
        if idx < 0:
            return (None, None, None)
        view = self.widget(idx)
        if view is None:
            return (None, None, None)
        scene = self._scenes.get(id(view))
        undo = self._undo_stacks.get(id(view))
        return (scene, view, undo)

    def _on_close(self, idx: int):
        view = self.widget(idx)
        if view is not None:
            self._undo_stacks.pop(id(view), None)
            self._scenes.pop(id(view), None)
        self.removeTab(idx)
        self.project_closed.emit(idx)
        if self.count() == 0:
            self.new_project("Sans titre")

    def _on_current_changed(self, _idx: int):
        scene, view, undo = self.active()
        if scene is not None:
            self.active_changed.emit(scene, view, undo)

    def rename_current(self, name: str):
        idx = self.currentIndex()
        if idx >= 0:
            self.setTabText(idx, name)

    def duplicate_current(self):
        scene, _view, _undo = self.active()
        if scene is None:
            return
        from ark.core.format import scene_to_payload
        payload = scene_to_payload(scene, name="Copie")
        current_name = self.tabText(self.currentIndex())
        self.new_project(f"{current_name} (copie)", payload=payload)
