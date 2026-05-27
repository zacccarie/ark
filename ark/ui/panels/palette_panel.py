"""Palette panel — arborescence des nodes par catégorie + double-clic spawn."""
from __future__ import annotations
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLineEdit, QTreeWidget, QTreeWidgetItem,
)

from ark.ui import design_tokens as T
from ark.nodes.registry import NodeRegistry


class PalettePanel(QWidget):
    node_double_clicked = Signal(str)

    def __init__(self, registry: NodeRegistry, parent=None):
        super().__init__(parent)
        self.registry = registry
        layout = QVBoxLayout(self)
        layout.setContentsMargins(T.S_2, T.S_2, T.S_2, T.S_2)
        layout.setSpacing(T.S_1)

        self.search = QLineEdit(self)
        self.search.setPlaceholderText("Filtrer...")
        self.search.textChanged.connect(self._refilter)
        layout.addWidget(self.search)

        self.tree = QTreeWidget(self)
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(12)
        self.tree.itemDoubleClicked.connect(self._on_double_clicked)
        layout.addWidget(self.tree)
        self._build_tree("")

    def _build_tree(self, filt: str):
        self.tree.clear()
        cats = self.registry.by_category()
        for cat, specs in cats.items():
            shown = [s for s in specs if not filt or
                      (filt.lower() in s.title.lower() or filt.lower() in s.name.lower())]
            if not shown:
                continue
            cat_item = QTreeWidgetItem([cat.upper()])
            font = QFont("Arial", T.FS_MICRO)
            font.setBold(True)
            cat_item.setFont(0, font)
            cat_item.setForeground(0, Qt.GlobalColor.darkGray)
            self.tree.addTopLevelItem(cat_item)
            for s in shown:
                child = QTreeWidgetItem([s.title])
                child.setData(0, Qt.ItemDataRole.UserRole, s.name)
                child.setToolTip(0, s.description or s.title)
                cat_item.addChild(child)
            cat_item.setExpanded(True)

    def _refilter(self, text: str):
        self._build_tree(text)

    def _on_double_clicked(self, item, col):
        type_name = item.data(0, Qt.ItemDataRole.UserRole)
        if type_name:
            self.node_double_clicked.emit(type_name)
