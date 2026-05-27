"""Search overlay : popup au double-clic vide pour rechercher + ajouter node."""
from __future__ import annotations
from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLineEdit, QListWidget, QListWidgetItem,
    QFrame,
)

from ark.ui import design_tokens as T
from ark.nodes.registry import NodeRegistry, NodeSpec


class SearchOverlay(QFrame):
    node_selected = Signal(str)

    def __init__(self, registry: NodeRegistry, parent=None):
        super().__init__(parent, Qt.WindowType.Popup)
        self.registry = registry
        self.setObjectName("search_overlay")
        self.setStyleSheet(f"""
            QFrame#search_overlay {{
                background-color: {T.BG_OVERLAY};
                border: 1px solid {T.BORDER_STRONG};
            }}
            QLineEdit {{
                background-color: {T.BG_DEEP};
                color: {T.FG_PRIMARY};
                border: none;
                border-bottom: 1px solid {T.BORDER_DEFAULT};
                padding: 8px 10px;
                font-family: {T.FONT_SANS_FAMILY};
                font-size: {T.FS_H1}px;
                selection-background-color: {T.ACCENT_DIM};
            }}
            QListWidget {{
                background-color: {T.BG_OVERLAY};
                color: {T.FG_PRIMARY};
                border: none;
                outline: 0;
                font-size: {T.FS_BODY}px;
            }}
            QListWidget::item {{
                padding: 6px 10px;
                border-bottom: 1px solid {T.BORDER_SUBTLE};
            }}
            QListWidget::item:selected {{
                background-color: {T.ACCENT_DIM};
                color: {T.FG_PRIMARY};
            }}
        """)
        self.setFixedSize(340, 360)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.search = QLineEdit(self)
        self.search.setPlaceholderText("Rechercher un node...")
        self.search.textChanged.connect(self._refilter)
        layout.addWidget(self.search)

        self.list = QListWidget(self)
        self.list.itemActivated.connect(self._pick)
        self.list.itemDoubleClicked.connect(self._pick)
        layout.addWidget(self.list)

        self._populate(self.registry.all())

    def _populate(self, specs: list[NodeSpec]):
        self.list.clear()
        for s in specs:
            item = QListWidgetItem(f"{s.title}   ·   {s.category}")
            item.setData(Qt.ItemDataRole.UserRole, s.name)
            self.list.addItem(item)
        if self.list.count():
            self.list.setCurrentRow(0)

    def _refilter(self, text: str):
        text = text.strip().lower()
        if not text:
            self._populate(self.registry.all())
            return
        specs = self.registry.all()
        filtered = []
        for s in specs:
            if text in s.title.lower() or text in s.name.lower() or text in s.category.lower():
                filtered.append(s)
        self._populate(filtered)

    def _pick(self, item: QListWidgetItem):
        name = item.data(Qt.ItemDataRole.UserRole)
        if name:
            self.node_selected.emit(name)
        self.close()

    def show_at(self, global_pos: QPoint):
        # Centre le popup sur le point
        self.move(global_pos.x() - self.width() // 2, global_pos.y() - 20)
        self.show()
        self.search.setFocus()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            it = self.list.currentItem()
            if it:
                self._pick(it)
            return
        if event.key() == Qt.Key.Key_Down:
            row = self.list.currentRow()
            self.list.setCurrentRow(min(row + 1, self.list.count() - 1))
            return
        if event.key() == Qt.Key.Key_Up:
            row = self.list.currentRow()
            self.list.setCurrentRow(max(row - 1, 0))
            return
        super().keyPressEvent(event)
