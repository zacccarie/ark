"""Phase 31 — Dialog 'Save as Group' : nomme + catégorise un sous-graphe."""
from __future__ import annotations
import re
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, QComboBox,
    QPlainTextEdit, QPushButton, QLabel, QFrame,
)

from ark.ui import design_tokens as T


class SaveGroupDialog(QDialog):
    """Demande name/title/category/description pour enregistrer un group."""

    def __init__(self, parent=None, n_selected: int = 0):
        super().__init__(parent)
        self.setWindowTitle("Enregistrer comme groupe")
        self.setMinimumWidth(420)
        self.setModal(True)
        self._build_ui(n_selected)

    def _build_ui(self, n_selected: int):
        v = QVBoxLayout(self)
        v.setContentsMargins(T.S_4, T.S_4, T.S_4, T.S_4)
        v.setSpacing(T.S_3)

        # Header
        h = QLabel(f"Sélection : {n_selected} nodes")
        h.setStyleSheet(f"font-size: {T.FS_H1}px; font-weight: 700; "
                          f"color: {T.FG_PRIMARY};")
        v.addWidget(h)

        sub = QLabel("Le sous-graphe sera enregistré dans ~/.ark/groups/<name>.json "
                      "et apparaîtra dans la palette au prochain lancement.")
        sub.setWordWrap(True)
        sub.setStyleSheet(f"font-size: {T.FS_LABEL}px; color: {T.FG_TERTIARY};")
        v.addWidget(sub)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color: {T.BORDER_DEFAULT};")
        v.addWidget(sep)

        # Form
        form = QFormLayout()
        form.setHorizontalSpacing(T.S_3)
        form.setVerticalSpacing(T.S_2)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("my_resnet_block (snake_case)")
        form.addRow("Name (id) :", self.name_edit)

        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("My ResNet Block")
        form.addRow("Title :", self.title_edit)

        self.category_combo = QComboBox()
        self.category_combo.setEditable(True)
        # Categories existantes courantes + User Groups par défaut
        for cat in ("User Groups", "Neural", "Neural Networks", "Sequential",
                      "Transformers", "Generative AI", "Adaptive Architectures",
                      "Dynamical Systems", "Biological Inspiration",
                      "Experimental Research"):
            self.category_combo.addItem(cat)
        form.addRow("Category :", self.category_combo)

        self.desc_edit = QPlainTextEdit()
        self.desc_edit.setPlaceholderText("Description courte du groupe...")
        self.desc_edit.setFixedHeight(80)
        form.addRow("Description :", self.desc_edit)

        v.addLayout(form)

        # Boutons
        h_btn = QHBoxLayout()
        h_btn.addStretch(1)
        self.btn_cancel = QPushButton("Annuler")
        self.btn_cancel.clicked.connect(self.reject)
        h_btn.addWidget(self.btn_cancel)
        self.btn_save = QPushButton("Enregistrer")
        self.btn_save.setDefault(True)
        self.btn_save.setStyleSheet(f"background-color: {T.ACCENT}; "
                                          f"color: white; font-weight: 600;")
        self.btn_save.clicked.connect(self._on_save)
        h_btn.addWidget(self.btn_save)
        v.addLayout(h_btn)

        # Auto-slug : title → name (en live)
        self.title_edit.textChanged.connect(self._autofill_name)

    def _autofill_name(self, txt: str):
        if not self.name_edit.text().strip():
            slug = re.sub(r"[^a-z0-9_]+", "_", txt.lower()).strip("_")
            self.name_edit.setText(slug)

    def _on_save(self):
        name = self.name_edit.text().strip()
        if not name or not re.match(r"^[a-z][a-z0-9_]*$", name):
            self.name_edit.setStyleSheet(f"border: 1px solid {T.ACCENT};")
            return
        self.accept()

    def get_payload(self) -> dict:
        return {
            "name": self.name_edit.text().strip(),
            "title": self.title_edit.text().strip() or self.name_edit.text().strip(),
            "category": self.category_combo.currentText().strip() or "User Groups",
            "description": self.desc_edit.toPlainText().strip(),
        }
