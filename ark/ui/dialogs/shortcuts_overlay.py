"""Phase 40 — Overlay raccourcis Cmd+/ : cheat-sheet keyboard shortcuts."""
from __future__ import annotations
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QWidget,
)

from ark.ui import design_tokens as T


SHORTCUTS = {
    "Fichier": [
        ("Ctrl+N", "Nouveau projet"),
        ("Ctrl+O", "Ouvrir architecture JSON"),
        ("Ctrl+S", "Sauvegarder"),
        ("Ctrl+Shift+T", "Parcourir templates"),
        ("Ctrl+Q", "Quitter"),
    ],
    "Édition": [
        ("Ctrl+Z", "Annuler"),
        ("Ctrl+Shift+Z", "Refaire"),
        ("Ctrl+C", "Copier sélection"),
        ("Ctrl+V", "Coller"),
        ("Ctrl+A", "Tout sélectionner"),
        ("Suppr / Backspace", "Supprimer sélection"),
        ("Ctrl+G", "Enregistrer sélection comme groupe"),
    ],
    "Affichage": [
        ("Ctrl+Alt+L", "Auto-layout nodes"),
        ("Ctrl+Shift+D", "Schématiser architecture (HTML)"),
        ("Ctrl+Shift+B", "Dashboard Benchmark"),
    ],
    "Navigation scene": [
        ("Molette souris", "Zoom (sous curseur)"),
        ("Bouton milieu / Espace+drag", "Pan"),
        ("Ctrl+F", "Search overlay nodes"),
        ("Double-clic vide", "Search overlay nodes"),
        ("Double-clic header node", "Toggle bypass"),
        ("Drag corner BR", "Resize node (W+H)"),
    ],
    "Aide": [
        ("F1", "Page d'aide complète"),
        ("Shift+F1", "Glossaire compact"),
        ("Cmd+/ ou Ctrl+/", "Cette cheat-sheet"),
    ],
}


class ShortcutsOverlay(QDialog):
    """Cheat-sheet raccourcis clavier ARK."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Raccourcis clavier · ARK")
        self.setMinimumSize(720, 560)
        self.setModal(False)
        self._build_ui()

    def _build_ui(self):
        v = QVBoxLayout(self)
        v.setContentsMargins(T.S_4, T.S_4, T.S_4, T.S_4)
        v.setSpacing(T.S_3)

        title = QLabel("Raccourcis Clavier", self)
        title.setStyleSheet(
            f"font-family: 'Arial'; font-size: {T.FS_HERO}px; "
            f"font-weight: 700; color: {T.ACCENT}; letter-spacing: 1px;")
        v.addWidget(title)

        sub = QLabel(
            "Vue d'ensemble des raccourcis disponibles dans ARK.", self)
        sub.setStyleSheet(f"color: {T.FG_TERTIARY}; font-size: {T.FS_LABEL}px;")
        v.addWidget(sub)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            f"background: {T.BG_PANEL}; border: 1px solid {T.BORDER_DEFAULT}; "
            f"border-radius: {T.R_MEDIUM}px;")
        inner = QWidget()
        iv = QVBoxLayout(inner)
        iv.setContentsMargins(T.S_3, T.S_3, T.S_3, T.S_3)
        iv.setSpacing(T.S_3)

        for category, shortcuts in SHORTCUTS.items():
            box = QFrame(inner)
            box.setStyleSheet(
                f"background: {T.BG_PANEL_RAISED}; "
                f"border-radius: {T.R_MEDIUM}px; "
                f"padding: {T.S_3}px;")
            bv = QVBoxLayout(box)
            bv.setContentsMargins(T.S_3, T.S_3, T.S_3, T.S_3)
            bv.setSpacing(T.S_2)

            cat_lbl = QLabel(category.upper(), box)
            cat_lbl.setStyleSheet(
                f"color: {T.ACCENT}; font-size: {T.FS_MICRO}px; "
                f"font-weight: 700; letter-spacing: 2px;")
            bv.addWidget(cat_lbl)

            grid = QGridLayout()
            grid.setSpacing(T.S_1)
            grid.setColumnStretch(1, 1)
            for row, (keys, desc) in enumerate(shortcuts):
                # Key chip
                k_lbl = QLabel(keys, box)
                k_lbl.setStyleSheet(
                    f"background: {T.BG_INPUT}; color: {T.FG_PRIMARY}; "
                    f"border: 1px solid {T.BORDER_DEFAULT}; "
                    f"border-radius: {T.R_SMALL}px; "
                    f"padding: 4px 8px; "
                    f"font-family: 'Menlo', 'Courier New', monospace; "
                    f"font-size: {T.FS_LABEL}px;")
                k_lbl.setMaximumWidth(220)
                grid.addWidget(k_lbl, row, 0)
                # Description
                d_lbl = QLabel(desc, box)
                d_lbl.setStyleSheet(
                    f"color: {T.FG_SECONDARY}; font-size: {T.FS_BODY}px; "
                    f"padding-left: {T.S_2}px;")
                grid.addWidget(d_lbl, row, 1)
            bv.addLayout(grid)
            iv.addWidget(box)

        iv.addStretch(1)
        scroll.setWidget(inner)
        v.addWidget(scroll, 1)

        footer = QHBoxLayout()
        footer.addStretch(1)
        close = QPushButton("Fermer", self)
        close.clicked.connect(self.close)
        footer.addWidget(close)
        v.addLayout(footer)
