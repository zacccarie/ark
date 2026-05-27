"""Templates browser dialog : preview + favoris + filtre."""
from __future__ import annotations
import json
from pathlib import Path
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QVBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QPushButton, QLineEdit, QFrame, QWidget,
)

from ark.ui import design_tokens as T
from ark.core.templates import TEMPLATES_LIST, get_template


FAV_FILE = Path.home() / ".ark" / "favorites.json"


def _load_favs() -> set[str]:
    if FAV_FILE.exists():
        try:
            return set(json.loads(FAV_FILE.read_text()))
        except Exception:
            return set()
    return set()


def _save_favs(favs: set[str]):
    FAV_FILE.parent.mkdir(parents=True, exist_ok=True)
    FAV_FILE.write_text(json.dumps(sorted(favs), indent=2))


class TemplatesBrowser(QDialog):
    """Popup browser pour templates : liste filtrable + preview + bouton ★ fav."""

    template_chosen = Signal(str)  # template name

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Templates ARK")
        self.resize(720, 460)
        self.setStyleSheet(f"""
            QDialog {{ background-color: {T.BG_DEEP}; color: {T.FG_PRIMARY}; }}
            QListWidget {{
                background-color: {T.BG_PANEL};
                border: 1px solid {T.BORDER_DEFAULT};
                font-size: {T.FS_BODY}px;
                outline: 0;
            }}
            QListWidget::item {{ padding: 8px 10px; border-bottom: 1px solid {T.BORDER_SUBTLE}; }}
            QListWidget::item:selected {{ background-color: {T.ACCENT_DIM}; color: {T.FG_PRIMARY}; }}
            QLineEdit {{
                background-color: {T.BG_DEEP};
                color: {T.FG_PRIMARY};
                border: 1px solid {T.BORDER_DEFAULT};
                padding: 6px 8px;
                font-family: {T.FONT_SANS_FAMILY};
                font-size: {T.FS_BODY}px;
            }}
            QLineEdit:focus {{ border-color: {T.ACCENT}; }}
            QLabel#preview_title {{
                font-size: {T.FS_H1}px; font-weight: 700; color: {T.FG_PRIMARY};
            }}
            QLabel#preview_micro {{
                font-size: {T.FS_MICRO}px; color: {T.FG_TERTIARY};
                text-transform: uppercase; letter-spacing: 1px;
            }}
            QLabel#preview_body {{
                font-size: {T.FS_BODY}px; color: {T.FG_SECONDARY};
                font-family: {T.FONT_MONO_FAMILY};
            }}
            QPushButton {{
                background-color: {T.BG_PANEL_RAISED};
                color: {T.FG_PRIMARY};
                border: 1px solid {T.BORDER_DEFAULT};
                padding: 6px 14px;
            }}
            QPushButton:hover {{ border-color: {T.BORDER_STRONG}; }}
            QPushButton[role="accent"] {{
                background-color: {T.ACCENT}; border-color: {T.ACCENT};
            }}
        """)

        self.favs = _load_favs()
        self._build_ui()
        self._refilter("")

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(T.S_3, T.S_3, T.S_3, T.S_3)
        root.setSpacing(T.S_3)

        # ----- Gauche : filtre + liste -----
        left = QVBoxLayout()
        left.setSpacing(T.S_2)
        self.search = QLineEdit(self)
        self.search.setPlaceholderText("Filtrer (nom, tag)...")
        self.search.textChanged.connect(self._refilter)
        left.addWidget(self.search)

        self.list = QListWidget(self)
        self.list.currentItemChanged.connect(self._on_select)
        self.list.itemDoubleClicked.connect(self._on_double_click)
        left.addWidget(self.list)

        left_host = QWidget()
        left_host.setLayout(left)
        left_host.setFixedWidth(280)
        root.addWidget(left_host)

        # ----- Droite : preview + actions -----
        right = QVBoxLayout()
        right.setSpacing(T.S_2)

        self.preview_title = QLabel("—", self)
        self.preview_title.setObjectName("preview_title")
        right.addWidget(self.preview_title)

        self.preview_micro = QLabel("", self)
        self.preview_micro.setObjectName("preview_micro")
        right.addWidget(self.preview_micro)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color: {T.BORDER_DEFAULT};")
        right.addWidget(sep)

        self.preview_body = QLabel("", self)
        self.preview_body.setObjectName("preview_body")
        self.preview_body.setWordWrap(True)
        self.preview_body.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        right.addWidget(self.preview_body, 1)

        # Actions
        btn_row = QHBoxLayout()
        btn_row.setSpacing(T.S_2)
        self.btn_fav = QPushButton("★ Favori", self)
        self.btn_fav.setCheckable(True)
        self.btn_fav.toggled.connect(self._on_fav_toggled)
        btn_row.addWidget(self.btn_fav)
        btn_row.addStretch(1)
        self.btn_cancel = QPushButton("Annuler", self)
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_cancel)
        self.btn_open = QPushButton("Ouvrir", self)
        self.btn_open.setProperty("role", "accent")
        self.btn_open.clicked.connect(self._on_open)
        btn_row.addWidget(self.btn_open)
        right.addLayout(btn_row)

        right_host = QWidget()
        right_host.setLayout(right)
        root.addWidget(right_host, 1)

    def _refilter(self, text: str):
        text = text.strip().lower()
        self.list.clear()
        # Favoris d'abord
        items = list(TEMPLATES_LIST)
        items.sort(key=lambda nl: (nl[0] not in self.favs, nl[1].lower()))
        for name, label in items:
            if text and text not in name.lower() and text not in label.lower():
                continue
            prefix = "★ " if name in self.favs else "  "
            it = QListWidgetItem(prefix + label)
            it.setData(Qt.ItemDataRole.UserRole, name)
            self.list.addItem(it)
        if self.list.count():
            self.list.setCurrentRow(0)

    def _on_select(self, current, _previous):
        if current is None:
            self.preview_title.setText("—")
            self.preview_micro.setText("")
            self.preview_body.setText("")
            return
        name = current.data(Qt.ItemDataRole.UserRole)
        payload = get_template(name)
        if payload is None:
            return
        self.preview_title.setText(payload.get("name", name))
        meta = payload.get("metadata", {})
        tags = ", ".join(meta.get("tags", [])) or "—"
        self.preview_micro.setText(f"{name}   ·   tags : {tags}")
        # Body : liste des nodes + connexions
        nodes = payload.get("nodes", [])
        conns = payload.get("connections", [])
        lines = []
        lines.append(f"  {len(nodes)} nodes, {len(conns)} connexions")
        lines.append("")
        for n in nodes[:18]:
            params = ", ".join(f"{k}={v}" for k, v in (n.get("params") or {}).items())
            lines.append(f"  · {n['type']:<22} {params[:60]}")
        if len(nodes) > 18:
            lines.append(f"  … +{len(nodes) - 18} autres")
        self.preview_body.setText("\n".join(lines))
        self.btn_fav.blockSignals(True)
        self.btn_fav.setChecked(name in self.favs)
        self.btn_fav.blockSignals(False)

    def _on_fav_toggled(self, on: bool):
        cur = self.list.currentItem()
        if cur is None:
            return
        name = cur.data(Qt.ItemDataRole.UserRole)
        if on:
            self.favs.add(name)
        else:
            self.favs.discard(name)
        _save_favs(self.favs)
        self._refilter(self.search.text())

    def _on_open(self):
        cur = self.list.currentItem()
        if cur is None:
            return
        name = cur.data(Qt.ItemDataRole.UserRole)
        self.template_chosen.emit(name)
        self.accept()

    def _on_double_click(self, _item):
        self._on_open()
