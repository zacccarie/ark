"""Glossaire ARK — dialog cherchable avec liste + détail."""
from __future__ import annotations
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QListWidget, QListWidgetItem,
    QTextBrowser, QLabel, QPushButton, QSplitter, QWidget,
)

from ark.core.glossary import GLOSSARY, find, by_category, get as glossary_get, GlossaryEntry
from ark.ui import design_tokens as T


class GlossaryDialog(QDialog):
    """Dialog modal liste + détail. Recherche live."""

    term_selected = Signal(str)

    def __init__(self, parent=None, initial_term: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("Glossaire ARK")
        self.setMinimumSize(900, 600)
        self.setModal(False)  # non-bloquant pour pouvoir manipuler le canvas en parallèle
        self._build_ui()
        self._populate_list("")
        if initial_term:
            self.select_term(initial_term)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Header : titre + count
        header = QHBoxLayout()
        title = QLabel("Glossaire")
        title.setStyleSheet(f"font-size: 18px; font-weight: 600; color: {T.FG_PRIMARY};")
        header.addWidget(title)
        header.addStretch()
        self._count_label = QLabel(f"{len(GLOSSARY)} termes")
        self._count_label.setStyleSheet(f"color: {T.FG_SECONDARY}; font-size: 11px;")
        header.addWidget(self._count_label)
        layout.addLayout(header)

        # Search bar
        self._search = QLineEdit()
        self._search.setPlaceholderText("Rechercher dans terme + définition…")
        self._search.textChanged.connect(self._on_search)
        self._search.setStyleSheet(
            f"QLineEdit {{ background:{T.BG_PANEL}; color:{T.FG_PRIMARY}; "
            f"border:1px solid {T.BORDER_DEFAULT}; border-radius:6px; padding:6px 10px; "
            f"font-size:12px; }} "
            f"QLineEdit:focus {{ border-color:{T.ACCENT}; }}"
        )
        layout.addWidget(self._search)

        # Splitter list / detail
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)

        # ---- List côté gauche ----
        self._list = QListWidget()
        self._list.setStyleSheet(
            f"QListWidget {{ background:{T.BG_PANEL}; color:{T.FG_PRIMARY}; "
            f"border:1px solid {T.BORDER_DEFAULT}; border-radius:6px; padding:4px; "
            f"font-size:12px; outline:0; }} "
            f"QListWidget::item {{ padding:6px 10px; border-radius:3px; }} "
            f"QListWidget::item:hover {{ background:{T.BG_HIGHLIGHT}; }} "
            f"QListWidget::item:selected {{ background:{T.ACCENT}; color:white; }}"
        )
        self._list.currentItemChanged.connect(self._on_select)
        splitter.addWidget(self._list)

        # ---- Detail côté droit ----
        detail_wrap = QWidget()
        detail_layout = QVBoxLayout(detail_wrap)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(8)

        self._detail = QTextBrowser()
        self._detail.setOpenExternalLinks(False)
        self._detail.anchorClicked.connect(self._on_anchor)
        # Anti-double resolve : on gère via anchorClicked à la place
        self._detail.setOpenLinks(False)
        self._detail.setStyleSheet(
            f"QTextBrowser {{ background:{T.BG_PANEL}; color:{T.FG_PRIMARY}; "
            f"border:1px solid {T.BORDER_DEFAULT}; border-radius:6px; padding:16px; "
            f"font-size:13px; }}"
        )
        detail_layout.addWidget(self._detail)
        splitter.addWidget(detail_wrap)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([280, 600])
        layout.addWidget(splitter, 1)

        # Footer : close button
        footer = QHBoxLayout()
        footer.addStretch()
        close = QPushButton("Fermer")
        close.clicked.connect(self.close)
        close.setStyleSheet(
            f"QPushButton {{ background:{T.BG_HIGHLIGHT}; color:{T.FG_PRIMARY}; "
            f"border:1px solid {T.BORDER_DEFAULT}; border-radius:6px; padding:6px 16px; "
            f"font-size:12px; }} "
            f"QPushButton:hover {{ background:{T.BG_OVERLAY}; border-color:{T.BORDER_STRONG}; }}"
        )
        footer.addWidget(close)
        layout.addLayout(footer)

        self.setStyleSheet(f"QDialog {{ background:{T.BG_DEEP}; }}")

    def _populate_list(self, query: str):
        self._list.clear()
        results = find(query)
        # Group by category, ordered
        cats: dict[str, list[GlossaryEntry]] = {}
        for e in results:
            cats.setdefault(e.category, []).append(e)
        for cat in sorted(cats):
            # Header item (non-selectable)
            header = QListWidgetItem(f"  {cat.upper()}")
            header.setFlags(Qt.ItemFlag.NoItemFlags)
            f = QFont()
            f.setBold(True)
            f.setPointSize(10)
            header.setFont(f)
            header.setForeground(Qt.GlobalColor.gray)
            self._list.addItem(header)
            for entry in sorted(cats[cat], key=lambda e: e.term.lower()):
                item = QListWidgetItem(f"   {entry.term}")
                item.setData(Qt.ItemDataRole.UserRole, entry.term)
                self._list.addItem(item)
        self._count_label.setText(f"{len(results)} terme{'s' if len(results) > 1 else ''}")
        if results and self._list.count() > 1:
            # Sélectionne le premier item non-header
            for i in range(self._list.count()):
                if self._list.item(i).flags() & Qt.ItemFlag.ItemIsSelectable:
                    self._list.setCurrentRow(i)
                    break

    def _on_search(self, text: str):
        self._populate_list(text)

    def _on_select(self, current: QListWidgetItem | None, _previous):
        if current is None:
            return
        term = current.data(Qt.ItemDataRole.UserRole)
        if not term:
            return
        entry = glossary_get(term)
        if entry:
            self._render_detail(entry)
            self.term_selected.emit(term)

    def _render_detail(self, entry: GlossaryEntry):
        related_html = ""
        if entry.related:
            links = " · ".join(
                f'<a href="ark:{r}" style="color:{T.ACCENT}; text-decoration:none;">{r}</a>'
                for r in entry.related
            )
            related_html = (
                f'<p style="margin-top:18px; color:{T.FG_SECONDARY}; font-size:11px;">'
                f'<b>Voir aussi :</b> {links}</p>'
            )
        html = f"""
        <div>
          <p style="color:{T.ACCENT}; font-size:10px; letter-spacing:0.1em;
                    text-transform:uppercase; margin:0 0 6px 0;">{entry.category}</p>
          <h2 style="margin:0 0 8px 0; color:{T.FG_PRIMARY}; font-size:20px;">{entry.term}</h2>
          <p style="color:{T.FG_SECONDARY}; font-style:italic; margin:0 0 16px 0;
                    font-size:13px; line-height:1.5;">{entry.short}</p>
          <p style="color:{T.FG_PRIMARY}; line-height:1.7; font-size:13px;
                    margin:0;">{entry.long}</p>
          {related_html}
        </div>
        """
        self._detail.setHtml(html)

    def _on_anchor(self, url):
        s = url.toString()
        if s.startswith("ark:"):
            self.select_term(s[4:])

    def select_term(self, term: str):
        """Sélectionne et affiche un terme par son nom."""
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == term:
                self._list.setCurrentRow(i)
                return
