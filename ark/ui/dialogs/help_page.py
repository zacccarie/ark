"""Phase 38 — Page d'aide complète ARK : glossaire + tutoriel + détail par concept.

QDialog non-modal pour parcourir 475+ concepts avec :
- Search bar live
- Tree par catégorie hiérarchique
- Détail rich HTML : titre / rôle / intuition / fonctionnement / équation /
  quand utiliser / exemple / piège / voir aussi

Inspiré du GlossaryDialog existant mais beaucoup plus exhaustif.
Niveau vulgarisation Bac+0/Bac+2 avec précision scientifique.
"""
from __future__ import annotations
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QTreeWidget,
    QTreeWidgetItem, QTextBrowser, QLabel, QPushButton, QSplitter,
    QWidget, QFrame,
)

from ark.core.glossary import GLOSSARY, find, by_category, GlossaryEntry
from ark.ui import design_tokens as T


class HelpPage(QDialog):
    """Page d'aide ARK avec recherche + tree + détail rich."""

    term_selected = Signal(str)

    def __init__(self, parent=None, initial_term: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("Aide ARK · Glossaire complet")
        self.setMinimumSize(1100, 720)
        self.setModal(False)
        self._build_ui()
        self._populate_tree("")
        if initial_term:
            self.select_term(initial_term)

    def _build_ui(self):
        v = QVBoxLayout(self)
        v.setContentsMargins(T.S_3, T.S_3, T.S_3, T.S_3)
        v.setSpacing(T.S_2)

        # Header
        header = QHBoxLayout()
        title = QLabel("Aide ARK", self)
        title.setStyleSheet(
            f"font-family: 'Arial'; font-size: {T.FS_HERO}px; "
            f"font-weight: 700; color: {T.ACCENT}; letter-spacing: 1px;")
        header.addWidget(title)
        header.addStretch(1)
        n_concepts = QLabel(f"{len(GLOSSARY)} concepts", self)
        n_concepts.setProperty("role", "badge")
        header.addWidget(n_concepts)
        v.addLayout(header)

        sub = QLabel(
            "Documentation exhaustive : intuition vulgarisée + précision technique. "
            "Niveau Bac → spécialiste.", self)
        sub.setStyleSheet(f"color: {T.FG_TERTIARY}; font-size: {T.FS_LABEL}px;")
        v.addWidget(sub)

        # Search bar
        self.search = QLineEdit(self)
        self.search.setPlaceholderText("Rechercher un concept, terme, équation...")
        self.search.textChanged.connect(self._on_search)
        v.addWidget(self.search)

        # Splitter : tree (left 30%) | detail (right 70%)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # Tree
        self.tree = QTreeWidget(self)
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(14)
        self.tree.currentItemChanged.connect(self._on_select)
        splitter.addWidget(self.tree)

        # Detail browser
        self.detail = QTextBrowser(self)
        self.detail.setOpenExternalLinks(False)
        self.detail.anchorClicked.connect(self._on_anchor)
        self.detail.setStyleSheet(
            f"background: {T.BG_PANEL_RAISED}; "
            f"border: 1px solid {T.BORDER_DEFAULT}; "
            f"border-radius: {T.R_MEDIUM}px; padding: {T.S_3}px;")
        splitter.addWidget(self.detail)

        splitter.setStretchFactor(0, 30)
        splitter.setStretchFactor(1, 70)
        splitter.setSizes([320, 780])
        v.addWidget(splitter, 1)

        # Footer
        footer = QHBoxLayout()
        footer.addStretch(1)
        close = QPushButton("Fermer", self)
        close.clicked.connect(self.close)
        footer.addWidget(close)
        v.addLayout(footer)

    def _populate_tree(self, query: str):
        self.tree.clear()
        entries = find(query) if query else list(GLOSSARY)
        cats: dict[str, list[GlossaryEntry]] = {}
        for e in entries:
            cats.setdefault(e.category, []).append(e)
        for cat in sorted(cats.keys()):
            cat_item = QTreeWidgetItem([f"{cat.upper()} ({len(cats[cat])})"])
            font = QFont("Arial", T.FS_MICRO)
            font.setBold(True)
            cat_item.setFont(0, font)
            cat_item.setForeground(0, Qt.GlobalColor.darkGray)
            self.tree.addTopLevelItem(cat_item)
            for e in sorted(cats[cat], key=lambda x: x.term.lower()):
                child = QTreeWidgetItem([e.term])
                child.setData(0, Qt.ItemDataRole.UserRole, e.term)
                cat_item.addChild(child)
            cat_item.setExpanded(bool(query))

    def _on_search(self, text: str):
        self._populate_tree(text)
        self.detail.setHtml(self._search_help_html(text))

    def _search_help_html(self, query: str) -> str:
        if not query:
            return self._home_html()
        return f"""
        <div style='color: {T.FG_SECONDARY}; font-family: Arial; padding: 16px;'>
            <h2 style='color: {T.ACCENT};'>Recherche : "{query}"</h2>
            <p>Sélectionne un résultat dans l'arbre à gauche.</p>
        </div>
        """

    def _on_select(self, current: QTreeWidgetItem | None, _previous):
        if current is None:
            return
        term = current.data(0, Qt.ItemDataRole.UserRole)
        if term:
            entry = next((e for e in GLOSSARY if e.term == term), None)
            if entry:
                self.detail.setHtml(self._render_entry(entry))
                self.term_selected.emit(term)

    def _render_entry(self, entry: GlossaryEntry) -> str:
        """Rich HTML : titre + sections didactiques."""
        sections = []

        # Header
        sections.append(f"""
        <div style='border-bottom: 2px solid {T.ACCENT}; padding-bottom: 8px;
                       margin-bottom: 16px;'>
            <h1 style='color: {T.FG_PRIMARY}; font-family: Arial; margin: 0;
                          font-size: 22px;'>{entry.term}</h1>
            <div style='color: {T.FG_TERTIARY}; font-size: 10px;
                           text-transform: uppercase; letter-spacing: 2px;
                           margin-top: 4px;'>{entry.category}</div>
        </div>
        """)

        # Rôle (short)
        if entry.short:
            sections.append(f"""
            <div style='background: {T.ACCENT_DIM}; border-left: 3px solid {T.ACCENT};
                          padding: 10px 14px; margin-bottom: 16px; border-radius: 4px;'>
                <span style='color: {T.FG_TERTIARY}; font-size: 10px;
                                text-transform: uppercase; letter-spacing: 1px;'>Rôle</span><br>
                <span style='color: {T.FG_PRIMARY}; font-size: 14px;
                                font-weight: 600;'>{entry.short}</span>
            </div>
            """)

        # Intuition (vulgarisée)
        if entry.intuition:
            sections.append(self._section_html(
                "Intuition", entry.intuition, "#00FF85", icon="💡"))

        # Comment ça marche (long)
        if entry.long and entry.long != entry.short:
            sections.append(self._section_html(
                "Comment ça marche", entry.long, T.INFO, icon="⚙"))

        # Équation
        if entry.equation:
            sections.append(f"""
            <div style='margin-bottom: 16px;'>
                <div style='color: {T.FG_TERTIARY}; font-size: 10px;
                                text-transform: uppercase; letter-spacing: 1px;
                                margin-bottom: 6px;'>Équation</div>
                <pre style='background: rgba(225, 29, 42, 0.08);
                              border-left: 2px solid {T.ACCENT};
                              padding: 12px 16px; margin: 0;
                              font-family: Menlo, monospace; font-size: 13px;
                              color: {T.ACCENT};'>{entry.equation}</pre>
            </div>
            """)

        # When to use
        if entry.when_to_use:
            sections.append(self._section_html(
                "Quand l'utiliser", entry.when_to_use, T.INFO, icon="🎯"))

        # Example
        if entry.example:
            sections.append(self._section_html(
                "Exemple concret", entry.example, T.SUCCESS, icon="📐",
                mono=True))

        # Pitfall
        if entry.pitfall:
            sections.append(self._section_html(
                "Piège fréquent", entry.pitfall, T.WARN, icon="⚠"))

        # Related
        if entry.related:
            links = " · ".join(
                f"<a href='term://{r}' style='color: {T.ACCENT};'>{r}</a>"
                for r in entry.related[:10]
            )
            sections.append(f"""
            <div style='margin-top: 24px; padding-top: 12px;
                          border-top: 1px solid {T.BORDER_DEFAULT};'>
                <div style='color: {T.FG_TERTIARY}; font-size: 10px;
                                text-transform: uppercase; letter-spacing: 1px;
                                margin-bottom: 6px;'>Voir aussi</div>
                <div>{links}</div>
            </div>
            """)

        body = "".join(sections)
        return f"""
        <html><body style='background: {T.BG_PANEL_RAISED}; color: {T.FG_SECONDARY};
                              font-family: Arial; line-height: 1.6; padding: 16px;'>
        {body}
        </body></html>
        """

    def _section_html(self, title: str, text: str, color: str,
                          icon: str = "", mono: bool = False) -> str:
        font_family = "Menlo, monospace" if mono else "Arial"
        return f"""
        <div style='margin-bottom: 16px;'>
            <div style='color: {T.FG_TERTIARY}; font-size: 10px;
                            text-transform: uppercase; letter-spacing: 1px;
                            margin-bottom: 6px;'>{icon} {title}</div>
            <div style='color: {T.FG_PRIMARY}; font-size: 13px;
                            font-family: {font_family}; padding-left: 12px;
                            border-left: 2px solid {color};'>{text}</div>
        </div>
        """

    def _home_html(self) -> str:
        cats = by_category()
        cat_list = "".join(
            f"<li><strong style='color: {T.ACCENT};'>{c}</strong> "
            f"<span style='color: {T.FG_TERTIARY};'>· {len(cats[c])} concepts</span></li>"
            for c in sorted(cats.keys())
        )
        return f"""
        <html><body style='background: {T.BG_PANEL_RAISED}; color: {T.FG_SECONDARY};
                              font-family: Arial; padding: 20px; line-height: 1.6;'>
            <h1 style='color: {T.ACCENT}; font-size: 24px;'>
                Bienvenue dans l'aide ARK
            </h1>
            <p>Cette page regroupe la documentation complète : <strong>{len(GLOSSARY)} concepts</strong>
            organisés en <strong>{len(cats)} catégories</strong>.</p>

            <h2 style='color: {T.FG_PRIMARY}; font-size: 16px; margin-top: 24px;'>
                Comment utiliser
            </h2>
            <ul>
                <li><strong>Rechercher</strong> par terme/équation dans la barre du haut</li>
                <li><strong>Parcourir</strong> l'arbre à gauche par catégorie</li>
                <li><strong>Cliquer</strong> un concept pour son détail complet</li>
                <li>Les liens <strong style='color: {T.ACCENT};'>colorés</strong> dans la section
                "Voir aussi" amènent aux concepts liés</li>
            </ul>

            <h2 style='color: {T.FG_PRIMARY}; font-size: 16px; margin-top: 24px;'>
                Catégories disponibles
            </h2>
            <ul style='line-height: 1.8;'>
                {cat_list}
            </ul>

            <div style='margin-top: 32px; padding: 12px; background: {T.ACCENT_DIM};
                          border-left: 3px solid {T.ACCENT}; border-radius: 4px;'>
                <strong style='color: {T.ACCENT};'>💡 Astuce</strong>
                <p style='margin: 4px 0 0 0; color: {T.FG_SECONDARY};'>
                    Pour ouvrir l'aide pour un node spécifique : sélectionne le node →
                    bouton "Aide" dans l'Inspector (ou double-clic).
                </p>
            </div>
        </body></html>
        """

    def _on_anchor(self, url):
        s = url.toString()
        if s.startswith("term://"):
            term = s[len("term://"):]
            self.select_term(term)

    def select_term(self, term: str):
        """Sélectionne et affiche le concept dans l'arbre."""
        for i in range(self.tree.topLevelItemCount()):
            cat = self.tree.topLevelItem(i)
            for j in range(cat.childCount()):
                child = cat.child(j)
                if child.data(0, Qt.ItemDataRole.UserRole) == term:
                    cat.setExpanded(True)
                    self.tree.setCurrentItem(child)
                    return
