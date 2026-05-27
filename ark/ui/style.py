"""ARK QSS (Qt Style Sheet) global. Génère depuis design_tokens."""
from __future__ import annotations
from ark.ui import design_tokens as T


def build_qss() -> str:
    """Renvoie le QSS complet pour QApplication.setStyleSheet."""
    return f"""
    /* ============================================================
       ARK — global QSS
       ============================================================ */

    QMainWindow, QWidget {{
        background-color: {T.BG_DEEP};
        color: {T.FG_PRIMARY};
        font-family: {T.FONT_SANS_FAMILY};
        font-size: {T.FS_BODY}px;
    }}

    /* Headers / labels */
    QLabel {{
        color: {T.FG_PRIMARY};
        background: transparent;
    }}
    QLabel[role="hero"] {{ font-size: {T.FS_HERO}px; font-weight: 700; letter-spacing: 0.5px; }}
    QLabel[role="h1"]   {{ font-size: {T.FS_H1}px; font-weight: 700; letter-spacing: 0.3px; }}
    QLabel[role="h2"]   {{ font-size: {T.FS_H2}px; font-weight: 600; color: {T.FG_SECONDARY}; }}
    QLabel[role="micro"] {{ font-size: {T.FS_MICRO}px; color: {T.FG_TERTIARY}; text-transform: uppercase; letter-spacing: 1px; }}
    QLabel[role="mono"] {{ font-family: {T.FONT_MONO_FAMILY}; color: {T.FG_SECONDARY}; }}

    /* DockWidget (panneaux) */
    QDockWidget {{
        background-color: {T.BG_PANEL};
        color: {T.FG_PRIMARY};
        titlebar-close-icon: none;
        titlebar-normal-icon: none;
    }}
    QDockWidget::title {{
        background-color: {T.BG_PANEL_RAISED};
        padding: {T.S_2}px {T.S_3}px;
        border-bottom: {T.B_HAIRLINE}px solid {T.BORDER_DEFAULT};
        font-size: {T.FS_MICRO}px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: {T.FG_SECONDARY};
        font-weight: 600;
    }}
    QDockWidget::close-button, QDockWidget::float-button {{
        background: transparent;
        border: none;
        icon-size: 12px;
        subcontrol-position: right;
        margin: {T.S_1}px;
    }}

    /* MenuBar + Menus */
    QMenuBar {{
        background-color: {T.BG_PANEL};
        color: {T.FG_PRIMARY};
        border-bottom: {T.B_HAIRLINE}px solid {T.BORDER_DEFAULT};
        padding: {T.S_1}px;
    }}
    QMenuBar::item {{
        background: transparent;
        padding: {T.S_1}px {T.S_3}px;
    }}
    QMenuBar::item:selected {{
        background-color: {T.BG_HIGHLIGHT};
    }}
    QMenu {{
        background-color: {T.BG_OVERLAY};
        color: {T.FG_PRIMARY};
        border: {T.B_HAIRLINE}px solid {T.BORDER_DEFAULT};
        padding: {T.S_1}px 0;
    }}
    QMenu::item {{
        padding: {T.S_2}px {T.S_4}px;
    }}
    QMenu::item:selected {{
        background-color: {T.BG_HIGHLIGHT};
        color: {T.FG_PRIMARY};
    }}
    QMenu::separator {{
        height: 1px;
        background: {T.BORDER_DEFAULT};
        margin: {T.S_1}px {T.S_2}px;
    }}

    /* StatusBar */
    QStatusBar {{
        background-color: {T.BG_PANEL};
        color: {T.FG_TERTIARY};
        border-top: {T.B_HAIRLINE}px solid {T.BORDER_DEFAULT};
        font-family: {T.FONT_MONO_FAMILY};
        font-size: {T.FS_MICRO}px;
    }}

    /* Boutons */
    QPushButton {{
        background-color: {T.BG_PANEL_RAISED};
        color: {T.FG_PRIMARY};
        border: {T.B_HAIRLINE}px solid {T.BORDER_DEFAULT};
        border-radius: {T.R_SMALL}px;
        padding: {T.S_1}px {T.S_3}px;
        font-size: {T.FS_BODY}px;
    }}
    QPushButton:hover {{
        border-color: {T.BORDER_STRONG};
        background-color: {T.BG_HIGHLIGHT};
    }}
    QPushButton:pressed {{
        background-color: {T.BG_OVERLAY};
    }}
    QPushButton[role="accent"] {{
        background-color: {T.ACCENT};
        color: {T.FG_PRIMARY};
        border-color: {T.ACCENT};
    }}
    QPushButton[role="accent"]:hover {{
        background-color: {T.ACCENT_HOVER};
    }}
    QPushButton[role="accent"]:pressed {{
        background-color: {T.ACCENT_PRESS};
    }}
    QPushButton:disabled {{
        color: {T.FG_DISABLED};
        background-color: {T.BG_DEEP};
        border-color: {T.BORDER_SUBTLE};
    }}

    /* Inputs */
    QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
        background-color: {T.BG_DEEP};
        color: {T.FG_PRIMARY};
        border: {T.B_HAIRLINE}px solid {T.BORDER_DEFAULT};
        border-radius: {T.R_SMALL}px;
        padding: {T.S_1}px {T.S_2}px;
        font-family: {T.FONT_MONO_FAMILY};
        font-size: {T.FS_BODY}px;
        selection-background-color: {T.ACCENT_DIM};
        selection-color: {T.FG_PRIMARY};
    }}
    QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
    QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {T.ACCENT};
    }}
    QComboBox::drop-down {{
        border: none; width: 18px;
    }}
    QComboBox::down-arrow {{
        width: 8px; height: 8px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {T.BG_OVERLAY};
        color: {T.FG_PRIMARY};
        selection-background-color: {T.BG_HIGHLIGHT};
        border: {T.B_HAIRLINE}px solid {T.BORDER_DEFAULT};
    }}

    /* TreeView (palette) */
    QTreeView, QListView {{
        background-color: {T.BG_PANEL};
        color: {T.FG_PRIMARY};
        border: none;
        outline: 0;
        font-size: {T.FS_BODY}px;
        alternate-background-color: {T.BG_PANEL_RAISED};
    }}
    QTreeView::item, QListView::item {{
        padding: {T.S_1}px {T.S_2}px;
    }}
    QTreeView::item:hover, QListView::item:hover {{
        background-color: {T.BG_HIGHLIGHT};
    }}
    QTreeView::item:selected, QListView::item:selected {{
        background-color: {T.ACCENT_DIM};
        color: {T.FG_PRIMARY};
    }}
    QHeaderView::section {{
        background-color: {T.BG_PANEL_RAISED};
        color: {T.FG_TERTIARY};
        padding: {T.S_1}px {T.S_2}px;
        border: none;
        border-right: {T.B_HAIRLINE}px solid {T.BORDER_DEFAULT};
        font-size: {T.FS_MICRO}px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    /* Tabs (sous-onglets dans panneaux) */
    QTabWidget::pane {{
        border: {T.B_HAIRLINE}px solid {T.BORDER_DEFAULT};
        background-color: {T.BG_PANEL};
        top: -1px;
    }}
    QTabBar::tab {{
        background-color: {T.BG_PANEL};
        color: {T.FG_TERTIARY};
        padding: {T.S_1}px {T.S_3}px;
        border: {T.B_HAIRLINE}px solid transparent;
        border-bottom: none;
        font-size: {T.FS_MICRO}px;
        text-transform: uppercase;
        letter-spacing: 1px;
        min-width: 60px;
    }}
    QTabBar::tab:selected {{
        background-color: {T.BG_PANEL_RAISED};
        color: {T.FG_PRIMARY};
        border-color: {T.BORDER_DEFAULT};
        border-bottom: 2px solid {T.ACCENT};
    }}
    QTabBar::tab:hover:!selected {{
        color: {T.FG_SECONDARY};
        background-color: {T.BG_HIGHLIGHT};
    }}

    /* ScrollBars (minimaux) */
    QScrollBar:vertical {{
        background: transparent;
        width: 10px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {T.BORDER_DEFAULT};
        border-radius: 1px;
        min-height: 24px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {T.BORDER_STRONG};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
        background: transparent; border: none; height: 0;
    }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 10px;
        margin: 0;
    }}
    QScrollBar::handle:horizontal {{
        background: {T.BORDER_DEFAULT};
        border-radius: 1px;
        min-width: 24px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: {T.BORDER_STRONG};
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
        background: transparent; border: none; width: 0;
    }}

    /* GroupBox */
    QGroupBox {{
        border: {T.B_HAIRLINE}px solid {T.BORDER_DEFAULT};
        border-radius: {T.R_SMALL}px;
        margin-top: {T.S_2}px;
        padding-top: {T.S_3}px;
        background-color: {T.BG_PANEL};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 {T.S_2}px;
        color: {T.FG_TERTIARY};
        font-size: {T.FS_MICRO}px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    /* ToolTip */
    QToolTip {{
        background-color: {T.BG_OVERLAY};
        color: {T.FG_PRIMARY};
        border: {T.B_HAIRLINE}px solid {T.BORDER_DEFAULT};
        padding: {T.S_1}px {T.S_2}px;
        font-size: {T.FS_MICRO}px;
    }}

    /* Splitter handles */
    QSplitter::handle {{
        background-color: {T.BORDER_SUBTLE};
    }}
    QSplitter::handle:hover {{
        background-color: {T.ACCENT};
    }}
    QSplitter::handle:horizontal {{
        width: 1px;
    }}
    QSplitter::handle:vertical {{
        height: 1px;
    }}
    """
