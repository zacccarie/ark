"""ARK QSS — Phase 41 premium CRM-inspired.

Inspiration : CRM Rénovation Énergétique (Inter+PP Neue Machina, R18 cards,
vert électrique primary, zinc neutrals, transitions douces).
"""
from __future__ import annotations
from ark.ui import design_tokens as T


def build_qss() -> str:
    return f"""
    /* ============================================================
       ARK Premium — Phase 41
       ============================================================ */

    QMainWindow, QWidget {{
        background-color: {T.BG_DEEP};
        color: {T.FG_PRIMARY};
        font-family: {T.FONT_SANS_FAMILY};
        font-size: {T.FS_BODY}px;
    }}

    /* Labels */
    QLabel {{
        color: {T.FG_PRIMARY};
        background: transparent;
    }}
    QLabel[role="hero"] {{
        font-family: {T.FONT_DISPLAY_FAMILY};
        font-size: {T.FS_HERO}px; font-weight: 800;
        letter-spacing: -0.5px;
    }}
    QLabel[role="h1"] {{
        font-family: {T.FONT_DISPLAY_FAMILY};
        font-size: {T.FS_H1}px; font-weight: 700;
        letter-spacing: -0.3px;
    }}
    QLabel[role="h2"] {{
        font-size: {T.FS_H2}px; font-weight: 600;
        color: {T.FG_SECONDARY}; letter-spacing: -0.2px;
    }}
    QLabel[role="micro"] {{
        font-size: {T.FS_MICRO}px; color: {T.FG_TERTIARY};
        text-transform: uppercase; letter-spacing: 1.5px;
        font-weight: 600;
    }}
    QLabel[role="mono"] {{
        font-family: {T.FONT_MONO_FAMILY};
        color: {T.FG_SECONDARY};
    }}

    /* DockWidget panneaux — premium spacing */
    QDockWidget {{
        background-color: {T.BG_PANEL};
        color: {T.FG_PRIMARY};
        titlebar-close-icon: none;
        titlebar-normal-icon: none;
    }}
    QDockWidget::title {{
        background-color: {T.BG_PANEL_RAISED};
        padding: {T.S_3}px {T.S_4}px;
        border-bottom: 1px solid {T.BORDER_DEFAULT};
        font-family: {T.FONT_DISPLAY_FAMILY};
        font-size: {T.FS_LABEL}px;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: {T.FG_SECONDARY};
        font-weight: 700;
    }}
    QDockWidget::close-button, QDockWidget::float-button {{
        background: transparent; border: none;
        icon-size: 14px;
        subcontrol-position: right; margin: {T.S_1}px;
    }}

    /* MenuBar + Menus */
    QMenuBar {{
        background-color: {T.BG_PANEL};
        color: {T.FG_PRIMARY};
        border-bottom: 1px solid {T.BORDER_DEFAULT};
        padding: {T.S_2}px {T.S_3}px;
        font-weight: 500;
    }}
    QMenuBar::item {{
        background: transparent;
        padding: {T.S_2}px {T.S_4}px;
        border-radius: {T.R_SMALL}px;
        margin: 0 2px;
    }}
    QMenuBar::item:selected {{
        background-color: {T.BG_HIGHLIGHT};
        color: {T.PRIMARY};
    }}
    QMenu {{
        background-color: {T.BG_OVERLAY};
        color: {T.FG_PRIMARY};
        border: 1px solid {T.BORDER_DEFAULT};
        border-radius: {T.R_MEDIUM}px;
        padding: {T.S_2}px 0;
    }}
    QMenu::item {{
        padding: {T.S_2}px {T.S_5}px;
        margin: 1px {T.S_2}px;
        border-radius: {T.R_SMALL}px;
    }}
    QMenu::item:selected {{
        background-color: {T.PRIMARY_DIM};
        color: {T.PRIMARY};
    }}
    QMenu::separator {{
        height: 1px;
        background: {T.BORDER_DEFAULT};
        margin: {T.S_2}px {T.S_3}px;
    }}

    /* StatusBar premium */
    QStatusBar {{
        background-color: {T.BG_PANEL};
        color: {T.FG_TERTIARY};
        border-top: 1px solid {T.BORDER_DEFAULT};
        font-family: {T.FONT_MONO_FAMILY};
        font-size: {T.FS_MICRO}px;
        padding: {T.S_1}px {T.S_3}px;
    }}

    /* Boutons premium CRM-style */
    QPushButton {{
        background-color: {T.BG_PANEL_RAISED};
        color: {T.FG_PRIMARY};
        border: 1px solid {T.BORDER_DEFAULT};
        border-radius: {T.R_MEDIUM}px;
        padding: {T.S_2}px {T.S_5}px;
        font-family: {T.FONT_SANS_FAMILY};
        font-size: {T.FS_BODY}px;
        font-weight: 500;
        min-height: 32px;
        letter-spacing: -0.1px;
    }}
    QPushButton:hover {{
        background-color: {T.BG_HIGHLIGHT};
        border-color: {T.BORDER_STRONG};
    }}
    QPushButton:pressed {{
        background-color: {T.BG_OVERLAY};
    }}
    QPushButton:disabled {{
        color: {T.FG_DISABLED};
        background-color: {T.BG_DEEP};
        border-color: {T.BORDER_SUBTLE};
    }}

    /* Primary = vert électrique CRM */
    QPushButton[role="primary"], QPushButton[role="accent"] {{
        background-color: {T.PRIMARY};
        color: {T.FG_INVERSE};
        border: 1px solid {T.PRIMARY};
        font-weight: 700;
    }}
    QPushButton[role="primary"]:hover, QPushButton[role="accent"]:hover {{
        background-color: {T.PRIMARY_HOVER};
        border-color: {T.PRIMARY_HOVER};
    }}
    QPushButton[role="primary"]:pressed, QPushButton[role="accent"]:pressed {{
        background-color: {T.PRIMARY_PRESS};
    }}

    /* Success = même que primary */
    QPushButton[role="success"] {{
        background-color: {T.PRIMARY_DIM};
        color: {T.PRIMARY};
        border: 1px solid {T.PRIMARY};
        font-weight: 600;
    }}
    QPushButton[role="success"]:hover {{
        background-color: {T.PRIMARY};
        color: {T.FG_INVERSE};
    }}

    /* Warn / Info */
    QPushButton[role="warn"] {{
        background-color: {T.WARN_DIM};
        color: {T.WARN};
        border: 1px solid {T.WARN};
        font-weight: 600;
    }}
    QPushButton[role="warn"]:hover {{
        background-color: {T.WARN}; color: {T.FG_INVERSE};
    }}
    QPushButton[role="info"] {{
        background-color: {T.INFO_DIM};
        color: {T.INFO};
        border: 1px solid {T.INFO};
        font-weight: 600;
    }}
    QPushButton[role="info"]:hover {{
        background-color: {T.INFO}; color: {T.FG_INVERSE};
    }}

    /* Danger (rouge ARK) */
    QPushButton[role="danger"] {{
        background-color: {T.ACCENT};
        color: {T.FG_PRIMARY};
        border: 1px solid {T.ACCENT};
        font-weight: 700;
    }}
    QPushButton[role="danger"]:hover {{
        background-color: {T.ACCENT_HOVER};
    }}

    /* Ghost (transparent) */
    QPushButton[role="ghost"] {{
        background-color: transparent;
        border: 1px solid transparent;
        color: {T.FG_SECONDARY};
        font-weight: 500;
    }}
    QPushButton[role="ghost"]:hover {{
        background-color: {T.BG_HIGHLIGHT};
        color: {T.FG_PRIMARY};
        border-color: {T.BORDER_DEFAULT};
    }}

    /* Status Badges Pills — vert primary par défaut */
    QLabel[role="badge"] {{
        background-color: {T.BG_HIGHLIGHT};
        border: 1px solid {T.BORDER_DEFAULT};
        border-radius: {T.R_PILL}px;
        padding: {T.S_1}px {T.S_3}px;
        font-size: {T.FS_MICRO}px;
        font-weight: 700;
        color: {T.FG_SECONDARY};
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    QLabel[role="badge-success"] {{
        background-color: {T.PRIMARY_DIM};
        border: 1px solid {T.PRIMARY};
        border-radius: {T.R_PILL}px;
        padding: {T.S_1}px {T.S_3}px;
        font-size: {T.FS_MICRO}px;
        font-weight: 700;
        color: {T.PRIMARY};
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    QLabel[role="badge-warn"] {{
        background-color: {T.WARN_DIM};
        border: 1px solid {T.WARN};
        border-radius: {T.R_PILL}px;
        padding: {T.S_1}px {T.S_3}px;
        font-size: {T.FS_MICRO}px;
        font-weight: 700;
        color: {T.WARN};
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    QLabel[role="badge-error"] {{
        background-color: {T.ERROR_DIM};
        border: 1px solid {T.ERROR};
        border-radius: {T.R_PILL}px;
        padding: {T.S_1}px {T.S_3}px;
        font-size: {T.FS_MICRO}px;
        font-weight: 700;
        color: {T.ERROR};
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    QLabel[role="badge-info"] {{
        background-color: {T.INFO_DIM};
        border: 1px solid {T.INFO};
        border-radius: {T.R_PILL}px;
        padding: {T.S_1}px {T.S_3}px;
        font-size: {T.FS_MICRO}px;
        font-weight: 700;
        color: {T.INFO};
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    /* Inputs premium */
    QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
        background-color: {T.BG_INPUT};
        color: {T.FG_PRIMARY};
        border: 1px solid {T.BORDER_DEFAULT};
        border-radius: {T.R_MEDIUM}px;
        padding: {T.S_3}px {T.S_4}px;
        font-family: {T.FONT_MONO_FAMILY};
        font-size: {T.FS_BODY}px;
        selection-background-color: {T.PRIMARY_DIM};
        selection-color: {T.PRIMARY};
        min-height: 28px;
    }}
    QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
    QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
        border: 2px solid {T.PRIMARY};
        padding: {T.S_3 - 1}px {T.S_4 - 1}px;
    }}

    /* QSpinBox arrows agrandies + visibles */
    QSpinBox, QDoubleSpinBox {{
        padding-right: 28px;
    }}
    QSpinBox::up-button, QDoubleSpinBox::up-button {{
        subcontrol-origin: border;
        subcontrol-position: top right;
        width: 22px;
        height: 18px;
        border-left: 1px solid {T.BORDER_DEFAULT};
        border-bottom: 1px solid {T.BORDER_DEFAULT};
        border-top-right-radius: {T.R_MEDIUM}px;
        background: {T.BG_PANEL_RAISED};
    }}
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        width: 22px;
        height: 18px;
        border-left: 1px solid {T.BORDER_DEFAULT};
        border-bottom-right-radius: {T.R_MEDIUM}px;
        background: {T.BG_PANEL_RAISED};
    }}
    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
        background: {T.PRIMARY_DIM};
    }}
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
        image: none;
        width: 10px; height: 10px;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-bottom: 6px solid {T.FG_SECONDARY};
    }}
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
        image: none;
        width: 10px; height: 10px;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 6px solid {T.FG_SECONDARY};
    }}

    QComboBox {{
        padding-right: 32px;
    }}
    QComboBox::drop-down {{
        border-left: 1px solid {T.BORDER_DEFAULT};
        width: 28px;
        background: {T.BG_PANEL_RAISED};
        border-top-right-radius: {T.R_MEDIUM}px;
        border-bottom-right-radius: {T.R_MEDIUM}px;
    }}
    QComboBox::drop-down:hover {{
        background: {T.PRIMARY_DIM};
    }}
    QComboBox::down-arrow {{
        image: none;
        width: 12px; height: 12px;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 6px solid {T.FG_SECONDARY};
    }}
    QComboBox QAbstractItemView {{
        background-color: {T.BG_OVERLAY};
        color: {T.FG_PRIMARY};
        selection-background-color: {T.PRIMARY_DIM};
        selection-color: {T.PRIMARY};
        border: 1px solid {T.BORDER_DEFAULT};
        border-radius: {T.R_MEDIUM}px;
        padding: {T.S_2}px;
        outline: 0;
    }}

    /* TreeView palette premium */
    QTreeView, QListView {{
        background-color: {T.BG_PANEL};
        color: {T.FG_PRIMARY};
        border: none;
        outline: 0;
        font-size: {T.FS_BODY}px;
        alternate-background-color: transparent;
        padding: {T.S_2}px;
    }}
    QTreeView::item, QListView::item {{
        padding: {T.S_2}px {T.S_3}px;
        border-radius: {T.R_SMALL}px;
        margin: 1px 0;
    }}
    QTreeView::item:hover, QListView::item:hover {{
        background-color: {T.BG_HIGHLIGHT};
    }}
    QTreeView::item:selected, QListView::item:selected {{
        background-color: {T.PRIMARY_DIM};
        color: {T.PRIMARY};
    }}
    QHeaderView::section {{
        background-color: {T.BG_PANEL_RAISED};
        color: {T.FG_TERTIARY};
        padding: {T.S_2}px {T.S_3}px;
        border: none;
        border-right: 1px solid {T.BORDER_DEFAULT};
        font-size: {T.FS_MICRO}px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 700;
    }}

    /* Tabs premium — fix bg gris/silver Qt natif */
    QTabWidget {{
        background-color: {T.BG_PANEL};
    }}
    QTabWidget::pane {{
        border: 1px solid {T.BORDER_DEFAULT};
        background-color: {T.BG_PANEL};
        border-radius: {T.R_MEDIUM}px;
        top: -1px;
    }}
    QTabWidget::tab-bar {{
        background-color: {T.BG_PANEL};
        alignment: left;
    }}
    QTabBar {{
        background-color: {T.BG_PANEL};
        border: none;
        qproperty-drawBase: 0;
    }}
    QTabBar::tab {{
        background-color: {T.BG_PANEL};
        color: {T.FG_TERTIARY};
        padding: {T.S_3}px {T.S_4}px;
        border: none;
        border-bottom: 2px solid transparent;
        font-family: {T.FONT_DISPLAY_FAMILY};
        font-size: {T.FS_LABEL}px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 700;
        min-width: 80px;
        margin: 0;
    }}
    QTabBar::tab:selected {{
        background-color: {T.BG_PANEL};
        color: {T.PRIMARY};
        border-bottom: 2px solid {T.PRIMARY};
    }}
    QTabBar::tab:hover:!selected {{
        color: {T.FG_PRIMARY};
        background-color: {T.BG_HIGHLIGHT};
    }}

    /* ScrollBars premium */
    QScrollBar:vertical {{
        background: transparent;
        width: 12px;
        margin: 4px 2px;
    }}
    QScrollBar::handle:vertical {{
        background: {T.BORDER_DEFAULT};
        border-radius: 4px;
        min-height: 32px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {T.PRIMARY};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
        background: transparent; border: none; height: 0;
    }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 12px;
        margin: 2px 4px;
    }}
    QScrollBar::handle:horizontal {{
        background: {T.BORDER_DEFAULT};
        border-radius: 4px;
        min-width: 32px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: {T.PRIMARY};
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
        background: transparent; border: none; width: 0;
    }}

    /* GroupBox premium */
    QGroupBox {{
        border: 1px solid {T.BORDER_DEFAULT};
        border-radius: {T.R_LARGE}px;
        margin-top: {T.S_3}px;
        padding-top: {T.S_4}px;
        background-color: {T.BG_PANEL};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 {T.S_3}px;
        color: {T.FG_TERTIARY};
        font-size: {T.FS_MICRO}px;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 700;
    }}

    /* ToolTip premium */
    QToolTip {{
        background-color: {T.BG_OVERLAY};
        color: {T.FG_PRIMARY};
        border: 1px solid {T.BORDER_DEFAULT};
        border-radius: {T.R_MEDIUM}px;
        padding: {T.S_3}px {T.S_4}px;
        font-size: {T.FS_LABEL}px;
        font-family: {T.FONT_SANS_FAMILY};
    }}

    /* Splitter handles */
    QSplitter::handle {{
        background-color: {T.BG_DEEP};
    }}
    QSplitter::handle:hover {{
        background-color: {T.PRIMARY};
    }}
    QSplitter::handle:horizontal {{
        width: 2px;
    }}
    QSplitter::handle:vertical {{
        height: 2px;
    }}

    /* CheckBox + RadioButton */
    QCheckBox, QRadioButton {{
        spacing: {T.S_2}px;
        color: {T.FG_PRIMARY};
        font-size: {T.FS_BODY}px;
    }}
    QCheckBox::indicator, QRadioButton::indicator {{
        width: 18px; height: 18px;
        border: 1px solid {T.BORDER_DEFAULT};
        background-color: {T.BG_INPUT};
    }}
    QCheckBox::indicator {{
        border-radius: {T.R_SMALL}px;
    }}
    QRadioButton::indicator {{
        border-radius: 9px;
    }}
    QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
        background-color: {T.PRIMARY};
        border-color: {T.PRIMARY};
    }}
    QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
        border-color: {T.BORDER_FOCUS};
    }}

    /* Frames horizontales */
    QFrame[frameShape="4"] {{  /* HLine */
        color: {T.BORDER_DEFAULT};
        background-color: {T.BORDER_DEFAULT};
        max-height: 1px;
    }}
    """
