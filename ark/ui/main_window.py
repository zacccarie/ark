"""ARK MainWindow — shell workspace multi-panneaux (QDockWidget).

Layout par défaut :
- Gauche : palette nodes (catégorisée).
- Centre : node editor (QGraphicsView).
- Droite : inspector (params node sélectionné) + outliner.
- Bas : terminal / log / charts (sous-onglets).

Tous panneaux QDockWidget natifs : drag, undock, redimension, fermeture.
"""
from __future__ import annotations
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QWidget, QVBoxLayout, QLabel,
    QTabWidget, QToolBar, QStatusBar, QApplication,
)

from ark.ui import design_tokens as T
from ark.ui.panels.palette_panel import PalettePanel
from ark.ui.panels.inspector_panel import InspectorPanel
from ark.ui.panels.terminal_panel import TerminalPanel
from ark.ui.node_editor.view import NodeEditorView
from ark.ui.node_editor.scene import NodeScene
from ark.nodes.registry import default_registry


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ARK")
        self.resize(1600, 1000)
        self.setMinimumSize(QSize(1000, 600))
        self.setDockOptions(
            QMainWindow.DockOption.AnimatedDocks
            | QMainWindow.DockOption.AllowNestedDocks
            | QMainWindow.DockOption.AllowTabbedDocks
        )

        self._registry = default_registry()
        self._build_menus()
        self._build_toolbar()
        self._build_status_bar()
        self._build_central()
        self._build_docks()

    # ------------------------------------------------------------
    # Menus
    # ------------------------------------------------------------
    def _build_menus(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("Fichier")
        a_new = QAction("Nouveau", self)
        a_new.setShortcut(QKeySequence.StandardKey.New)
        a_new.triggered.connect(self._on_new)
        file_menu.addAction(a_new)

        a_open = QAction("Ouvrir...", self)
        a_open.setShortcut(QKeySequence.StandardKey.Open)
        a_open.triggered.connect(self._on_open)
        file_menu.addAction(a_open)

        a_save = QAction("Enregistrer", self)
        a_save.setShortcut(QKeySequence.StandardKey.Save)
        a_save.triggered.connect(self._on_save)
        file_menu.addAction(a_save)

        file_menu.addSeparator()
        a_quit = QAction("Quitter", self)
        a_quit.setShortcut(QKeySequence.StandardKey.Quit)
        a_quit.triggered.connect(QApplication.instance().quit)
        file_menu.addAction(a_quit)

        edit_menu = menubar.addMenu("Édition")
        edit_menu.addAction("Annuler", self._on_undo, QKeySequence.StandardKey.Undo)
        edit_menu.addAction("Refaire", self._on_redo, QKeySequence.StandardKey.Redo)
        edit_menu.addSeparator()
        edit_menu.addAction("Tout sélectionner", lambda: None, QKeySequence.StandardKey.SelectAll)

        view_menu = menubar.addMenu("Affichage")
        view_menu.addAction("Reset layout", self._on_reset_layout)
        view_menu.addAction("Layout : Workspace", lambda: self._apply_preset("workspace"))
        view_menu.addAction("Layout : Compact", lambda: self._apply_preset("compact"))
        view_menu.addAction("Layout : Code-first", lambda: self._apply_preset("code"))

        run_menu = menubar.addMenu("Exécution")
        a_run = QAction("Compiler", self)
        a_run.setShortcut(QKeySequence("Ctrl+R"))
        a_run.triggered.connect(self._on_compile)
        run_menu.addAction(a_run)

        a_train = QAction("Entraîner", self)
        a_train.setShortcut(QKeySequence("Ctrl+T"))
        a_train.triggered.connect(self._on_train)
        run_menu.addAction(a_train)

        help_menu = menubar.addMenu("Aide")
        help_menu.addAction("À propos d'ARK", self._on_about)

    # ------------------------------------------------------------
    # Toolbar (top)
    # ------------------------------------------------------------
    def _build_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(QSize(16, 16))
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)
        tb.addAction("▶ Compiler", self._on_compile)
        tb.addAction("⏵ Entraîner", self._on_train)
        tb.addAction("■ Stop", self._on_stop)
        tb.addSeparator()
        tb.addAction("⌕ Rechercher node", self._on_search)

    # ------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------
    def _build_status_bar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        sb.showMessage("ARK · ready")

    # ------------------------------------------------------------
    # Central widget : node editor
    # ------------------------------------------------------------
    def _build_central(self):
        self.scene = NodeScene(registry=self._registry)
        self.view = NodeEditorView(self.scene)
        self.setCentralWidget(self.view)

    # ------------------------------------------------------------
    # Docks
    # ------------------------------------------------------------
    def _build_docks(self):
        # Palette gauche
        self.palette = PalettePanel(registry=self._registry, parent=self)
        self.palette.node_double_clicked.connect(self._on_palette_node_dropped)
        dock_pal = QDockWidget("Palette", self)
        dock_pal.setObjectName("dock_palette")
        dock_pal.setWidget(self.palette)
        dock_pal.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock_pal)
        self.dock_palette = dock_pal

        # Inspector droit
        self.inspector = InspectorPanel(parent=self)
        dock_insp = QDockWidget("Inspector", self)
        dock_insp.setObjectName("dock_inspector")
        dock_insp.setWidget(self.inspector)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock_insp)
        self.dock_inspector = dock_insp

        # Terminal / logs en bas (tabs)
        self.terminal = TerminalPanel(parent=self)
        dock_term = QDockWidget("Terminal", self)
        dock_term.setObjectName("dock_terminal")
        dock_term.setWidget(self.terminal)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock_term)
        self.dock_terminal = dock_term

        # Connection sélection node ↔ inspector
        self.scene.selectionChanged.connect(self._on_scene_selection_changed)

    # ------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------
    def _on_palette_node_dropped(self, type_name: str):
        # Spawn node au centre de la vue
        center = self.view.mapToScene(self.view.viewport().rect().center())
        self.scene.create_node(type_name, x=center.x(), y=center.y())
        self.statusBar().showMessage(f"Ajouté · {type_name}", 1500)

    def _on_scene_selection_changed(self):
        sel = self.scene.selectedItems()
        from ark.ui.node_editor.node_item import NodeItem
        nodes = [i for i in sel if isinstance(i, NodeItem)]
        if len(nodes) == 1:
            self.inspector.set_node(nodes[0])
        else:
            self.inspector.set_node(None)

    def _on_new(self):
        self.scene.clear_graph()
        self.statusBar().showMessage("Nouveau graphe", 1500)

    def _on_open(self):
        self.statusBar().showMessage("Ouverture non implémentée", 2000)

    def _on_save(self):
        self.statusBar().showMessage("Sauvegarde non implémentée", 2000)

    def _on_undo(self):
        self.statusBar().showMessage("Annuler — non implémenté", 1500)

    def _on_redo(self):
        self.statusBar().showMessage("Refaire — non implémenté", 1500)

    def _on_reset_layout(self):
        self.statusBar().showMessage("Reset layout — TODO presets", 1500)

    def _apply_preset(self, name: str):
        self.statusBar().showMessage(f"Layout preset : {name} — TODO", 2000)

    def _on_compile(self):
        n_nodes = len(self.scene.nodes())
        self.statusBar().showMessage(f"Compile · {n_nodes} nodes — TODO", 2000)

    def _on_train(self):
        self.statusBar().showMessage("Train — TODO", 2000)

    def _on_stop(self):
        self.statusBar().showMessage("Stop", 1500)

    def _on_search(self):
        self.view.open_search_overlay()

    def _on_about(self):
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.about(self, "À propos d'ARK",
                          "<h2>ARK</h2><p>Architecture builder node-based.</p>"
                          "<p>Refonte Python (PySide6) de ai-arch-builder.</p>")
