"""ARK MainWindow — shell workspace multi-panneaux (QDockWidget).

Layout par défaut :
- Gauche : palette nodes (catégorisée).
- Centre : node editor (QGraphicsView).
- Droite : inspector (params node sélectionné) + outliner.
- Bas : terminal / log / charts (sous-onglets).

Tous panneaux QDockWidget natifs : drag, undock, redimension, fermeture.
"""
from __future__ import annotations
from PySide6.QtCore import Qt, QSize, QThread
from PySide6.QtGui import QAction, QKeySequence, QUndoStack
from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QWidget, QVBoxLayout, QLabel,
    QTabWidget, QToolBar, QStatusBar, QApplication, QFileDialog, QMessageBox,
)

from ark.ui import design_tokens as T
from ark.ui.panels.palette_panel import PalettePanel
from ark.ui.panels.inspector_panel import InspectorPanel
from ark.ui.panels.terminal_panel import TerminalPanel
from ark.ui.node_editor.view import NodeEditorView
from ark.ui.node_editor.scene import NodeScene
from ark.ui.project_tabs import ProjectTabs
from ark.nodes.registry import default_registry
from ark.core.templates import TEMPLATES_LIST, get_template


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
        self.undo_stack = QUndoStack(self)
        self.undo_stack.setUndoLimit(200)
        self._current_file: str | None = None
        # Phase 50 : setup logger central
        try:
            from ark.core.logger import setup_logging
            setup_logging()
        except Exception:
            pass
        self._build_menus()
        self._build_status_bar()
        self._build_central()
        self._build_docks()
        # Phase 42 : overlays flottants au lieu du toolbar texte
        self._setup_floating_overlays()
        # Phase 50 : drag-drop accept JSON
        self.setAcceptDrops(True)

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
        # Templates
        a_browse = QAction("Parcourir templates...", self)
        a_browse.setShortcut(QKeySequence("Ctrl+Shift+T"))
        a_browse.triggered.connect(self._on_browse_templates)
        file_menu.addAction(a_browse)
        tpl_menu = file_menu.addMenu("Nouveau depuis template")
        for tpl_name, tpl_label in TEMPLATES_LIST:
            tpl_menu.addAction(tpl_label, lambda n=tpl_name: self._load_template(n))
        file_menu.addSeparator()
        # Export
        exp_menu = file_menu.addMenu("Exporter")
        exp_menu.addAction("Python standalone (.py)", self._on_export_python)
        exp_menu.addAction("ONNX (.onnx)", self._on_export_onnx)
        file_menu.addSeparator()
        a_quit = QAction("Quitter", self)
        a_quit.setShortcut(QKeySequence.StandardKey.Quit)
        a_quit.triggered.connect(QApplication.instance().quit)
        file_menu.addAction(a_quit)

        edit_menu = menubar.addMenu("Édition")
        undo_act = self.undo_stack.createUndoAction(self, "Annuler")
        undo_act.setShortcut(QKeySequence.StandardKey.Undo)
        edit_menu.addAction(undo_act)
        redo_act = self.undo_stack.createRedoAction(self, "Refaire")
        redo_act.setShortcut(QKeySequence.StandardKey.Redo)
        edit_menu.addAction(redo_act)
        edit_menu.addSeparator()
        edit_menu.addAction("Copier", self._on_copy, QKeySequence.StandardKey.Copy)
        edit_menu.addAction("Coller", self._on_paste, QKeySequence.StandardKey.Paste)
        edit_menu.addSeparator()
        edit_menu.addAction("Tout sélectionner", self._on_select_all,
                              QKeySequence.StandardKey.SelectAll)
        edit_menu.addSeparator()
        a_group = QAction("Enregistrer sélection comme groupe...", self)
        a_group.setShortcut(QKeySequence("Ctrl+G"))
        a_group.triggered.connect(self._on_save_selection_as_group)
        edit_menu.addAction(a_group)
        a_reload_groups = QAction("Recharger groupes utilisateur", self)
        a_reload_groups.triggered.connect(self._on_reload_user_groups)
        edit_menu.addAction(a_reload_groups)

        view_menu = menubar.addMenu("Affichage")
        view_menu.addAction("Reset layout", self._on_reset_layout)
        view_menu.addSeparator()
        a_auto = QAction("Auto-layout nodes", self)
        a_auto.setShortcut(QKeySequence("Ctrl+Alt+L"))
        a_auto.triggered.connect(self._on_auto_layout)
        view_menu.addAction(a_auto)
        view_menu.addSeparator()
        a_arch_diag = QAction("Schématiser architecture (HTML)", self)
        a_arch_diag.setShortcut(QKeySequence("Ctrl+Shift+D"))
        a_arch_diag.triggered.connect(self._on_show_arch_diagram)
        view_menu.addAction(a_arch_diag)
        a_bench_dash = QAction("Dashboard Benchmark", self)
        a_bench_dash.setShortcut(QKeySequence("Ctrl+Shift+B"))
        a_bench_dash.triggered.connect(self._on_open_benchmark_dashboard)
        view_menu.addAction(a_bench_dash)
        view_menu.addSeparator()
        view_menu.addAction("Layout : Workspace", lambda: self._apply_preset("workspace"))
        view_menu.addAction("Layout : Compact", lambda: self._apply_preset("compact"))
        view_menu.addAction("Layout : Code-first", lambda: self._apply_preset("code"))
        view_menu.addAction("Layout : Training-focus", lambda: self._apply_preset("training"))
        view_menu.addSeparator()
        view_menu.addAction("Sauver layout courant...", self._on_save_layout)
        view_menu.addAction("Charger layout...", self._on_load_layout)

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
        a_help = QAction("Aide complète (glossaire + tutoriel)…", self)
        a_help.setShortcut(QKeySequence("F1"))
        a_help.setToolTip("475+ concepts didactiques + intuition + exemples + équations.")
        a_help.triggered.connect(self._on_open_help_page)
        help_menu.addAction(a_help)
        a_shortcuts = QAction("Raccourcis clavier (cheat-sheet)…", self)
        a_shortcuts.setShortcut(QKeySequence("Ctrl+/"))
        a_shortcuts.setToolTip("Vue d'ensemble des raccourcis.")
        a_shortcuts.triggered.connect(self._on_open_shortcuts)
        help_menu.addAction(a_shortcuts)
        a_glossary = QAction("Glossaire compact…", self)
        a_glossary.setShortcut(QKeySequence("Shift+F1"))
        a_glossary.setToolTip("Définitions courtes (ancien dialog).")
        a_glossary.triggered.connect(self._on_open_glossary)
        help_menu.addAction(a_glossary)
        help_menu.addSeparator()
        help_menu.addAction("À propos d'ARK", self._on_about)

    # ------------------------------------------------------------
    # Toolbar (top)
    # ------------------------------------------------------------
    def _build_toolbar(self):
        """Phase 42 : toolbar texte remplacé par overlays floating (cf _setup_floating_overlays)."""
        # Actions toujours accessibles via raccourcis Cmd+R/T/S + menus
        pass

    def _setup_floating_overlays(self):
        """Phase 42/49/50 : overlays au-dessus du canvas central."""
        from ark.ui.floating_overlays import (
            LeftRail, ActionBar, StatsOverlay, ViewControls, GlobalProgress,
        )
        from ark.ui.minimap import Minimap

        host = self.tabs
        self._left_rail = LeftRail(host)
        self._action_bar = ActionBar(host)
        self._stats_overlay = StatsOverlay(host)
        self._view_controls = ViewControls(host)
        self._global_progress = GlobalProgress(host)
        self._global_progress.hide()
        # Phase 50 : minimap bottom-right above ViewControls
        self._minimap = Minimap(self.scene, self.view, host)
        self._minimap.show()

        for w in (self._left_rail, self._action_bar,
                    self._stats_overlay, self._view_controls,
                    self._global_progress, self._minimap):
            w.raise_()
        for w in (self._left_rail, self._action_bar,
                    self._stats_overlay, self._view_controls):
            w.show()

        # Wire ActionBar signals
        self._action_bar.compile_clicked.connect(self._on_action_run)
        self._action_bar.stop_clicked.connect(self._on_stop)
        self._action_bar.extensions_clicked.connect(self._on_browse_templates)
        self._action_bar.btn_inspector_toggle.clicked.connect(
            self._toggle_inspector)

        # Wire LeftRail
        self._left_rail.panel_toggled.connect(self._on_left_rail_panel)
        self._left_rail.action_triggered.connect(self._on_left_rail_action)
        # Initial states
        self._left_rail.btn_palette.setChecked(True)
        self._left_rail.btn_inspector.setChecked(True)

        # Wire ViewControls
        self._view_controls.zoom_changed.connect(self._on_zoom_changed)

        # Initial stats
        self._update_overlay_stats()

        # Reposition timer
        self._reposition_overlays()

    def _on_action_run(self):
        """Phase 42 : routage selon mode dropdown ActionBar."""
        mode = self._action_bar.mode_combo.currentText()
        if mode == "Compiler":
            self._on_compile()
        elif mode == "Entraîner":
            self._on_train()
        elif mode == "Bench":
            self._on_open_benchmark_dashboard()

    def _on_left_rail_panel(self, name: str):
        """Toggle dock panels selon name."""
        mapping = {
            "palette": getattr(self, "dock_palette", None),
            "inspector": getattr(self, "dock_inspector", None),
            "terminal": getattr(self, "dock_terminal", None),
        }
        dock = mapping.get(name)
        if dock is not None:
            dock.setVisible(not dock.isVisible())

    def _on_left_rail_action(self, name: str):
        """Actions one-shot depuis left rail."""
        if name == "help":
            self._on_open_help_page()
        elif name == "shortcuts":
            self._on_open_shortcuts()
        elif name == "settings":
            self.statusBar().showMessage("Préférences (à venir)", 2000)

    def _toggle_inspector(self):
        if hasattr(self, "dock_inspector"):
            self.dock_inspector.setVisible(not self.dock_inspector.isVisible())

    def _on_zoom_changed(self, factor: float):
        if hasattr(self, "view") and self.view is not None:
            self.view.resetTransform()
            self.view.scale(factor, factor)

    def _update_overlay_stats(self):
        if not hasattr(self, "_stats_overlay"):
            return
        try:
            scene = self.scene
            nodes = list(scene.nodes()) if scene is not None else []
            sel = sum(1 for n in nodes if n.isSelected())
            self._stats_overlay.set_nodes(len(nodes), sel)
        except Exception:
            pass

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_left_rail"):
            self._reposition_overlays()

    # Phase 50 : drag-drop JSON import
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith((".json", ".jsonl")):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        from ark.core.format import load_scene
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith((".json", ".jsonl")):
                try:
                    load_scene(self.scene, path)
                    self._current_file = path
                    self.statusBar().showMessage(
                        f"✓ Chargé via drag-drop : {path}", 4000)
                    event.acceptProposedAction()
                    return
                except Exception as e:
                    self.statusBar().showMessage(
                        f"✗ Erreur import : {e}", 5000)
        event.ignore()

    def _reposition_overlays(self):
        if not hasattr(self, "_left_rail"):
            return
        host = self.tabs
        if host is None:
            return
        host_rect = host.rect()
        margin = 16

        # LeftRail : full height left
        rail_w = self._left_rail.width()
        self._left_rail.setGeometry(0, 0, rail_w, host_rect.height())

        # ActionBar : top-right
        ab = self._action_bar
        ab.adjustSize()
        ab_w = ab.sizeHint().width()
        ab_h = ab.sizeHint().height()
        ab.setGeometry(host_rect.width() - ab_w - margin, margin, ab_w, ab_h)

        # GlobalProgress (Phase 49) : top-center sous ActionBar
        if hasattr(self, "_global_progress"):
            gp = self._global_progress
            gp_w = 360
            gp.setGeometry(rail_w + margin,
                              margin,
                              gp_w, 36)

        # StatsOverlay : bottom-left (above left rail bottom icons)
        so = self._stats_overlay
        so.adjustSize()
        so_w = so.sizeHint().width()
        so_h = so.sizeHint().height()
        so.setGeometry(rail_w + margin,
                          host_rect.height() - so_h - margin,
                          so_w, so_h)

        # ViewControls : bottom-right
        vc = self._view_controls
        vc.adjustSize()
        vc_w = vc.sizeHint().width()
        vc_h = vc.sizeHint().height()
        vc.setGeometry(host_rect.width() - vc_w - margin,
                          host_rect.height() - vc_h - margin,
                          vc_w, vc_h)

        # Phase 50 : Minimap au-dessus ViewControls
        if hasattr(self, "_minimap"):
            mm_w = self._minimap.width()
            mm_h = self._minimap.height()
            self._minimap.setGeometry(
                host_rect.width() - mm_w - margin,
                host_rect.height() - vc_h - margin - mm_h - 8,
                mm_w, mm_h)

    # ------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------
    def _build_status_bar(self):
        from PySide6.QtWidgets import QLabel
        sb = QStatusBar()
        self.setStatusBar(sb)
        # Phase 36 : global state badge (right side)
        self._state_badge = QLabel("READY", self)
        self._state_badge.setProperty("role", "badge-success")
        sb.addPermanentWidget(self._state_badge)
        sb.showMessage("ARK · ready")

    def set_global_state(self, state: str, message: str = ""):
        """Phase 36 : update global state badge (ready/compiling/training/error)."""
        states = {
            "ready": ("READY", "badge-success"),
            "compiling": ("COMPILING", "badge-info"),
            "training": ("TRAINING", "badge-info"),
            "error": ("ERROR", "badge-error"),
            "warning": ("WARNING", "badge-warn"),
            "idle": ("IDLE", "badge"),
        }
        label, role = states.get(state, ("READY", "badge"))
        if hasattr(self, "_state_badge"):
            self._state_badge.setText(f"  ●  {label}")
            self._state_badge.setProperty("role", role)
            self._state_badge.style().unpolish(self._state_badge)
            self._state_badge.style().polish(self._state_badge)
        if message:
            self.statusBar().showMessage(message, 3000)

    # ------------------------------------------------------------
    # Central widget : node editor
    # ------------------------------------------------------------
    def _build_central(self):
        # Workspace multi-projets (Phase 12)
        self.tabs = ProjectTabs(registry=self._registry, parent=self)
        # Premier projet vide
        self.tabs.new_project("Sans titre")
        # Wire active changed → update scene/undo refs
        self.tabs.active_changed.connect(self._on_active_project_changed)
        # Init scene/view/undo aux valeurs du tab actif
        s, v, u = self.tabs.active()
        self.scene = s
        self.view = v
        if u is not None:
            self.undo_stack = u
        self.setCentralWidget(self.tabs)

    def _on_active_project_changed_refresh_minimap(self):
        """Phase 50 : update minimap scene/view ref après tab switch."""
        if hasattr(self, "_minimap"):
            self._minimap.set_scene_view(self.scene, self.view)

    def _on_active_project_changed(self, scene, view, undo):
        # Disconnect ancienne scène
        try:
            self.scene.selectionChanged.disconnect(self._on_scene_selection_changed)
        except (TypeError, RuntimeError):
            pass
        self.scene = scene
        self.view = view
        self.undo_stack = undo
        self.scene.selectionChanged.connect(self._on_scene_selection_changed)
        # Sync undo actions menubar (les actions sont créées sur un stack
        # spécifique ; pour simplifier on reconstruit le edit menu undo si besoin)
        self.statusBar().showMessage(f"Projet actif : {self.tabs.tabText(self.tabs.currentIndex())}", 1500)

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

        # Connection sélection node ↔ inspector (sera reconnecté à chaque switch tab)
        self.scene.selectionChanged.connect(self._on_scene_selection_changed)
        self.tabs.active_changed.connect(lambda s, v, u: None)  # déjà wired plus haut

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
        # Phase 12 : ouvre un NOUVEAU tab (vs effacer l'actuel)
        self.tabs.new_project("Sans titre")
        self.statusBar().showMessage("Nouveau projet", 1500)

    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Ouvrir une architecture", "",
            "Tous formats JSON (*.json)"
        )
        if not path:
            return
        try:
            from ark.core.import_legacy import load_any
            payload, fmt = load_any(path)
            name = payload.get("name") or path.split("/")[-1].replace(".json", "")
            # Phase 12 : ouvre dans NOUVEAU tab
            self.tabs.new_project(name, payload=payload)
            self._current_file = path
            extra = ""
            if fmt == "electron":
                skipped = payload.get("metadata", {}).get("unknown_types_skipped", [])
                if skipped:
                    extra = f" (skipped: {set(skipped)})"
            self.statusBar().showMessage(f"Chargé [{fmt}] : {path}{extra}", 4000)
            self.terminal.append_log(
                f"[load:{fmt}] {name} — "
                f"{len(self.scene.nodes())} nodes, "
                f"{len(self.scene.connections())} conns{extra}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Erreur de chargement", str(e))

    def _on_save(self):
        if self._current_file:
            path = self._current_file
        else:
            path, _ = QFileDialog.getSaveFileName(self, "Enregistrer", "untitled.json",
                                                    "ARK JSON (*.json)")
            if not path:
                return
        try:
            from ark.core.format import save_scene
            save_scene(self.scene, path, name=path.split('/')[-1].replace('.json', ''))
            self._current_file = path
            self.statusBar().showMessage(f"Sauvé : {path}", 3000)
            self.terminal.append_log(f"[save] {path}")
        except Exception as e:
            QMessageBox.critical(self, "Erreur de sauvegarde", str(e))

    def _on_reset_layout(self):
        self._apply_preset("workspace")

    def _apply_preset(self, name: str):
        """Reset puis applique un preset layout déterministe."""
        # Restore base : remove all docks puis re-add
        for d in (self.dock_palette, self.dock_inspector, self.dock_terminal):
            self.removeDockWidget(d)
            d.setVisible(True)
            d.setFloating(False)
        if name == "workspace":
            # Palette gauche large, Inspector droite, Terminal bas
            self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_palette)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_inspector)
            self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.dock_terminal)
            self.resizeDocks([self.dock_palette], [240], Qt.Orientation.Horizontal)
            self.resizeDocks([self.dock_inspector], [280], Qt.Orientation.Horizontal)
            self.resizeDocks([self.dock_terminal], [180], Qt.Orientation.Vertical)
        elif name == "compact":
            # Palette + Inspector côte à côte à gauche, Terminal fermé
            self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_palette)
            self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_inspector)
            self.tabifyDockWidget(self.dock_palette, self.dock_inspector)
            self.dock_palette.raise_()
            self.dock_terminal.setVisible(False)
            self.resizeDocks([self.dock_palette], [260], Qt.Orientation.Horizontal)
        elif name == "code":
            # Terminal/Python prend la moitié droite, palette gauche fine
            self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_palette)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_terminal)
            self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.dock_inspector)
            self.resizeDocks([self.dock_palette], [200], Qt.Orientation.Horizontal)
            self.resizeDocks([self.dock_terminal], [480], Qt.Orientation.Horizontal)
            self.resizeDocks([self.dock_inspector], [160], Qt.Orientation.Vertical)
        elif name == "training":
            # Pas de palette (cachée), Inspector + Terminal grands pour suivre training
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_inspector)
            self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.dock_terminal)
            self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_palette)
            self.dock_palette.setVisible(False)
            self.resizeDocks([self.dock_inspector], [320], Qt.Orientation.Horizontal)
            self.resizeDocks([self.dock_terminal], [320], Qt.Orientation.Vertical)
        self.statusBar().showMessage(f"Layout : {name}", 2000)

    def _on_save_layout(self):
        path, _ = QFileDialog.getSaveFileName(self, "Sauver layout", "layout.bin",
                                                "Layout (*.bin)")
        if not path:
            return
        with open(path, "wb") as f:
            f.write(self.saveState())
        self.statusBar().showMessage(f"Layout sauvé : {path}", 2000)

    def _on_load_layout(self):
        path, _ = QFileDialog.getOpenFileName(self, "Charger layout", "",
                                                "Layout (*.bin)")
        if not path:
            return
        with open(path, "rb") as f:
            self.restoreState(f.read())
        self.statusBar().showMessage(f"Layout chargé : {path}", 2000)

    def _on_compile(self):
        """Phase 49 : statuts visuels par node + GlobalProgress animée."""
        from ark.core.graph_compiler import compile_scene, GraphCompileError
        from PySide6.QtCore import QTimer

        nodes = list(self.scene.nodes())
        # Reset all to compiling
        for n in nodes:
            n.set_status("compiling")
            n.set_progress(0.0)
            n.set_error(None)
        if hasattr(self, "_global_progress"):
            self._global_progress.show()
            self._global_progress.raise_()
            self._global_progress.set_progress(0.0, "Compile", "compiling")

        # Animate progress per node sequentially (instant compile but UI feedback)
        try:
            self._compiled = compile_scene(self.scene, num_classes=None)
            # Iterate nodes : set OK pour ceux qui sont dans modules_dict
            modules_dict = getattr(self._compiled.model, "modules_dict", {})
            compiled_ids = set(modules_dict.keys()) if hasattr(modules_dict, "keys") else set()

            def step(i):
                if i >= len(nodes):
                    if hasattr(self, "_global_progress"):
                        self._global_progress.set_progress(1.0, "Compiled ✓", "ok")
                        QTimer.singleShot(2500, self._global_progress.hide)
                    return
                node = nodes[i]
                if node.id in compiled_ids or len(compiled_ids) == 0:
                    node.set_status("ok")
                else:
                    node.set_status("warn")
                node.set_progress(1.0)
                pct = (i + 1) / max(len(nodes), 1)
                if hasattr(self, "_global_progress"):
                    self._global_progress.set_progress(
                        pct, f"Compile {i+1}/{len(nodes)}", "compiling")
                QTimer.singleShot(40, lambda j=i+1: step(j))

            step(0)
            n_params = sum(p.numel() for p in self._compiled.model.parameters())
            msg = (f"✓ compile · {len(nodes)} nodes · "
                   f"{n_params:,} params · in={self._compiled.input_shape} → "
                   f"out={self._compiled.output_shape}")
            self.statusBar().showMessage(msg, 5000)
            self.terminal.append_log(msg)
        except GraphCompileError as e:
            # Mark all not-yet-ok as error
            for n in nodes:
                if n.status != "ok":
                    n.set_status("error")
            if hasattr(self, "_global_progress"):
                self._global_progress.set_progress(1.0, "Compile failed", "error")
                QTimer.singleShot(3500, self._global_progress.hide)
            self.statusBar().showMessage(f"✗ compile error : {e}", 8000)
            self.terminal.append_log(f"[compile error] {e}")

    def _on_train(self):
        from ark.core.graph_compiler import compile_scene, GraphCompileError
        from ark.core.trainer import TrainerWorker, mnist_loaders, pick_device
        try:
            self._compiled = compile_scene(self.scene, num_classes=10)
        except GraphCompileError as e:
            QMessageBox.critical(self, "Compile error", str(e))
            return
        # Phase 49 : marque tous nodes training + show progress
        for n in self.scene.nodes():
            n.set_status("training")
            n.set_progress(0.0)
        if hasattr(self, "_global_progress"):
            self._global_progress.show()
            self._global_progress.raise_()
            self._global_progress.set_progress(0.0, "Training", "training")
        self._thread = QThread(self)
        train_loader, val_loader = mnist_loaders(n_train=3000, n_val=1000, batch_size=128)
        # Détecte si auto_mutate doit être activé (présence de trigger_rule + vg_block)
        types = {n.spec.name for n in self.scene.nodes()}
        auto_mut = ("trigger_rule" in types and "vg_block" in types)
        self._worker = TrainerWorker(self._compiled, train_loader, val_loader,
                                       epochs=2, device=pick_device(),
                                       auto_mutate=auto_mut)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.status_changed.connect(lambda s: self.terminal.append_log(f"[train] {s}"))
        self._worker.batch_progressed.connect(self._on_batch)
        self._worker.epoch_finished.connect(self._on_epoch)
        self._worker.observations.connect(self._on_observations)
        self._worker.auto_mutated.connect(self._on_auto_mutated)
        self._worker.error.connect(lambda e: self.terminal.append_log(f"[train error] {e}"))
        self._worker.done.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()
        msg = "Training démarré (auto_mutate)" if auto_mut else "Training démarré"
        self.statusBar().showMessage(msg, 2000)

    def _on_observations(self, info: dict):
        epoch = info.get("epoch")
        observers = info.get("observers", {})
        # Store history + push to inspector live chart
        for nid, metrics in observers.items():
            for n in self.scene.nodes():
                if n.id == nid:
                    hist = getattr(n, "_metric_history", None)
                    if hist is None:
                        hist = []
                        n._metric_history = hist
                    hist.append(dict(metrics))
                    # Cap rolling
                    if len(hist) > 60:
                        n._metric_history = hist[-60:]
            self.inspector.push_live_metrics(nid, metrics)
        summary = " | ".join(
            f"{nid[:8]}: {', '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in m.items())}"
            for nid, m in observers.items()
        )
        self.terminal.append_log(f"[obs ep{epoch}] {summary}")

    def _on_auto_mutated(self, info: dict):
        epoch = info.get("epoch")
        for a in info.get("triggered", []):
            if a.get("applied"):
                tb = a.get("triggered_by", {})
                self.terminal.append_log(
                    f"[mutate ep{epoch}] {a['node_id'][:8]} {a['op']} +{a['delta']} → "
                    f"h={a.get('new_hidden_dim', '?')} ({tb.get('metric')}={tb.get('value'):.3f} "
                    f"{tb.get('op')} {tb.get('threshold')})"
                )
                # Update visual node
                for n in self.scene.nodes():
                    if n.id == a["node_id"]:
                        n.params["start_hidden"] = a.get("new_hidden_dim", n.params.get("start_hidden"))
                        n.update()
                        break
            elif a.get("error"):
                self.terminal.append_log(f"[mutate ep{epoch} fail] {a['error']}")

    def _on_batch(self, info: dict):
        # Phase 50 : route metrics vers Dashboard si ouvert
        if getattr(self, "_bench_dlg", None) is not None and \
              self._bench_dlg.isVisible():
            try:
                self._bench_dlg.push_train_metrics(info)
            except Exception:
                pass
        # Phase 49 : update global progress
        if hasattr(self, "_global_progress"):
            ep = info.get("epoch", 1)
            ep_total = info.get("epochs", 1) or 1
            batch = info.get("batch", 0)
            total = info.get("total_batches", 1) or 1
            local = batch / total
            overall = ((ep - 1) + local) / ep_total
            self._global_progress.set_progress(
                overall, f"Train ep {ep}/{ep_total}", "training")
        self.statusBar().showMessage(
            f"epoch {info['epoch']} · batch {info['batch']}/{info['total_batches']} · "
            f"loss={info['loss']:.4f} · acc={info['acc']*100:.2f}%", 1500
        )

    def _on_epoch(self, info: dict):
        msg = (f"epoch {info['epoch']}/{info['epochs']} — "
               f"train_loss={info['train_loss']:.4f} train_acc={info['train_acc']*100:.2f}% · "
               f"val_loss={info['val_loss']:.4f if info['val_loss'] is not None else 0:.4f} "
               f"val_acc={info['val_acc']*100 if info['val_acc'] else 0:.2f}% "
               f"({info['elapsed']:.1f}s)")
        self.terminal.append_log(msg)
        self.statusBar().showMessage(msg, 4000)
        # Phase 49 : final epoch → set ok
        if hasattr(self, "_global_progress") and info.get("epoch") == info.get("epochs"):
            from PySide6.QtCore import QTimer
            self._global_progress.set_progress(1.0, "Trained ✓", "ok")
            QTimer.singleShot(3500, self._global_progress.hide)
            # Mark all training nodes ok
            for n in self.scene.nodes():
                if n.status == "training":
                    n.set_status("ok")

    def _on_stop(self):
        if hasattr(self, "_worker") and self._worker is not None:
            self._worker.stop()
            self.statusBar().showMessage("Stop demandé", 1500)
        # Phase 49 : reset nodes idle + hide progress
        for n in self.scene.nodes():
            if n.status in ("compiling", "training"):
                n.set_status("idle")
                n.set_progress(0.0)
        if hasattr(self, "_global_progress"):
            self._global_progress.hide()

    def _on_search(self):
        self.view.open_search_overlay()

    def _on_auto_layout(self):
        from ark.core.layout import auto_layout_scene
        auto_layout_scene(self.scene)
        self.statusBar().showMessage("Auto-layout appliqué", 2000)

    def _on_save_selection_as_group(self):
        """Phase 31 — Ctrl+G : enregistre sélection comme groupe réutilisable."""
        from ark.ui.node_editor.node_item import NodeItem
        from ark.ui.dialogs.save_group_dialog import SaveGroupDialog
        from ark.core.groups import (
            serialize_group, save_group, register_groups,
        )
        from ark.core.graph_compiler import HANDLERS

        sel = [it for it in self.scene.selectedItems() if isinstance(it, NodeItem)]
        if not sel:
            self.statusBar().showMessage("Sélectionne d'abord ≥1 node", 2500)
            return
        dlg = SaveGroupDialog(self, n_selected=len(sel))
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        meta = dlg.get_payload()
        try:
            payload = serialize_group(self.scene, sel,
                                          name=meta["name"], title=meta["title"],
                                          category=meta["category"],
                                          description=meta["description"])
            path = save_group(payload)
            # Auto-register live
            registry = self.scene.registry
            n = register_groups(registry, HANDLERS)
            # Refresh palette (sans relance app)
            if hasattr(self, "palette") and hasattr(self.palette, "refresh"):
                self.palette.refresh()
            self.statusBar().showMessage(
                f"Groupe '{meta['name']}' enregistré → {path}  "
                f"({n} groupes chargés)", 4000)
        except Exception as e:
            self.statusBar().showMessage(f"Erreur save group : {e}", 5000)

    def _on_open_benchmark_dashboard(self):
        """Phase 39 — ouvre dashboard benchmark non-modal."""
        from ark.ui.dialogs.benchmark_dashboard import BenchmarkDashboard
        if getattr(self, "_bench_dlg", None) is not None and \
              self._bench_dlg.isVisible():
            self._bench_dlg.raise_()
            self._bench_dlg.activateWindow()
            return
        self._bench_dlg = BenchmarkDashboard(self, scene=self.scene)
        self._bench_dlg.show()

    def _on_show_arch_diagram(self):
        """Phase 35 — génère HTML organigramme de la scene + ouvre browser."""
        from ark.core.arch_viz_nodes import ArchDiagram
        from pathlib import Path
        out_path = Path.home() / ".ark" / "arch_diagrams" / "current.html"
        arch = ArchDiagram(
            output_path=str(out_path),
            title=f"ARK · Architecture ({len(list(self.scene.nodes()))} nodes)",
            auto_open=True,
        )
        arch.set_scene(self.scene)
        arch.forward()
        self.statusBar().showMessage(
            f"Architecture exportée → {out_path}", 4000)

    def _on_reload_user_groups(self):
        """Recharge ~/.ark/groups/ sans redémarrer."""
        from ark.core.groups import register_groups
        from ark.core.graph_compiler import HANDLERS
        registry = self.scene.registry
        n = register_groups(registry, HANDLERS)
        if hasattr(self, "palette") and hasattr(self.palette, "refresh"):
            self.palette.refresh()
        self.statusBar().showMessage(f"{n} groupes chargés depuis ~/.ark/groups/",
                                          3000)

    def _on_copy(self):
        from ark.ui.node_editor.node_item import NodeItem
        sel = [it for it in self.scene.selectedItems() if isinstance(it, NodeItem)]
        if not sel:
            self.statusBar().showMessage("Rien à copier", 1500)
            return
        clip = []
        for n in sel:
            pos = n.pos()
            clip.append({
                "type": n.spec.name,
                "params": dict(n.params),
                "bypass": n.bypass,
                "width": n.width,
                "rel_x": float(pos.x()), "rel_y": float(pos.y()),
            })
        self._clipboard = clip
        self.statusBar().showMessage(f"{len(clip)} node(s) copié(s)", 1500)

    def _on_paste(self):
        clip = getattr(self, "_clipboard", None)
        if not clip:
            self.statusBar().showMessage("Presse-papier vide", 1500)
            return
        center = self.view.mapToScene(self.view.viewport().rect().center())
        base_x = min(c["rel_x"] for c in clip)
        base_y = min(c["rel_y"] for c in clip)
        for c in clip:
            dx = c["rel_x"] - base_x + 40
            dy = c["rel_y"] - base_y + 40
            new_node = self.scene.create_node(c["type"],
                                                x=center.x() + dx,
                                                y=center.y() + dy)
            if new_node is not None:
                for k, v in c["params"].items():
                    if k in new_node.params:
                        new_node.params[k] = v
                new_node.bypass = c.get("bypass", False)
                try:
                    new_node.width = int(c.get("width", new_node.width))
                    new_node._layout_sockets()
                except Exception:
                    pass
        self.statusBar().showMessage(f"{len(clip)} node(s) collé(s)", 1500)

    def _on_select_all(self):
        from ark.ui.node_editor.node_item import NodeItem
        for it in self.scene.items():
            if isinstance(it, NodeItem):
                it.setSelected(True)

    def _on_about(self):
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.about(self, "À propos d'ARK",
                          "<h2>ARK</h2><p>Architecture builder node-based.</p>"
                          "<p>Refonte Python (PySide6) de ai-arch-builder.</p>")

    def _on_open_glossary(self, initial_term: str | None = None):
        """Ouvre le glossaire (non-modal). Accept optionnellement un terme initial."""
        from ark.ui.dialogs.glossary_dialog import GlossaryDialog
        if getattr(self, "_glossary_dlg", None) is not None and self._glossary_dlg.isVisible():
            if initial_term:
                self._glossary_dlg.select_term(initial_term)
            self._glossary_dlg.raise_()
            self._glossary_dlg.activateWindow()
            return
        self._glossary_dlg = GlossaryDialog(self, initial_term=initial_term)
        self._glossary_dlg.show()

    def _on_open_shortcuts(self):
        """Phase 40 — Ctrl+/ overlay cheat-sheet."""
        from ark.ui.dialogs.shortcuts_overlay import ShortcutsOverlay
        if getattr(self, "_shortcuts_dlg", None) is not None and \
              self._shortcuts_dlg.isVisible():
            self._shortcuts_dlg.raise_()
            self._shortcuts_dlg.activateWindow()
            return
        self._shortcuts_dlg = ShortcutsOverlay(self)
        self._shortcuts_dlg.show()

    def _on_open_help_page(self, initial_term: str | None = None):
        """Phase 38 : ouvre la page d'aide complète (non-modal)."""
        from ark.ui.dialogs.help_page import HelpPage
        if getattr(self, "_help_dlg", None) is not None and self._help_dlg.isVisible():
            if initial_term:
                self._help_dlg.select_term(initial_term)
            self._help_dlg.raise_()
            self._help_dlg.activateWindow()
            return
        self._help_dlg = HelpPage(self, initial_term=initial_term)
        self._help_dlg.show()

    def _on_browse_templates(self):
        from ark.ui.dialogs.templates_browser import TemplatesBrowser
        dlg = TemplatesBrowser(self)
        dlg.template_chosen.connect(self._load_template)
        dlg.exec()

    def _load_template(self, name: str):
        payload = get_template(name)
        if payload is None:
            QMessageBox.warning(self, "Template", f"Template '{name}' introuvable")
            return
        # Phase 12 : ouvre dans un NOUVEAU tab (préserve projets existants)
        try:
            label = next((lbl for n, lbl in TEMPLATES_LIST if n == name), name)
            self.tabs.new_project(label, payload=payload)
            self.statusBar().showMessage(f"Template chargé : {name}", 3000)
            self.terminal.append_log(f"[template] {name} → nouvel onglet")
        except Exception as e:
            QMessageBox.critical(self, "Erreur template", str(e))

    def _on_export_python(self):
        if not self.scene.nodes():
            QMessageBox.information(self, "Export", "Graphe vide")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Python", "ark_model.py",
                                                "Python (*.py)")
        if not path:
            return
        try:
            from ark.core.export import export_python
            out = export_python(self.scene, path)
            self.statusBar().showMessage(f"Exporté : {out}", 4000)
            self.terminal.append_log(f"[export.py] {out}")
        except Exception as e:
            QMessageBox.critical(self, "Erreur export", str(e))

    def _on_export_onnx(self):
        if not self.scene.nodes():
            QMessageBox.information(self, "Export", "Graphe vide")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export ONNX", "ark_model.onnx",
                                                "ONNX (*.onnx)")
        if not path:
            return
        try:
            from ark.core.export import export_onnx
            out = export_onnx(self.scene, path, num_classes=10)
            self.statusBar().showMessage(f"Exporté : {out}", 4000)
            self.terminal.append_log(f"[export.onnx] {out}")
        except Exception as e:
            QMessageBox.critical(self, "Erreur export ONNX", str(e))
