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
        self.undo_stack = QUndoStack(self)
        self.undo_stack.setUndoLimit(200)
        self._current_file: str | None = None
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
        undo_act = self.undo_stack.createUndoAction(self, "Annuler")
        undo_act.setShortcut(QKeySequence.StandardKey.Undo)
        edit_menu.addAction(undo_act)
        redo_act = self.undo_stack.createRedoAction(self, "Refaire")
        redo_act.setShortcut(QKeySequence.StandardKey.Redo)
        edit_menu.addAction(redo_act)
        edit_menu.addSeparator()
        edit_menu.addAction("Tout sélectionner", lambda: None, QKeySequence.StandardKey.SelectAll)

        view_menu = menubar.addMenu("Affichage")
        view_menu.addAction("Reset layout", self._on_reset_layout)
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
        self.scene = NodeScene(registry=self._registry, undo_stack=self.undo_stack)
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
        path, _ = QFileDialog.getOpenFileName(self, "Ouvrir une architecture", "",
                                                "ARK JSON (*.json)")
        if not path:
            return
        try:
            from ark.core.format import load_scene
            payload = load_scene(self.scene, path)
            self._current_file = path
            self.undo_stack.clear()
            self.statusBar().showMessage(f"Chargé : {path}", 3000)
            self.terminal.append_log(f"[load] {payload.get('name', '?')} — "
                                       f"{len(self.scene.nodes())} nodes, "
                                       f"{len(self.scene.connections())} conns")
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
        from ark.core.graph_compiler import compile_scene, GraphCompileError
        try:
            self._compiled = compile_scene(self.scene, num_classes=None)
            n_params = sum(p.numel() for p in self._compiled.model.parameters())
            msg = (f"✓ compile · {len(self.scene.nodes())} nodes · "
                   f"{n_params:,} params · in={self._compiled.input_shape} → "
                   f"out={self._compiled.output_shape}")
            self.statusBar().showMessage(msg, 5000)
            self.terminal.append_log(msg)
        except GraphCompileError as e:
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

    def _on_stop(self):
        if hasattr(self, "_worker") and self._worker is not None:
            self._worker.stop()
            self.statusBar().showMessage("Stop demandé", 1500)

    def _on_search(self):
        self.view.open_search_overlay()

    def _on_about(self):
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.about(self, "À propos d'ARK",
                          "<h2>ARK</h2><p>Architecture builder node-based.</p>"
                          "<p>Refonte Python (PySide6) de ai-arch-builder.</p>")
