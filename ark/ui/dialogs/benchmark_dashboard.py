"""Phase 39 — Dashboard Benchmark : fenêtre dédiée monitoring + audit.

QDialog non-modal qui :
1. Affiche metrics du modèle compilé courant
2. Bouton 'Run Audit' → ModelAudit + ouvre HTML
3. Bouton 'Run Benchmark Inference' → BenchmarkInference + HTML
4. Bouton 'Activation Stats' → forward hooks + HTML
5. Bouton 'Param Distribution' → histogrammes weights HTML
6. Bouton 'Gradient Flow' → diagnostic gradients (need backward done)
7. Stats permanent : params, FLOPs, depth, latency estim
"""
from __future__ import annotations
from pathlib import Path
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QFrame, QPlainTextEdit, QGroupBox, QSizePolicy, QWidget,
)

from ark.ui import design_tokens as T


class StatCard(QWidget):
    """Carte stat avec valeur grosse + label."""

    def __init__(self, label: str, value: str = "—",
                 color: str = T.FG_PRIMARY, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(T.S_3, T.S_3, T.S_3, T.S_3)
        v.setSpacing(T.S_1)
        self.setStyleSheet(
            f"background-color: {T.BG_PANEL_RAISED}; "
            f"border: 1px solid {T.BORDER_DEFAULT}; "
            f"border-radius: {T.R_MEDIUM}px;")
        self.value_lbl = QLabel(value, self)
        self.value_lbl.setStyleSheet(
            f"font-size: {T.FS_HERO}px; font-weight: 700; color: {color}; "
            f"background: transparent;")
        v.addWidget(self.value_lbl)
        self.label_lbl = QLabel(label, self)
        self.label_lbl.setStyleSheet(
            f"font-size: {T.FS_MICRO}px; color: {T.FG_TERTIARY}; "
            f"text-transform: uppercase; letter-spacing: 1px; "
            f"background: transparent;")
        v.addWidget(self.label_lbl)

    def set_value(self, val: str, color: str | None = None):
        self.value_lbl.setText(val)
        if color:
            self.value_lbl.setStyleSheet(
                f"font-size: {T.FS_HERO}px; font-weight: 700; color: {color}; "
                f"background: transparent;")


class BenchmarkDashboard(QDialog):
    """Dashboard dédiée benchmark + audit + monitoring."""

    def __init__(self, parent=None, scene=None):
        super().__init__(parent)
        self.setWindowTitle("Dashboard Benchmark · ARK")
        self.setMinimumSize(900, 640)
        self.setModal(False)
        self._scene = scene
        self._model = None
        self._build_ui()
        if scene is not None:
            self._try_compile_scene()

    def _build_ui(self):
        v = QVBoxLayout(self)
        v.setContentsMargins(T.S_4, T.S_4, T.S_4, T.S_4)
        v.setSpacing(T.S_3)

        # Header
        header = QHBoxLayout()
        title = QLabel("Benchmark · Audit", self)
        title.setStyleSheet(
            f"font-family: 'Arial'; font-size: {T.FS_HERO}px; "
            f"font-weight: 700; color: {T.ACCENT}; letter-spacing: 1px;")
        header.addWidget(title)
        header.addStretch(1)
        self.state_badge = QLabel("idle", self)
        self.state_badge.setProperty("role", "badge")
        header.addWidget(self.state_badge)
        v.addLayout(header)

        sub = QLabel(
            "Mesure performance, audite l'architecture, "
            "diagnostique vanishing/exploding, monitor activations.", self)
        sub.setStyleSheet(f"color: {T.FG_TERTIARY}; font-size: {T.FS_LABEL}px;")
        v.addWidget(sub)

        # Stats grid (4 cards)
        stats = QGridLayout()
        stats.setSpacing(T.S_3)
        self.card_params = StatCard("Params", "—", T.ACCENT)
        self.card_flops = StatCard("FLOPs", "—", T.ACCENT_GREEN)
        self.card_depth = StatCard("Depth", "—", T.INFO)
        self.card_memory = StatCard("Memory MB", "—", T.WARN)
        stats.addWidget(self.card_params, 0, 0)
        stats.addWidget(self.card_flops, 0, 1)
        stats.addWidget(self.card_depth, 0, 2)
        stats.addWidget(self.card_memory, 0, 3)
        v.addLayout(stats)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color: {T.BORDER_DEFAULT};")
        v.addWidget(sep)

        # Actions buttons
        actions = QGroupBox("Outils", self)
        actions.setStyleSheet(
            f"QGroupBox {{ color: {T.FG_TERTIARY}; "
            f"font-size: {T.FS_MICRO}px; "
            f"border: 1px solid {T.BORDER_DEFAULT}; "
            f"border-radius: {T.R_MEDIUM}px; "
            f"margin-top: 12px; padding-top: 12px; }} "
            f"QGroupBox::title {{ subcontrol-origin: margin; "
            f"left: 12px; padding: 0 6px; }}")
        ag = QGridLayout(actions)
        ag.setSpacing(T.S_2)

        self.btn_audit = QPushButton("🔍  Run Model Audit")
        self.btn_audit.setProperty("role", "accent")
        self.btn_audit.clicked.connect(self._run_audit)

        self.btn_bench = QPushButton("⏱  Benchmark Inference")
        self.btn_bench.setProperty("role", "info")
        self.btn_bench.clicked.connect(self._run_inference_bench)

        self.btn_params = QPushButton("📊  Param Distribution")
        self.btn_params.setProperty("role", "success")
        self.btn_params.clicked.connect(self._run_param_dist)

        self.btn_activ = QPushButton("⚡  Activation Stats")
        self.btn_activ.setProperty("role", "info")
        self.btn_activ.clicked.connect(self._run_activations)

        self.btn_grad = QPushButton("📉  Gradient Flow")
        self.btn_grad.setProperty("role", "warn")
        self.btn_grad.clicked.connect(self._run_grad_flow)

        self.btn_refresh = QPushButton("🔄  Refresh Model")
        self.btn_refresh.setProperty("role", "ghost")
        self.btn_refresh.clicked.connect(self._try_compile_scene)

        ag.addWidget(self.btn_audit, 0, 0)
        ag.addWidget(self.btn_bench, 0, 1)
        ag.addWidget(self.btn_params, 1, 0)
        ag.addWidget(self.btn_activ, 1, 1)
        ag.addWidget(self.btn_grad, 2, 0)
        ag.addWidget(self.btn_refresh, 2, 1)
        v.addWidget(actions)

        # Log
        log_lbl = QLabel("Output log", self)
        log_lbl.setStyleSheet(
            f"color: {T.FG_TERTIARY}; font-size: {T.FS_MICRO}px; "
            f"text-transform: uppercase; letter-spacing: 1px;")
        v.addWidget(log_lbl)

        self.log = QPlainTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setStyleSheet(
            f"background-color: {T.BG_INPUT}; "
            f"color: {T.FG_PRIMARY}; "
            f"font-family: 'Menlo', 'Courier New', monospace; "
            f"font-size: {T.FS_LABEL}px; "
            f"border: 1px solid {T.BORDER_DEFAULT}; "
            f"border-radius: {T.R_MEDIUM}px; "
            f"padding: {T.S_2}px;")
        v.addWidget(self.log, 1)

        # Footer
        footer = QHBoxLayout()
        footer.addStretch(1)
        close = QPushButton("Fermer", self)
        close.clicked.connect(self.close)
        footer.addWidget(close)
        v.addLayout(footer)

    # ============ Logic ============

    def _log(self, msg: str):
        self.log.appendPlainText(msg)

    def push_train_metrics(self, info: dict):
        """Phase 50 : appelé live par MainWindow._on_batch."""
        ep = info.get("epoch", 0)
        batch = info.get("batch", 0)
        total = info.get("total_batches", 0)
        loss = info.get("loss", 0.0)
        acc = info.get("acc", 0.0)
        self._log(f"[train] ep {ep} batch {batch}/{total} "
                     f"loss={loss:.4f} acc={acc*100:.1f}%")

    def _set_state(self, state: str):
        roles = {
            "idle": "badge",
            "running": "badge-info",
            "success": "badge-success",
            "error": "badge-error",
        }
        self.state_badge.setText(state.upper())
        self.state_badge.setProperty("role", roles.get(state, "badge"))
        self.state_badge.style().unpolish(self.state_badge)
        self.state_badge.style().polish(self.state_badge)

    def _try_compile_scene(self):
        if self._scene is None:
            self._log("Pas de scene reliée.")
            return
        try:
            from ark.core.graph_compiler import compile_scene
            cg = compile_scene(self._scene)
            self._model = cg.model
            input_shape = cg.input_shape
            self._update_stats(input_shape)
            self._log(f"Modèle compilé : input_shape={input_shape}")
            self._set_state("success")
        except Exception as e:
            self._log(f"Erreur compile : {e}")
            self._set_state("error")

    def _update_stats(self, input_shape):
        if self._model is None:
            return
        from ark.core.benchmark_nodes import count_params, estimate_flops
        params = count_params(self._model)
        try:
            flops = estimate_flops(self._model, tuple(input_shape))
        except Exception:
            flops = 0
        depth = sum(1 for _ in self._model.modules())
        mem = params["total"] * 4 / 1e6  # float32

        self.card_params.set_value(f"{params['total']:,}")
        self.card_flops.set_value(
            f"{flops / 1e6:.1f}M" if flops > 1e6 else f"{flops:,}")
        self.card_depth.set_value(str(depth))
        self.card_memory.set_value(f"{mem:.1f}")

    # ============ Actions ============

    def _run_audit(self):
        if self._model is None:
            self._log("Pas de modèle.")
            return
        from ark.core.benchmark_nodes import ModelAudit
        out = str(Path.home() / ".ark" / "benchmarks" / "audit.html")
        try:
            audit = ModelAudit(output_path=out, input_shape="1,28,28",
                                  auto_open=True)
            audit(self._model)
            self._log(f"✓ Audit → {out}")
            self._set_state("success")
        except Exception as e:
            self._log(f"✗ Audit erreur : {e}")
            self._set_state("error")

    def _run_inference_bench(self):
        if self._model is None:
            return
        from ark.core.benchmark_nodes import BenchmarkInference
        out = str(Path.home() / ".ark" / "benchmarks" / "inference.html")
        self._set_state("running")
        try:
            bench = BenchmarkInference(output_path=out, n_runs=10,
                                              input_shape="1,28,28",
                                              auto_open=True)
            bench(self._model)
            self._log(f"✓ Benchmark inference → {out}")
            self._set_state("success")
        except Exception as e:
            self._log(f"✗ Bench erreur : {e}")
            self._set_state("error")

    def _run_param_dist(self):
        if self._model is None:
            return
        from ark.core.benchmark_nodes import ParamDistribution
        out = str(Path.home() / ".ark" / "benchmarks" / "param_dist.html")
        try:
            pd = ParamDistribution(output_path=out, auto_open=True)
            pd(self._model)
            self._log(f"✓ Param dist → {out}")
            self._set_state("success")
        except Exception as e:
            self._log(f"✗ Param dist : {e}")
            self._set_state("error")

    def _run_activations(self):
        if self._model is None:
            return
        from ark.core.benchmark_nodes import ActivationStats
        out = str(Path.home() / ".ark" / "benchmarks" / "activ.html")
        try:
            stats = ActivationStats(output_path=out,
                                          input_shape="1,28,28",
                                          auto_open=True)
            stats(self._model)
            self._log(f"✓ Activations → {out}")
            self._set_state("success")
        except Exception as e:
            self._log(f"✗ Activations : {e}")
            self._set_state("error")

    def _run_grad_flow(self):
        if self._model is None:
            return
        from ark.core.benchmark_nodes import GradientFlow
        out = str(Path.home() / ".ark" / "benchmarks" / "grad_flow.html")
        # Force backward pass aléatoire pour avoir des gradients
        try:
            import torch
            x = torch.randn(2, 1, 28, 28)
            self._model.train()
            out_t = self._model(x)
            if isinstance(out_t, tuple):
                out_t = out_t[0]
            loss = out_t.sum()
            loss.backward()
            gf = GradientFlow(output_path=out, auto_open=True)
            gf(self._model)
            self._log(f"✓ Gradient flow → {out}")
            self._set_state("success")
        except Exception as e:
            self._log(f"✗ Gradient flow : {e}")
            self._set_state("error")
