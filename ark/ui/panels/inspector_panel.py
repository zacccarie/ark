"""Inspector — propriétés du node sélectionné (params éditables)."""
from __future__ import annotations
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QFormLayout, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QSpinBox,
    QDoubleSpinBox, QComboBox, QCheckBox, QPlainTextEdit, QFrame, QPushButton,
    QFileDialog,
)

from ark.ui import design_tokens as T
from ark.ui.widgets.metric_chart import MetricChart
from ark.nodes.registry import ParamSpec


class InspectorPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._node = None
        self._metric_chart: MetricChart | None = None
        v = QVBoxLayout(self)
        v.setContentsMargins(T.S_3, T.S_3, T.S_3, T.S_3)
        v.setSpacing(T.S_2)
        self._main_layout = v

        self.title = QLabel("Aucune sélection", self)
        self.title.setProperty("role", "h1")
        self.title.setStyleSheet(f"font-size: {T.FS_H1}px; font-weight: 700; color: {T.FG_PRIMARY};")
        v.addWidget(self.title)

        self.subtitle = QLabel("", self)
        self.subtitle.setStyleSheet(f"font-size: {T.FS_MICRO}px; color: {T.FG_TERTIARY}; "
                                       "text-transform: uppercase; letter-spacing: 1px;")
        v.addWidget(self.subtitle)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color: {T.BORDER_DEFAULT};")
        v.addWidget(sep)

        # Bloc explicatif (rôle + fonction + équation + related) — Phase 30
        self._info_host = QWidget(self)
        self._info_layout = QVBoxLayout(self._info_host)
        self._info_layout.setContentsMargins(0, 0, 0, T.S_2)
        self._info_layout.setSpacing(T.S_2)

        self.role_label = QLabel("", self._info_host)
        self.role_label.setWordWrap(True)
        self.role_label.setStyleSheet(
            f"font-size: {T.FS_BODY}px; color: {T.FG_PRIMARY}; "
            "font-weight: 600; line-height: 1.4;"
        )
        self._info_layout.addWidget(self.role_label)

        self.function_label = QLabel("", self._info_host)
        self.function_label.setWordWrap(True)
        self.function_label.setStyleSheet(
            f"font-size: {T.FS_LABEL}px; color: {T.FG_SECONDARY}; line-height: 1.5;"
        )
        self._info_layout.addWidget(self.function_label)

        self.sockets_label = QLabel("", self._info_host)
        self.sockets_label.setWordWrap(False)
        self.sockets_label.setStyleSheet(
            f"font-family: 'Menlo','Courier New',monospace; "
            f"font-size: {T.FS_MICRO}px; color: {T.FG_TERTIARY}; "
            f"padding: {T.S_1}px 0; letter-spacing: 0.5px;"
        )
        self._info_layout.addWidget(self.sockets_label)

        self.equation_label = QLabel("", self._info_host)
        self.equation_label.setWordWrap(True)
        self.equation_label.setStyleSheet(
            f"font-family: 'Menlo','Courier New',monospace; "
            f"font-size: {T.FS_LABEL}px; color: {T.ACCENT}; "
            f"background-color: rgba(225,29,42,0.08); "
            f"padding: {T.S_2}px {T.S_3}px; border-left: 2px solid {T.ACCENT};"
        )
        self._info_layout.addWidget(self.equation_label)

        self.related_label = QLabel("", self._info_host)
        self.related_label.setWordWrap(True)
        self.related_label.setStyleSheet(
            f"font-size: {T.FS_MICRO}px; color: {T.FG_TERTIARY}; "
            "text-transform: uppercase; letter-spacing: 1px;"
        )
        self._info_layout.addWidget(self.related_label)

        self._info_sep = QFrame()
        self._info_sep.setFrameShape(QFrame.Shape.HLine)
        self._info_sep.setStyleSheet(f"color: {T.BORDER_DEFAULT};")
        self._info_layout.addWidget(self._info_sep)

        v.addWidget(self._info_host)
        self._info_host.hide()

        # Chart placeholder (visible seulement pour nodes observer)
        self._chart_host = QWidget(self)
        from PySide6.QtWidgets import QVBoxLayout as _VBL
        self._chart_layout = _VBL(self._chart_host)
        self._chart_layout.setContentsMargins(0, 0, 0, T.S_2)
        v.addWidget(self._chart_host)
        self._chart_host.hide()

        self._form_host = QWidget(self)
        self._form = QFormLayout(self._form_host)
        self._form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._form.setHorizontalSpacing(T.S_3)
        self._form.setVerticalSpacing(T.S_2)
        v.addWidget(self._form_host)
        v.addStretch(1)

    def set_node(self, node):
        self._clear_form()
        self._clear_chart()
        self._node = node
        if node is None:
            self.title.setText("Aucune sélection")
            self.subtitle.setText("")
            self._info_host.hide()
            self._chart_host.hide()
            return
        self.title.setText(node.spec.title)
        self.subtitle.setText(f"{node.spec.category}   ·   id {node.id[:14]}")
        # Bloc explicatif (Phase 30)
        self._populate_info_block(node.spec)
        # Chart pour observer / vg_block
        if node.spec.name == "observer":
            self._metric_chart = MetricChart(self._chart_host, width=260, height=140)
            self._chart_layout.addWidget(self._metric_chart)
            self._chart_host.show()
            # Replay si historique
            history = getattr(node, "_metric_history", None)
            if history:
                for m in history:
                    self._metric_chart.push_metrics(m)
        else:
            self._chart_host.hide()
        # Tooltip global sur le titre = description du node
        if node.spec.description:
            self.title.setToolTip(node.spec.description)
            self.subtitle.setToolTip(node.spec.description)
        for p in node.spec.params:
            w = self._build_param_widget(p, node.params.get(p.name, p.default), node)
            if w:
                help_txt = node.spec.param_help.get(p.name, "") if node.spec.param_help else ""
                if help_txt:
                    w.setToolTip(help_txt)
                from PySide6.QtWidgets import QLabel as _QL
                label = _QL(p.name + " :")
                if help_txt:
                    label.setToolTip(help_txt)
                self._form.addRow(label, w)

    def _populate_info_block(self, spec):
        """Phase 30 — remplit bloc explicatif depuis NodeSpec.

        Hiérarchie :
        - role (bold) : short summary (1 phrase max)
        - function (regular) : explication détaillée
        - sockets : inputs → outputs
        - equation : formule si présente
        - related : concepts liés si présents
        """
        # Toujours clear les labels (fix persistance text Phase 30b)
        self.role_label.clear()
        self.function_label.clear()
        self.equation_label.clear()
        self.related_label.clear()

        role = (getattr(spec, "role", "") or getattr(spec, "short", "")).strip()
        function = (getattr(spec, "function", "") or "").strip()
        description = (spec.description or "").strip()

        # Fallback role : 1ère phrase de description si pas de role explicite
        if not role and description:
            first_sentence = description.split(".")[0].strip()
            if len(first_sentence) < 80:
                role = first_sentence + "."

        # Fallback function : description complète si pas de function explicite
        if not function:
            function = description

        # Évite redondance : si function commence par role (legacy nodes),
        # garde seulement la suite après role
        if role and function and function.startswith(role):
            tail = function[len(role):].strip()
            function = tail if tail else ""

        equation = (getattr(spec, "equation", "") or "").strip()
        related = getattr(spec, "related", ()) or ()

        # Sockets summary
        in_types = " · ".join(s.socket_type for s in spec.inputs) or "—"
        out_types = " · ".join(s.socket_type for s in spec.outputs) or "—"
        sockets_text = f"IN : {in_types}    →    OUT : {out_types}"

        any_shown = False

        if role:
            self.role_label.setText(role)
            self.role_label.show()
            any_shown = True
        else:
            self.role_label.hide()

        if function:
            self.function_label.setText(function)
            self.function_label.show()
            any_shown = True
        else:
            self.function_label.hide()

        # Sockets toujours affichés (sauf si node sans sockets)
        if spec.inputs or spec.outputs:
            self.sockets_label.setText(sockets_text)
            self.sockets_label.show()
            any_shown = True
        else:
            self.sockets_label.hide()

        if equation:
            self.equation_label.setText(f"  {equation}")
            self.equation_label.show()
            any_shown = True
        else:
            self.equation_label.hide()

        if related:
            self.related_label.setText(
                "Voir aussi : " + " · ".join(str(r) for r in related[:6]))
            self.related_label.show()
            any_shown = True
        else:
            self.related_label.hide()

        if any_shown:
            self._info_host.show()
            self._info_sep.show()
        else:
            self._info_host.hide()

    def push_live_metrics(self, node_id: str, metrics: dict):
        """Push depuis trainer.observations vers chart si node sélectionné."""
        if self._node is None or self._node.id != node_id:
            return
        if self._metric_chart is None:
            return
        self._metric_chart.push_metrics(metrics)

    def _clear_chart(self):
        if self._metric_chart is not None:
            self._chart_layout.removeWidget(self._metric_chart)
            self._metric_chart.deleteLater()
            self._metric_chart = None

    def _clear_form(self):
        while self._form.rowCount() > 0:
            self._form.removeRow(0)

    def _build_param_widget(self, p: ParamSpec, value, node) -> QWidget | None:
        if p.kind == "int":
            w = QSpinBox()
            w.setRange(int(p.min) if p.min is not None else -10**9,
                        int(p.max) if p.max is not None else 10**9)
            w.setValue(int(value))
            w.valueChanged.connect(lambda v, n=p.name: node.set_param(n, v))
            return w
        if p.kind == "float":
            w = QDoubleSpinBox()
            w.setRange(float(p.min) if p.min is not None else -1e9,
                        float(p.max) if p.max is not None else 1e9)
            w.setDecimals(6)
            w.setValue(float(value))
            w.valueChanged.connect(lambda v, n=p.name: node.set_param(n, v))
            return w
        if p.kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(value))
            w.toggled.connect(lambda v, n=p.name: node.set_param(n, v))
            return w
        if p.kind == "choice":
            w = QComboBox()
            for opt in p.options:
                w.addItem(str(opt))
            if value in p.options:
                w.setCurrentIndex(p.options.index(value))
            w.currentTextChanged.connect(lambda v, n=p.name: node.set_param(n, v))
            return w
        if p.kind == "multiline":
            w = QPlainTextEdit()
            w.setPlainText(str(value))
            w.setFixedHeight(120)
            w.textChanged.connect(lambda n=p.name, ww=w: node.set_param(n, ww.toPlainText()))
            return w
        if p.kind in ("filepath", "dirpath"):
            # Widget composite : QLineEdit + bouton "..." → QFileDialog
            container = QWidget()
            h = QHBoxLayout(container)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(T.S_1)
            le = QLineEdit(container)
            le.setText(str(value) if value is not None else "")
            le.setPlaceholderText("Chemin fichier..." if p.kind == "filepath"
                                       else "Chemin dossier...")
            le.textChanged.connect(lambda v, n=p.name: node.set_param(n, v))
            btn = QPushButton("…", container)
            btn.setFixedWidth(28)
            btn.setStyleSheet(f"background-color: {T.ACCENT}; color: white; "
                                  "font-weight: 700;")
            is_dir = p.kind == "dirpath"
            file_filter = getattr(p, "file_filter", "") or "All files (*)"

            def _pick(_=None, le_=le, n=p.name, dir_mode=is_dir, flt=file_filter):
                cur = le_.text() or str(Path.home())
                if dir_mode:
                    chosen = QFileDialog.getExistingDirectory(
                        self, "Choisir dossier", cur)
                else:
                    chosen, _filter = QFileDialog.getOpenFileName(
                        self, "Choisir fichier", cur, flt)
                if chosen:
                    le_.setText(chosen)
                    node.set_param(n, chosen)
            btn.clicked.connect(_pick)
            h.addWidget(le, 1)
            h.addWidget(btn)
            return container
        # default : str
        w = QLineEdit()
        w.setText(str(value))
        w.textChanged.connect(lambda v, n=p.name: node.set_param(n, v))
        return w
