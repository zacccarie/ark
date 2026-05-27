"""Inspector — propriétés du node sélectionné (params éditables)."""
from __future__ import annotations
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QFormLayout, QVBoxLayout, QLabel, QLineEdit, QSpinBox,
    QDoubleSpinBox, QComboBox, QCheckBox, QPlainTextEdit, QFrame,
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
            self._chart_host.hide()
            return
        self.title.setText(node.spec.title)
        self.subtitle.setText(f"{node.spec.category}   ·   id {node.id[:14]}")
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
        # default : str
        w = QLineEdit()
        w.setText(str(value))
        w.textChanged.connect(lambda v, n=p.name: node.set_param(n, v))
        return w
