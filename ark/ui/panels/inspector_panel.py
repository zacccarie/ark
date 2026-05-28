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

        # Phase 38 : intuition (vulgarisation, vert électrique)
        self.intuition_label = QLabel("", self._info_host)
        self.intuition_label.setWordWrap(True)
        self.intuition_label.setStyleSheet(
            f"font-size: {T.FS_LABEL}px; color: {T.SUCCESS}; "
            f"font-style: italic; line-height: 1.5; "
            f"padding: {T.S_2}px {T.S_3}px; "
            f"background-color: {T.SUCCESS_DIM}; "
            f"border-left: 2px solid {T.SUCCESS};"
        )
        self._info_layout.addWidget(self.intuition_label)

        self.equation_label = QLabel("", self._info_host)
        self.equation_label.setWordWrap(True)
        self.equation_label.setStyleSheet(
            f"font-family: 'Menlo','Courier New',monospace; "
            f"font-size: {T.FS_LABEL}px; color: {T.ACCENT}; "
            f"background-color: rgba(225,29,42,0.08); "
            f"padding: {T.S_2}px {T.S_3}px; border-left: 2px solid {T.ACCENT};"
        )
        self._info_layout.addWidget(self.equation_label)

        # Phase 38 : when_to_use (cyan)
        self.when_label = QLabel("", self._info_host)
        self.when_label.setWordWrap(True)
        self.when_label.setStyleSheet(
            f"font-size: {T.FS_LABEL}px; color: {T.INFO}; "
            f"padding: {T.S_2}px {T.S_3}px; "
            f"background-color: {T.INFO_DIM}; "
            f"border-left: 2px solid {T.INFO}; line-height: 1.5;"
        )
        self._info_layout.addWidget(self.when_label)

        # Phase 38 : example (mono, accent rouge)
        self.example_label = QLabel("", self._info_host)
        self.example_label.setWordWrap(True)
        self.example_label.setStyleSheet(
            f"font-family: 'Menlo','Courier New',monospace; "
            f"font-size: {T.FS_LABEL}px; color: {T.FG_PRIMARY}; "
            f"padding: {T.S_2}px {T.S_3}px; "
            f"background-color: {T.BG_OVERLAY}; "
            f"border-left: 2px solid {T.FG_SECONDARY};"
        )
        self._info_layout.addWidget(self.example_label)

        # Phase 38 : pitfall (warn amber)
        self.pitfall_label = QLabel("", self._info_host)
        self.pitfall_label.setWordWrap(True)
        self.pitfall_label.setStyleSheet(
            f"font-size: {T.FS_LABEL}px; color: {T.WARN}; "
            f"padding: {T.S_2}px {T.S_3}px; "
            f"background-color: {T.WARN_DIM}; "
            f"border-left: 2px solid {T.WARN}; line-height: 1.5;"
        )
        self._info_layout.addWidget(self.pitfall_label)

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

        # Phase 52 : preview thumbnail si node a _preview_qimage
        self._preview_label = QLabel("", self)
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setStyleSheet(
            f"background-color: {T.BG_DEEP}; "
            f"border: 1px solid {T.PRIMARY}; "
            f"border-radius: {T.R_MEDIUM}px; "
            f"padding: 4px;")
        self._preview_label.setMinimumHeight(200)
        self._preview_label.setMaximumHeight(280)
        v.addWidget(self._preview_label)
        self._preview_label.hide()

        # Phase 55 : live output (text/value/predictions) après compile
        self._output_host = QWidget(self)
        self._output_layout = QVBoxLayout(self._output_host)
        self._output_layout.setContentsMargins(0, 0, 0, T.S_2)
        self._output_layout.setSpacing(T.S_1)
        self._output_title = QLabel("Live Output", self._output_host)
        self._output_title.setStyleSheet(
            f"font-size: {T.FS_MICRO}px; color: {T.SUCCESS}; "
            f"text-transform: uppercase; letter-spacing: 1px; font-weight: 700;")
        self._output_layout.addWidget(self._output_title)
        self._output_text = QPlainTextEdit("", self._output_host)
        self._output_text.setReadOnly(True)
        self._output_text.setMaximumHeight(220)
        self._output_text.setStyleSheet(
            f"font-family: 'Menlo','Courier New',monospace; "
            f"font-size: {T.FS_LABEL}px; color: {T.FG_PRIMARY}; "
            f"background-color: {T.BG_DEEP}; "
            f"border: 1px solid {T.SUCCESS}; "
            f"border-radius: {T.R_MEDIUM}px; padding: 6px;")
        self._output_layout.addWidget(self._output_text)
        v.addWidget(self._output_host)
        self._output_host.hide()

        # Phase 56b : Audio player widget (▶/⏸/💾)
        self._audio_host = QWidget(self)
        self._audio_layout = QVBoxLayout(self._audio_host)
        self._audio_layout.setContentsMargins(0, 0, 0, T.S_2)
        self._audio_layout.setSpacing(T.S_1)
        self._audio_title = QLabel("Audio Player", self._audio_host)
        self._audio_title.setStyleSheet(
            f"font-size: {T.FS_MICRO}px; color: {T.SUCCESS}; "
            f"text-transform: uppercase; letter-spacing: 1px; font-weight: 700;")
        self._audio_layout.addWidget(self._audio_title)
        self._audio_path_label = QLabel("", self._audio_host)
        self._audio_path_label.setStyleSheet(
            f"font-family: 'Menlo',monospace; font-size: {T.FS_MICRO}px; "
            f"color: {T.FG_TERTIARY};")
        self._audio_path_label.setWordWrap(True)
        self._audio_layout.addWidget(self._audio_path_label)
        # Button row
        _btn_row = QWidget(self._audio_host)
        _btn_row_h = QHBoxLayout(_btn_row)
        _btn_row_h.setContentsMargins(0, 0, 0, 0)
        _btn_row_h.setSpacing(T.S_2)
        self._audio_play_btn = QPushButton("▶  Play", _btn_row)
        self._audio_play_btn.setStyleSheet(
            f"background-color: {T.SUCCESS}; color: {T.BG_DEEP}; "
            f"border: none; border-radius: {T.R_MEDIUM}px; "
            f"padding: 6px 12px; font-weight: 700;")
        self._audio_play_btn.clicked.connect(self._on_audio_play_toggle)
        _btn_row_h.addWidget(self._audio_play_btn)
        self._audio_dl_btn = QPushButton("⬇  Download", _btn_row)
        self._audio_dl_btn.setStyleSheet(
            f"background-color: {T.BG_PANEL_RAISED}; color: {T.FG_PRIMARY}; "
            f"border: 1px solid {T.BORDER_DEFAULT}; "
            f"border-radius: {T.R_MEDIUM}px; padding: 6px 12px;")
        self._audio_dl_btn.clicked.connect(self._on_audio_download)
        _btn_row_h.addWidget(self._audio_dl_btn)
        _btn_row_h.addStretch(1)
        self._audio_layout.addWidget(_btn_row)
        v.addWidget(self._audio_host)
        self._audio_host.hide()

        # Audio playback (subprocess afplay/aplay/powershell)
        self._audio_proc = None
        self._poll_timer = None
        self._is_playing = False

        # Stored compiled modules_dict (set by MainWindow post-compile)
        self._modules_dict: dict = {}

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
        # Phase 52 : preview thumbnail si présent
        self._update_preview(node)
        # Phase 55 : live output (text_viewer, number_display, tiny_llm_generate, top_k)
        self._update_live_output(node)
        # Phase 56b : audio player widget
        self._update_audio_player(node)
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
        """Phase 30/38 — remplit bloc explicatif exhaustif depuis NodeSpec + glossaire.

        Hiérarchie (Phase 38) :
        - role (bold) : short summary
        - function (regular) : comment ça marche
        - intuition (italic vert) : analogie vulgarisée
        - sockets : inputs → outputs
        - equation : formule
        - when_to_use (cyan) : cas d'usage concret
        - example (mono) : exemple chiffré
        - pitfall (amber) : piège à éviter
        - related : concepts liés
        """
        # Toujours clear les labels (fix persistance text Phase 30b)
        self.role_label.clear()
        self.function_label.clear()
        self.equation_label.clear()
        self.related_label.clear()
        # Phase 38 : enrichissement depuis GLOSSARY + DIDACTIC_EXTRA
        from ark.core.glossary import GLOSSARY
        glossary_entry = next(
            (g for g in GLOSSARY if g.term == spec.title), None)
        # Fallback : lookup didactic par spec.name (snake_case)
        try:
            from ark.core.didactic_data import DIDACTIC_EXTRA
            didactic = DIDACTIC_EXTRA.get(spec.name, {})
        except ImportError:
            didactic = {}

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

        # Phase 38 : enrichissement multi-source (GLOSSARY + DIDACTIC_EXTRA + spec)
        intuition = didactic.get("intuition", "")
        when_to_use = didactic.get("when_to_use", "")
        example = didactic.get("example", "")
        pitfall = didactic.get("pitfall", "")
        # Fallback GlossaryEntry
        if glossary_entry is not None:
            intuition = intuition or glossary_entry.intuition
            when_to_use = when_to_use or glossary_entry.when_to_use
            example = example or glossary_entry.example
            pitfall = pitfall or glossary_entry.pitfall
            if not equation:
                equation = glossary_entry.equation
            if not related and glossary_entry.related:
                related = glossary_entry.related

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

        # Phase 38 : sections didactiques
        for label_attr, content, prefix in [
            ("intuition_label", intuition, "💡 "),
            ("when_label", when_to_use, "🎯 "),
            ("example_label", example, ""),
            ("pitfall_label", pitfall, "⚠ "),
        ]:
            lbl = getattr(self, label_attr, None)
            if lbl is None:
                continue
            lbl.clear()
            if content:
                lbl.setText(prefix + content)
                lbl.show()
                any_shown = True
            else:
                lbl.hide()

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

    def _update_preview(self, node):
        """Phase 52 : affiche preview si node a _preview_qimage."""
        qimg = getattr(node, "_preview_qimage", None)
        if qimg is None:
            self._preview_label.hide()
            self._preview_label.clear()
            return
        try:
            from PySide6.QtGui import QPixmap
            pix = QPixmap.fromImage(qimg)
            # Scale preserving aspect ratio
            max_w = self._preview_label.width() or 280
            max_h = 260
            scaled = pix.scaled(max_w, max_h,
                                  Qt.AspectRatioMode.KeepAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation)
            self._preview_label.setPixmap(scaled)
            self._preview_label.show()
        except Exception:
            self._preview_label.hide()

    def _update_audio_player(self, node):
        """Phase 56b : affiche play/pause/download si node a last_audio_path.

        Sources path (priorité) :
        1. modules_dict[node.id].last_audio_path (post-compile)
        2. ~/.ark/audio_player/<node.id>.wav (cache disque)
        """
        if node is None:
            self._audio_host.hide()
            return
        # Only show for audio_player node type
        if not hasattr(node, "spec") or node.spec.name != "audio_player":
            self._audio_host.hide()
            return
        path = None
        mod = self._modules_dict.get(node.id) if self._modules_dict else None
        if mod is not None:
            p = getattr(mod, "last_audio_path", None)
            if p and not p.startswith("err:") and Path(p).exists():
                path = p
        if not path:
            # Disk fallback : ~/.ark/audio_player/<node.id>.wav
            disk = Path.home() / ".ark" / "audio_player" / f"{node.id}.wav"
            if disk.exists():
                path = str(disk)
        if not path:
            self._audio_host.hide()
            self._stop_audio()
            return
        self._audio_path_label.setText(path)
        self._audio_host.show()
        self._stop_audio()

    def _on_audio_play_toggle(self):
        """Phase 56b : play/pause via subprocess (afplay macOS / aplay linux /
        start windows). Plus fiable que QMediaPlayer."""
        path = self._audio_path_label.text()
        if not path or not Path(path).exists():
            return
        if self._is_playing:
            self._stop_audio()
            return
        try:
            import subprocess, sys, platform
            sysname = platform.system()
            if sysname == "Darwin":
                cmd = ["afplay", path]
            elif sysname == "Linux":
                cmd = ["aplay", "-q", path]
            else:  # Windows
                cmd = ["powershell", "-c",
                          f"(New-Object Media.SoundPlayer '{path}').PlaySync();"]
            self._audio_proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._is_playing = True
            self._audio_play_btn.setText("⏸  Stop")
            # Polling via QTimer pour reset bouton à fin lecture
            from PySide6.QtCore import QTimer
            self._poll_timer = QTimer(self)
            self._poll_timer.timeout.connect(self._poll_audio_proc)
            self._poll_timer.start(200)
        except Exception as e:
            self._audio_path_label.setText(f"play error: {e}")

    def _poll_audio_proc(self):
        proc = getattr(self, "_audio_proc", None)
        if proc is None or proc.poll() is not None:
            self._stop_audio()
            if getattr(self, "_poll_timer", None) is not None:
                self._poll_timer.stop()

    def _stop_audio(self):
        proc = getattr(self, "_audio_proc", None)
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
            self._audio_proc = None
        self._audio_play_btn.setText("▶  Play")
        self._is_playing = False

    def _on_audio_download(self):
        path = self._audio_path_label.text()
        if not path or not Path(path).exists():
            return
        try:
            dest, _ = QFileDialog.getSaveFileName(
                self, "Save audio as…",
                str(Path.home() / "Downloads" / Path(path).name),
                "Audio (*.wav *.mp3);;All files (*)")
            if dest:
                import shutil
                shutil.copy2(path, dest)
        except Exception:
            pass

    def set_modules_dict(self, modules_dict: dict):
        """Phase 55 : appelé par MainWindow post-compile pour exposer outputs."""
        self._modules_dict = modules_dict or {}
        # Refresh affichage si node actuel concerné
        if self._node is not None:
            self._update_live_output(self._node)
            self._update_audio_player(self._node)

    def _update_live_output(self, node):
        """Phase 55 : affiche last_text/last_value/last_predictions du module compilé."""
        if not self._modules_dict or node is None:
            self._output_host.hide()
            return
        mod = self._modules_dict.get(node.id)
        if mod is None:
            self._output_host.hide()
            return
        lines = []
        if hasattr(mod, "last_generated") and mod.last_generated:
            lines.append(f"▸ generated:\n{mod.last_generated}")
        if hasattr(mod, "last_text") and mod.last_text:
            lines.append(f"▸ text:\n{mod.last_text}")
        if hasattr(mod, "last_value") and mod.last_value not in ("", None):
            lines.append(f"▸ value: {mod.last_value}")
        if hasattr(mod, "last_predictions") and mod.last_predictions:
            lines.append("▸ top-K:")
            for p in mod.last_predictions[:10]:
                lines.append(f"  #{p['rank']:>2} {p['label']:<24} {p['prob']:.4f}")
        if hasattr(mod, "last_path") and mod.last_path:
            lines.append(f"▸ output file:\n{mod.last_path}")
        if not lines:
            self._output_host.hide()
            return
        self._output_text.setPlainText("\n\n".join(lines))
        self._output_host.show()

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
