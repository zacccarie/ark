"""Phase 42 — Overlays flottants au-dessus du canvas.

Inspiré agencement node editors desktop modernes :
- LeftRail        : strip vertical d'icônes (palette/inspector/terminal/...)
- ActionBar       : barre top-right flottante (extensions + batch + RUN bleu + cancel + status)
- StatsOverlay    : stats compactes bottom-left (T, I, N, V, FPS)
- ViewControls    : zoom + select tool dropdown bottom-right + minimap toggle

Tous widgets standalone overlay-able sur QGraphicsView via .raise_() + manual positioning.
Style : dark + blur subtle + bordures rounded + accent bleu sur action primaire.
"""
from __future__ import annotations
import time
from PySide6.QtCore import Qt, Signal, QTimer, QSize, QRect, QPointF
from PySide6.QtGui import QFont, QPainter, QColor, QPen, QBrush, QPainterPath
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QSpinBox, QComboBox, QSizePolicy, QGraphicsDropShadowEffect,
)

from ark.ui import design_tokens as T


# ============================================================
# Left vertical icon rail
# ============================================================

class IconButton(QPushButton):
    """Icon-only button with tooltip and active state."""

    def __init__(self, glyph: str, tooltip: str, parent=None):
        super().__init__(glyph, parent)
        self.setToolTip(tooltip)
        self.setFixedSize(32, 32)
        self.setCheckable(True)
        self.setStyleSheet(
            f"QPushButton {{"
            f"  background: transparent;"
            f"  border: 1px solid transparent;"
            f"  border-radius: {T.R_SMALL}px;"
            f"  color: {T.FG_TERTIARY};"
            f"  font-size: 14px;"
            f"  padding: 0;"
            f"}}"
            f"QPushButton:hover {{"
            f"  background: {T.BG_HIGHLIGHT};"
            f"  color: {T.FG_PRIMARY};"
            f"}}"
            f"QPushButton:checked {{"
            f"  background: {T.ACCENT_BLUE_DIM};"
            f"  color: {T.ACCENT_BLUE};"
            f"  border-color: {T.ACCENT_BLUE};"
            f"}}"
        )


class LeftRail(QWidget):
    """Vertical icon rail à gauche du canvas. Émet panel_toggled(name)."""

    panel_toggled = Signal(str)
    action_triggered = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(44)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"background-color: {T.BG_PANEL}; "
            f"border-right: 1px solid {T.BORDER_DEFAULT};"
        )

        v = QVBoxLayout(self)
        v.setContentsMargins(6, 8, 6, 8)
        v.setSpacing(4)

        # Top section : panel toggles
        self.btn_palette = self._add_icon(v, "⊞", "Palette de nodes", "palette")
        self.btn_inspector = self._add_icon(v, "≣", "Inspector", "inspector")
        self.btn_outline = self._add_icon(v, "⌗", "Plan / arborescence", "outline")
        self.btn_groups = self._add_icon(v, "◫", "Mes groupes", "groups")
        self.btn_templates = self._add_icon(v, "✦", "Templates", "templates")
        self.btn_layout = self._add_icon(v, "▦", "Layouts", "layouts")

        v.addStretch(1)

        # Bottom section : help + terminal + settings
        self.btn_help = self._add_icon(v, "?", "Aide (F1)", "help",
                                            checkable=False)
        self.btn_terminal = self._add_icon(v, "⌘", "Terminal", "terminal")
        self.btn_shortcuts = self._add_icon(v, "⌨", "Raccourcis (Ctrl+/)",
                                                  "shortcuts", checkable=False)
        self.btn_settings = self._add_icon(v, "⚙", "Préférences", "settings",
                                                checkable=False)

    def _add_icon(self, layout, glyph, tooltip, name, checkable=True):
        btn = IconButton(glyph, tooltip, self)
        btn.setCheckable(checkable)
        if checkable:
            btn.clicked.connect(lambda checked, n=name: self.panel_toggled.emit(n))
        else:
            btn.clicked.connect(lambda: self.action_triggered.emit(name))
        layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignHCenter)
        return btn


# ============================================================
# Top-right floating action bar
# ============================================================

class ActionBar(QWidget):
    """Barre flottante top-right : Extensions + batch + RUN bleu + cancel + status."""

    compile_clicked = Signal()
    train_clicked = Signal()
    stop_clicked = Signal()
    extensions_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"ActionBar {{ "
            f"  background-color: {T.BG_PANEL}; "
            f"  border: 1px solid {T.BORDER_DEFAULT}; "
            f"  border-radius: {T.R_LARGE}px; "
            f"}}"
        )

        # Shadow elevation
        sh = QGraphicsDropShadowEffect(self)
        sh.setBlurRadius(24)
        sh.setColor(QColor(0, 0, 0, 140))
        sh.setOffset(0, 6)
        self.setGraphicsEffect(sh)

        h = QHBoxLayout(self)
        h.setContentsMargins(6, 4, 6, 4)
        h.setSpacing(4)

        # Extensions button — labeled
        self.btn_ext = QPushButton("Extensions", self)
        self.btn_ext.setFixedHeight(28)
        self.btn_ext.setToolTip("Gérer les extensions / templates")
        self.btn_ext.setStyleSheet(
            f"QPushButton {{"
            f"  background: transparent;"
            f"  border: 1px solid {T.BORDER_DEFAULT};"
            f"  border-radius: {T.R_SMALL}px;"
            f"  color: {T.FG_SECONDARY};"
            f"  font-size: {T.FS_LABEL}px;"
            f"  padding: 0 10px;"
            f"}}"
            f"QPushButton:hover {{ background: {T.BG_HIGHLIGHT}; "
            f"  color: {T.FG_PRIMARY}; }}"
        )
        self.btn_ext.clicked.connect(self.extensions_clicked.emit)
        h.addWidget(self.btn_ext)

        # Batch label + spinbox
        batch_lbl = QLabel("Batch", self)
        batch_lbl.setStyleSheet(
            f"color: {T.FG_TERTIARY}; font-size: {T.FS_MICRO}px; "
            "text-transform: uppercase; letter-spacing: 1px;")
        h.addWidget(batch_lbl)
        self.batch_spin = QSpinBox(self)
        self.batch_spin.setRange(1, 1000)
        self.batch_spin.setValue(1)
        self.batch_spin.setFixedWidth(60)
        self.batch_spin.setFixedHeight(28)
        self.batch_spin.setToolTip("Nombre d'itérations / batches")
        h.addWidget(self.batch_spin)

        # PRIMARY RUN button — bleu compact
        self.btn_run = QPushButton("▶  Compiler", self)
        self.btn_run.setFixedHeight(30)
        self.btn_run.setStyleSheet(
            f"QPushButton {{"
            f"  background-color: {T.ACCENT_BLUE};"
            f"  color: white;"
            f"  border: 1px solid {T.ACCENT_BLUE};"
            f"  border-radius: {T.R_SMALL}px;"
            f"  padding: 4px 14px;"
            f"  font-weight: 700;"
            f"  font-size: {T.FS_LABEL}px;"
            f"  letter-spacing: -0.2px;"
            f"}}"
            f"QPushButton:hover {{ background-color: {T.ACCENT_BLUE_HOVER}; }}"
            f"QPushButton:pressed {{ background-color: {T.ACCENT_BLUE_PRESS}; }}"
        )
        self.btn_run.clicked.connect(self.compile_clicked.emit)
        h.addWidget(self.btn_run)

        # Mode dropdown
        self.mode_combo = QComboBox(self)
        self.mode_combo.addItems(["Compiler", "Entraîner", "Bench"])
        self.mode_combo.setFixedWidth(130)
        self.mode_combo.setFixedHeight(28)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        h.addWidget(self.mode_combo)

        # Cancel button (danger) — labeled
        self.btn_cancel = QPushButton("Stop", self)
        self.btn_cancel.setFixedHeight(28)
        self.btn_cancel.setToolTip("Arrêter")
        self.btn_cancel.setStyleSheet(
            f"QPushButton {{"
            f"  background-color: transparent;"
            f"  color: {T.ACCENT};"
            f"  border: 1px solid {T.ACCENT_DIM};"
            f"  border-radius: {T.R_SMALL}px;"
            f"  font-weight: 700; font-size: {T.FS_LABEL}px;"
            f"  padding: 0 10px;"
            f"}}"
            f"QPushButton:hover {{ background-color: {T.ACCENT}; color: white; }}"
        )
        self.btn_cancel.clicked.connect(self.stop_clicked.emit)
        h.addWidget(self.btn_cancel)

        # Active count badge — compact
        self.active_lbl = QLabel("0 actif", self)
        self.active_lbl.setToolTip("Tâches actives")
        self.active_lbl.setStyleSheet(
            f"color: {T.FG_SECONDARY}; "
            f"font-family: {T.FONT_MONO_FAMILY}; "
            f"font-size: {T.FS_MICRO}px; "
            f"font-weight: 700; "
            f"padding: 2px 6px; "
            f"background: {T.BG_PANEL_RAISED}; "
            f"border: 1px solid {T.BORDER_DEFAULT}; "
            f"border-radius: {T.R_SMALL}px;"
        )
        h.addWidget(self.active_lbl)

        # Toggle inspector button — labeled
        self.btn_inspector_toggle = QPushButton("Insp.", self)
        self.btn_inspector_toggle.setFixedHeight(28)
        self.btn_inspector_toggle.setToolTip("Toggle Inspector")
        self.btn_inspector_toggle.setStyleSheet(
            f"QPushButton {{"
            f"  background: transparent;"
            f"  border: 1px solid {T.BORDER_DEFAULT};"
            f"  border-radius: {T.R_SMALL}px;"
            f"  color: {T.FG_SECONDARY};"
            f"  font-size: {T.FS_LABEL}px;"
            f"  padding: 0 10px;"
            f"}}"
            f"QPushButton:hover {{ background: {T.BG_HIGHLIGHT}; color: {T.FG_PRIMARY}; }}"
        )
        h.addWidget(self.btn_inspector_toggle)

    def _on_mode_changed(self, mode):
        labels = {"Compiler": "▶  Compiler", "Entraîner": "▶  Entraîner",
                    "Bench": "▶  Bench"}
        self.btn_run.setText(labels.get(mode, "▶  Run"))

    def set_active_count(self, n: int):
        self.active_lbl.setText(str(n))


# ============================================================
# Bottom-left stats overlay (compact mono)
# ============================================================

class StatsOverlay(QWidget):
    """Compact stats : T (time), I (iter), N (nodes), V (val), FPS."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"StatsOverlay {{ "
            f"  background-color: rgba(24, 24, 27, 200); "
            f"  border-radius: {T.R_SMALL}px; "
            f"  padding: 2px 6px; "
            f"}}"
        )

        h = QHBoxLayout(self)
        h.setContentsMargins(6, 2, 6, 2)
        h.setSpacing(8)

        # Phase 58 : compact 1-line stats — nodes + FPS only
        self.n_lbl = self._make_lbl("N 0")
        self.fps_lbl = self._make_lbl("60fps")
        for lbl in (self.n_lbl, self.fps_lbl):
            h.addWidget(lbl)
        # Hidden labels kept for API compat (push_* methods write into them)
        self.t_lbl = self._make_lbl("")
        self.i_lbl = self._make_lbl("")
        self.v_lbl = self._make_lbl("")
        for lbl in (self.t_lbl, self.i_lbl, self.v_lbl):
            lbl.hide()

        self._t0 = time.perf_counter()
        self._frame_count = 0
        self._last_fps_t = self._t0

        # Auto-refresh time + fps
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(500)

    def _make_lbl(self, text):
        l = QLabel(text, self)
        l.setStyleSheet(
            f"color: {T.FG_TERTIARY}; "
            f"font-family: {T.FONT_MONO_FAMILY}; "
            f"font-size: 9px; "
            f"font-weight: 600; "
            f"background: transparent;"
        )
        return l

    def _tick(self):
        elapsed = time.perf_counter() - self._t0
        self.t_lbl.setText(f"{elapsed:.0f}s")

    def set_iter(self, i: int):
        self.i_lbl.setText(f"I{i}")

    def set_nodes(self, n: int, selected: int = 0):
        if selected:
            self.n_lbl.setText(f"N {n}·{selected}")
        else:
            self.n_lbl.setText(f"N {n}")

    def set_val(self, v):
        self.v_lbl.setText(f"V{v}")

    def set_fps(self, fps: float):
        self.fps_lbl.setText(f"{fps:.0f}fps")

    def reset_timer(self):
        self._t0 = time.perf_counter()


# ============================================================
# Bottom-right view controls (zoom + tool + minimap)
# ============================================================

class GlobalProgress(QWidget):
    """Phase 49 : barre progression globale top canvas. Cachée si idle."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(36)
        self._progress = 0.0
        self._label_text = "READY"
        self._state = "idle"  # idle | compiling | training | ok | error
        self.setStyleSheet(
            f"GlobalProgress {{ "
            f"  background-color: {T.BG_PANEL}; "
            f"  border: 1px solid {T.BORDER_DEFAULT}; "
            f"  border-radius: {T.R_LARGE}px; "
            f"}}"
        )

        sh = QGraphicsDropShadowEffect(self)
        sh.setBlurRadius(20)
        sh.setColor(QColor(0, 0, 0, 140))
        sh.setOffset(0, 4)
        self.setGraphicsEffect(sh)

        h = QHBoxLayout(self)
        h.setContentsMargins(14, 6, 14, 6)
        h.setSpacing(10)

        self.dot = QLabel("●", self)
        self.dot.setStyleSheet(
            f"color: {T.FG_TERTIARY}; font-size: 14px; background: transparent;")
        h.addWidget(self.dot)

        self.label = QLabel("READY", self)
        self.label.setStyleSheet(
            f"color: {T.FG_PRIMARY}; "
            f"font-family: {T.FONT_DISPLAY_FAMILY}; "
            f"font-size: {T.FS_LABEL}px; "
            f"font-weight: 700; "
            f"text-transform: uppercase; "
            f"letter-spacing: 1.5px; "
            f"background: transparent;")
        h.addWidget(self.label)

        self.pct = QLabel("", self)
        self.pct.setStyleSheet(
            f"color: {T.FG_SECONDARY}; "
            f"font-family: {T.FONT_MONO_FAMILY}; "
            f"font-size: {T.FS_LABEL}px; "
            f"font-weight: 600; "
            f"background: transparent;")
        h.addWidget(self.pct)

        h.addStretch(1)

    def set_progress(self, value: float, label: str = "", state: str = ""):
        self._progress = max(0.0, min(1.0, float(value)))
        if label:
            self._label_text = label
            self.label.setText(label.upper())
        if state:
            self._state = state
        self.pct.setText(f"{int(self._progress * 100)}%")
        colors = {
            "idle": T.FG_TERTIARY,
            "compiling": T.INFO,
            "training": T.INFO,
            "ok": T.STATE_OK,
            "error": T.STATE_ERROR,
        }
        c = colors.get(self._state, T.FG_TERTIARY)
        self.dot.setStyleSheet(
            f"color: {c}; font-size: 14px; background: transparent;")
        self.update()

    def paintEvent(self, ev):
        super().paintEvent(ev)
        if self._state == "idle" or self._progress <= 0:
            return
        # Barre progression au bottom
        from PySide6.QtGui import QPainter, QColor as _QC
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        bar_h = 3
        bar_y = rect.height() - bar_h - 2
        bar_w = int((rect.width() - 24) * self._progress)
        col_map = {
            "compiling": T.INFO, "training": T.INFO,
            "ok": T.STATE_OK, "error": T.STATE_ERROR,
        }
        col = _QC(col_map.get(self._state, T.PRIMARY))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(col)
        p.drawRoundedRect(12, bar_y, bar_w, bar_h, 1.5, 1.5)
        p.end()


class ViewControls(QWidget):
    """Tool dropdown + zoom % + minimap toggle + snap toggle."""

    zoom_changed = Signal(float)
    tool_changed = Signal(str)
    minimap_toggled = Signal(bool)
    snap_toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"ViewControls {{"
            f"  background-color: {T.BG_PANEL}; "
            f"  border: 1px solid {T.BORDER_DEFAULT}; "
            f"  border-radius: {T.R_LARGE}px; "
            f"}}"
        )

        sh = QGraphicsDropShadowEffect(self)
        sh.setBlurRadius(20)
        sh.setColor(QColor(0, 0, 0, 130))
        sh.setOffset(0, 4)
        self.setGraphicsEffect(sh)

        h = QHBoxLayout(self)
        h.setContentsMargins(4, 3, 4, 3)
        h.setSpacing(3)

        # Tool selector — compact
        self.tool_combo = QComboBox(self)
        self.tool_combo.addItems(["Select", "Pan", "Box"])
        self.tool_combo.setFixedWidth(80)
        self.tool_combo.setFixedHeight(24)
        self.tool_combo.currentTextChanged.connect(
            lambda t: self.tool_changed.emit(t.split()[-1].lower()))
        h.addWidget(self.tool_combo)

        # Fit-to-view button
        self.btn_fit = QPushButton("Fit", self)
        self.btn_fit.setFixedHeight(24)
        self.btn_fit.setToolTip("Centrer + ajuster vue")
        self.btn_fit.setStyleSheet(self._icon_style())
        h.addWidget(self.btn_fit)

        # Zoom %
        self.zoom_combo = QComboBox(self)
        self.zoom_combo.addItems(["25%", "50%", "75%", "100%", "150%", "200%",
                                         "400%"])
        self.zoom_combo.setCurrentText("100%")
        self.zoom_combo.setFixedWidth(58)
        self.zoom_combo.setFixedHeight(24)
        self.zoom_combo.currentTextChanged.connect(self._on_zoom)
        h.addWidget(self.zoom_combo)

        # Minimap toggle
        self.btn_map = QPushButton("Map", self)
        self.btn_map.setFixedHeight(24)
        self.btn_map.setCheckable(True)
        self.btn_map.setChecked(True)
        self.btn_map.setToolTip("Toggle minimap")
        self.btn_map.setStyleSheet(self._icon_style())
        self.btn_map.toggled.connect(self.minimap_toggled.emit)
        h.addWidget(self.btn_map)

        # Snap toggle
        self.btn_snap = QPushButton("Snap", self)
        self.btn_snap.setFixedHeight(24)
        self.btn_snap.setCheckable(True)
        self.btn_snap.setToolTip("Snap to grid")
        self.btn_snap.setStyleSheet(self._icon_style())
        self.btn_snap.toggled.connect(self.snap_toggled.emit)
        h.addWidget(self.btn_snap)

    def _icon_style(self):
        return (
            f"QPushButton {{"
            f"  background: transparent;"
            f"  border: 1px solid {T.BORDER_DEFAULT};"
            f"  border-radius: {T.R_SMALL}px;"
            f"  color: {T.FG_SECONDARY};"
            f"  font-size: {T.FS_LABEL}px;"
            f"  padding: 0 8px;"
            f"}}"
            f"QPushButton:hover {{ background: {T.BG_HIGHLIGHT}; "
            f"  color: {T.FG_PRIMARY}; }}"
            f"QPushButton:checked {{ background: {T.ACCENT_BLUE_DIM}; "
            f"  color: {T.ACCENT_BLUE}; border-color: {T.ACCENT_BLUE}; }}"
        )

    def _on_zoom(self, text):
        try:
            pct = float(text.rstrip("%"))
            self.zoom_changed.emit(pct / 100.0)
        except ValueError:
            pass

    def set_zoom(self, factor: float):
        pct = int(factor * 100)
        text = f"{pct}%"
        self.zoom_combo.blockSignals(True)
        # Find or set
        idx = self.zoom_combo.findText(text)
        if idx >= 0:
            self.zoom_combo.setCurrentIndex(idx)
        else:
            self.zoom_combo.setEditText(text)
        self.zoom_combo.blockSignals(False)
