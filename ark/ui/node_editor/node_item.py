"""NodeItem — node visuel + sockets + handles redim + bypass."""
from __future__ import annotations
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QBrush, QPen, QColor, QFont, QPainterPath, QCursor
from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsTextItem, QGraphicsProxyWidget,
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QFileDialog,
    QPlainTextEdit,
)

from ark.ui import design_tokens as T
from ark.ui.node_editor.socket_item import SocketItem
from ark.nodes.registry import NodeSpec, SocketSpec
import uuid


class NodeItem(QGraphicsItem):
    DEFAULT_WIDTH = 220
    HEADER_H = 32
    ROW_H = 22
    PARAM_H = 18
    SOCKET_MARGIN = 14
    PADDING_BOTTOM = 8
    HANDLE_SIZE = 8

    def __init__(self, spec: NodeSpec):
        super().__init__()
        self.id = f"{spec.name}_{uuid.uuid4().hex[:6]}"
        self.spec = spec
        self.params: dict = {p.name: p.default for p in spec.params}
        self.bypass = False
        self.error: str | None = None
        self.width = self.DEFAULT_WIDTH
        self.user_height_extra = 0  # Phase 33 : redim vertical user (px additionnels)
        # Phase 36 : status sémantique
        self.status: str = "idle"  # idle | compiling | training | ok | error | warn
        self._hover_scale = 1.0  # animé subtilement sur hover
        # Phase 49 : progress par node 0..1 (animé loading si > 0)
        self.progress: float = 0.0
        # Phase 51 : preview image (tensor C, H, W) pour save_image nodes
        self._preview_image_tensor = None
        self._preview_qimage = None

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setZValue(T.Z_NODE)
        self.setAcceptHoverEvents(True)
        self._hover = False

        # Tooltip riche (HTML) — survol node = description + I/O + params
        self.setToolTip(self._build_tooltip())

        # Sockets
        self.input_sockets: list[SocketItem] = []
        self.output_sockets: list[SocketItem] = []
        for i, s in enumerate(spec.inputs):
            sock = SocketItem(self, s.label, s.socket_type, is_input=True, index=i)
            self.input_sockets.append(sock)
        for i, s in enumerate(spec.outputs):
            sock = SocketItem(self, s.label, s.socket_type, is_input=False, index=i)
            self.output_sockets.append(sock)
        self._layout_sockets()

        # Resize state — Phase 33 : 2D (W + H)
        self._resizing = False
        self._resize_start_w = 0
        self._resize_start_h_extra = 0
        self._resize_start_pos = QPointF()

        # Phase 56f/57 : embed widgets IN node (audio_player, text_input)
        self._embed_proxy: QGraphicsProxyWidget | None = None
        self._embed_host: QWidget | None = None
        self._audio_play_btn = None
        self._audio_dl_btn = None
        self._audio_proc = None
        self._audio_is_playing = False
        self._audio_poll_timer = None
        self._text_input_widget: QPlainTextEdit | None = None
        if spec.name == "audio_player":
            self._setup_audio_player_widget()
        elif spec.name == "text_input":
            self._setup_text_input_widget()

    # ------------------------------------------------------------
    # Size / layout
    # ------------------------------------------------------------
    @property
    def natural_height(self) -> int:
        """Phase 33 : hauteur calculée depuis content (sans extra user)."""
        n_sockets = max(len(self.input_sockets), len(self.output_sockets))
        socket_h = max(n_sockets, 1) * self.ROW_H
        param_h = len(self.spec.params) * self.PARAM_H
        thumb_h = 58 if self.spec.has_thumbnail else 0
        err_h = 16 if self.error else 0
        # Phase 56f/57 : embedded widgets (audio_player 80px, text_input 90px)
        embed_h = 0
        if self.spec.name == "audio_player":
            embed_h = 80
        elif self.spec.name == "text_input":
            embed_h = 90
        return (self.HEADER_H + socket_h + param_h + thumb_h + err_h
                + embed_h + self.PADDING_BOTTOM)

    @property
    def height(self) -> int:
        return self.natural_height + int(self.user_height_extra)

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.width, self.height + self.HANDLE_SIZE)

    def _layout_sockets(self):
        for i, sock in enumerate(self.input_sockets):
            sock.setPos(0, self.HEADER_H + self.ROW_H * i + self.ROW_H / 2)
        for i, sock in enumerate(self.output_sockets):
            sock.setPos(self.width, self.HEADER_H + self.ROW_H * i + self.ROW_H / 2)

    # ------------------------------------------------------------
    # Paint — Phase 50d try/except global (Python exception dans paint
    # cause SIGSEGV via PySide6 bridge)
    # ------------------------------------------------------------
    def paint(self, painter, option, widget=None):
        try:
            self._paint_safe(painter, option, widget)
        except Exception as e:
            import sys
            import traceback
            print(f"[NodeItem.paint EXCEPTION] {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            # Fallback minimal : juste fill rect rouge pour debug visuel
            try:
                painter.fillRect(QRectF(0, 0, self.width, max(self.height, 30)),
                                  QColor("#3A2A05"))
            except Exception:
                pass

    def _paint_safe(self, painter, option, widget=None):
        rect = QRectF(0, 0, self.width, self.height)
        path = QPainterPath()
        path.addRoundedRect(rect, T.R_MEDIUM, T.R_MEDIUM)

        # Body
        painter.setBrush(QBrush(QColor(T.BG_PANEL_RAISED)))
        border_col = T.ACCENT if self.error else (T.BORDER_STRONG if self.isSelected()
                                                     else (T.BORDER_DEFAULT if not self._hover else T.BORDER_STRONG))
        pen = QPen(QColor(border_col))
        pen.setWidth(2 if (self.isSelected() or self.error) else 1)
        painter.setPen(pen)
        painter.drawPath(path)

        # Header band — Phase 41 plus haut + radius medium
        header_rect = QRectF(0, 0, self.width, self.HEADER_H)
        header_path = QPainterPath()
        header_path.addRoundedRect(header_rect, T.R_MEDIUM, T.R_MEDIUM)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(T.BG_PANEL_RAISED)))
        painter.drawPath(header_path)
        # Cut rounded bottom
        painter.fillRect(QRectF(0, self.HEADER_H - 4, self.width, 4),
                          QColor(T.BG_PANEL_RAISED))
        # Subtle accent bar at top
        if self.isSelected() or self.status == "ok":
            painter.fillRect(QRectF(0, 0, self.width, 2),
                              QColor(T.PRIMARY))

        # Title
        title_pen = QPen(QColor(T.FG_PRIMARY if not self.bypass else T.FG_DISABLED))
        painter.setPen(title_pen)
        font = QFont("Arial", T.FS_BODY)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(QRectF(8, 0, self.width - 16, self.HEADER_H),
                         Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                         self.spec.title)

        # Bypass indicator (rouge si actif)
        if self.bypass:
            painter.fillRect(QRectF(self.width - 8, 4, 4, self.HEADER_H - 8),
                              QColor(T.ACCENT))

        # Phase 36 : Status dot (top-right header)
        status_color = {
            "idle": T.STATE_IDLE,
            "compiling": T.STATE_COMPILING,
            "training": T.STATE_TRAINING,
            "ok": T.STATE_OK,
            "error": T.STATE_ERROR,
            "warn": T.STATE_WARN,
        }.get(self.status, T.STATE_IDLE)
        dot_size = 8
        dot_x = self.width - 16 if not self.bypass else self.width - 22
        dot_y = 8
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(status_color)))
        painter.drawEllipse(int(dot_x), dot_y, dot_size, dot_size)
        # Glow autour si state actif (non-idle)
        if self.status not in ("idle", ""):
            painter.setBrush(QBrush(QColor(status_color)))
            painter.setOpacity(0.25)
            painter.drawEllipse(int(dot_x) - 2, dot_y - 2, dot_size + 4, dot_size + 4)
            painter.setOpacity(1.0)

        # Phase 49 : progress arc top edge si node actif avec progression
        if self.status in ("compiling", "training") and self.progress > 0:
            # Barre horizontale top edge animée
            bar_w = int(self.width * min(self.progress, 1.0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(T.STATE_TRAINING)))
            painter.drawRect(QRectF(0, 0, bar_w, 3))
        elif self.status == "ok":
            # Border accent vert plus marqué (success visible)
            painter.setPen(QPen(QColor(T.STATE_OK), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            ok_path = QPainterPath()
            ok_path.addRoundedRect(QRectF(0, 0, self.width, self.height),
                                         T.R_MEDIUM, T.R_MEDIUM)
            painter.drawPath(ok_path)

        # Category microlabel
        font_micro = QFont("Arial", T.FS_MICRO)
        painter.setFont(font_micro)
        painter.setPen(QPen(QColor(T.FG_TERTIARY)))
        painter.drawText(QRectF(8, self.HEADER_H + 2, self.width - 16, 12),
                         Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight,
                         self.spec.category.upper())

        # Socket labels
        font_label = QFont("Arial", T.FS_LABEL)
        painter.setFont(font_label)
        painter.setPen(QPen(QColor(T.FG_SECONDARY)))
        for i, s in enumerate(self.spec.inputs):
            y = self.HEADER_H + self.ROW_H * i + self.ROW_H / 2 + 4
            painter.drawText(QRectF(10, y - 8, self.width / 2, 12),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                             s.label)
        for i, s in enumerate(self.spec.outputs):
            y = self.HEADER_H + self.ROW_H * i + self.ROW_H / 2 + 4
            painter.drawText(QRectF(self.width / 2, y - 8, self.width / 2 - 10, 12),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                             s.label)

        # Params summary (mono)
        n_sockets = max(len(self.input_sockets), len(self.output_sockets))
        param_y0 = self.HEADER_H + n_sockets * self.ROW_H + 2
        font_mono = QFont("SF Mono", T.FS_MICRO)
        painter.setFont(font_mono)
        painter.setPen(QPen(QColor(T.FG_TERTIARY)))
        # Phase 57 : skip params hidden by embedded widget (prompt for text_input,
        # all params for audio_player which has full Play+Download row)
        _hide_params = set()
        if self.spec.name == "text_input":
            _hide_params = {"prompt"}
        elif self.spec.name == "audio_player":
            _hide_params = {p.name for p in self.spec.params}
        for i, p in enumerate(self.spec.params):
            if p.name in _hide_params:
                continue
            v = self.params.get(p.name, p.default)
            txt = f"{p.name} = {v}"
            painter.drawText(QRectF(8, param_y0 + i * self.PARAM_H, self.width - 16, self.PARAM_H),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                             txt)

        # Thumbnail (Phase 11) — affichage compact preview
        if self.spec.has_thumbnail:
            n_sockets = max(len(self.input_sockets), len(self.output_sockets))
            thumb_y = self.HEADER_H + n_sockets * self.ROW_H + len(self.spec.params) * self.PARAM_H + 4
            thumb_rect = QRectF(8, thumb_y, self.width - 16, 50)
            self._paint_thumbnail(painter, thumb_rect)

        # Error message bar
        if self.error:
            err_y = self.height - 14
            painter.fillRect(QRectF(0, err_y, self.width, 14), QColor(T.ACCENT_DIM))
            painter.setPen(QPen(QColor(T.ACCENT)))
            painter.drawText(QRectF(8, err_y, self.width - 16, 14),
                             Qt.AlignmentFlag.AlignVCenter,
                             f"⚠ {self.error}")

        # Resize handle (bottom-right corner)
        h_rect = QRectF(self.width - self.HANDLE_SIZE, self.height,
                          self.HANDLE_SIZE, self.HANDLE_SIZE)
        painter.fillRect(h_rect, QColor(T.BORDER_DEFAULT))

    def _paint_thumbnail(self, painter, rect):
        """Render mini-aperçu selon type de node."""
        # Background placeholder
        painter.setPen(QPen(QColor(T.BORDER_SUBTLE), 1))
        painter.setBrush(QBrush(QColor(T.BG_DEEP)))
        painter.drawRect(rect)

        if self.spec.name in ("load_image", "save_image"):
            self._paint_image_placeholder(painter, rect)
        elif self.spec.name == "webcam":
            self._paint_webcam_placeholder(painter, rect)
        elif self.spec.name == "microphone":
            self._paint_audio_waveform(painter, rect)
        elif self.spec.name == "observer":
            self._paint_observer_sparkline(painter, rect)
        elif self.spec.name in ("plot_scatter", "plot_line"):
            self._paint_plot_placeholder(painter, rect)
        else:
            painter.setPen(QPen(QColor(T.FG_TERTIARY)))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             f"⟨ preview {self.spec.name} ⟩")

    def _paint_image_placeholder(self, painter, rect):
        # Phase 52 : preserve aspect ratio (no width stretch)
        if self._preview_qimage is not None:
            try:
                from PySide6.QtCore import QRectF as _QR
                img = self._preview_qimage
                iw, ih = img.width(), img.height()
                rw, rh = rect.width(), rect.height()
                if iw <= 0 or ih <= 0:
                    raise ValueError("invalid image")
                scale = min(rw / iw, rh / ih)
                dw, dh = iw * scale, ih * scale
                dx = rect.x() + (rw - dw) / 2
                dy = rect.y() + (rh - dh) / 2
                target = _QR(dx, dy, dw, dh)
                # Background fond noir pour letterboxing
                painter.fillRect(rect, QColor(T.BG_DEEP))
                painter.drawImage(target, img)
                # Border 1px PRIMARY
                painter.setPen(QPen(QColor(T.PRIMARY), 1))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(target)
                return
            except Exception:
                pass
        # Damier subtil pour signaler "image" (pas de preview encore)
        from PySide6.QtCore import QRectF as _QR
        tile = 6
        c1 = QColor(T.BG_PANEL)
        c2 = QColor(T.BG_HIGHLIGHT)
        n_x = int(rect.width()) // tile
        n_y = int(rect.height()) // tile
        for i in range(n_x):
            for j in range(n_y):
                col = c1 if (i + j) % 2 == 0 else c2
                painter.fillRect(_QR(rect.x() + i * tile, rect.y() + j * tile,
                                       tile, tile), col)
        painter.setPen(QPen(QColor(T.FG_TERTIARY)))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "🖼 image")

    def _paint_webcam_placeholder(self, painter, rect):
        from PySide6.QtCore import QRectF as _QR
        painter.fillRect(rect, QColor(T.BG_OVERLAY))
        # cercle rouge "rec"
        painter.setBrush(QBrush(QColor(T.ACCENT)))
        painter.setPen(Qt.PenStyle.NoPen)
        cx = rect.x() + 14
        cy = rect.y() + rect.height() / 2
        painter.drawEllipse(QPointF(cx, cy), 4, 4)
        painter.setPen(QPen(QColor(T.FG_PRIMARY)))
        painter.drawText(_QR(rect.x() + 24, rect.y(), rect.width() - 24, rect.height()),
                          Qt.AlignmentFlag.AlignVCenter,
                          f"WEBCAM · {self.params.get('fps', 30)}fps")

    def _paint_audio_waveform(self, painter, rect):
        # Fake waveform synthétique
        from PySide6.QtGui import QPainterPath
        import math
        path = QPainterPath()
        n = 40
        for i in range(n):
            t = i / n
            x = rect.x() + 4 + (rect.width() - 8) * t
            amp = math.sin(t * math.pi * 8) * math.cos(t * math.pi * 3) * 12
            y = rect.y() + rect.height() / 2 + amp
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        painter.setPen(QPen(QColor(T.ACCENT), 1.2))
        painter.drawPath(path)

    def _paint_observer_sparkline(self, painter, rect):
        # Mini sparkline depuis _metric_history (eff_rank)
        from PySide6.QtGui import QPainterPath
        hist = getattr(self, "_metric_history", None)
        if not hist:
            painter.setPen(QPen(QColor(T.FG_TERTIARY)))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "◉ no live data")
            return
        vals = [m.get("eff_rank") for m in hist if m.get("eff_rank") is not None]
        if not vals:
            painter.setPen(QPen(QColor(T.FG_TERTIARY)))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "◉ ...")
            return
        vmin, vmax = min(vals), max(vals)
        rng = max(vmax - vmin, 1e-6)
        path = QPainterPath()
        n = len(vals)
        for i, v in enumerate(vals):
            x = rect.x() + 4 + (rect.width() - 8) * (i / max(n - 1, 1))
            y = rect.bottom() - 4 - (v - vmin) / rng * (rect.height() - 8)
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        painter.setPen(QPen(QColor(T.ACCENT), 1.4))
        painter.drawPath(path)
        # Légende last value
        painter.setPen(QPen(QColor(T.FG_TERTIARY)))
        painter.drawText(rect.adjusted(4, 0, 0, 0), Qt.AlignmentFlag.AlignTop,
                         f"eff_rank={vals[-1]:.1f}")

    def _paint_plot_placeholder(self, painter, rect):
        # Mini axes + dots aléatoires fixes (déterministe par id)
        from PySide6.QtCore import QRectF as _QR
        painter.setPen(QPen(QColor(T.BORDER_DEFAULT), 1))
        painter.drawLine(int(rect.x() + 8), int(rect.bottom() - 4),
                          int(rect.right() - 4), int(rect.bottom() - 4))
        painter.drawLine(int(rect.x() + 8), int(rect.y() + 4),
                          int(rect.x() + 8), int(rect.bottom() - 4))
        import random
        rng = random.Random(self.id)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(T.ACCENT)))
        for _ in range(15):
            px = rect.x() + 10 + rng.random() * (rect.width() - 14)
            py = rect.y() + 6 + rng.random() * (rect.height() - 12)
            painter.drawEllipse(QPointF(px, py), 1.5, 1.5)

    # ------------------------------------------------------------
    # Interaction : hover, double-click toggle bypass, resize
    # ------------------------------------------------------------
    def hoverEnterEvent(self, event):
        self._hover = True
        self.update()

    def hoverLeaveEvent(self, event):
        self._hover = False
        self.update()

    def mouseDoubleClickEvent(self, event):
        # Bypass toggle au double-clic header band
        if event.pos().y() < self.HEADER_H:
            self.bypass = not self.bypass
            self.update()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            # Repaint connections liées
            for sock in self.input_sockets + self.output_sockets:
                for conn in sock.connections:
                    conn.update_path()
        return super().itemChange(change, value)

    def set_param(self, name: str, value):
        if name in self.params:
            self.params[name] = value
            # Phase 57 : sync embedded text_input widget
            if (self.spec.name == "text_input" and name == "prompt"
                    and self._text_input_widget is not None):
                if self._text_input_widget.toPlainText() != str(value):
                    self._text_input_widget.blockSignals(True)
                    self._text_input_widget.setPlainText(str(value))
                    self._text_input_widget.blockSignals(False)
            self.update()

    def set_error(self, msg: str | None):
        self.error = msg
        if msg:
            self.status = "error"
        self.update()
        self.setToolTip(self._build_tooltip())

    def set_status(self, status: str):
        """Phase 36 : update status (idle|compiling|training|ok|error|warn)."""
        if status not in ("idle", "compiling", "training", "ok", "error", "warn"):
            return
        self.status = status
        if status != "error":
            # Clear error si on passe à autre chose
            pass
        self.update()

    def set_progress(self, value: float):
        """Phase 49 : update progress 0..1 (visible si status compiling/training)."""
        self.progress = max(0.0, min(1.0, float(value)))
        self.update()

    def set_preview_image(self, tensor):
        """Phase 51 : tensor (C, H, W) → QImage preview thumbnail."""
        from PySide6.QtGui import QImage
        try:
            if tensor is None:
                self._preview_image_tensor = None
                self._preview_qimage = None
                self.update()
                return
            self._preview_image_tensor = tensor
            arr = tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            import numpy as np
            arr = (arr * 255).astype(np.uint8)
            if arr.shape[2] == 4:
                arr = arr[..., :3]
            h, w, c = arr.shape
            bytes_per_line = c * w
            qimg = QImage(arr.tobytes(), w, h, bytes_per_line,
                            QImage.Format.Format_RGB888).copy()
            self._preview_qimage = qimg
            self.update()
        except Exception:
            self._preview_qimage = None

    # ------------------------------------------------------------
    # Phase 56f/57 : embedded widgets (audio player ▶/⬇, text_input editor)
    # ------------------------------------------------------------
    def _setup_audio_player_widget(self):
        """Embed Play (full-width) + Download buttons directly in node body."""
        try:
            host = QWidget()
            host.setStyleSheet(
                f"QWidget {{ background: {T.BG_DEEP}; border-radius: 8px; }}")
            lay = QVBoxLayout(host)
            lay.setContentsMargins(8, 6, 8, 6)
            lay.setSpacing(6)
            self._audio_play_btn = QPushButton("▶  Play Audio")
            self._audio_play_btn.setMinimumHeight(34)
            self._audio_play_btn.setStyleSheet(
                f"QPushButton {{ background-color: {T.SUCCESS}; color: #0A0A0A; "
                f"border: none; border-radius: 6px; padding: 6px 12px; "
                f"font-weight: 700; font-size: 12px; }}"
                f"QPushButton:hover {{ background-color: #00CC6A; }}")
            self._audio_play_btn.clicked.connect(self._on_node_audio_play)
            lay.addWidget(self._audio_play_btn)
            self._audio_dl_btn = QPushButton("⬇  Download .wav")
            self._audio_dl_btn.setMinimumHeight(28)
            self._audio_dl_btn.setStyleSheet(
                f"QPushButton {{ background-color: {T.BG_PANEL_RAISED}; "
                f"color: {T.FG_PRIMARY}; border: 1px solid {T.BORDER_DEFAULT}; "
                f"border-radius: 6px; padding: 4px 10px; font-size: 11px; }}"
                f"QPushButton:hover {{ background-color: {T.BG_OVERLAY}; }}")
            self._audio_dl_btn.clicked.connect(self._on_node_audio_download)
            lay.addWidget(self._audio_dl_btn)
            self._embed_host = host
            self._embed_proxy = QGraphicsProxyWidget(self)
            self._embed_proxy.setWidget(host)
            self._reposition_embed()
        except Exception:
            self._embed_proxy = None

    def _setup_text_input_widget(self):
        """Embed QPlainTextEdit directly in text_input node for typing."""
        try:
            host = QWidget()
            host.setStyleSheet(
                f"QWidget {{ background: {T.BG_DEEP}; border-radius: 8px; }}")
            lay = QVBoxLayout(host)
            lay.setContentsMargins(8, 4, 8, 6)
            lay.setSpacing(4)
            self._text_input_widget = QPlainTextEdit()
            self._text_input_widget.setPlainText(
                str(self.params.get("prompt", "")))
            self._text_input_widget.setPlaceholderText(
                "Tape ton prompt ici…")
            self._text_input_widget.setStyleSheet(
                f"QPlainTextEdit {{ background-color: {T.BG_PANEL}; "
                f"color: {T.FG_PRIMARY}; border: 1px solid {T.PRIMARY}; "
                f"border-radius: 6px; padding: 6px; font-size: 12px; "
                f"font-family: 'Menlo', monospace; }}"
                f"QPlainTextEdit:focus {{ border: 1px solid {T.SUCCESS}; }}")
            self._text_input_widget.setMinimumHeight(70)
            # Wire to set_param so compile picks up the value
            self._text_input_widget.textChanged.connect(
                self._on_text_input_changed)
            lay.addWidget(self._text_input_widget)
            self._embed_host = host
            self._embed_proxy = QGraphicsProxyWidget(self)
            self._embed_proxy.setWidget(host)
            self._reposition_embed()
        except Exception:
            self._embed_proxy = None

    def _on_text_input_changed(self):
        if self._text_input_widget is None:
            return
        txt = self._text_input_widget.toPlainText()
        self.params["prompt"] = txt
        self.update()

    def _reposition_embed(self):
        """Phase 57 : responsive sizing — widget fills node width."""
        if self._embed_proxy is None or self._embed_host is None:
            return
        n_sockets = max(len(self.input_sockets), len(self.output_sockets))
        socket_h = max(n_sockets, 1) * self.ROW_H
        param_h = len(self.spec.params) * self.PARAM_H
        # Position : after header + sockets + param labels
        y = self.HEADER_H + socket_h + param_h + 4
        # Width : full node width minus margins
        inner_w = max(120, self.width - 12)
        self._embed_host.setFixedWidth(inner_w)
        # Height : let widget compute (already minimum set)
        self._embed_proxy.setPos(6, y)

    def _resolve_audio_path(self) -> str | None:
        """Find audio file: disk cache <id>.wav or most recent in cache dir."""
        from pathlib import Path
        cache_dir = Path.home() / ".ark" / "audio_player"
        # Match this node
        candidate = cache_dir / f"{self.id}.wav"
        if candidate.exists():
            return str(candidate)
        # Recent fallback
        if cache_dir.exists():
            wavs = sorted(cache_dir.glob("*.wav"),
                              key=lambda p: p.stat().st_mtime, reverse=True)
            if wavs:
                return str(wavs[0])
        return None

    def _on_node_audio_play(self):
        from pathlib import Path
        import subprocess, platform
        if self._audio_is_playing:
            if self._audio_proc is not None:
                try:
                    self._audio_proc.terminate()
                except Exception:
                    pass
                self._audio_proc = None
            self._audio_is_playing = False
            if self._audio_play_btn is not None:
                self._audio_play_btn.setText("▶ Play")
            return
        path = self._resolve_audio_path()
        if not path or not Path(path).exists():
            if self._audio_play_btn is not None:
                self._audio_play_btn.setText("no audio")
                from PySide6.QtCore import QTimer
                QTimer.singleShot(1500, lambda: self._audio_play_btn.setText("▶ Play"))
            return
        try:
            sysname = platform.system()
            if sysname == "Darwin":
                cmd = ["afplay", path]
            elif sysname == "Linux":
                cmd = ["aplay", "-q", path]
            else:
                cmd = ["powershell", "-c",
                          f"(New-Object Media.SoundPlayer '{path}').PlaySync();"]
            self._audio_proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._audio_is_playing = True
            if self._audio_play_btn is not None:
                self._audio_play_btn.setText("⏸ Stop")
            # Poll
            from PySide6.QtCore import QTimer
            self._audio_poll_timer = QTimer()
            self._audio_poll_timer.timeout.connect(self._poll_node_audio)
            self._audio_poll_timer.start(200)
        except Exception:
            pass

    def _poll_node_audio(self):
        if self._audio_proc is None or self._audio_proc.poll() is not None:
            self._audio_proc = None
            self._audio_is_playing = False
            if self._audio_play_btn is not None:
                self._audio_play_btn.setText("▶ Play")
            if self._audio_poll_timer is not None:
                self._audio_poll_timer.stop()

    def _on_node_audio_download(self):
        from pathlib import Path
        path = self._resolve_audio_path()
        if not path or not Path(path).exists():
            return
        try:
            dest, _ = QFileDialog.getSaveFileName(
                None, "Save audio as…",
                str(Path.home() / "Downloads" / Path(path).name),
                "Audio (*.wav);;All files (*)")
            if dest:
                import shutil
                shutil.copy2(path, dest)
        except Exception:
            pass

    def _build_tooltip(self) -> str:
        """HTML tooltip riche : description + I/O + erreur."""
        lines = [f"<b>{self.spec.title}</b>",
                  f"<span style='color:#888;font-size:10px'>{self.spec.category} · {self.spec.name}</span>"]
        if self.spec.description:
            lines.append(f"<br><i>{self.spec.description}</i>")
        if self.spec.inputs:
            ins = ", ".join(f"{i.label}:{i.socket_type}" for i in self.spec.inputs)
            lines.append(f"<br><b>In</b> · {ins}")
        if self.spec.outputs:
            outs = ", ".join(f"{o.label}:{o.socket_type}" for o in self.spec.outputs)
            lines.append(f"<b>Out</b> · {outs}")
        if self.error:
            lines.append(f"<br><span style='color:#E11D2A'><b>Erreur</b> · {self.error}</span>")
        lines.append("<br><span style='color:#888;font-size:10px'>Double-clic : voir glossaire</span>")
        return "<br>".join(lines)

    # ------------------------------------------------------------
    # Multi-input dynamique (vague residual_add, concat)
    # ------------------------------------------------------------
    def add_multi_input_slot(self):
        """Ajoute un nouveau slot d'entrée si tous les existants sont connectés.

        Type pris depuis le 1er socket (ou 'any' si vide).
        """
        if not self.spec.is_multi_input:
            return
        # Ne crée un nouveau que si tous les existants sont connectés
        unconnected = [s for s in self.input_sockets if not s.connections]
        if unconnected:
            return
        # Type ref
        sock_type = self.input_sockets[0].socket_type if self.input_sockets else "any"
        idx = len(self.input_sockets)
        label = chr(ord('a') + idx) if idx < 26 else f"in{idx}"
        new_sock = SocketItem(self, label, sock_type, is_input=True, index=idx)
        self.input_sockets.append(new_sock)
        # Update spec.inputs aussi (pour rendu)
        self.spec.inputs.append(SocketSpec(label, sock_type))
        self._layout_sockets()
        self.prepareGeometryChange()
        self.update()

    # ------------------------------------------------------------
    # Resize handle (corner BR)
    # ------------------------------------------------------------
    def _in_resize_handle(self, local_pos: QPointF) -> bool:
        return (local_pos.x() >= self.width - self.HANDLE_SIZE
                and local_pos.y() >= self.height - 2
                and local_pos.y() <= self.height + self.HANDLE_SIZE)

    def mousePressEvent(self, event):
        if self._in_resize_handle(event.pos()):
            self._resizing = True
            self._resize_start_w = self.width
            self._resize_start_h_extra = self.user_height_extra
            self._resize_start_pos = event.scenePos()
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._resizing:
            dx = event.scenePos().x() - self._resize_start_pos.x()
            dy = event.scenePos().y() - self._resize_start_pos.y()
            new_w = max(140, int(self._resize_start_w + dx))
            new_h_extra = max(0, int(self._resize_start_h_extra + dy))
            changed = (new_w != self.width
                          or new_h_extra != self.user_height_extra)
            if changed:
                self.prepareGeometryChange()
                self.width = new_w
                self.user_height_extra = new_h_extra
                self._layout_sockets()
                # Phase 57 : reposition embedded widgets (audio/text_input)
                self._reposition_embed()
                # Update connections liées (output socket position changes)
                for sock in self.output_sockets + self.input_sockets:
                    for conn in sock.connections:
                        conn.update_path()
                self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._resizing:
            self._resizing = False
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def hoverMoveEvent(self, event):
        if self._in_resize_handle(event.pos()):
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        else:
            self.unsetCursor()
        super().hoverMoveEvent(event)
