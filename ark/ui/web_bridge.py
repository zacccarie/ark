"""Phase 61 — QWebChannel bridge for React WebEngine panels.

Expose Python methods + Qt signals to React via QWebChannel. React side
loads qrc:///qtwebchannel/qwebchannel.js, instantiates QWebChannel, gets
'bridge' object with these slots/signals.
"""
from __future__ import annotations
import base64
from io import BytesIO
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot, Qt


class ArkBridge(QObject):
    """Façade Python → JS. All Slots() callable from JS side.

    Signals JS-side subscribed via `bridge.signalName.connect(cb)`.
    """

    # Signals
    selection_changed = Signal(object)   # node data dict or null
    log_appended = Signal(str)            # one log line
    template_loaded = Signal(str)         # template name
    compile_progress = Signal(float, str)  # 0..1, state

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._mw = main_window  # weakref-ish, set lazy

    # ============================================================
    # Palette
    # ============================================================

    @Slot(result=list)
    def list_categories(self):
        from ark.nodes.registry import default_registry
        r = default_registry()
        cats = sorted({s.category for s in r.all()})
        return cats

    @Slot(str, result=list)
    def list_nodes_in_category(self, category: str):
        from ark.nodes.registry import default_registry
        r = default_registry()
        nodes = [s for s in r.all() if s.category == category]
        nodes.sort(key=lambda s: s.title)
        return [{"name": s.name, "title": s.title} for s in nodes]

    # ============================================================
    # Inspector
    # ============================================================

    @Slot(result=object)
    def get_selected_node(self):
        if self._mw is None or not hasattr(self._mw, "scene"):
            return None
        sel = [it for it in self._mw.scene.selectedItems()
                if hasattr(it, "spec")]
        if not sel:
            return None
        return self._serialize_node(sel[0])

    def _serialize_node(self, node) -> dict:
        spec = node.spec
        out = {
            "id": node.id,
            "type": spec.name,
            "title": spec.title,
            "category": spec.category,
            "description": spec.description,
            "role": getattr(spec, "role", ""),
            "function": getattr(spec, "function", ""),
            "params": [
                {
                    "name": p.name,
                    "kind": p.kind,
                    "default": p.default,
                    "options": list(p.options) if p.options else [],
                    "min": p.min,
                    "max": p.max,
                }
                for p in spec.params
            ],
            "values": dict(node.params),
        }
        # Live outputs
        mod = None
        if hasattr(self._mw, "_compiled") and self._mw._compiled is not None:
            md = getattr(self._mw._compiled.model, "modules_dict", {})
            try:
                mod = md.get(node.id)
            except Exception:
                mod = None
        if mod is not None:
            for attr in ("last_text", "last_value", "last_generated",
                          "last_audio_path"):
                v = getattr(mod, attr, None)
                if v not in (None, ""):
                    out[attr] = v
        # Preview image base64 if present
        qimg = getattr(node, "_preview_qimage", None)
        if qimg is not None:
            try:
                from PySide6.QtCore import QBuffer, QByteArray, QIODevice
                buf = QBuffer()
                buf.open(QIODevice.OpenModeFlag.WriteOnly)
                qimg.save(buf, "PNG")
                b64 = base64.b64encode(bytes(buf.data())).decode("ascii")
                out["preview_image"] = f"data:image/png;base64,{b64}"
            except Exception:
                pass
        return out

    @Slot(str)
    def spawn_node(self, node_type: str):
        """Phase 62 : palette click → create node at viewport center."""
        if self._mw is None or not hasattr(self._mw, "scene"):
            return
        try:
            view = getattr(self._mw, "view", None)
            if view is not None:
                from PySide6.QtCore import QPointF
                center = view.mapToScene(view.viewport().rect().center())
                x, y = center.x(), center.y()
            else:
                x, y = 200.0, 200.0
            self._mw.scene._raw_create_node(node_type, float(x), float(y))
        except Exception as e:
            self.emit_log(f"[spawn_node error] {e}")

    @Slot(str, str, "QVariant")
    def set_node_param(self, node_id: str, name: str, value: Any):
        if self._mw is None or not hasattr(self._mw, "scene"):
            return
        for it in self._mw.scene.items():
            if getattr(it, "id", None) == node_id and hasattr(it, "set_param"):
                it.set_param(name, value)
                return

    # ============================================================
    # Toolbar actions
    # ============================================================

    @Slot(result=list)
    def list_templates(self):
        from ark.core.templates import TEMPLATES_LIST
        return [{"name": n, "label": l} for n, l in TEMPLATES_LIST]

    @Slot(str)
    def load_template(self, name: str):
        if self._mw and hasattr(self._mw, "_load_template"):
            self._mw._load_template(name)
            self.template_loaded.emit(name)

    @Slot(int)
    def compile(self, batch: int = 1):
        if self._mw and hasattr(self._mw, "_on_compile"):
            self._mw._on_compile()

    @Slot(int)
    def train(self, batch: int = 1):
        if self._mw and hasattr(self._mw, "_on_train"):
            self._mw._on_train()

    @Slot()
    def stop(self):
        if self._mw and hasattr(self._mw, "_on_stop"):
            try:
                self._mw._on_stop()
            except Exception:
                pass

    @Slot()
    def open_extensions(self):
        # Phase 64 : Extensions button → templates browser (most useful action)
        if self._mw is None:
            return
        for attr in ("_open_extensions", "_on_browse_templates"):
            fn = getattr(self._mw, attr, None)
            if callable(fn):
                fn()
                return

    @Slot()
    def open_settings(self):
        if self._mw is None:
            return
        for attr in ("_open_settings", "_on_open_shortcuts",
                          "_on_open_help_page"):
            fn = getattr(self._mw, attr, None)
            if callable(fn):
                fn()
                return

    @Slot()
    def toggle_inspector(self):
        if self._mw and hasattr(self._mw, "_toggle_inspector"):
            self._mw._toggle_inspector()

    # ============================================================
    # View controls (zoom / fit / map / snap)
    # ============================================================

    @Slot(float)
    def set_zoom(self, factor: float):
        if self._mw is None or not hasattr(self._mw, "view"):
            return
        v = self._mw.view
        try:
            v.resetTransform()
            v.scale(factor, factor)
        except Exception:
            pass

    @Slot()
    def fit_view(self):
        if self._mw is None or not hasattr(self._mw, "view"):
            return
        v = self._mw.view
        try:
            rect = self._mw.scene.itemsBoundingRect()
            if rect.isEmpty():
                return
            v.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        except Exception:
            pass

    @Slot(bool)
    def toggle_minimap(self, on: bool):
        if self._mw and hasattr(self._mw, "_minimap"):
            try:
                self._mw._minimap.setVisible(on)
            except Exception:
                pass

    @Slot(bool)
    def toggle_snap(self, on: bool):
        if self._mw and hasattr(self._mw, "scene"):
            try:
                self._mw.scene.snap_to_grid = on
            except Exception:
                pass

    # ============================================================
    # Audio
    # ============================================================

    @Slot(str)
    def play_audio(self, path: str):
        if not path or not Path(path).exists():
            return
        import platform, subprocess
        sysname = platform.system()
        if sysname == "Darwin":
            subprocess.Popen(["afplay", path],
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
        elif sysname == "Linux":
            subprocess.Popen(["aplay", "-q", path],
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
        else:
            subprocess.Popen(
                ["powershell", "-c",
                 f"(New-Object Media.SoundPlayer '{path}').PlaySync();"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @Slot(str)
    def download_audio(self, src: str):
        if not src or not Path(src).exists():
            return
        from PySide6.QtWidgets import QFileDialog
        dest, _ = QFileDialog.getSaveFileName(
            self._mw, "Save audio as…",
            str(Path.home() / "Downloads" / Path(src).name),
            "Audio (*.wav);;All files (*)")
        if dest:
            import shutil
            shutil.copy2(src, dest)

    # ============================================================
    # Lifecycle hooks (called by MainWindow)
    # ============================================================

    def emit_selection(self, node):
        try:
            data = self._serialize_node(node) if node else None
            self.selection_changed.emit(data)
        except Exception:
            self.selection_changed.emit(None)

    def emit_log(self, line: str):
        self.log_appended.emit(line)
