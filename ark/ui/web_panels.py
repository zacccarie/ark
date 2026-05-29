"""Phase 61 — QWebEngineView panel wrappers.

Each panel = a QDockWidget hosting a QWebEngineView pointed at
ark/ui_web/dist/<panel>.html. QWebChannel exposes ArkBridge for
bidirectional communication.

Activation:
    export ARK_WEB=1
    ark
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QUrl, Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QDockWidget, QLabel
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebChannel import QWebChannel


_DIST_DIR = Path(__file__).parent.parent / "ui_web" / "dist"


def _panel_url(name: str) -> Optional[QUrl]:
    """Resolve file:// URL to the built panel HTML. Returns None if not built."""
    html = _DIST_DIR / f"{name}.html"
    if not html.exists():
        return None
    return QUrl.fromLocalFile(str(html.absolute()))


def is_web_ui_available() -> bool:
    return _DIST_DIR.exists() and (_DIST_DIR / "palette.html").exists()


class WebPanel(QWidget):
    """Generic QWebEngineView host for a React panel."""

    def __init__(self, panel_name: str, bridge, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        url = _panel_url(panel_name)
        if url is None:
            err = QLabel(
                f"Web UI not built.\n\n"
                f"Run:\n  cd ark/ui_web && npm install && npm run build\n\n"
                f"Then relaunch ARK.\n\nMissing: {_DIST_DIR}/{panel_name}.html",
                self)
            err.setAlignment(Qt.AlignmentFlag.AlignCenter)
            err.setStyleSheet(
                "color: #B8B8C0; background: #18181B; padding: 24px;"
                "font-family: Menlo, monospace; font-size: 11px;")
            layout.addWidget(err)
            self.view = None
            return
        self.view = QWebEngineView(self)
        layout.addWidget(self.view)
        # Channel
        self.channel = QWebChannel(self.view.page())
        self.channel.registerObject("bridge", bridge)
        self.view.page().setWebChannel(self.channel)
        self.view.load(url)


def build_web_docks(main_window, bridge) -> dict[str, QDockWidget]:
    """Create QDockWidgets containing WebPanel for each web-rendered panel.

    Returns dict {name: QDockWidget}. Caller is responsible for addDockWidget.
    """
    docks: dict[str, QDockWidget] = {}
    sizes = {
        "palette": (260, 0),       # min width, min height
        "inspector": (300, 0),
        "terminal": (0, 180),
        "architect": (320, 0),
        "compute": (320, 0),
    }
    for name, title in [
        ("palette", "Palette"),
        ("inspector", "Inspector"),
        ("terminal", "Terminal"),
        ("architect", "Architect"),
        ("compute", "Compute"),
    ]:
        dock = QDockWidget(title, main_window)
        dock.setObjectName(f"dock_web_{name}")
        panel = WebPanel(name, bridge, dock)
        dock.setWidget(panel)
        mw, mh = sizes[name]
        if mw:
            dock.setMinimumWidth(mw)
        if mh:
            dock.setMinimumHeight(mh)
        docks[name] = dock
    return docks
