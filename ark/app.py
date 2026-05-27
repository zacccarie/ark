"""ARK QApplication setup."""
from __future__ import annotations
import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont

from ark.ui.style import build_qss
from ark.ui import design_tokens as T


def create_app(argv: list[str] | None = None) -> QApplication:
    """Crée ou récupère le singleton QApplication."""
    existing = QApplication.instance()
    if existing is not None:
        return existing
    if argv is None:
        argv = sys.argv
    QApplication.setApplicationName("ARK")
    QApplication.setOrganizationName("Zaccarie Aglas")
    app = QApplication(argv)
    f = QFont("Arial", T.FS_BODY)
    app.setFont(f)
    app.setStyleSheet(build_qss())
    return app
