"""Terminal / log panel avec sous-onglets."""
from __future__ import annotations
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QPlainTextEdit

from ark.ui import design_tokens as T


class TerminalPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.tabs = QTabWidget(self)
        self.tabs.setDocumentMode(True)
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        v.addWidget(self.tabs)

        self.log = QPlainTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setStyleSheet(f"font-family: {T.FONT_MONO_FAMILY}; "
                                  f"font-size: {T.FS_MICRO}px; "
                                  f"background-color: {T.BG_DEEP};")
        self.tabs.addTab(self.log, "Log")

        self.code = QPlainTextEdit(self)
        self.code.setStyleSheet(f"font-family: {T.FONT_MONO_FAMILY}; "
                                   f"font-size: {T.FS_BODY}px; "
                                   f"background-color: {T.BG_DEEP};")
        self.code.setPlainText("# Python REPL — ⌘+Enter pour exécuter\n")
        self.tabs.addTab(self.code, "Python")

        self.metrics = QPlainTextEdit(self)
        self.metrics.setReadOnly(True)
        self.metrics.setStyleSheet(f"font-family: {T.FONT_MONO_FAMILY}; "
                                       f"font-size: {T.FS_MICRO}px; "
                                       f"background-color: {T.BG_DEEP};")
        self.tabs.addTab(self.metrics, "Metrics")

    def append_log(self, line: str):
        self.log.appendPlainText(line)

    def set_metrics(self, text: str):
        self.metrics.setPlainText(text)
