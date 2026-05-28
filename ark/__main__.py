"""ARK entry point — python -m ark / ark."""
from __future__ import annotations
import sys
import traceback
from pathlib import Path


def _install_crash_handler():
    """Phase 50c : capture toutes les exceptions non gérées dans ~/.ark/crash.log."""
    crash_log = Path.home() / ".ark" / "crash.log"
    crash_log.parent.mkdir(parents=True, exist_ok=True)

    def _hook(exc_type, exc_value, exc_tb):
        with crash_log.open("a") as f:
            f.write("=== UNHANDLED ===\n")
            traceback.print_exception(exc_type, exc_value, exc_tb, file=f)
            f.write("\n")
        # Print stderr aussi
        traceback.print_exception(exc_type, exc_value, exc_tb)

    sys.excepthook = _hook

    # Qt message handler
    try:
        from PySide6.QtCore import qInstallMessageHandler, QtMsgType
        def _qt_msg(mode, ctx, msg):
            if mode in (QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg):
                with crash_log.open("a") as f:
                    f.write(f"=== Qt {mode} ===\n{msg}\n\n")
        qInstallMessageHandler(_qt_msg)
    except Exception:
        pass


def main():
    _install_crash_handler()
    from ark.app import create_app
    from ark.ui.main_window import MainWindow
    app = create_app(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
