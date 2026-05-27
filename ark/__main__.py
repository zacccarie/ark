"""ARK entry point — python -m ark."""
from __future__ import annotations
import sys

from ark.app import create_app
from ark.ui.main_window import MainWindow


def main():
    app = create_app(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
