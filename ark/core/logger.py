"""Phase 50 — Logger central ARK.

Replace print(file=sys.stderr) par logging avec levels + formatter unifié.
Niveaux : DEBUG / INFO / WARNING / ERROR / CRITICAL.
Format : [HH:MM:SS][LEVEL][module] message
"""
from __future__ import annotations
import logging
import os
import sys
from pathlib import Path


_DEFAULT_LEVEL = os.environ.get("ARK_LOG_LEVEL", "INFO").upper()
_LOG_FILE = Path.home() / ".ark" / "ark.log"


def setup_logging(level: str = _DEFAULT_LEVEL,
                    file_logging: bool = True) -> None:
    """Configure root logger. Appelé 1× au démarrage ARK."""
    root = logging.getLogger("ark")
    if root.handlers:
        return  # déjà configuré
    root.setLevel(getattr(logging, level, logging.INFO))

    fmt = logging.Formatter(
        "[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        datefmt="%H:%M:%S")

    # Console
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    # File rotation
    if file_logging:
        try:
            _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
            fh.setFormatter(fmt)
            root.addHandler(fh)
        except Exception:
            pass


def get_logger(name: str) -> logging.Logger:
    """Helper : ark.core.foo → 'ark.foo'."""
    base = name.replace("ark.core.", "ark.").replace("ark.ui.", "ark.ui.")
    return logging.getLogger(base)
