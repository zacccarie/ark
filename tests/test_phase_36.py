"""Tests Phase 36 — Design refresh + state indicators + animations."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QLabel, QPushButton

from ark.app import create_app
from ark.ui import design_tokens as T
from ark.ui.animations import (
    fade_in, fade_out, pulse_opacity, animate_property,
    animate_item_float, AnimatedFloat,
)
from ark.ui.node_editor.node_item import NodeItem
from ark.nodes.registry import default_registry


@pytest.fixture(scope="module")
def app():
    return create_app()


# ============ Tokens ============

def test_new_tokens_exist():
    assert hasattr(T, "ACCENT_GREEN")
    assert T.ACCENT_GREEN == "#00FF85"
    assert hasattr(T, "SUCCESS")
    assert hasattr(T, "WARN")
    assert hasattr(T, "INFO")
    assert hasattr(T, "ERROR")
    assert hasattr(T, "STATE_TRAINING")
    assert hasattr(T, "DUR_FAST")
    assert hasattr(T, "DUR_NORMAL")
    assert hasattr(T, "DUR_PULSE")


def test_radius_softer():
    """R_SMALL passe de 2 → 4 (softer)."""
    assert T.R_SMALL >= 4
    assert T.R_MEDIUM >= 6
    assert T.R_PILL == 999


def test_font_sizes_larger():
    """Sizes augmentées légèrement pour readability."""
    assert T.FS_BODY >= 12  # était 11
    assert T.FS_HERO >= 20  # était 18


def test_shadow_tokens():
    assert hasattr(T, "SHADOW_SM")
    assert hasattr(T, "SHADOW_MD")
    assert hasattr(T, "SHADOW_LG")


def test_socket_color_diversified():
    """Sockets ont couleurs distinctes (vs old grey-only)."""
    assert T.SOCKET_NUMBER != T.SOCKET_TENSOR
    assert T.SOCKET_IMAGE != T.SOCKET_TENSOR


# ============ NodeItem status ============

def test_node_status_default(app):
    r = default_registry()
    spec = r.get("dense")
    n = NodeItem(spec)
    assert n.status == "idle"


def test_node_set_status(app):
    r = default_registry()
    spec = r.get("dense")
    n = NodeItem(spec)
    for s in ("compiling", "training", "ok", "error", "warn", "idle"):
        n.set_status(s)
        assert n.status == s


def test_node_set_error_triggers_status(app):
    """set_error change auto status à 'error'."""
    r = default_registry()
    spec = r.get("dense")
    n = NodeItem(spec)
    assert n.status == "idle"
    n.set_error("Bug")
    assert n.status == "error"


def test_node_invalid_status_ignored(app):
    r = default_registry()
    n = NodeItem(r.get("dense"))
    n.set_status("idle")
    n.set_status("bogus_state")
    assert n.status == "idle"


# ============ Animations ============

def test_fade_in(app):
    w = QLabel("test")
    w.show()
    anim = fade_in(w, duration=50)
    assert anim is not None
    assert anim.duration() == 50


def test_fade_out(app):
    w = QLabel("test")
    w.show()
    anim = fade_out(w, duration=30)
    assert anim is not None


def test_pulse_opacity(app):
    w = QLabel("test")
    w.show()
    seq = pulse_opacity(w, min_op=0.3, max_op=1.0, duration=200)
    assert seq.loopCount() == -1


def test_animated_float_proxy(app):
    """AnimatedFloat anime attribut d'un objet sans QProperty natif."""
    class Item:
        def __init__(self):
            self._v = 0.0
        @property
        def v(self): return self._v
        @v.setter
        def v(self, x): self._v = x

    item = Item()
    item.v = 0.5

    proxy = AnimatedFloat(item, "v")
    assert proxy.get_value() == 0.5
    proxy.set_value(0.8)
    assert item.v == 0.8


def test_animate_item_float(app):
    """Anime un attribut d'un QGraphicsItem."""
    r = default_registry()
    n = NodeItem(r.get("dense"))
    n._hover_scale = 1.0
    anim, proxy = animate_item_float(n, "_hover_scale", 1.05,
                                            duration=50)
    assert anim is not None


# ============ Style QSS ============

def test_qss_contains_new_roles(app):
    from ark.ui.style import build_qss
    qss = build_qss()
    assert "badge-success" in qss
    assert "badge-warn" in qss
    assert "badge-error" in qss
    assert "badge-info" in qss
    assert 'role="success"' in qss or "[role=\"success\"]" in qss
    assert 'role="ghost"' in qss or "[role=\"ghost\"]" in qss


# ============ MainWindow global state ============

def test_main_window_set_global_state(app):
    from ark.ui.main_window import MainWindow
    w = MainWindow()
    w.set_global_state("training", "Training MNIST")
    assert "TRAINING" in w._state_badge.text()
    w.set_global_state("error", "Crash")
    assert "ERROR" in w._state_badge.text()
    w.set_global_state("ready")
    assert "READY" in w._state_badge.text()


def test_main_window_state_badge_role(app):
    from ark.ui.main_window import MainWindow
    w = MainWindow()
    w.set_global_state("training")
    assert w._state_badge.property("role") == "badge-info"
    w.set_global_state("error")
    assert w._state_badge.property("role") == "badge-error"
    w.set_global_state("ready")
    assert w._state_badge.property("role") == "badge-success"
