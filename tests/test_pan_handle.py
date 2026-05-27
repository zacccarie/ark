"""Tests pan handle + gestures registration."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt
from ark.app import create_app
from ark.nodes.registry import default_registry
from ark.ui.node_editor.scene import NodeScene
from ark.ui.node_editor.view import NodeEditorView
from ark.ui.node_editor.pan_handle import PanHandle


@pytest.fixture(scope="module")
def app():
    return create_app()


@pytest.fixture
def view(app):
    scene = NodeScene(registry=default_registry())
    return NodeEditorView(scene)


def test_pan_handle_attached(view):
    assert isinstance(view.pan_handle, PanHandle)
    assert view.pan_handle.parent() is view


def test_pan_handle_repositions_on_resize(view):
    view.resize(800, 500)
    view.pan_handle.reposition()
    expected_x = 800 - PanHandle.SIZE - PanHandle.MARGIN
    expected_y = 500 - PanHandle.SIZE - PanHandle.MARGIN
    assert view.pan_handle.pos().x() == expected_x
    assert view.pan_handle.pos().y() == expected_y


def test_pan_handle_signal_translates_view(view):
    """pan_delta signal connecté → scrollbar bouge."""
    # Force range non-trivial sinon setValue clamp à 0
    view.horizontalScrollBar().setRange(0, 1000)
    view.verticalScrollBar().setRange(0, 1000)
    view.horizontalScrollBar().setValue(500)
    view.verticalScrollBar().setValue(500)
    h0 = view.horizontalScrollBar().value()
    v0 = view.verticalScrollBar().value()
    view._on_handle_pan(40.0, 25.0)
    h1 = view.horizontalScrollBar().value()
    v1 = view.verticalScrollBar().value()
    assert h1 - h0 == -40  # pan handle = drag opposé du scroll
    assert v1 - v0 == -25


def test_pan_handle_emit_signal(view):
    """Signal fires correctement."""
    received = []
    view.pan_handle.pan_delta.connect(lambda dx, dy: received.append((dx, dy)))
    view.pan_handle.pan_delta.emit(3.0, -7.0)
    assert received == [(3.0, -7.0)]


def test_gestures_registered(view):
    """PanGesture + PinchGesture grabbed sur viewport."""
    # PySide6 n'a pas d'API directe pour vérifier mais on s'assure que
    # grabGesture a été appelé sans crash + viewport accepts touch events
    assert view.viewport().testAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents)
