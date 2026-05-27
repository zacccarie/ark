"""Tests headless ARK node editor (Phase 0 + 1)."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from ark.app import create_app
from ark.nodes.registry import default_registry
from ark.ui.node_editor.scene import NodeScene
from ark.ui.node_editor.socket_item import types_compatible
from ark.ui.node_editor.connection_item import ConnectionItem


@pytest.fixture(scope="module")
def app():
    return create_app()


@pytest.fixture
def scene(app):
    return NodeScene(registry=default_registry())


# ---- Phase 0 ----


def test_registry_specs_count(app):
    r = default_registry()
    assert len(r.all()) >= 18
    cats = r.by_category()
    assert "Neural" in cats
    assert "Input" in cats
    assert "Math" in cats


def test_create_node(scene):
    n = scene.create_node("dense", x=0, y=0)
    assert n is not None
    assert n.spec.name == "dense"
    assert len(scene.nodes()) == 1


def test_unknown_node_type_returns_none(scene):
    n = scene.create_node("__missing__")
    assert n is None


def test_remove_node(scene):
    n = scene.create_node("conv2d")
    scene.remove_node(n)
    assert len(scene.nodes()) == 0


# ---- Phase 1 ----


def test_types_compatible():
    assert types_compatible("tensor", "tensor")
    assert types_compatible("tensor", "any")
    assert types_compatible("any", "image")
    assert types_compatible("number", "tensor")
    assert not types_compatible("image", "string")
    assert not types_compatible("model", "number")


def test_connection_created(scene):
    a = scene.create_node("image_input")
    b = scene.create_node("conv2d")
    out_sock = a.output_sockets[0]
    in_sock = b.input_sockets[0]
    conn = ConnectionItem(start_socket=out_sock, end_socket=in_sock)
    out_sock.add_connection(conn)
    in_sock.add_connection(conn)
    scene.add_connection(conn)
    assert len(scene.connections()) == 1
    assert conn in out_sock.connections
    assert conn in in_sock.connections


def test_multi_input_grows(scene):
    a = scene.create_node("image_input")
    b = scene.create_node("conv2d")
    r = scene.create_node("residual_add")
    assert r.spec.is_multi_input
    n0 = len(r.input_sockets)
    # Connect both initial inputs
    for src_node, slot_idx in [(a, 0), (b, 1)]:
        c = ConnectionItem(start_socket=src_node.output_sockets[0],
                            end_socket=r.input_sockets[slot_idx])
        src_node.output_sockets[0].add_connection(c)
        r.input_sockets[slot_idx].add_connection(c)
        scene.add_connection(c)
    r.add_multi_input_slot()
    assert len(r.input_sockets) == n0 + 1
    # New slot has correct type (héritage de la 1ère socket)
    assert r.input_sockets[-1].socket_type == r.input_sockets[0].socket_type


def test_multi_input_skip_if_unconnected(scene):
    r = scene.create_node("residual_add")
    n0 = len(r.input_sockets)
    r.add_multi_input_slot()  # Aucun connecté → ne crée pas
    assert len(r.input_sockets) == n0


def test_resize_node_width(scene):
    a = scene.create_node("dense")
    a.width = 320
    a._layout_sockets()
    assert a.width == 320
    # Output socket repositionné à width
    assert a.output_sockets[0].pos().x() == 320


def test_clear_graph(scene):
    scene.create_node("dense")
    scene.create_node("conv2d")
    a = scene.create_node("image_input")
    b = scene.create_node("dense")
    c = ConnectionItem(start_socket=a.output_sockets[0], end_socket=b.input_sockets[0])
    a.output_sockets[0].add_connection(c)
    b.input_sockets[0].add_connection(c)
    scene.add_connection(c)
    assert len(scene.nodes()) == 4
    assert len(scene.connections()) == 1
    scene.clear_graph()
    assert len(scene.nodes()) == 0
    assert len(scene.connections()) == 0


def test_node_set_param(scene):
    n = scene.create_node("dense")
    n.set_param("neurons", 256)
    assert n.params["neurons"] == 256


def test_node_bypass_toggle(scene):
    n = scene.create_node("dense")
    assert not n.bypass
    n.bypass = True
    assert n.bypass


def test_node_error_state(scene):
    n = scene.create_node("dense")
    n.set_error("dim mismatch")
    assert n.error == "dim mismatch"
    n.set_error(None)
    assert n.error is None
