"""Tests ARK Phases 21 (plugin system) + 24 (auto-layout) + 23 mini (copy/paste)."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
from pathlib import Path
import pytest

from ark.app import create_app
from ark.nodes.registry import default_registry, NodeRegistry
from ark.ui.node_editor.scene import NodeScene
from ark.ui.node_editor.connection_item import ConnectionItem
from ark.core.plugins import discover_plugins, register_plugins, example_plugin_text
from ark.core.layout import compute_layout, auto_layout_scene


@pytest.fixture(scope="module")
def app():
    return create_app()


# ============ Phase 21 Plugin system ============


def test_example_plugin_text_compiles(tmp_path):
    """Le texte d'exemple plugin doit être un Python valide."""
    src = example_plugin_text()
    p = tmp_path / "scale.py"
    p.write_text(src)
    compile(src, str(p), "exec")  # syntax check


def test_discover_plugins(tmp_path):
    p = tmp_path / "myplugin.py"
    p.write_text(example_plugin_text())
    plugins = discover_plugins(tmp_path)
    assert len(plugins) == 1
    assert plugins[0]["spec"].name == "scale"


def test_register_plugins(tmp_path, app):
    p = tmp_path / "myplugin.py"
    p.write_text(example_plugin_text())
    registry = NodeRegistry()
    handlers = {}
    n = register_plugins(registry, handlers, plugins_dir=tmp_path)
    assert n == 1
    assert registry.get("scale") is not None
    assert "scale" in handlers


def test_plugin_handler_works(tmp_path, app):
    p = tmp_path / "myplugin.py"
    p.write_text(example_plugin_text())
    registry = NodeRegistry()
    handlers = {}
    register_plugins(registry, handlers, plugins_dir=tmp_path)
    spec = registry.get("scale")
    # Fake node compatible avec handler signature
    class _FakeNode:
        def __init__(self):
            self.params = {"factor": 3.0}
    mod, out_shape = handlers["scale"](_FakeNode(), [(16,)])
    assert out_shape == (16,)
    import torch
    x = torch.ones(2, 16)
    y = mod(x)
    assert torch.allclose(y, torch.ones(2, 16) * 3.0)


def test_bad_plugin_silently_skipped(tmp_path, capsys):
    bad = tmp_path / "broken.py"
    bad.write_text("raise SyntaxError('intentional')")
    plugins = discover_plugins(tmp_path)
    assert plugins == []  # silently skipped after traceback print


def test_plugin_underscore_prefix_ignored(tmp_path):
    p = tmp_path / "_helper.py"
    p.write_text(example_plugin_text())
    plugins = discover_plugins(tmp_path)
    assert plugins == []  # _ prefix → ignored


# ============ Phase 24 Auto-layout ============


def test_compute_layout_chain(app):
    s = NodeScene(registry=default_registry())
    a = s.create_node("image_input", x=0, y=0)
    b = s.create_node("dense", x=0, y=0)
    c = s.create_node("dense", x=0, y=0)
    for src, dst in [(a, b), (b, c)]:
        conn = ConnectionItem(start_socket=src.output_sockets[0],
                                end_socket=dst.input_sockets[0])
        src.output_sockets[0].add_connection(conn)
        dst.input_sockets[0].add_connection(conn)
        s.add_connection(conn)
    pos = compute_layout(s, h_spacing=200, x_origin=0)
    # Chain : x croissant strict
    xs = [pos[a.id][0], pos[b.id][0], pos[c.id][0]]
    assert xs[0] < xs[1] < xs[2]
    assert xs[1] - xs[0] == 200
    assert xs[2] - xs[1] == 200


def test_auto_layout_scene_applies_positions(app):
    s = NodeScene(registry=default_registry())
    a = s.create_node("image_input")
    b = s.create_node("dense")
    conn = ConnectionItem(start_socket=a.output_sockets[0],
                            end_socket=b.input_sockets[0])
    a.output_sockets[0].add_connection(conn)
    b.input_sockets[0].add_connection(conn)
    s.add_connection(conn)
    auto_layout_scene(s, h_spacing=300)
    # a (level 0) à gauche de b (level 1)
    assert a.pos().x() < b.pos().x()
    assert (b.pos().x() - a.pos().x()) == 300


def test_compute_layout_parallel_levels(app):
    """Deux nodes au même level → stagger Y différent."""
    s = NodeScene(registry=default_registry())
    a = s.create_node("image_input")
    b = s.create_node("dense")
    c = s.create_node("dense")  # même niveau que b si connecté depuis a
    # a → b, a → c
    for dst in (b, c):
        conn = ConnectionItem(start_socket=a.output_sockets[0],
                                end_socket=dst.input_sockets[0])
        a.output_sockets[0].add_connection(conn)
        dst.input_sockets[0].add_connection(conn)
        s.add_connection(conn)
    pos = compute_layout(s, v_spacing=100, y_origin=200)
    # b et c au même x
    assert pos[b.id][0] == pos[c.id][0]
    # mais y différents (stagger)
    assert pos[b.id][1] != pos[c.id][1]


def test_compute_layout_empty():
    from ark.nodes.registry import default_registry as dr
    s = NodeScene(registry=dr())
    pos = compute_layout(s)
    assert pos == {}
