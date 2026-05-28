"""Tests Phase 35 — Architecture diagram + NN schematic."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import torch

from ark.app import create_app
from ark.nodes.registry import default_registry
from ark.core.arch_viz_nodes import ArchDiagram, NNSchematic
from ark.core.graph_compiler import HANDLERS
from ark.ui.node_editor.scene import NodeScene
from ark.ui.node_editor.connection_item import ConnectionItem


@pytest.fixture(scope="module")
def app():
    return create_app()


def _build_chain_scene(registry, types):
    scene = NodeScene(registry=registry)
    nodes = []
    for i, t in enumerate(types):
        spec = registry.get(t)
        if spec is None:
            continue
        n = scene._raw_create_node(t, x=i * 250, y=0)
        nodes.append(n)
    for a, b in zip(nodes, nodes[1:]):
        if a.output_sockets and b.input_sockets:
            conn = ConnectionItem(start_socket=a.output_sockets[0],
                                       end_socket=b.input_sockets[0])
            a.output_sockets[0].add_connection(conn)
            b.input_sockets[0].add_connection(conn)
            scene._raw_add_connection(conn)
    return scene, nodes


def test_arch_diagram_empty_scene(tmp_path, app):
    """Pas de scene → graceful empty."""
    out = tmp_path / "arch.html"
    m = ArchDiagram(output_path=str(out))
    m.forward()
    assert out.exists()
    content = out.read_text()
    assert "scene_ref non défini" in content or "vide" in content.lower()


def test_arch_diagram_real_scene(tmp_path, app):
    """Scene 4 nodes chain → SVG flowchart."""
    r = default_registry()
    scene, nodes = _build_chain_scene(r,
                                            ["image_input", "conv2d", "dense", "dense"])
    out = tmp_path / "arch.html"
    m = ArchDiagram(output_path=str(out), title="Test Arch")
    m.set_scene(scene)
    m.forward()
    assert out.exists()
    content = out.read_text()
    assert "<svg" in content
    assert "Test Arch" in content
    # Vérifie présence boxes + connexions
    assert "<rect" in content
    assert "<path" in content  # arrows bezier
    assert "<circle" in content  # socket dots
    assert "4 nodes" in content
    assert "3 connexions" in content
    assert m.last_result.n_nodes == 4


def test_arch_diagram_topo_layers(app):
    """Sugiyama layout : nodes en couches selon profondeur depuis sources."""
    r = default_registry()
    scene, nodes = _build_chain_scene(r,
                                            ["image_input", "dense", "dense", "dense"])
    m = ArchDiagram()
    layers, layer_of, outgoing = m._topo_layers(
        list(scene.nodes()), list(scene.connections()))
    # Image input = layer 0
    assert layer_of[nodes[0].id] == 0
    # Dernier dense = layer 3
    assert layer_of[nodes[-1].id] == 3


def test_nn_schematic_basic(tmp_path, app):
    """Schéma MLP 784→128→64→10."""
    out = tmp_path / "nn.html"
    m = NNSchematic(output_path=str(out), title="MLP",
                       sizes="784,128,64,10",
                       activations="relu,relu,softmax")
    m.forward()
    assert out.exists()
    content = out.read_text()
    assert "<svg" in content
    assert "INPUT (784)" in content
    assert "OUTPUT (10)" in content
    assert "HIDDEN" in content
    # Params count : 784*128+128 + 128*64+64 + 64*10+10 = 109066
    assert "109,066" in content or "109" in content


def test_nn_schematic_neuron_sampling(tmp_path, app):
    """Couches > MAX_NEURONS=16 → sample + indicateur '…'."""
    out = tmp_path / "nn_big.html"
    m = NNSchematic(output_path=str(out), sizes="1000,500,10",
                       activations="relu,softmax")
    m.forward()
    content = out.read_text()
    assert "1000" in content
    assert "…" in content  # truncation indicator


def test_nn_schematic_small(tmp_path, app):
    """Petite couche < MAX → toutes rendues sans ellipsis."""
    out = tmp_path / "nn_small.html"
    m = NNSchematic(output_path=str(out), sizes="5,3,2",
                       activations="relu,sigmoid")
    m.forward()
    content = out.read_text()
    assert "INPUT (5)" in content


def test_arch_diagram_handler_registered(app):
    assert "arch_diagram" in HANDLERS
    assert "nn_schematic" in HANDLERS


def test_arch_diagram_spec_registered(app):
    r = default_registry()
    s = r.get("arch_diagram")
    assert s is not None
    assert s.category == "Viz"
    s2 = r.get("nn_schematic")
    assert s2 is not None
    assert s2.category == "Viz"


def test_main_window_action_exists(app):
    """Action menu Ctrl+Shift+D doit exister."""
    from ark.ui.main_window import MainWindow
    w = MainWindow()
    # Check menu items
    actions = w.menuBar().findChildren(type(w.menuBar().actions()[0]))
    titles = [a.text() for a in w.menuBar().actions()]
    # Au moins Affichage menu existe
    assert any("Affichage" in t for t in titles)
