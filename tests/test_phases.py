"""Tests ARK Phases 1.5 + 2 + 3 + 4."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import pytest
from PySide6.QtGui import QUndoStack
from ark.app import create_app
from ark.nodes.registry import default_registry
from ark.ui.node_editor.scene import NodeScene
from ark.ui.node_editor.connection_item import ConnectionItem
from ark.core.graph_compiler import compile_scene, GraphCompileError
from ark.core.format import save_scene, load_scene


@pytest.fixture(scope="module")
def app():
    return create_app()


@pytest.fixture
def scene_with_undo(app):
    s = NodeScene(registry=default_registry())
    s.undo_stack = QUndoStack()
    return s


# ============ Phase 1.5 Undo/Redo ============


def test_undo_add_node(scene_with_undo):
    s = scene_with_undo
    n = s.create_node("dense", x=0, y=0)
    assert n is not None
    assert len(s.nodes()) == 1
    s.undo_stack.undo()
    assert len(s.nodes()) == 0
    s.undo_stack.redo()
    assert len(s.nodes()) == 1


def test_undo_remove_node_restores_connections(scene_with_undo):
    s = scene_with_undo
    a = s.create_node("image_input")
    b = s.create_node("conv2d")
    conn = ConnectionItem(start_socket=a.output_sockets[0],
                            end_socket=b.input_sockets[0])
    a.output_sockets[0].add_connection(conn)
    b.input_sockets[0].add_connection(conn)
    s.add_connection(conn)
    assert len(s.connections()) == 1
    s.remove_node(a)
    assert len(s.connections()) == 0
    s.undo_stack.undo()
    assert len(s.nodes()) == 2
    assert len(s.connections()) == 1


# ============ Phase 2 compile ============


def test_compile_simple_mlp(scene_with_undo):
    s = scene_with_undo
    a = s.create_node("image_input")
    a.params["shape"] = "[1, 28, 28]"
    b = s.create_node("dense")
    b.params["neurons"] = 64
    conn = ConnectionItem(start_socket=a.output_sockets[0],
                            end_socket=b.input_sockets[0])
    a.output_sockets[0].add_connection(conn)
    b.input_sockets[0].add_connection(conn)
    s.add_connection(conn)
    cg = compile_scene(s, num_classes=10)
    assert cg.input_shape == (1, 28, 28)
    assert cg.output_shape == (10,)
    import torch
    x = torch.randn(2, 1, 28, 28)
    y = cg.model(x)
    assert y.shape == (2, 10)


def test_compile_missing_input_raises(scene_with_undo):
    s = scene_with_undo
    s.create_node("dense")
    with pytest.raises(GraphCompileError):
        compile_scene(s)


def test_compile_attention_shape_check(scene_with_undo):
    s = scene_with_undo
    a = s.create_node("text_input")
    a.params["max_length"] = 16
    a.params["embed_dim"] = 64
    b = s.create_node("attention")
    b.params["n_heads"] = 4
    conn = ConnectionItem(start_socket=a.output_sockets[0],
                            end_socket=b.input_sockets[0])
    a.output_sockets[0].add_connection(conn)
    b.input_sockets[0].add_connection(conn)
    s.add_connection(conn)
    cg = compile_scene(s)
    assert cg.output_shape == (16, 64)


def test_compile_error_sets_node_red(scene_with_undo):
    s = scene_with_undo
    a = s.create_node("image_input")
    # Plug image (3D) into attention (needs 2D) → erreur sur attention
    b = s.create_node("attention")
    conn = ConnectionItem(start_socket=a.output_sockets[0],
                            end_socket=b.input_sockets[0])
    a.output_sockets[0].add_connection(conn)
    b.input_sockets[0].add_connection(conn)
    s.add_connection(conn)
    with pytest.raises(GraphCompileError):
        compile_scene(s)
    assert b.error is not None
    assert "T, D" in b.error or "input" in b.error.lower()


# ============ Phase 3 save/load ============


def test_save_load_roundtrip(scene_with_undo):
    s = scene_with_undo
    a = s.create_node("image_input")
    a.params["shape"] = "[3, 32, 32]"
    b = s.create_node("conv2d")
    b.params["filters"] = 16
    c = s.create_node("dense")
    c.params["neurons"] = 64
    for src, dst in [(a, b), (b, c)]:
        conn = ConnectionItem(start_socket=src.output_sockets[0],
                                end_socket=dst.input_sockets[0])
        src.output_sockets[0].add_connection(conn)
        dst.input_sockets[0].add_connection(conn)
        s.add_connection(conn)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        path = tf.name
    save_scene(s, path, name="roundtrip")
    s.clear_graph()
    assert len(s.nodes()) == 0
    load_scene(s, path)
    assert len(s.nodes()) == 3
    assert len(s.connections()) == 2
    # Verify params preserved
    nodes_by_type = {n.spec.name: n for n in s.nodes()}
    assert nodes_by_type["image_input"].params["shape"] == "[3, 32, 32]"
    assert nodes_by_type["conv2d"].params["filters"] == 16
    assert nodes_by_type["dense"].params["neurons"] == 64


# ============ Phase 4 ComfyUI specs ============


def test_registry_has_diffusion(app):
    r = default_registry()
    cats = r.by_category()
    assert "Diffusion" in cats
    diffusion_names = {s.name for s in cats["Diffusion"]}
    assert "k_sampler" in diffusion_names
    assert "vae_encode" in diffusion_names
    assert "vae_decode" in diffusion_names
    assert "clip_text_encode" in diffusion_names
    assert "empty_latent_image" in diffusion_names
    assert "lora_loader" in diffusion_names


def test_registry_has_huggingface(app):
    r = default_registry()
    cats = r.by_category()
    assert "HuggingFace" in cats
    assert any(s.name == "hf_model_loader" for s in cats["HuggingFace"])


def test_registry_has_devices(app):
    r = default_registry()
    cats = r.by_category()
    assert "Devices" in cats
    names = {s.name for s in cats["Devices"]}
    assert "webcam" in names
    assert "microphone" in names


def test_k_sampler_specs(app):
    r = default_registry()
    ks = r.get("k_sampler")
    assert ks is not None
    assert len(ks.inputs) == 4   # model, positive, negative, latent
    assert len(ks.outputs) == 1
    param_names = [p.name for p in ks.params]
    for p in ("seed", "steps", "cfg", "sampler", "scheduler", "denoise"):
        assert p in param_names
