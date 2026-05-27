"""Tests ARK Phases 8 (export) + 11 (thumbnails) + 13 (templates)."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import pytest
from PySide6.QtGui import QUndoStack
from ark.app import create_app
from ark.nodes.registry import default_registry
from ark.ui.node_editor.scene import NodeScene
from ark.core.format import payload_to_scene
from ark.core.templates import (
    TEMPLATES_LIST, get_template,
    template_mlp_mnist, template_cnn_cifar, template_resnet_block,
    template_transformer_mini, template_vg_mutable, template_diffusion_basic,
)
from ark.core.export import export_python, export_onnx
from ark.core.graph_compiler import compile_scene


@pytest.fixture(scope="module")
def app():
    return create_app()


@pytest.fixture
def scene(app):
    return NodeScene(registry=default_registry())


# ============ Phase 13 templates ============


def test_templates_list_complete():
    names = [n for n, _ in TEMPLATES_LIST]
    assert "mlp_mnist" in names
    assert "cnn_cifar" in names
    assert "vg_mutable" in names
    assert "diffusion_basic" in names


@pytest.mark.parametrize("name", [n for n, _ in TEMPLATES_LIST])
def test_template_loads(scene, name):
    payload = get_template(name)
    assert payload is not None
    assert payload["version"] == "ark-arch-v1"
    payload_to_scene(scene, payload)
    assert len(scene.nodes()) > 0


def test_template_mlp_mnist_compiles(scene):
    payload = template_mlp_mnist()
    payload_to_scene(scene, payload)
    cg = compile_scene(scene, num_classes=10)
    assert cg.input_shape == (1, 28, 28)
    assert cg.output_shape == (10,)


def test_template_cnn_cifar_compiles(scene):
    payload = template_cnn_cifar()
    payload_to_scene(scene, payload)
    cg = compile_scene(scene, num_classes=10)
    assert cg.input_shape == (3, 32, 32)
    assert cg.output_shape == (10,)


def test_template_resnet_block_has_residual(scene):
    payload = template_resnet_block()
    payload_to_scene(scene, payload)
    types = {n.spec.name for n in scene.nodes()}
    assert "residual_add" in types
    # Vérifie qu'il y a une connexion skip (2 inputs sur le node residual)
    res_node = next(n for n in scene.nodes() if n.spec.name == "residual_add")
    assert len(res_node.input_sockets) == 2
    n_conn_on_res = sum(len(s.connections) for s in res_node.input_sockets)
    assert n_conn_on_res == 2


def test_template_vg_mutable_has_trigger_target(scene):
    payload = template_vg_mutable()
    payload_to_scene(scene, payload)
    rule = next(n for n in scene.nodes() if n.spec.name == "trigger_rule")
    vg = next(n for n in scene.nodes() if n.spec.name == "vg_block")
    # Le target_node du trigger_rule a été remappé sur le vg_block présent
    assert rule.params["target_node"] == vg.id


def test_template_diffusion_basic_loads(scene):
    payload = template_diffusion_basic()
    payload_to_scene(scene, payload)
    types = {n.spec.name for n in scene.nodes()}
    for t in ("diffusion_model_loader", "clip_text_encode", "k_sampler",
               "vae_decode", "save_image", "empty_latent_image"):
        assert t in types
    assert len(scene.connections()) >= 8


# ============ Phase 8 export ============


def test_export_python_mlp(scene):
    payload = template_mlp_mnist()
    payload_to_scene(scene, payload)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tf:
        path = tf.name
    out = export_python(scene, path, num_classes=10, name="exported_mlp")
    src = out.read_text()
    assert "class ArkModel" in src
    assert "def build_model" in src
    assert "def forward" in src
    assert "torch" in src
    assert "INPUT_SHAPE" in src


def test_export_python_runs(scene, tmp_path):
    """Le code exporté tourne (exec_module + forward dummy)."""
    payload = template_mlp_mnist()
    payload_to_scene(scene, payload)
    path = tmp_path / "exported_mlp.py"
    export_python(scene, str(path), num_classes=10, name="exported_mlp")
    import importlib.util
    spec = importlib.util.spec_from_file_location("exported_mlp", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.build_model()
    import torch
    x = torch.randn(2, 1, 28, 28)
    y = model(x)
    # LazyLinear → in_features résolu, output (2, 10)
    assert y.shape == (2, 10)


def test_export_onnx_mlp(scene, tmp_path):
    pytest.importorskip("onnxscript", reason="onnxscript not installed for ONNX export")
    payload = template_mlp_mnist()
    payload_to_scene(scene, payload)
    path = tmp_path / "exported.onnx"
    out = export_onnx(scene, str(path), num_classes=10)
    assert out.exists()
    assert len(out.read_bytes()) > 0


# ============ Phase 11 thumbnails ============


def test_thumbnail_observer_has_history_after_metrics(scene):
    n = scene.create_node("observer")
    # Simulate metric_history accumulation
    n._metric_history = [
        {"eff_rank": 5.0, "intrinsic_dim": 2.0, "mean_abs": 0.1, "power_slope": -1.0},
        {"eff_rank": 6.0, "intrinsic_dim": 2.2, "mean_abs": 0.11, "power_slope": -1.1},
        {"eff_rank": 7.0, "intrinsic_dim": 2.4, "mean_abs": 0.12, "power_slope": -1.2},
    ]
    # _paint_thumbnail ne crash pas (testé via paint() implicite — on vérifie la
    # méthode existe et le state est cohérent)
    assert n.spec.has_thumbnail
    assert hasattr(n, "_paint_observer_sparkline")


def test_thumbnail_nodes_have_flag(scene):
    """Certains nodes ont has_thumbnail=True, d'autres pas."""
    from ark.nodes.registry import default_registry
    r = default_registry()
    thumbed = {s.name for s in r.all() if s.has_thumbnail}
    assert "observer" in thumbed
    assert "load_image" in thumbed
    assert "webcam" in thumbed
    # Inversement, dense pas de thumb
    dense_spec = r.get("dense")
    assert not dense_spec.has_thumbnail
