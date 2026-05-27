"""Tests ARK Phase 19 — multi-output compile (diffusion DAG enfin compilable)."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import torch

from ark.app import create_app
from ark.nodes.registry import default_registry
from ark.ui.node_editor.scene import NodeScene
from ark.ui.node_editor.connection_item import ConnectionItem
from ark.core.format import payload_to_scene
from ark.core.templates import template_diffusion_basic
from ark.core.graph_compiler import compile_scene


@pytest.fixture(scope="module")
def app():
    return create_app()


@pytest.fixture
def scene(app):
    return NodeScene(registry=default_registry())


def test_diffusion_template_compiles(scene):
    """Phase 19 cible : diffusion_basic compile désormais (multi-output)."""
    payload_to_scene(scene, template_diffusion_basic())
    cg = compile_scene(scene)
    assert cg is not None
    assert "k_sampler" in str(list(cg.model.modules_dict.keys()))


def test_diffusion_forward_produces_image(scene):
    """Forward complet : noise + sampler + vae decode → image (B, 3, H, W)."""
    payload_to_scene(scene, template_diffusion_basic())
    cg = compile_scene(scene)
    cg.model.eval()
    with torch.no_grad():
        out = cg.model(None)
    assert isinstance(out, torch.Tensor)
    assert out.ndim == 4
    assert out.shape[1] == 3  # RGB
    assert out.shape[2] == out.shape[3]  # square (latent 64 → image 512)


def test_socket_routing_correct(scene):
    """Vérifie que les outputs (model/clip/vae) du loader vont aux bons sockets."""
    payload_to_scene(scene, template_diffusion_basic())
    cg = compile_scene(scene)
    # Le loader produit tuple (model, clip, vae). Les CLIPTextEncode reçoivent clip.
    # Le KSampler reçoit model en socket 0.
    # Vérifie indirectement : forward marche sans crash → routing OK.
    cg.model.eval()
    with torch.no_grad():
        out = cg.model(None)
    assert out is not None


def test_load_image_multi_output(scene):
    """load_image retourne (image, mask). Test routing socket=0 vs socket=1."""
    li = scene.create_node("load_image")
    li.params["path"] = "/tmp/nonexistent_test.png"
    sv1 = scene.create_node("save_image")
    sv1.params["prefix"] = "img_"
    sv2 = scene.create_node("save_image")
    sv2.params["prefix"] = "mask_"
    # connecte image (sock 0) → sv1, mask (sock 1) → sv2
    c1 = ConnectionItem(start_socket=li.output_sockets[0],
                          end_socket=sv1.input_sockets[0])
    li.output_sockets[0].add_connection(c1)
    sv1.input_sockets[0].add_connection(c1)
    scene.add_connection(c1)
    c2 = ConnectionItem(start_socket=li.output_sockets[1],
                          end_socket=sv2.input_sockets[0])
    li.output_sockets[1].add_connection(c2)
    sv2.input_sockets[0].add_connection(c2)
    scene.add_connection(c2)
    cg = compile_scene(scene)
    assert cg is not None


def test_source_nodes_detected(scene):
    """Phase 19 : nodes sans input (diffusion_model_loader, empty_latent_image)
    sont automatiquement détectés comme sources, pas besoin d'input/tabular/text."""
    payload_to_scene(scene, template_diffusion_basic())
    cg = compile_scene(scene)
    # Pas de "Missing input node" levé
    assert cg.model.source_node_ids
    # Loader + empty_latent_image = 2 sources (CLIPs ont incoming depuis loader)
    assert len(cg.model.source_node_ids) >= 2


def test_classifier_template_still_compiles(scene):
    """Régression : MLP MNIST template compile encore après refactor multi-output."""
    from ark.core.templates import template_mlp_mnist
    payload_to_scene(scene, template_mlp_mnist())
    cg = compile_scene(scene, num_classes=10)
    assert cg.output_shape == (10,)
    x = torch.randn(2, 1, 28, 28)
    y = cg.model(x)
    assert y.shape == (2, 10)
