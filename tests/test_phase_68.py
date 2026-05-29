"""Tests ARK Phase 68 — headless scene compile + Modal backend scaffold."""
import torch
import pytest


def test_headless_compile_mlp():
    from ark.core.templates import get_template
    from ark.core.headless_scene import compile_payload
    t = get_template("mlp_mnist")
    cg = compile_payload(t, num_classes=10)
    assert cg.output_shape == (10,)
    x = torch.randn(2, 1, 28, 28)
    out = cg.model(x)
    if isinstance(out, tuple):
        out = out[0]
    assert out.shape == (2, 10)


def test_headless_compile_cnn():
    from ark.core.templates import get_template
    from ark.core.headless_scene import compile_payload
    t = get_template("cnn_cifar")
    cg = compile_payload(t, num_classes=10)
    n = sum(p.numel() for p in cg.model.parameters())
    assert n > 0


def test_headless_scene_api():
    from ark.core.headless_scene import HeadlessScene
    from ark.core.templates import get_template
    t = get_template("mlp_mnist")
    s = HeadlessScene(t)
    nodes = s.nodes()
    assert len(nodes) == len(t["nodes"])
    for n in nodes:
        assert hasattr(n, "id") and hasattr(n, "spec") and hasattr(n, "params")
        assert hasattr(n, "input_sockets") and hasattr(n, "output_sockets")
        n.set_error(None)  # API surface
    conns = s.connections()
    for c in conns:
        assert c.start_socket.parent_node is not None
        assert c.end_socket.parent_node is not None


def test_modal_backend_importable_without_modal():
    # Module must import even if modal absent
    from ark.core import modal_backend
    assert hasattr(modal_backend, "modal_available")
    assert hasattr(modal_backend, "submit_training")
    assert isinstance(modal_backend.GPU_CHOICES, list)
    assert "A10G" in modal_backend.GPU_CHOICES


def test_modal_token_helpers():
    from ark.core.modal_backend import token_configured, save_token
    # token_configured returns bool without crashing
    assert isinstance(token_configured(), bool)


def test_bridge_modal_config_json():
    import json
    from ark.ui.web_bridge import ArkBridge
    b = ArkBridge()
    cfg = json.loads(b.modal_get_config())
    assert "available" in cfg
    assert "token_configured" in cfg
    assert "gpus" in cfg
