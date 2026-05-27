"""Tests ARK Phase 16 (agent_api) + Phase 4 stub (diffusion handlers)."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import pytest
import torch

from ark.app import create_app


@pytest.fixture(scope="module")
def app():
    return create_app()


# ============ Phase 16 agent_api ============


def test_build_quick_mlp(app):
    from ark.agent_api import build_quick_mlp, simulate_forward
    g = build_quick_mlp(layers=[128, 64, 10], input_features=20)
    assert g["version"] == "ark-arch-v1"
    sim = simulate_forward(g, num_classes=10)
    assert sim["output_shape"][-1] == 10
    assert sim["n_params"] > 0


def test_build_quick_cnn(app):
    from ark.agent_api import build_quick_cnn, simulate_forward
    g = build_quick_cnn(filters=[8, 16], dense_layers=[64, 10], input_shape=[1, 28, 28])
    sim = simulate_forward(g, num_classes=10)
    assert sim["output_shape"] == [2, 10]


def test_build_quick_resnet(app):
    from ark.agent_api import build_quick_resnet, simulate_forward
    g = build_quick_resnet(filters=16, input_shape=[3, 32, 32])
    types = {n["type"] for n in g["nodes"]}
    assert "residual_add" in types
    sim = simulate_forward(g, num_classes=10)
    assert sim["output_shape"] == [2, 10]


def test_build_quick_transformer(app):
    from ark.agent_api import build_quick_transformer, simulate_forward
    g = build_quick_transformer(n_blocks=2, d_model=32, n_heads=4)
    sim = simulate_forward(g, num_classes=10)
    assert sim["output_shape"] == [2, 10]


def test_compile_payload(app):
    from ark.agent_api import build_quick_mlp, compile_payload
    g = build_quick_mlp(layers=[32, 10])
    s = compile_payload(g, num_classes=10)
    assert s["n_params"] > 0
    assert s["output_shape"] == [10]


def test_export_payload_python(app, tmp_path):
    from ark.agent_api import build_quick_mlp, export_payload
    g = build_quick_mlp(layers=[16, 10])
    out_path = tmp_path / "out.py"
    p = export_payload(g, "python", str(out_path), num_classes=10)
    assert tmp_path / "out.py" == out_path
    src = out_path.read_text()
    assert "class ArkModel" in src


# ============ Phase 4 stub diffusion ============


def test_diffusion_stubs_loaded():
    from ark.core.graph_compiler import HANDLERS
    for t in ("k_sampler", "clip_text_encode", "vae_decode", "vae_encode",
              "diffusion_model_loader", "empty_latent_image",
              "load_image", "save_image", "lora_loader", "controlnet_apply"):
        assert t in HANDLERS, f"diffusion handler {t} missing"


def test_diffusion_modules_forward_pass():
    """Vérifie que les stubs ne crashent pas en forward."""
    from ark.core.diffusion_stubs import (
        DiffusionModelLoaderStub, CLIPTextEncodeStub, EmptyLatentImageStub,
        KSamplerStub, VAEDecodeStub, VAEEncodeStub, LoadImageStub, SaveImageStub,
        LoRALoaderStub, ControlNetApplyStub,
    )
    loader = DiffusionModelLoaderStub("model.safetensors")
    model, clip, vae = loader()
    assert hasattr(clip, "encode") and hasattr(vae, "decode")

    enc_pos = CLIPTextEncodeStub("a cat")
    pos = enc_pos(clip)
    assert pos.shape == (1, 77, 768)

    enc_neg = CLIPTextEncodeStub("blurry")
    neg = enc_neg(clip)

    latent = EmptyLatentImageStub(512, 512, 1)()
    assert latent.shape == (1, 4, 64, 64)

    sampler = KSamplerStub(seed=42, steps=5, cfg=7.0)
    sampled = sampler(model, pos, neg, latent)
    assert sampled.shape == latent.shape

    img = VAEDecodeStub()(vae, sampled)
    assert img.shape[1] == 3  # RGB
    # 64*8 = 512
    assert img.shape[2] == 512

    # Determinism (même seed)
    s1 = KSamplerStub(seed=1, steps=3)(model, pos, neg, latent)
    s2 = KSamplerStub(seed=1, steps=3)(model, pos, neg, latent)
    assert torch.allclose(s1, s2)


def test_clip_text_encode_deterministic_per_text():
    from ark.core.diffusion_stubs import CLIPTextEncodeStub, _CLIPSentinel
    clip = _CLIPSentinel("test")
    p1 = CLIPTextEncodeStub("hello")(clip)
    p2 = CLIPTextEncodeStub("hello")(clip)
    assert torch.allclose(p1, p2)
    p3 = CLIPTextEncodeStub("different")(clip)
    assert not torch.allclose(p1, p3)


def test_diffusion_template_loads_compilable_nodes(app):
    """Le template diffusion charge bien mais ne compile pas en classifier
    (graph DAG diffusion ≠ classifier classique)."""
    from ark.core.templates import template_diffusion_basic
    g = template_diffusion_basic()
    assert len(g["nodes"]) >= 6
    types = {n["type"] for n in g["nodes"]}
    assert "k_sampler" in types
    assert "vae_decode" in types
    # Tous les types diffusion sont enregistrés dans HANDLERS
    from ark.core.graph_compiler import HANDLERS
    for t in types:
        assert t in HANDLERS, f"missing handler for {t}"


def test_load_image_stub_deterministic():
    from ark.core.diffusion_stubs import LoadImageStub
    n1 = LoadImageStub("/tmp/nonexistent_image_test_abc.png")
    img1, mask1 = n1()
    n2 = LoadImageStub("/tmp/nonexistent_image_test_abc.png")
    img2, mask2 = n2()
    assert torch.allclose(img1, img2)
    assert img1.shape == (1, 3, 512, 512)
