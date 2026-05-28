"""Tests Phase 4 réelle — diffusion_real dispatch + fallback."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from types import SimpleNamespace

from ark.core.diffusion_real import (
    make_diffusion_loader, make_clip_encode, make_ksampler,
    make_vae_decode, make_vae_encode, _ensure_diffusers,
)
from ark.core.diffusion_stubs import (
    DiffusionModelLoaderStub, CLIPTextEncodeStub, KSamplerStub,
    VAEDecodeStub, VAEEncodeStub,
)
from ark.core.graph_compiler import HANDLERS


def test_stub_default():
    """real_backend=False → stubs."""
    m = make_diffusion_loader("foo.safetensors", "cpu", real_backend=False)
    assert isinstance(m, DiffusionModelLoaderStub)
    m2 = make_clip_encode("hello", real_backend=False)
    assert isinstance(m2, CLIPTextEncodeStub)
    m3 = make_ksampler(42, 20, 7.0, "euler", "normal", 1.0, real_backend=False)
    assert isinstance(m3, KSamplerStub)
    assert isinstance(make_vae_decode(False), VAEDecodeStub)
    assert isinstance(make_vae_encode(False), VAEEncodeStub)


def test_real_requires_diffusers():
    """real_backend=True sans diffusers → RuntimeError clair."""
    try:
        import diffusers  # noqa
        has = True
    except ImportError:
        has = False
    if has:
        pytest.skip("diffusers installé, fallback non testable")
    with pytest.raises(RuntimeError, match="diffusers"):
        m = make_diffusion_loader("dummy", "cpu", real_backend=True)
        m.forward()  # déclenche _load() → _ensure_diffusers


def test_handler_dispatch_stub():
    """Handler dispatch via param real_backend=False."""
    node = SimpleNamespace(id="x", params={
        "checkpoint": "x.safetensors", "device": "cpu",
        "real_backend": False,
    })
    m, shape = HANDLERS["diffusion_model_loader"](node, [])
    assert isinstance(m, DiffusionModelLoaderStub)


def test_handler_ksampler_param_dispatch():
    node = SimpleNamespace(id="x", params={
        "seed": 1, "steps": 5, "cfg": 7.0,
        "sampler": "euler", "scheduler": "normal", "denoise": 1.0,
        "real_backend": False,
    })
    m, shape = HANDLERS["k_sampler"](node, [])
    assert isinstance(m, KSamplerStub)


def test_spec_has_real_backend_param():
    """Specs exposent param real_backend pour 5 nodes."""
    from ark.nodes.registry import default_registry
    r = default_registry()
    for name in ("diffusion_model_loader", "clip_text_encode", "k_sampler",
                  "vae_encode", "vae_decode"):
        spec = r.get(name)
        assert spec is not None
        names = [p.name for p in spec.params]
        assert "real_backend" in names, f"{name} manque real_backend"
