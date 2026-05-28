"""Tests Phase 29 — Glossary massive + auto-register."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import torch
from types import SimpleNamespace

from ark.app import create_app
from ark.nodes.registry import default_registry
from ark.core.graph_compiler import HANDLERS
from ark.core.glossary_data import CONCEPTS, by_category, by_kind
from ark.core.auto_register import HANDLER_FACTORIES


@pytest.fixture(scope="module")
def app():
    return create_app()


def test_concepts_count(app):
    assert len(CONCEPTS) >= 400, f"trop peu concepts : {len(CONCEPTS)}"


def test_categories_coverage(app):
    cats = by_category()
    expected = ["Foundations", "Neural Networks", "Sequential", "Transformers",
                  "NLP", "Computer Vision", "Generative AI", "Reinforcement Learning",
                  "Graph Systems", "Unsupervised", "Evolutionary Systems",
                  "Dynamical Systems", "Biological Inspiration",
                  "Cognitive Architectures", "Interpretability", "Optimization",
                  "Training", "Distributed Systems", "Hardware",
                  "Simulation", "Multi-Agent Systems", "Adaptive Architectures",
                  "Experimental Research"]
    for c in expected:
        assert c in cats, f"catégorie manquante : {c}"


def test_registry_grew(app):
    r = default_registry()
    assert len(r.all()) >= 180, f"registry trop petit : {len(r.all())}"


def test_handlers_grew(app):
    assert len(HANDLERS) >= 70, f"handlers trop peu : {len(HANDLERS)}"


# ============ Tests fonctionnels handlers ============

def _node(name, **params):
    return SimpleNamespace(id="x", params=params)


def test_activations_work(app):
    x = torch.randn(2, 16)
    for name in ["act_sigmoid", "act_tanh", "act_relu", "act_gelu", "act_silu",
                  "act_softmax"]:
        m, _ = HANDLERS[name](_node(name), [(16,)])
        out = m(x)
        assert out.shape == x.shape


def test_layers_work(app):
    # Existing conv2d handler utilise param 'filters' (legacy)
    m, _ = HANDLERS["conv2d"](_node("conv2d", filters=16, kernel_size=3), [(3, 32, 32)])
    out = m(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 16, 32, 32)
    # Nouveau conv1d auto-registered
    m1, _ = HANDLERS["conv1d"](_node("conv1d", in_channels=8, out_channels=16,
                                            kernel_size=3), [(8, 100)])
    out1 = m1(torch.randn(2, 8, 100))
    assert out1.shape[1] == 16


def test_lstm_works(app):
    m, _ = HANDLERS["lstm_layer"](_node("lstm_layer", input_size=64,
                                              hidden_size=128, num_layers=1), [(10, 64)])
    out, _ = m(torch.randn(2, 10, 64))
    assert out.shape == (2, 10, 128)


def test_self_attention_works(app):
    m, _ = HANDLERS["self_attention"](_node("self_attention", d_model=64, d_k=16),
                                            [(10, 64)])
    out = m(torch.randn(2, 10, 64))
    assert out.shape == (2, 10, 64)


def test_positional_encoding(app):
    m, _ = HANDLERS["positional_encoding"](_node("positional_encoding",
                                                        d_model=64, max_len=100), [(10, 64)])
    out = m(torch.randn(2, 10, 64))
    assert out.shape == (2, 10, 64)


def test_reaction_diffusion(app):
    m, _ = HANDLERS["reaction_diffusion"](_node("reaction_diffusion"), [(2, 32, 32)])
    x = torch.ones(1, 2, 32, 32)
    x[..., 14:18, 14:18] = 0.5
    out = m(x)
    assert out.shape == x.shape


def test_lorenz(app):
    m, _ = HANDLERS["lorenz_attractor"](_node("lorenz_attractor"), [(3,)])
    out = m(torch.tensor([[1.0, 1.0, 1.0]]))
    assert out.shape == (1, 3)
    assert not torch.isnan(out).any()


def test_cellular_automaton(app):
    m, _ = HANDLERS["cellular_automaton"](_node("cellular_automaton", rule=110,
                                                      size=32, steps=10), [(32,)])
    x = torch.zeros(1, 32)
    out = m(x)
    assert out.shape == (1, 32)


def test_kuramoto(app):
    m, _ = HANDLERS["kuramoto_sync"](_node("kuramoto_sync", n=20, K=1.0),
                                          [(20,)])
    x = torch.rand(1, 20) * 6.28
    out = m(x)
    assert out.shape == (1, 20)


def test_entropy_kl(app):
    m_h, _ = HANDLERS["entropy"](_node("entropy"), [(10,)])
    out = m_h(torch.randn(2, 10))
    assert out.shape == (2,)


def test_l2_norm(app):
    m, _ = HANDLERS["l2_norm"](_node("l2_norm"), [(10,)])
    out = m(torch.randn(2, 10))
    assert out.shape == (2,)


def test_autoencoder_compose(app):
    m, _ = HANDLERS["autoencoder"](_node("autoencoder", in_dim=128, latent_dim=8),
                                         [(128,)])
    out = m(torch.randn(2, 128))
    assert out.shape == (2, 128)


def test_vae_compose(app):
    m, _ = HANDLERS["vae"](_node("vae", in_dim=128, latent_dim=8), [(128,)])
    out = m(torch.randn(2, 128))
    assert out.shape == (2, 128)


def test_boids_runs(app):
    m, _ = HANDLERS["boids"](_node("boids", n_boids=20, perception_radius=2.0),
                                  [(20, 4)])
    pos = torch.randn(1, 20, 4)
    out = m(pos)
    assert out.shape == (1, 20, 4)
    assert not torch.isnan(out).any()


def test_glossary_specs_in_palette(app):
    """Vérifie qu'on retrouve les nouveaux nodes dans catégories UI."""
    r = default_registry()
    for name in ["act_gelu", "self_attention", "reaction_diffusion",
                  "lorenz_attractor", "cellular_automaton", "kmeans"]:
        spec = r.get(name)
        assert spec is not None, f"NodeSpec absent : {name}"


def test_count_handler_factories(app):
    assert len(HANDLER_FACTORIES) >= 50, \
        f"handler factories insuffisantes : {len(HANDLER_FACTORIES)}"
