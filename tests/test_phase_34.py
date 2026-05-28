"""Tests Phase 34 — Plot3D + BertViz."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import torch

from ark.app import create_app
from ark.nodes.registry import default_registry
from ark.core.graph_compiler import HANDLERS
from ark.core.plot3d_nodes import (
    PCA3D, TSNE3D, UMAP3D, Scatter3D, Line3D, Surface3D, CurvePlot3D,
)
from ark.core.bertviz_nodes import (
    AttentionHeadView, AttentionModelView, AttentionNeuronView,
)


@pytest.fixture(scope="module")
def app():
    return create_app()


# ============ 3D plots ============

def test_pca_3d(tmp_path, app):
    out = tmp_path / "pca.html"
    m = PCA3D(output_path=str(out), title="t", auto_open=False)
    x = torch.randn(50, 10)
    coords = m(x)
    assert coords.shape == (50, 3)
    assert out.exists()
    assert out.stat().st_size > 1000


def test_tsne_3d(tmp_path, app):
    out = tmp_path / "tsne.html"
    m = TSNE3D(output_path=str(out), perplexity=10, n_iter=250)
    x = torch.randn(30, 8)
    coords = m(x)
    assert coords.shape == (30, 3)
    assert out.exists()


def test_umap_3d(tmp_path, app):
    out = tmp_path / "umap.html"
    m = UMAP3D(output_path=str(out), n_neighbors=5)
    x = torch.randn(40, 12)
    coords = m(x)
    assert coords.shape == (40, 3)
    assert out.exists()


def test_scatter_3d(tmp_path, app):
    out = tmp_path / "sc.html"
    m = Scatter3D(output_path=str(out))
    x = torch.randn(20, 3)
    y = m(x)
    assert y.shape == (20, 3)
    assert out.exists()


def test_line_3d(tmp_path, app):
    out = tmp_path / "ln.html"
    m = Line3D(output_path=str(out))
    # Trajectoire Lorenz-like
    t = torch.linspace(0, 10, 100)
    traj = torch.stack([torch.sin(t), torch.cos(t), t / 10], dim=-1)
    y = m(traj)
    assert y.shape == (100, 3)
    assert out.exists()


def test_surface_3d(tmp_path, app):
    out = tmp_path / "sf.html"
    m = Surface3D(output_path=str(out))
    z = torch.randn(20, 30)
    y = m(z)
    assert y.shape == (20, 30)
    assert out.exists()


def test_surface_3d_1d_reshape(tmp_path, app):
    """1D auto-reshape √n × √n."""
    out = tmp_path / "sf1.html"
    m = Surface3D(output_path=str(out))
    z = torch.randn(64)  # → 8x8
    y = m(z)
    assert out.exists()


def test_curves_3d(tmp_path, app):
    out = tmp_path / "cv.html"
    m = CurvePlot3D(output_path=str(out))
    series = torch.randn(10, 100)  # 10 séries de 100 points
    y = m(series)
    assert y.shape == (10, 100)
    assert out.exists()


# ============ BertViz ============

def test_attention_head_view(tmp_path, app):
    out = tmp_path / "hv.html"
    m = AttentionHeadView(output_path=str(out),
                                tokens="the,cat,sat,on,mat",
                                layer_idx=0)
    attn = torch.softmax(torch.randn(2, 4, 5, 5), dim=-1)  # 2 layers, 4 heads
    y = m(attn)
    assert out.exists()
    content = out.read_text()
    assert "Head" in content


def test_attention_model_view(tmp_path, app):
    out = tmp_path / "mv.html"
    m = AttentionModelView(output_path=str(out), tokens="a,b,c,d")
    attn = torch.softmax(torch.randn(3, 4, 4, 4), dim=-1)
    y = m(attn)
    assert out.exists()
    content = out.read_text()
    assert "Layer" in content


def test_attention_neuron_view(tmp_path, app):
    out = tmp_path / "nv.html"
    m = AttentionNeuronView(output_path=str(out), tokens="x,y,z,w,v",
                                  layer_idx=0, head_idx=0)
    attn = torch.softmax(torch.randn(2, 4, 5, 5), dim=-1)
    y = m(attn)
    assert out.exists()
    content = out.read_text()
    assert "svg" in content.lower()


# ============ Registration ============

def test_all_handlers_registered(app):
    for name in ("pca_3d", "tsne_3d", "umap_3d", "scatter_3d", "line_3d",
                  "surface_3d", "curves_3d",
                  "attention_head_view", "attention_model_view",
                  "attention_neuron_view"):
        assert name in HANDLERS, f"handler {name} absent"


def test_all_specs_registered(app):
    r = default_registry()
    for name, cat in [("pca_3d", "Viz"), ("tsne_3d", "Viz"),
                        ("umap_3d", "Viz"), ("scatter_3d", "Viz"),
                        ("line_3d", "Viz"), ("surface_3d", "Viz"),
                        ("curves_3d", "Viz"),
                        ("attention_head_view", "Interpretability"),
                        ("attention_model_view", "Interpretability"),
                        ("attention_neuron_view", "Interpretability")]:
        spec = r.get(name)
        assert spec is not None
        assert spec.category == cat


def test_handler_dispatch_pca(app):
    from types import SimpleNamespace
    node = SimpleNamespace(id="p", params={
        "output_path": "/tmp/pca_test.html",
        "title": "t", "auto_open": False,
        "color_by_index": False,
    })
    m, shape = HANDLERS["pca_3d"](node, [(10, 5)])
    x = torch.randn(10, 5)
    out = m(x)
    assert out.shape == (10, 3)
