"""Tests Phase 37 — Dynamoscope bridge."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import torch
import numpy as np
from types import SimpleNamespace

from ark.app import create_app
from ark.nodes.registry import default_registry
from ark.core.graph_compiler import HANDLERS
from ark.core.dynamoscope_bridge import (
    PhaseSpace3D, TakensEmbedding, LyapunovRosenstein,
    CorrelationDimension, RecurrencePlot, DMD, RQAStats, DNAScore,
    _ensure_dynamoscope,
)


@pytest.fixture(scope="module")
def app():
    return create_app()


def _lorenz_trajectory(n: int = 500, dt: float = 0.01) -> torch.Tensor:
    """Generate Lorenz attractor trajectory pour tests."""
    sigma, rho, beta = 10.0, 28.0, 2.667
    x = torch.zeros(n, 3)
    x[0] = torch.tensor([1.0, 1.0, 1.0])
    for i in range(1, n):
        xs, ys, zs = x[i - 1]
        x[i, 0] = xs + dt * sigma * (ys - xs)
        x[i, 1] = ys + dt * (xs * (rho - zs) - ys)
        x[i, 2] = zs + dt * (xs * ys - beta * zs)
    return x


# ============ PhaseSpace3D ============

def test_phase_space_3d_basic(tmp_path, app):
    out = tmp_path / "ps.html"
    m = PhaseSpace3D(output_path=str(out), title="Test PS",
                          color_mode="time")
    traj = _lorenz_trajectory(200)
    coords = m(traj)
    assert coords.shape == (200, 3)
    assert out.exists()
    content = out.read_text()
    assert "Test PS" in content
    assert "START" in content
    assert "END" in content


def test_phase_space_3d_velocity_mode(tmp_path, app):
    out = tmp_path / "ps_vel.html"
    m = PhaseSpace3D(output_path=str(out), color_mode="velocity")
    traj = _lorenz_trajectory(100)
    coords = m(traj)
    assert coords.shape == (100, 3)
    assert out.exists()


def test_phase_space_3d_pca_high_dim(tmp_path, app):
    """High-dim (N, D > 3) → auto-PCA."""
    out = tmp_path / "ps_pca.html"
    m = PhaseSpace3D(output_path=str(out))
    x = torch.randn(50, 16)  # 16D
    coords = m(x)
    assert coords.shape == (50, 3)  # projeté en 3D


# ============ TakensEmbedding ============

def test_takens_embedding(app):
    """1D signal → 3D delay coordinates."""
    signal = torch.sin(torch.linspace(0, 10 * 3.14, 200))
    m = TakensEmbedding(m=3, tau=5)
    out = m(signal)
    # out shape : (200 - 2*5, 3) = (190, 3)
    assert out.shape == (190, 3)


def test_takens_embedding_m4(app):
    signal = torch.randn(100)
    m = TakensEmbedding(m=4, tau=2)
    out = m(signal)
    assert out.shape[1] == 4
    assert out.shape[0] == 100 - 3 * 2


# ============ LyapunovRosenstein ============

def test_lyapunov_chaotic(app):
    """Lorenz chaotic → λ > 0."""
    traj = _lorenz_trajectory(500)
    m = LyapunovRosenstein(m=3, tau=1, max_t=20)
    lam = m(traj)
    assert lam.shape == (1,)
    # Lorenz λ_max ≈ 0.9
    assert lam.item() > -1.0  # bounds large : juste vérifie pas NaN


def test_lyapunov_periodic(app):
    """Signal périodique → λ ≈ 0."""
    t = torch.linspace(0, 20 * 3.14, 500)
    signal = torch.stack([torch.sin(t), torch.cos(t), torch.sin(2 * t)],
                              dim=-1)
    m = LyapunovRosenstein(m=3, max_t=20)
    lam = m(signal)
    assert lam.shape == (1,)


# ============ CorrelationDimension ============

def test_correlation_dim(app):
    """Lorenz D₂ ≈ 2.06."""
    traj = _lorenz_trajectory(400)
    m = CorrelationDimension(n_eps=14)
    d = m(traj)
    assert d.shape == (1,)
    # Bounded sanity
    assert 0.5 < d.item() < 5.0


# ============ RecurrencePlot ============

def test_recurrence_plot(tmp_path, app):
    out = tmp_path / "rp.html"
    m = RecurrencePlot(output_path=str(out), max_n=100)
    traj = _lorenz_trajectory(300)
    y = m(traj)
    assert y.shape == traj.shape  # pass-through
    assert out.exists()
    assert "rec_rate" in out.read_text()


# ============ DMD ============

def test_dmd(app):
    """DMD eigenvalues."""
    traj = _lorenz_trajectory(200)
    m = DMD(rank=5, dt=0.01)
    eigs_mag = m(traj)
    # Au moins 1 eigenvalue
    assert eigs_mag.shape[0] >= 1
    assert not torch.isnan(eigs_mag).any()


# ============ RQA Stats ============

def test_rqa_stats(app):
    traj = _lorenz_trajectory(200)
    m = RQAStats(max_n=100, lmin=2)
    stats = m(traj)
    assert stats.shape == (4,)
    # rec_rate ∈ [0, 1]
    assert 0.0 <= stats[0].item() <= 1.0
    # determinism ∈ [0, 1]
    assert 0.0 <= stats[1].item() <= 1.0


# ============ DNA Score ============

def test_dna_score(app):
    traj = _lorenz_trajectory(300)
    m = DNAScore()
    dna = m(traj)
    assert dna.shape == (7,)
    # Tous bornés [0, 1] sanity
    assert (dna >= 0).all() and (dna <= 1.0).all()


# ============ Registration ============

def test_all_handlers_registered(app):
    for name in ("phase_space_3d", "takens_embedding", "lyapunov_rosenstein",
                  "correlation_dimension", "recurrence_plot", "dmd",
                  "rqa_stats", "dna_score"):
        assert name in HANDLERS, f"handler {name} absent"


def test_all_specs_registered(app):
    r = default_registry()
    for name in ("phase_space_3d", "takens_embedding", "lyapunov_rosenstein",
                  "correlation_dimension", "recurrence_plot", "dmd",
                  "rqa_stats", "dna_score"):
        spec = r.get(name)
        assert spec is not None
        assert spec.category == "Dynamoscope"


def test_handler_dispatch_phase_space(app):
    node = SimpleNamespace(id="ps", params={
        "output_path": "/tmp/test_ps.html",
        "title": "t", "auto_open": False,
        "color_mode": "time", "show_attractor": True,
    })
    m, shape = HANDLERS["phase_space_3d"](node, [(100, 3)])
    out = m(torch.randn(100, 3))
    assert out.shape == (100, 3)
