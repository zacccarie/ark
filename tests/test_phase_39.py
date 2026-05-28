"""Tests Phase 39 — Benchmark + audit + dashboard."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import torch
import torch.nn as nn

from ark.app import create_app
from ark.nodes.registry import default_registry
from ark.core.graph_compiler import HANDLERS
from ark.core.benchmark_nodes import (
    BenchmarkInference, ModelAudit, ParamDistribution,
    GradientFlow, ActivationStats, count_params, estimate_flops,
)


@pytest.fixture(scope="module")
def app():
    return create_app()


def _simple_model():
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * 28 * 28, 64), nn.ReLU(),
        nn.Linear(64, 10),
    )


def test_count_params(app):
    m = _simple_model()
    p = count_params(m)
    assert p["total"] > 0
    assert p["trainable"] == p["total"]
    assert "Conv2d" in p["by_module_type"]
    assert "Linear" in p["by_module_type"]


def test_estimate_flops(app):
    m = _simple_model()
    flops = estimate_flops(m, (1, 28, 28))
    assert flops > 0


def test_model_audit(tmp_path, app):
    out = tmp_path / "audit.html"
    audit = ModelAudit(output_path=str(out), input_shape="1,28,28",
                          title="Test Audit")
    m = _simple_model()
    audit(m)
    assert audit.last_result is not None
    assert audit.last_result.ok
    assert out.exists()
    metrics = audit.last_result.metrics
    assert metrics["total_params"] > 0
    assert metrics["estimated_flops"] > 0
    assert "Conv2d" in metrics["by_module_type"]


def test_benchmark_inference(tmp_path, app):
    out = tmp_path / "bench.html"
    bench = BenchmarkInference(output_path=str(out), batch_sizes="1,4",
                                      n_runs=3, input_shape="1,28,28")
    m = _simple_model()
    bench(m)
    assert bench.last_result is not None
    metrics = bench.last_result.metrics
    assert metrics["latency_ms"][0] >= 0
    assert len(metrics["latency_ms"]) == 2
    assert out.exists()


def test_param_distribution(tmp_path, app):
    out = tmp_path / "pd.html"
    pd = ParamDistribution(output_path=str(out))
    m = _simple_model()
    pd(m)
    assert pd.last_result is not None
    assert pd.last_result.ok
    metrics = pd.last_result.metrics
    assert len(metrics["layers"]) > 0
    assert "mean" in metrics["layers"][0]
    assert out.exists()


def test_gradient_flow(tmp_path, app):
    out = tmp_path / "gf.html"
    m = _simple_model()
    # Trigger backward
    x = torch.randn(2, 1, 28, 28)
    y = m(x)
    loss = y.sum()
    loss.backward()
    gf = GradientFlow(output_path=str(out))
    gf(m)
    assert gf.last_result is not None
    metrics = gf.last_result.metrics
    assert len(metrics["layers"]) > 0
    assert "grad_norm" in metrics["layers"][0]
    assert out.exists()


def test_activation_stats(tmp_path, app):
    out = tmp_path / "as.html"
    stats = ActivationStats(output_path=str(out), input_shape="1,28,28")
    m = _simple_model()
    stats(m)
    assert stats.last_result is not None
    metrics = stats.last_result.metrics
    assert "activations" in metrics
    assert out.exists()


def test_handlers_registered(app):
    for name in ("benchmark_inference", "model_audit", "param_distribution",
                  "gradient_flow", "activation_stats"):
        assert name in HANDLERS, f"handler {name} absent"


def test_specs_registered(app):
    r = default_registry()
    for name in ("benchmark_inference", "model_audit", "param_distribution",
                  "gradient_flow", "activation_stats"):
        spec = r.get(name)
        assert spec is not None
        assert spec.category == "Benchmark"


def test_dashboard_constructs(app):
    from ark.ui.dialogs.benchmark_dashboard import BenchmarkDashboard, StatCard
    dlg = BenchmarkDashboard()
    assert dlg.card_params is not None
    assert dlg.btn_audit is not None
    assert dlg.log is not None


def test_stat_card_set_value(app):
    from ark.ui.dialogs.benchmark_dashboard import StatCard
    card = StatCard("Test", "—")
    card.set_value("42")
    assert card.value_lbl.text() == "42"
    card.set_value("99", color="#00FF85")
    assert card.value_lbl.text() == "99"


def test_main_window_dashboard_action(app):
    from ark.ui.main_window import MainWindow
    w = MainWindow()
    assert hasattr(w, "_on_open_benchmark_dashboard")
    w._on_open_benchmark_dashboard()
    assert hasattr(w, "_bench_dlg")
    assert w._bench_dlg is not None
