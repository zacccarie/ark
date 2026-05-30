"""Tests ARK Phase 82 — benchmark (params/FLOPs/latence) + références."""
import torch
import torch.nn as nn


def test_human_readable():
    from ark.core.bench import human
    assert human(1_500_000) == "1.50M"
    assert human(2_300_000_000) == "2.30G"
    assert human(750) == "750"


def test_count_flops_linear():
    from ark.core.bench import count_flops
    model = nn.Sequential(nn.Flatten(), nn.Linear(10, 5))
    flops = count_flops(model, torch.zeros(1, 10))
    assert flops == 2 * 10 * 5  # 100


def test_count_flops_conv():
    from ark.core.bench import count_flops
    model = nn.Conv2d(3, 8, kernel_size=3, padding=1)
    flops = count_flops(model, torch.zeros(1, 3, 16, 16))
    # 2 * Cin * k*k * Cout * Hout * Wout * B = 2*3*9*8*16*16*1
    assert flops == 2 * 3 * 9 * 8 * 16 * 16


def test_benchmark_mlp_demo():
    from ark.core.templates import get_template
    from ark.core.headless_scene import compile_payload
    from ark.core.bench import benchmark_model
    cg = compile_payload(get_template("mlp_mnist_demo"), num_classes=10)
    b = benchmark_model(cg, device=torch.device("cpu"), n_iter=3)
    assert b["n_params"] > 0
    assert b["flops"] > 0
    assert b["latency_ms"] is not None and b["latency_ms"] > 0
    assert b["throughput"] > 0
    assert isinstance(b["params_str"], str)
    assert b["param_mb"] > 0


def test_reference_table():
    from ark.core.bench import reference_table
    t = reference_table()
    assert len(t) >= 5
    assert all("params_str" in r and "value" in r for r in t)
    names = {r["name"] for r in t}
    assert any("ResNet" in n for n in names)


def test_bridge_benchmark_no_scene():
    import json
    from ark.ui.web_bridge import ArkBridge
    b = ArkBridge()
    r = json.loads(b.benchmark_model())
    assert r["status"] == "error"  # pas de scene sans mw
