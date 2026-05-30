"""Phase 82 — Benchmark d'un modèle compilé.

Mesure les éléments discriminants : nombre de params, taille mémoire, FLOPs
(estimation Linear+Conv), latence (ms/échantillon), débit, device. Plus une
table de modèles de référence pour situer le modèle sur la frontière de Pareto
(précision vs coût).
"""
from __future__ import annotations
import time
from typing import Optional

import torch
import torch.nn as nn


# ============================================================
# FLOPs (estimation via hooks : Linear + Conv2d couvrent MLP/CNN)
# ============================================================

def count_flops(model, dummy) -> int:
    total = [0]
    hooks = []

    def lin_hook(m, inp, out):
        x = inp[0]
        if not torch.is_tensor(x):
            return
        rows = x.numel() // x.shape[-1] if x.dim() >= 1 and x.shape[-1] else 1
        total[0] += 2 * m.in_features * m.out_features * rows

    def conv_hook(m, inp, out):
        o = out[0] if isinstance(out, (tuple, list)) else out
        if not torch.is_tensor(o) or o.dim() < 4:
            return
        B, _, Hout, Wout = o.shape[0], o.shape[1], o.shape[2], o.shape[3]
        k = m.kernel_size[0] * m.kernel_size[1]
        total[0] += 2 * (m.in_channels // m.groups) * k * m.out_channels \
            * Hout * Wout * B

    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(lin_hook))
        elif isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(conv_hook))
    with torch.no_grad():
        try:
            model(dummy)
        except Exception:
            pass
    for h in hooks:
        h.remove()
    return total[0]


# ============================================================
# Latence
# ============================================================

def measure_latency(model, dummy, n_iter: int = 20, warmup: int = 3,
                    device=None) -> Optional[float]:
    model.eval()
    is_mps = device is not None and device.type == "mps"
    is_cuda = device is not None and device.type == "cuda"
    with torch.no_grad():
        for _ in range(warmup):
            try:
                model(dummy)
            except Exception:
                return None
        if is_mps:
            torch.mps.synchronize()
        if is_cuda:
            torch.cuda.synchronize()
        ts = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            model(dummy)
            if is_mps:
                torch.mps.synchronize()
            if is_cuda:
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]  # médiane


# ============================================================
# Helpers
# ============================================================

def human(n: float) -> str:
    for unit, div in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if n >= div:
            return f"{n / div:.2f}{unit}"
    return str(int(n))


def _make_dummy(input_shape, device):
    if (isinstance(input_shape, (tuple, list)) and input_shape and
            all(isinstance(d, int) and d > 0 for d in input_shape)):
        return torch.zeros(1, *input_shape, device=device)
    return torch.zeros(1, 1, 28, 28, device=device)


# ============================================================
# Benchmark principal
# ============================================================

def benchmark_model(compiled, device=None, n_iter: int = 20) -> dict:
    model = getattr(compiled, "model", compiled)
    device = device or torch.device("cpu")
    try:
        model = model.to(device).eval()
    except Exception:
        device = torch.device("cpu")
        model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    ish = getattr(compiled, "input_shape", None)
    dummy = _make_dummy(ish, device)
    flops = count_flops(model, dummy)
    lat = measure_latency(model, dummy, n_iter, device=device)
    return {
        "n_params": n_params,
        "params_str": human(n_params),
        "param_mb": round(param_bytes / 1e6, 3),
        "flops": flops,
        "flops_str": human(flops) if flops else "—",
        "latency_ms": round(lat, 3) if lat else None,
        "throughput": round(1000.0 / lat, 1) if lat else None,
        "device": str(device),
        "input_shape": list(ish) if isinstance(ish, (tuple, list)) else None,
        "output_shape": list(getattr(compiled, "output_shape", []) or []),
    }


# ============================================================
# Modèles de référence (situer sur la frontière params↔précision)
# ============================================================

REFERENCE_MODELS = [
    {"name": "LeNet-5", "params": 61706, "metric": "accuracy",
     "value": 0.991, "dataset": "MNIST", "year": 1998, "type": "CNN"},
    {"name": "MLP-784 (ce démo)", "params": 109386, "metric": "accuracy",
     "value": 0.98, "dataset": "MNIST", "year": 2026, "type": "MLP"},
    {"name": "ResNet-18", "params": 11689512, "metric": "top1",
     "value": 0.698, "dataset": "ImageNet", "year": 2015, "type": "CNN"},
    {"name": "ResNet-50", "params": 25557032, "metric": "top1",
     "value": 0.761, "dataset": "ImageNet", "year": 2015, "type": "CNN"},
    {"name": "ViT-B/16", "params": 86567656, "metric": "top1",
     "value": 0.779, "dataset": "ImageNet", "year": 2020, "type": "Transformer"},
    {"name": "GPT-2 small", "params": 124000000, "metric": "perplexity",
     "value": 29.4, "dataset": "WebText", "year": 2019, "type": "Transformer"},
]


def reference_table() -> list:
    out = []
    for r in REFERENCE_MODELS:
        d = dict(r)
        d["params_str"] = human(r["params"])
        out.append(d)
    return out
