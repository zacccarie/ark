"""Phase 39 — Benchmark + audit nodes ARK.

Modules :
1. BenchmarkCompile  : mesure compile time + param count per scene
2. BenchmarkInference : forward latency + throughput per batch size
3. BenchmarkTraining : loss/accuracy/epoch time tracker
4. ModelAudit        : params + FLOPs estim + module breakdown
5. ParamDistribution : histogramme weights par couche
6. GradientFlow      : norms gradients per layer (vanishing/exploding)
7. ActivationStats   : moyennes + variances activations
8. LatencyProfiler   : per-op timings

Tous output : dict métrique + HTML Plotly interactif optionnel.
"""
from __future__ import annotations
import json
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn


ACCENT = "#E11D2A"
ACCENT_GREEN = "#00FF85"
INFO = "#38BDF8"
WARN = "#FBBF24"
BG = "#0E0E11"
BG_PANEL = "#1C1C25"
FG = "#F4F4F5"
FG_MUTED = "#8E8E98"


@dataclass
class BenchResult:
    ok: bool
    metrics: dict = field(default_factory=dict)
    html_path: Optional[str] = None
    error: str = ""


def _save_html(html: str, path: str) -> str:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return str(out)


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    import numpy as np
    return np.asarray(x)


# ============================================================
# Helper : count params + FLOPs estimation
# ============================================================

def count_params(model: nn.Module) -> dict:
    total = 0
    trainable = 0
    by_type: dict[str, int] = {}
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    for m in model.modules():
        t = type(m).__name__
        if any(p.requires_grad for p in m.parameters(recurse=False)):
            np_m = sum(p.numel() for p in m.parameters(recurse=False))
            by_type[t] = by_type.get(t, 0) + np_m
    return {
        "total": total, "trainable": trainable,
        "by_module_type": by_type,
        "frozen": total - trainable,
    }


def estimate_flops(model: nn.Module, input_shape: tuple) -> int:
    """Estimation rough FLOPs via hooks."""
    flops = [0]

    def conv_hook(m, inp, out):
        # 2 × in_ch × out_ch × out_H × out_W × kernel_size²
        bs = out.shape[0]
        c_out = out.shape[1]
        h_out = out.shape[2] if out.dim() > 2 else 1
        w_out = out.shape[3] if out.dim() > 3 else 1
        c_in = inp[0].shape[1]
        k = (m.kernel_size if isinstance(m.kernel_size, int)
              else m.kernel_size[0]) ** 2
        flops[0] += 2 * bs * c_in * c_out * h_out * w_out * k

    def lin_hook(m, inp, out):
        flops[0] += 2 * inp[0].shape[0] * m.in_features * m.out_features

    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(lin_hook))

    try:
        x = torch.zeros(1, *input_shape)
        with torch.no_grad():
            model.eval()(x)
    except Exception:
        pass
    for h in hooks:
        h.remove()
    return flops[0]


# ============================================================
# 1. BenchmarkInference : forward latency + throughput
# ============================================================

class BenchmarkInference(nn.Module):
    """Mesure forward latency + throughput sur batch_sizes différents."""

    def __init__(self, output_path: str = "./output/bench_inference.html",
                 batch_sizes: str = "1,4,16,64", n_runs: int = 20,
                 input_shape: str = "1,28,28", title: str = "Inference Benchmark",
                 auto_open: bool = False):
        super().__init__()
        self.output_path = output_path
        self.batch_sizes = [int(b) for b in batch_sizes.split(",")]
        self.n_runs = n_runs
        self.input_shape = tuple(int(s) for s in input_shape.split(","))
        self.title = title
        self.auto_open = auto_open
        self.last_result: BenchResult | None = None

    def benchmark(self, model: nn.Module) -> dict:
        device = next(model.parameters()).device if list(model.parameters()) \
                  else torch.device("cpu")
        model.eval()
        results = {"batch_sizes": self.batch_sizes,
                    "latency_ms": [], "throughput_samples_s": []}
        for bs in self.batch_sizes:
            try:
                x = torch.randn(bs, *self.input_shape, device=device)
                # Warmup
                with torch.no_grad():
                    for _ in range(3):
                        _ = model(x)
                # Measure
                timings = []
                with torch.no_grad():
                    for _ in range(self.n_runs):
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        _ = model(x)
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        timings.append((time.perf_counter() - t0) * 1000)
                lat_med = sorted(timings)[len(timings) // 2]
                tp = bs * 1000 / lat_med if lat_med > 0 else 0
                results["latency_ms"].append(lat_med)
                results["throughput_samples_s"].append(tp)
            except Exception as e:
                results["latency_ms"].append(-1)
                results["throughput_samples_s"].append(-1)
        return results

    def forward(self, x):
        """Forward attendu : input = (model, optional batch_size)."""
        if isinstance(x, nn.Module):
            metrics = self.benchmark(x)
            html_path = self._render_html(metrics)
            self.last_result = BenchResult(True, metrics, html_path)
        return x

    def _render_html(self, metrics: dict) -> str:
        try:
            import plotly.graph_objects as go
        except ImportError:
            return ""
        bs = metrics["batch_sizes"]
        lat = metrics["latency_ms"]
        tp = metrics["throughput_samples_s"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bs, y=lat, mode="lines+markers",
                                       name="Latency (ms)",
                                       line=dict(color=ACCENT, width=2),
                                       marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=bs, y=tp, mode="lines+markers",
                                       name="Throughput (samples/s)",
                                       line=dict(color=ACCENT_GREEN, width=2),
                                       marker=dict(size=8), yaxis="y2"))
        fig.update_layout(
            title=dict(text=self.title, font=dict(color=FG)),
            paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=FG),
            xaxis=dict(title="Batch size", color=FG, type="log",
                          gridcolor="#3A3A45"),
            yaxis=dict(title="Latency (ms)", color=ACCENT,
                          gridcolor="#3A3A45"),
            yaxis2=dict(title="Throughput", color=ACCENT_GREEN,
                           overlaying="y", side="right"),
            margin=dict(l=60, r=60, t=40, b=40),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=FG)),
        )
        html = fig.to_html(include_plotlyjs="cdn")
        path = _save_html(html, self.output_path)
        if self.auto_open:
            webbrowser.open(f"file://{Path(path).absolute()}")
        return path


# ============================================================
# 2. ModelAudit : params + FLOPs + modules breakdown
# ============================================================

class ModelAudit(nn.Module):
    """Audit complet : params, FLOPs, modules, mémoire."""

    def __init__(self, output_path: str = "./output/audit.html",
                 input_shape: str = "1,28,28",
                 title: str = "Model Audit", auto_open: bool = False):
        super().__init__()
        self.output_path = output_path
        self.input_shape = tuple(int(s) for s in input_shape.split(","))
        self.title = title
        self.auto_open = auto_open
        self.last_result: BenchResult | None = None

    def audit(self, model: nn.Module) -> dict:
        params = count_params(model)
        flops = estimate_flops(model, self.input_shape)
        # Mémoire estimée (params + activations rough)
        mem_params_mb = params["total"] * 4 / 1e6  # float32
        # Module breakdown
        modules = []
        for name, m in model.named_modules():
            if name == "":
                continue
            np_m = sum(p.numel() for p in m.parameters(recurse=False))
            if np_m > 0:
                modules.append({"name": name, "type": type(m).__name__,
                                  "params": np_m})
        return {
            "total_params": params["total"],
            "trainable_params": params["trainable"],
            "frozen_params": params["frozen"],
            "by_module_type": params["by_module_type"],
            "modules": modules,
            "estimated_flops": flops,
            "estimated_memory_mb": mem_params_mb,
            "input_shape": list(self.input_shape),
        }

    def forward(self, x):
        if isinstance(x, nn.Module):
            metrics = self.audit(x)
            html_path = self._render_html(metrics)
            self.last_result = BenchResult(True, metrics, html_path)
        return x

    def _render_html(self, m: dict) -> str:
        try:
            import plotly.graph_objects as go
            import plotly.subplots as sp
        except ImportError:
            return ""

        # Pie chart par type module
        types = list(m["by_module_type"].keys())
        counts = list(m["by_module_type"].values())
        # Bar chart top modules
        modules_sorted = sorted(m["modules"], key=lambda x: -x["params"])[:15]

        fig = sp.make_subplots(rows=1, cols=2,
                                  specs=[[{"type": "domain"}, {"type": "xy"}]],
                                  subplot_titles=("Params par type",
                                                    "Top 15 modules"))

        fig.add_trace(go.Pie(labels=types, values=counts, hole=0.4,
                                marker=dict(colors=[ACCENT, ACCENT_GREEN, INFO,
                                                          WARN, "#A78BFA", "#F472B6"])),
                       row=1, col=1)
        fig.add_trace(go.Bar(
            x=[m_["params"] for m_ in modules_sorted],
            y=[m_["name"][:30] for m_ in modules_sorted],
            orientation="h",
            marker=dict(color=ACCENT),
        ), row=1, col=2)

        stats = (
            f"Total params: {m['total_params']:,}  ·  "
            f"Trainable: {m['trainable_params']:,}  ·  "
            f"FLOPs: {m['estimated_flops']:,}  ·  "
            f"Memory: {m['estimated_memory_mb']:.1f} MB"
        )

        fig.update_layout(
            title=dict(text=f"{self.title}<br><sub>{stats}</sub>",
                          font=dict(color=FG)),
            paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=FG),
            margin=dict(l=20, r=20, t=80, b=20),
            showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=FG)),
        )
        html = fig.to_html(include_plotlyjs="cdn")
        path = _save_html(html, self.output_path)
        if self.auto_open:
            webbrowser.open(f"file://{Path(path).absolute()}")
        return path


# ============================================================
# 3. ParamDistribution : histogramme weights
# ============================================================

class ParamDistribution(nn.Module):
    """Distribution des weights par couche (histogramme)."""

    def __init__(self, output_path: str = "./output/param_dist.html",
                 title: str = "Param Distribution", auto_open: bool = False,
                 n_bins: int = 50):
        super().__init__()
        self.output_path = output_path
        self.title = title
        self.auto_open = auto_open
        self.n_bins = n_bins
        self.last_result: BenchResult | None = None

    def analyze(self, model: nn.Module) -> dict:
        result = {"layers": []}
        for name, p in model.named_parameters():
            if "weight" not in name:
                continue
            arr = p.detach().cpu().numpy().flatten()
            result["layers"].append({
                "name": name,
                "shape": list(p.shape),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "sparsity": float((arr == 0).mean()),
                "histogram": arr.tolist()[:5000],  # cap pour HTML
            })
        return result

    def forward(self, x):
        if isinstance(x, nn.Module):
            metrics = self.analyze(x)
            html_path = self._render_html(metrics)
            self.last_result = BenchResult(True, metrics, html_path)
        return x

    def _render_html(self, metrics: dict) -> str:
        try:
            import plotly.graph_objects as go
        except ImportError:
            return ""

        fig = go.Figure()
        for layer in metrics["layers"][:8]:  # max 8 layers
            fig.add_trace(go.Histogram(
                x=layer["histogram"],
                name=f"{layer['name']} (σ={layer['std']:.3f})",
                opacity=0.7,
                nbinsx=self.n_bins,
            ))
        fig.update_layout(
            title=dict(text=self.title, font=dict(color=FG)),
            paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=FG),
            xaxis=dict(title="Weight value", color=FG, gridcolor="#3A3A45"),
            yaxis=dict(title="Count", color=FG, gridcolor="#3A3A45"),
            barmode="overlay",
            margin=dict(l=60, r=20, t=40, b=40),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=FG)),
        )
        html = fig.to_html(include_plotlyjs="cdn")
        path = _save_html(html, self.output_path)
        if self.auto_open:
            webbrowser.open(f"file://{Path(path).absolute()}")
        return path


# ============================================================
# 4. GradientFlow : norms gradients per layer
# ============================================================

class GradientFlow(nn.Module):
    """Diagnostic vanishing/exploding gradients."""

    def __init__(self, output_path: str = "./output/grad_flow.html",
                 title: str = "Gradient Flow", auto_open: bool = False):
        super().__init__()
        self.output_path = output_path
        self.title = title
        self.auto_open = auto_open
        self.last_result: BenchResult | None = None

    def analyze(self, model: nn.Module) -> dict:
        """Suppose qu'un backward() vient d'être fait."""
        result = {"layers": []}
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            grad = p.grad.detach().cpu()
            result["layers"].append({
                "name": name,
                "grad_norm": float(grad.norm().item()),
                "grad_mean": float(grad.mean().item()),
                "grad_max": float(grad.abs().max().item()),
                "param_norm": float(p.detach().cpu().norm().item()),
            })
        return result

    def forward(self, x):
        if isinstance(x, nn.Module):
            metrics = self.analyze(x)
            html_path = self._render_html(metrics)
            self.last_result = BenchResult(True, metrics, html_path)
        return x

    def _render_html(self, m: dict) -> str:
        try:
            import plotly.graph_objects as go
        except ImportError:
            return ""

        layers = m["layers"]
        names = [l["name"][:30] for l in layers]
        norms = [l["grad_norm"] for l in layers]

        # Color : amber si très faible (< 1e-6), rouge si très haut (> 100)
        colors = []
        for n in norms:
            if n < 1e-6:
                colors.append(WARN)  # vanishing
            elif n > 100:
                colors.append(ACCENT)  # exploding
            else:
                colors.append(ACCENT_GREEN)  # OK

        fig = go.Figure(data=go.Bar(
            x=norms, y=names, orientation="h",
            marker=dict(color=colors),
            text=[f"{n:.2e}" for n in norms],
            textposition="auto",
        ))
        fig.update_layout(
            title=dict(text=f"{self.title} · "
                              f"({sum(1 for n in norms if n < 1e-6)} vanishing, "
                              f"{sum(1 for n in norms if n > 100)} exploding)",
                          font=dict(color=FG)),
            paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=FG),
            xaxis=dict(title="Gradient norm (log)", color=FG, type="log",
                          gridcolor="#3A3A45"),
            yaxis=dict(color=FG, autorange="reversed"),
            margin=dict(l=200, r=20, t=40, b=40),
        )
        html = fig.to_html(include_plotlyjs="cdn")
        path = _save_html(html, self.output_path)
        if self.auto_open:
            webbrowser.open(f"file://{Path(path).absolute()}")
        return path


# ============================================================
# 5. ActivationStats : statistics activations per layer
# ============================================================

class ActivationStats(nn.Module):
    """Capture activations stats via forward hooks."""

    def __init__(self, output_path: str = "./output/activ_stats.html",
                 title: str = "Activation Stats",
                 input_shape: str = "1,28,28",
                 auto_open: bool = False):
        super().__init__()
        self.output_path = output_path
        self.title = title
        self.input_shape = tuple(int(s) for s in input_shape.split(","))
        self.auto_open = auto_open
        self.last_result: BenchResult | None = None

    def analyze(self, model: nn.Module) -> dict:
        activations: dict[str, dict] = {}

        def make_hook(name):
            def hook(_m, _inp, out):
                t = out if not isinstance(out, tuple) else out[0]
                a = t.detach().cpu().numpy()
                activations[name] = {
                    "shape": list(t.shape),
                    "mean": float(a.mean()),
                    "std": float(a.std()),
                    "min": float(a.min()),
                    "max": float(a.max()),
                    "saturated": float((a.abs() > 0.99).mean()),
                    "sparsity": float((a == 0).mean()),
                }
            return hook

        hooks = []
        for name, m in model.named_modules():
            if name == "" or len(list(m.children())) > 0:
                continue
            hooks.append(m.register_forward_hook(make_hook(name)))

        try:
            x = torch.randn(1, *self.input_shape)
            with torch.no_grad():
                model.eval()(x)
        except Exception:
            pass
        for h in hooks:
            h.remove()

        return {"activations": activations}

    def forward(self, x):
        if isinstance(x, nn.Module):
            metrics = self.analyze(x)
            html_path = self._render_html(metrics)
            self.last_result = BenchResult(True, metrics, html_path)
        return x

    def _render_html(self, metrics: dict) -> str:
        try:
            import plotly.graph_objects as go
        except ImportError:
            return ""
        activs = metrics["activations"]
        names = list(activs.keys())[:30]
        means = [activs[n]["mean"] for n in names]
        stds = [activs[n]["std"] for n in names]
        sats = [activs[n]["saturated"] for n in names]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=names, y=means, name="mean",
                                marker=dict(color=ACCENT_GREEN)))
        fig.add_trace(go.Bar(x=names, y=stds, name="std",
                                marker=dict(color=INFO)))
        fig.add_trace(go.Scatter(x=names, y=sats, name="saturated (%)",
                                       mode="markers+lines",
                                       marker=dict(color=ACCENT, size=10),
                                       yaxis="y2"))
        fig.update_layout(
            title=dict(text=self.title, font=dict(color=FG)),
            paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=FG),
            xaxis=dict(color=FG, tickangle=-45),
            yaxis=dict(title="mean / std", color=FG, gridcolor="#3A3A45"),
            yaxis2=dict(title="saturated", overlaying="y", side="right",
                           color=ACCENT, range=[0, 1]),
            barmode="group",
            margin=dict(l=60, r=60, t=40, b=120),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=FG)),
        )
        html = fig.to_html(include_plotlyjs="cdn")
        path = _save_html(html, self.output_path)
        if self.auto_open:
            webbrowser.open(f"file://{Path(path).absolute()}")
        return path


# ============================================================
# Handlers
# ============================================================

def _make_handler(cls, default_params: dict):
    def handler(node, in_shapes):
        params = {k: node.params.get(k, v) for k, v in default_params.items()}
        m = cls(**params)
        m._ark_node_id = getattr(node, "id", "bench")
        return m, in_shapes[0] if in_shapes else (1,)
    return handler


def register_benchmark_handlers(handlers_dict: dict) -> bool:
    handlers_dict["benchmark_inference"] = _make_handler(
        BenchmarkInference, dict(
            output_path="./output/bench_inference.html",
            batch_sizes="1,4,16,64", n_runs=20,
            input_shape="1,28,28", title="Inference Benchmark",
            auto_open=False))
    handlers_dict["model_audit"] = _make_handler(
        ModelAudit, dict(
            output_path="./output/audit.html",
            input_shape="1,28,28", title="Model Audit",
            auto_open=False))
    handlers_dict["param_distribution"] = _make_handler(
        ParamDistribution, dict(
            output_path="./output/param_dist.html",
            title="Param Distribution", auto_open=False,
            n_bins=50))
    handlers_dict["gradient_flow"] = _make_handler(
        GradientFlow, dict(
            output_path="./output/grad_flow.html",
            title="Gradient Flow", auto_open=False))
    handlers_dict["activation_stats"] = _make_handler(
        ActivationStats, dict(
            output_path="./output/activ_stats.html",
            title="Activation Stats", input_shape="1,28,28",
            auto_open=False))
    return True


def register_benchmark_specs(registry) -> bool:
    from ark.nodes.registry import NodeSpec, ParamSpec, SocketSpec

    CAT = "Benchmark"

    registry.register(NodeSpec(
        name="benchmark_inference", title="Benchmark Inference",
        category=CAT,
        inputs=[SocketSpec("model", "model")],
        outputs=[SocketSpec("model", "model")],
        params=[
            ParamSpec("output_path", "str", "./output/bench_inference.html"),
            ParamSpec("batch_sizes", "str", "1,4,16,64"),
            ParamSpec("n_runs", "int", 20, min=3, max=1000),
            ParamSpec("input_shape", "str", "1,28,28"),
            ParamSpec("title", "str", "Inference Benchmark"),
            ParamSpec("auto_open", "bool", False),
        ],
        description="Mesure latency + throughput pour plusieurs batch sizes. "
                       "Génère HTML Plotly latency-vs-bs + throughput-vs-bs.",
        role="Model performance benchmark.",
        function="Warmup 3 runs + n_runs measure × batch sizes. "
                    "Median latency en ms, throughput = bs/latency.",
        has_thumbnail=True,
    ))

    registry.register(NodeSpec(
        name="model_audit", title="Model Audit (full)",
        category=CAT,
        inputs=[SocketSpec("model", "model")],
        outputs=[SocketSpec("model", "model")],
        params=[
            ParamSpec("output_path", "str", "./output/audit.html"),
            ParamSpec("input_shape", "str", "1,28,28"),
            ParamSpec("title", "str", "Model Audit"),
            ParamSpec("auto_open", "bool", False),
        ],
        description="Audit complet : params (total/trainable/frozen), FLOPs estim, "
                       "memory MB, breakdown par type module, top 15 modules.",
        role="Comprehensive model accounting.",
        function="count_params via named_parameters + estimate_flops via "
                    "forward hooks Conv/Linear. Render pie + bar HTML.",
        has_thumbnail=True,
    ))

    registry.register(NodeSpec(
        name="param_distribution", title="Param Distribution",
        category=CAT,
        inputs=[SocketSpec("model", "model")],
        outputs=[SocketSpec("model", "model")],
        params=[
            ParamSpec("output_path", "str", "./output/param_dist.html"),
            ParamSpec("title", "str", "Param Distribution"),
            ParamSpec("auto_open", "bool", False),
            ParamSpec("n_bins", "int", 50, min=10, max=200),
        ],
        description="Histogrammes des weights par couche. Détecte init mauvais, "
                       "dead neurons, distribution skewed.",
        role="Weight distribution diagnostic.",
        function="Pour chaque param 'weight' : flatten + histogramme. "
                    "Overlay max 8 couches en transparence.",
        has_thumbnail=True,
    ))

    registry.register(NodeSpec(
        name="gradient_flow", title="Gradient Flow Diagnostic",
        category=CAT,
        inputs=[SocketSpec("model", "model")],
        outputs=[SocketSpec("model", "model")],
        params=[
            ParamSpec("output_path", "str", "./output/grad_flow.html"),
            ParamSpec("title", "str", "Gradient Flow"),
            ParamSpec("auto_open", "bool", False),
        ],
        description="Norms gradients per layer. Auto-détecte vanishing (<1e-6) en "
                       "amber et exploding (>100) en rouge. Diagnostic backward critique.",
        role="Vanishing/exploding gradient diagnostic.",
        function="Lit p.grad pour chaque param. Bars colorées par seuil. "
                    "REQUIRES backward() préalable.",
        has_thumbnail=True,
        param_help={
            "output_path": "HTML output.",
        },
    ))

    registry.register(NodeSpec(
        name="activation_stats", title="Activation Stats (live)",
        category=CAT,
        inputs=[SocketSpec("model", "model")],
        outputs=[SocketSpec("model", "model")],
        params=[
            ParamSpec("output_path", "str", "./output/activ_stats.html"),
            ParamSpec("title", "str", "Activation Stats"),
            ParamSpec("input_shape", "str", "1,28,28"),
            ParamSpec("auto_open", "bool", False),
        ],
        description="Mean/std/min/max/saturated/sparsity des activations per layer "
                       "via forward hooks. Diagnostic dead ReLU + saturation.",
        role="Activations distribution diagnostic.",
        function="register_forward_hook sur tous les leaf modules. "
                    "Sample 1 forward avec random input.",
        has_thumbnail=True,
    ))
    return True
