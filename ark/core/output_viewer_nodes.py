"""Phase 55 — Output viewer nodes.

Visualisateurs résultats modèles. Tous pass-through avec attr `last_value`
récupéré par Inspector pour affichage UI.

Nodes :
- text_viewer            : affiche string (output LLM, log, JSON)
- number_display         : affiche scalar (loss, acc, latency)
- image_viewer           : affiche tensor comme image (preview)
- top_k_predictions      : top-K labels + probabilités
- prob_distribution      : softmax → bar chart HTML Plotly
- confusion_matrix_viewer: confusion N×N heatmap HTML
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn


# ============================================================
# Text viewer
# ============================================================

class TextViewer(nn.Module):
    """Pass-through string display."""

    def __init__(self, title: str = "Text Output", max_chars: int = 2000):
        super().__init__()
        self.title = title
        self.max_chars = max_chars
        self.last_text: str = ""

    def forward(self, x):
        # Accept str, list[str], tensor
        if isinstance(x, str):
            self.last_text = x[: self.max_chars]
        elif isinstance(x, (list, tuple)):
            joined = "\n".join(str(item) for item in x[:50])
            self.last_text = joined[: self.max_chars]
        elif hasattr(x, "shape"):
            self.last_text = f"<tensor {tuple(x.shape)}>"
        else:
            self.last_text = str(x)[: self.max_chars]
        return x


# ============================================================
# Number display
# ============================================================

class NumberDisplay(nn.Module):
    """Affiche scalar(s) avec format configurable."""

    def __init__(self, title: str = "Value", fmt: str = ".4f",
                 unit: str = ""):
        super().__init__()
        self.title = title
        self.fmt = fmt
        self.unit = unit
        self.last_value: float | str = ""

    def forward(self, x):
        try:
            if hasattr(x, "item"):
                v = float(x.item()) if x.numel() == 1 else float(x.mean().item())
            elif isinstance(x, (int, float)):
                v = float(x)
            elif isinstance(x, (list, tuple)) and len(x) > 0:
                v = float(x[0])
            else:
                v = 0.0
            try:
                fmt_str = "{:" + self.fmt + "}"
                txt = fmt_str.format(v)
            except (ValueError, KeyError):
                txt = str(v)
            self.last_value = f"{txt} {self.unit}".strip()
        except Exception as e:
            self.last_value = f"err: {e}"
        return x


# ============================================================
# Image viewer (preview tensor)
# ============================================================

class ImageViewer(nn.Module):
    """Affiche tensor comme image. Comme save_image mais sans disque."""

    def __init__(self, title: str = "Image"):
        super().__init__()
        self.title = title
        self.last_image_tensor: torch.Tensor | None = None
        self._ark_node_ref = None

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            return x
        try:
            if x.dim() == 4:  # (B, C, H, W) → take batch 0
                img = x[0]
            elif x.dim() == 3:  # (C, H, W)
                img = x
            else:
                return x
            self.last_image_tensor = img.detach().cpu().clamp(0, 1)
            # Push to NodeItem if linked
            if self._ark_node_ref is not None:
                try:
                    self._ark_node_ref.set_preview_image(self.last_image_tensor)
                except Exception:
                    pass
        except Exception:
            pass
        return x


# ============================================================
# Top-K predictions
# ============================================================

class TopKPredictions(nn.Module):
    """Argmax + top-K labels. Optionnellement avec labels dict."""

    def __init__(self, title: str = "Top-K", k: int = 5,
                 labels: str = ""):
        super().__init__()
        self.title = title
        self.k = k
        # labels = "label1,label2,..." OR JSON list
        self.label_list = self._parse_labels(labels)
        self.last_predictions: list = []

    def _parse_labels(self, s):
        if not s:
            return []
        try:
            import json
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return [x.strip() for x in s.split(",") if x.strip()]

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            return x
        try:
            logits = x
            if logits.dim() > 1:
                logits = logits[0]  # take first batch
            if logits.dim() > 1:
                logits = logits.flatten()
            probs = torch.softmax(logits, dim=-1)
            k = min(self.k, probs.numel())
            top_v, top_i = torch.topk(probs, k)
            self.last_predictions = []
            for v, i in zip(top_v.tolist(), top_i.tolist()):
                label = self.label_list[i] if i < len(self.label_list) else f"#{i}"
                self.last_predictions.append({
                    "rank": len(self.last_predictions) + 1,
                    "index": int(i),
                    "label": label,
                    "prob": float(v),
                })
        except Exception:
            self.last_predictions = []
        return x


# ============================================================
# Probability distribution → Plotly bar HTML
# ============================================================

class ProbabilityDistribution(nn.Module):
    """Softmax → bar chart HTML interactif."""

    def __init__(self, output_path: str = "./output/prob_dist.html",
                 title: str = "Distribution", auto_open: bool = False,
                 max_bars: int = 50):
        super().__init__()
        self.output_path = output_path
        self.title = title
        self.auto_open = auto_open
        self.max_bars = max_bars
        self.last_path: Optional[str] = None

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            return x
        try:
            import plotly.graph_objects as go
            logits = x
            if logits.dim() > 1:
                logits = logits[0]
            if logits.dim() > 1:
                logits = logits.flatten()
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            # Cap visualisation
            n = min(len(probs), self.max_bars)
            sorted_idx = (-probs).argsort()[:n]
            top_p = probs[sorted_idx]
            labels = [f"#{int(i)}" for i in sorted_idx]
            fig = go.Figure(data=[go.Bar(
                x=labels, y=top_p,
                marker=dict(color="#00FF85"),
            )])
            fig.update_layout(
                title=self.title,
                paper_bgcolor="#0A0A0A", plot_bgcolor="#0A0A0A",
                font=dict(color="#F2F2F2"),
                xaxis=dict(title="class", color="#F2F2F2",
                              gridcolor="#2E2E37"),
                yaxis=dict(title="probability", color="#F2F2F2",
                              gridcolor="#2E2E37"),
            )
            out = Path(self.output_path).expanduser()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(fig.to_html(include_plotlyjs="cdn"),
                              encoding="utf-8")
            self.last_path = str(out)
            if self.auto_open:
                import webbrowser
                webbrowser.open(f"file://{out.absolute()}")
        except Exception:
            pass
        return x


# ============================================================
# Confusion matrix viewer
# ============================================================

class ConfusionMatrixViewer(nn.Module):
    """N×N matrix → Plotly heatmap HTML."""

    def __init__(self, output_path: str = "./output/confusion.html",
                 title: str = "Confusion Matrix", auto_open: bool = False):
        super().__init__()
        self.output_path = output_path
        self.title = title
        self.auto_open = auto_open
        self.last_path: Optional[str] = None

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            return x
        try:
            import plotly.graph_objects as go
            mat = x
            if mat.dim() == 3:
                mat = mat[0]
            mat_np = mat.detach().cpu().numpy()
            fig = go.Figure(data=go.Heatmap(
                z=mat_np,
                colorscale=[[0, "#0A0A0A"], [1, "#00FF85"]],
            ))
            fig.update_layout(
                title=self.title,
                paper_bgcolor="#0A0A0A", plot_bgcolor="#0A0A0A",
                font=dict(color="#F2F2F2"),
                xaxis=dict(title="predicted", color="#F2F2F2"),
                yaxis=dict(title="true", color="#F2F2F2", autorange="reversed"),
            )
            out = Path(self.output_path).expanduser()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(fig.to_html(include_plotlyjs="cdn"),
                              encoding="utf-8")
            self.last_path = str(out)
            if self.auto_open:
                import webbrowser
                webbrowser.open(f"file://{out.absolute()}")
        except Exception:
            pass
        return x


# ============================================================
# Handlers + Specs
# ============================================================

def h_text_viewer(node, in_shapes):
    m = TextViewer(
        title=node.params.get("title", "Text Output"),
        max_chars=int(node.params.get("max_chars", 2000)),
    )
    m._ark_node_id = getattr(node, "id", "tv")
    return m, in_shapes[0] if in_shapes else (1,)


def h_number_display(node, in_shapes):
    m = NumberDisplay(
        title=node.params.get("title", "Value"),
        fmt=node.params.get("fmt", ".4f"),
        unit=node.params.get("unit", ""),
    )
    m._ark_node_id = getattr(node, "id", "nd")
    return m, in_shapes[0] if in_shapes else (1,)


def h_image_viewer(node, in_shapes):
    m = ImageViewer(title=node.params.get("title", "Image"))
    m._ark_node_id = getattr(node, "id", "iv")
    m._ark_node_ref = node
    return m, in_shapes[0] if in_shapes else (1,)


def h_top_k_predictions(node, in_shapes):
    m = TopKPredictions(
        title=node.params.get("title", "Top-K"),
        k=int(node.params.get("k", 5)),
        labels=node.params.get("labels", ""),
    )
    m._ark_node_id = getattr(node, "id", "tk")
    return m, in_shapes[0] if in_shapes else (1,)


def h_prob_distribution(node, in_shapes):
    m = ProbabilityDistribution(
        output_path=node.params.get("output_path", "./output/prob_dist.html"),
        title=node.params.get("title", "Distribution"),
        auto_open=bool(node.params.get("auto_open", False)),
        max_bars=int(node.params.get("max_bars", 50)),
    )
    m._ark_node_id = getattr(node, "id", "pd")
    return m, in_shapes[0] if in_shapes else (1,)


def h_confusion_matrix_viewer(node, in_shapes):
    m = ConfusionMatrixViewer(
        output_path=node.params.get("output_path", "./output/confusion.html"),
        title=node.params.get("title", "Confusion Matrix"),
        auto_open=bool(node.params.get("auto_open", False)),
    )
    m._ark_node_id = getattr(node, "id", "cm")
    return m, in_shapes[0] if in_shapes else (1,)


def register_output_viewer_handlers(handlers_dict: dict) -> bool:
    handlers_dict["text_viewer"] = h_text_viewer
    handlers_dict["number_display"] = h_number_display
    handlers_dict["image_viewer"] = h_image_viewer
    handlers_dict["top_k_predictions"] = h_top_k_predictions
    handlers_dict["prob_distribution"] = h_prob_distribution
    handlers_dict["confusion_matrix_viewer"] = h_confusion_matrix_viewer
    return True


def register_output_viewer_specs(registry) -> bool:
    from ark.nodes.registry import NodeSpec, ParamSpec, SocketSpec

    CAT = "Output Viewer"

    registry.register(NodeSpec(
        name="text_viewer", title="Text Viewer",
        category=CAT,
        inputs=[SocketSpec("text", "any")],
        outputs=[SocketSpec("text", "any")],
        params=[
            ParamSpec("title", "str", "Text Output"),
            ParamSpec("max_chars", "int", 2000, min=10, max=100000),
        ],
        description="Affiche texte/string dans Inspector. Compatible LLM "
                       "output, JSON, logs. Pass-through.",
        role="Display string/text result.",
        function="Stocke last_text. Inspector lit attr et affiche QPlainTextEdit.",
        has_thumbnail=True,
    ))

    registry.register(NodeSpec(
        name="number_display", title="Number Display",
        category=CAT,
        inputs=[SocketSpec("value", "any")],
        outputs=[SocketSpec("value", "any")],
        params=[
            ParamSpec("title", "str", "Value"),
            ParamSpec("fmt", "str", ".4f"),
            ParamSpec("unit", "str", ""),
        ],
        description="Affiche scalar (loss, accuracy, latency) format configurable.",
        role="Scalar display.",
        function="Format via Python {:fmt}. Multi-dim tensor → mean().",
        has_thumbnail=True,
    ))

    registry.register(NodeSpec(
        name="image_viewer", title="Image Viewer",
        category=CAT,
        inputs=[SocketSpec("image", "tensor")],
        outputs=[SocketSpec("image", "tensor")],
        params=[
            ParamSpec("title", "str", "Image"),
        ],
        description="Tensor → image preview thumbnail dans node + Inspector. "
                     "Sans écriture disque (vs save_image).",
        role="Preview tensor as image.",
        function="Convert (C, H, W) → QImage via NodeItem.set_preview_image.",
        has_thumbnail=True,
    ))

    registry.register(NodeSpec(
        name="top_k_predictions", title="Top-K Predictions",
        category=CAT,
        inputs=[SocketSpec("logits", "tensor")],
        outputs=[SocketSpec("logits", "tensor")],
        params=[
            ParamSpec("title", "str", "Top-K"),
            ParamSpec("k", "int", 5, min=1, max=1000),
            ParamSpec("labels", "multiline", ""),
        ],
        description="Argmax + top-K labels + probabilités. Labels CSV ou JSON list.",
        role="Top-K classification predictions.",
        function="softmax → topk → list[{rank, index, label, prob}]. "
                    "Affiché Inspector.",
        has_thumbnail=True,
        param_help={
            "labels": "CSV ex 'cat,dog,bird' ou JSON ['cat','dog']. "
                          "Vide = '#index' fallback.",
        },
    ))

    registry.register(NodeSpec(
        name="prob_distribution", title="Probability Distribution",
        category=CAT,
        inputs=[SocketSpec("logits", "tensor")],
        outputs=[SocketSpec("logits", "tensor")],
        params=[
            ParamSpec("output_path", "str", "./output/prob_dist.html"),
            ParamSpec("title", "str", "Distribution"),
            ParamSpec("auto_open", "bool", False),
            ParamSpec("max_bars", "int", 50, min=2, max=10000),
        ],
        description="Softmax → bar chart Plotly HTML. Top-N classes triées proba.",
        role="Classification probability bars.",
        function="Plotly Bar dark theme + vert PRIMARY. HTML output dir.",
        has_thumbnail=True,
    ))

    registry.register(NodeSpec(
        name="confusion_matrix_viewer", title="Confusion Matrix",
        category=CAT,
        inputs=[SocketSpec("matrix", "tensor")],
        outputs=[SocketSpec("matrix", "tensor")],
        params=[
            ParamSpec("output_path", "str", "./output/confusion.html"),
            ParamSpec("title", "str", "Confusion Matrix"),
            ParamSpec("auto_open", "bool", False),
        ],
        description="N×N matrix → Plotly heatmap dark+vert PRIMARY.",
        role="Confusion matrix visualization.",
        function="go.Heatmap colorscale BG_DEEP→PRIMARY. Y reversed.",
        has_thumbnail=True,
    ))
    return True
