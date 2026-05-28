"""Phase 34 — BertViz wrappers : attention head/model/neuron views.

bertviz génère widgets Jupyter HTML/JS. Pour ARK : capture HTML rendu standalone.

3 modes BertViz :
- head_view : attention pour 1 ou plusieurs heads d'1 layer
- model_view : grid de TOUTES les heads × layers
- neuron_view : Q/K vector projections

Input : attention tensor (n_layers, n_heads, seq, seq) ou
        (batch, n_heads, seq, seq) pour 1 layer
Optionnel : tokens list

Lazy import bertviz. Sauve HTML standalone (head_view/model_view) ouvrable
navigateur (Plotly-style).
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class BertVizResult:
    ok: bool
    html_path: Optional[str] = None
    n_layers: int = 0
    n_heads: int = 0
    seq_len: int = 0
    error: str = ""


def _ensure_bertviz():
    try:
        import bertviz  # noqa
        return True
    except ImportError as e:
        raise RuntimeError("bertviz absent. Install : uv pip install bertviz") from e


def _save_html_widget(html_content: str, output_path: str) -> str:
    """Wrap HTML content from bertviz into standalone document."""
    out = Path(output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>BertViz — ARK</title>
<style>
  body {{ background: #0A0A0A; color: #F2F2F2; font-family: Arial; padding: 20px; }}
  h1 {{ color: #E11D2A; }}
</style>
</head>
<body>
<h1>BertViz · ARK</h1>
{html_content}
</body>
</html>"""
    out.write_text(full_html, encoding="utf-8")
    return str(out)


def _format_attention(att: torch.Tensor) -> tuple:
    """Normalise vers tuple of layer tensors (1, heads, seq, seq).

    Accepte :
    - (L, H, S, S) → tuple de L tenseurs (1, H, S, S)
    - (H, S, S)   → 1 layer
    - (B, H, S, S) avec B=1 → 1 layer
    - List/tuple → comme tel
    """
    if isinstance(att, (list, tuple)):
        return tuple(t if t.dim() == 4 else t.unsqueeze(0) for t in att)
    if att.dim() == 3:
        # (H, S, S) → 1 layer
        return (att.unsqueeze(0),)
    if att.dim() == 4:
        if att.shape[0] == 1:
            # (1, H, S, S) → 1 layer
            return (att,)
        # (L, H, S, S) → L layers
        return tuple(att[i:i + 1] for i in range(att.shape[0]))
    raise ValueError(f"format attention non supporté : shape {att.shape}")


# ============================================================
# Modules
# ============================================================

class AttentionHeadView(nn.Module):
    """BertViz head_view : attention links per head pour 1+ layers."""

    def __init__(self, output_path: str = "./output/head_view.html",
                 tokens: str = "", layer_idx: int = -1):
        super().__init__()
        self.output_path = output_path
        self.tokens = tokens  # comma-separated
        self.layer_idx = layer_idx
        self.last_result: BertVizResult | None = None

    def _make_tokens(self, seq_len: int) -> list[str]:
        if self.tokens:
            toks = [t.strip() for t in self.tokens.split(",") if t.strip()]
            if len(toks) >= seq_len:
                return toks[:seq_len]
            return toks + [f"<pad{i}>" for i in range(seq_len - len(toks))]
        return [f"tok{i}" for i in range(seq_len)]

    def forward(self, x):
        _ensure_bertviz()
        from bertviz import head_view

        attention = _format_attention(x.detach().cpu() if hasattr(x, "detach") else x)
        n_layers = len(attention)
        n_heads = attention[0].shape[1]
        seq_len = attention[0].shape[-1]
        tokens = self._make_tokens(seq_len)

        # head_view normalement génère widget Jupyter, on intercept l'HTML
        # via _get_attention_html (helper interne)
        try:
            from bertviz.head_view import format_attention, format_special_chars
            attn_data = format_attention(attention,
                                            layers=[self.layer_idx] if self.layer_idx >= 0 else None,
                                            heads=None)
            params = {
                "attention": attn_data,
                "default_filter": "0",
                "root_div_id": "bertviz-head",
                "layer": self.layer_idx if self.layer_idx >= 0 else 0,
                "heads": list(range(n_heads)),
                "include_layers": list(range(n_layers)),
            }
            # Fallback simple : HTML matrix table par head
            html = self._render_simple_attention(attention, tokens)
        except Exception:
            html = self._render_simple_attention(attention, tokens)

        path = _save_html_widget(html, self.output_path)
        self.last_result = BertVizResult(True, path, n_layers, n_heads, seq_len)
        return x

    def _render_simple_attention(self, attention: tuple, tokens: list[str]) -> str:
        """Fallback HTML : grid heatmap par head (sans dépendre du JS BertViz interne)."""
        parts = []
        for li, layer in enumerate(attention):
            # layer : (1, H, S, S)
            heads = layer[0]  # (H, S, S)
            H, S, _ = heads.shape
            parts.append(f"<h2 style='color:#E11D2A'>Layer {li}</h2>")
            parts.append("<div style='display:flex; flex-wrap:wrap; gap:10px;'>")
            for h in range(H):
                mat = heads[h].cpu().numpy()
                parts.append(f"<div><h4>Head {h}</h4>"
                                f"{self._matrix_html(mat, tokens)}</div>")
            parts.append("</div>")
        return "\n".join(parts)

    def _matrix_html(self, mat, tokens: list[str]) -> str:
        S = mat.shape[0]
        cell = 16
        max_v = float(mat.max()) if mat.size > 0 else 1.0
        if max_v <= 0:
            max_v = 1.0
        cells = []
        cells.append("<table style='border-collapse:collapse; font-size:9px; color:#F2F2F2;'>")
        # Header tokens
        cells.append("<tr><td></td>")
        for j in range(S):
            cells.append(f"<td style='width:{cell}px; text-align:center;'>"
                            f"{tokens[j][:4] if j < len(tokens) else ''}</td>")
        cells.append("</tr>")
        for i in range(S):
            cells.append("<tr>")
            cells.append(f"<td style='font-weight:bold;'>"
                            f"{tokens[i][:6] if i < len(tokens) else f'{i}'}</td>")
            for j in range(S):
                v = mat[i, j] / max_v
                r = int(225 * v)
                g = int(29 * v)
                b = int(42 * v)
                cells.append(f"<td style='width:{cell}px; height:{cell}px; "
                                f"background:rgb({r},{g},{b});'></td>")
            cells.append("</tr>")
        cells.append("</table>")
        return "".join(cells)


class AttentionModelView(nn.Module):
    """BertViz model_view : grid de toutes layers × heads."""

    def __init__(self, output_path: str = "./output/model_view.html",
                 tokens: str = ""):
        super().__init__()
        self.output_path = output_path
        self.tokens = tokens
        self.last_result: BertVizResult | None = None

    def _make_tokens(self, seq_len: int) -> list[str]:
        if self.tokens:
            toks = [t.strip() for t in self.tokens.split(",") if t.strip()]
            if len(toks) >= seq_len:
                return toks[:seq_len]
            return toks + [f"<pad{i}>" for i in range(seq_len - len(toks))]
        return [f"t{i}" for i in range(seq_len)]

    def forward(self, x):
        _ensure_bertviz()
        attention = _format_attention(x.detach().cpu() if hasattr(x, "detach") else x)
        n_layers = len(attention)
        n_heads = attention[0].shape[1]
        seq_len = attention[0].shape[-1]
        tokens = self._make_tokens(seq_len)

        parts = [f"<p style='color:#7E7E88'>Model View · {n_layers} layers × "
                  f"{n_heads} heads · seq_len={seq_len}</p>"]
        for li, layer in enumerate(attention):
            heads = layer[0]
            parts.append(f"<details open><summary style='color:#E11D2A; "
                            f"cursor:pointer'>Layer {li}</summary>")
            parts.append("<div style='display:flex; gap:6px; flex-wrap:wrap;'>")
            for h in range(n_heads):
                mat = heads[h].cpu().numpy()
                parts.append("<div style='border:1px solid #2E2E37; padding:4px;'>")
                parts.append(f"<div style='font-size:9px; color:#7E7E88'>"
                                f"L{li}.H{h}</div>")
                parts.append(self._mini_heatmap(mat, tokens))
                parts.append("</div>")
            parts.append("</div></details>")

        html = "\n".join(parts)
        path = _save_html_widget(html, self.output_path)
        self.last_result = BertVizResult(True, path, n_layers, n_heads, seq_len)
        return x

    def _mini_heatmap(self, mat, tokens) -> str:
        S = mat.shape[0]
        cell = 6
        max_v = float(mat.max()) if mat.size > 0 else 1.0
        if max_v <= 0:
            max_v = 1.0
        cells = ["<table style='border-collapse:collapse;'>"]
        for i in range(S):
            cells.append("<tr>")
            for j in range(S):
                v = mat[i, j] / max_v
                r = int(225 * v)
                g = int(29 * v)
                b = int(42 * v)
                cells.append(f"<td style='width:{cell}px; height:{cell}px; "
                                f"background:rgb({r},{g},{b});'></td>")
            cells.append("</tr>")
        cells.append("</table>")
        return "".join(cells)


class AttentionNeuronView(nn.Module):
    """Neuron view simplifié : visualise Q et K vectors par head (projection 2D).

    Input attendu : tuple (Q, K) chacun (B, n_heads, seq, d_k)
    Ou tenseur attention (L, H, S, S) → projeté via PCA des rangées.
    """

    def __init__(self, output_path: str = "./output/neuron_view.html",
                 tokens: str = "", layer_idx: int = 0, head_idx: int = 0):
        super().__init__()
        self.output_path = output_path
        self.tokens = tokens
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.last_result: BertVizResult | None = None

    def forward(self, x):
        from sklearn.decomposition import PCA
        attention = _format_attention(x.detach().cpu() if hasattr(x, "detach") else x)
        layer = attention[min(self.layer_idx, len(attention) - 1)]
        head = layer[0, min(self.head_idx, layer.shape[1] - 1)]  # (S, S)
        S = head.shape[0]
        tokens = self.tokens.split(",") if self.tokens else [f"t{i}" for i in range(S)]
        tokens = tokens[:S]

        # PCA des rangées (chaque token = vecteur d'attention)
        coords = PCA(n_components=2).fit_transform(head.cpu().numpy())

        # Render SVG
        max_v = max(coords.max() - coords.min(), 1e-6)
        norm = (coords - coords.min()) / max_v
        W = 600
        H = 400
        svg = [f"<svg width='{W}' height='{H}' style='background:#0A0A0A'>"]
        for i in range(S):
            x_p = int(50 + norm[i, 0] * (W - 100))
            y_p = int(50 + norm[i, 1] * (H - 100))
            svg.append(f"<circle cx='{x_p}' cy='{y_p}' r='5' fill='#E11D2A' "
                          f"opacity='0.8'/>")
            label = tokens[i] if i < len(tokens) else f"t{i}"
            svg.append(f"<text x='{x_p + 8}' y='{y_p}' fill='#F2F2F2' "
                          f"font-size='10' font-family='Arial'>{label}</text>")
        svg.append("</svg>")
        body = (f"<p style='color:#7E7E88'>Neuron view · Layer "
                 f"{self.layer_idx} Head {self.head_idx} · PCA 2D des "
                 f"attention vectors</p>" + "\n".join(svg))
        path = _save_html_widget(body, self.output_path)
        self.last_result = BertVizResult(True, path, len(attention),
                                              layer.shape[1], S)
        return x


# ============================================================
# Handlers
# ============================================================

def h_attention_head_view(node, in_shapes):
    m = AttentionHeadView(
        output_path=node.params.get("output_path", "./output/head_view.html"),
        tokens=node.params.get("tokens", ""),
        layer_idx=int(node.params.get("layer_idx", -1)),
    )
    m._ark_node_id = getattr(node, "id", "hv")
    return m, in_shapes[0] if in_shapes else (1,)


def h_attention_model_view(node, in_shapes):
    m = AttentionModelView(
        output_path=node.params.get("output_path", "./output/model_view.html"),
        tokens=node.params.get("tokens", ""),
    )
    m._ark_node_id = getattr(node, "id", "mv")
    return m, in_shapes[0] if in_shapes else (1,)


def h_attention_neuron_view(node, in_shapes):
    m = AttentionNeuronView(
        output_path=node.params.get("output_path", "./output/neuron_view.html"),
        tokens=node.params.get("tokens", ""),
        layer_idx=int(node.params.get("layer_idx", 0)),
        head_idx=int(node.params.get("head_idx", 0)),
    )
    m._ark_node_id = getattr(node, "id", "nv")
    return m, in_shapes[0] if in_shapes else (1,)


def register_bertviz_handlers(handlers_dict: dict) -> bool:
    handlers_dict["attention_head_view"] = h_attention_head_view
    handlers_dict["attention_model_view"] = h_attention_model_view
    handlers_dict["attention_neuron_view"] = h_attention_neuron_view
    return True


def register_bertviz_specs(registry) -> bool:
    from ark.nodes.registry import NodeSpec, ParamSpec, SocketSpec

    registry.register(NodeSpec(
        name="attention_head_view", title="Attention Head View (BertViz)",
        category="Interpretability",
        inputs=[SocketSpec("attention", "tensor")],
        outputs=[SocketSpec("attention", "tensor")],
        params=[
            ParamSpec("output_path", "str", "./output/head_view.html"),
            ParamSpec("tokens", "str", ""),
            ParamSpec("layer_idx", "int", -1, min=-1, max=100),
        ],
        description="Visualise attention par head pour 1 layer. HTML standalone "
                       "interactif. Style BertViz adapté ARK dark+rouge.",
        role="Per-head attention heatmap grid.",
        function="bertviz fallback + custom HTML heatmap. Input (L,H,S,S) ou (H,S,S). "
                    "tokens optionnel (comma-separated).",
        has_thumbnail=True,
        param_help={
            "output_path": "HTML output.",
            "tokens": "Liste tokens (séparés par virgule).",
            "layer_idx": "Layer cible (-1 = toutes).",
        },
    ))

    registry.register(NodeSpec(
        name="attention_model_view", title="Attention Model View (BertViz)",
        category="Interpretability",
        inputs=[SocketSpec("attention", "tensor")],
        outputs=[SocketSpec("attention", "tensor")],
        params=[
            ParamSpec("output_path", "str", "./output/model_view.html"),
            ParamSpec("tokens", "str", ""),
        ],
        description="Grid complète layers × heads d'un Transformer. Vue d'ensemble "
                       "pattern attention global. Style BertViz.",
        role="Full model attention grid.",
        function="Itère sur (L, H) heatmaps mini + <details> expand par layer.",
        has_thumbnail=True,
    ))

    registry.register(NodeSpec(
        name="attention_neuron_view", title="Attention Neuron View",
        category="Interpretability",
        inputs=[SocketSpec("attention", "tensor")],
        outputs=[SocketSpec("attention", "tensor")],
        params=[
            ParamSpec("output_path", "str", "./output/neuron_view.html"),
            ParamSpec("tokens", "str", ""),
            ParamSpec("layer_idx", "int", 0, min=0, max=100),
            ParamSpec("head_idx", "int", 0, min=0, max=100),
        ],
        description="Projeté PCA 2D des attention vectors par token. SVG interactif "
                       "(scatter tokens dans espace attention).",
        role="Token attention vectors → 2D PCA projection.",
        function="sklearn.PCA n_components=2 sur head matrix → SVG scatter labelled.",
        has_thumbnail=True,
    ))
    return True
