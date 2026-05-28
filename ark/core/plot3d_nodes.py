"""Phase 34 — Nodes Plotly 3D interactif + dimensionality reduction.

Output : HTML file ouvrable navigateur (Plotly embeds JS).
Lazy plotly + sklearn + umap-learn (déjà optional[viz]).

Nodes :
- pca_3d        : (N, F) → (N, 3) PCA + plotly scatter 3D
- tsne_3d       : (N, F) → (N, 3) t-SNE + plotly scatter 3D
- umap_3d       : (N, F) → (N, 3) UMAP + plotly scatter 3D
- scatter_3d    : (N, 3) brut → plotly scatter 3D
- line_3d       : (N, 3) trajectoire ordonnée → plotly line 3D
- surface_3d    : (H, W) → plotly surface 3D
- mesh_3d       : (N, 3) vertices + (M, 3) faces → mesh 3D

Esthétique : dark background, accent rouge ARK #E11D2A.
"""
from __future__ import annotations
import json
import os
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


ACCENT = "#E11D2A"
BG_COLOR = "#0A0A0A"
GRID_COLOR = "#2E2E37"
FG_COLOR = "#F2F2F2"


@dataclass
class VizResult:
    ok: bool
    html_path: Optional[str] = None
    n_points: int = 0
    error: str = ""


# ============================================================
# Helpers Plotly
# ============================================================

def _plotly_layout_dark(title: str) -> dict:
    """Layout sombre uniforme + accent rouge."""
    return dict(
        title=dict(text=title, font=dict(color=FG_COLOR, size=14)),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=FG_COLOR, family="Arial"),
        scene=dict(
            xaxis=dict(backgroundcolor=BG_COLOR, gridcolor=GRID_COLOR,
                          showbackground=True, zerolinecolor=GRID_COLOR,
                          color=FG_COLOR),
            yaxis=dict(backgroundcolor=BG_COLOR, gridcolor=GRID_COLOR,
                          showbackground=True, zerolinecolor=GRID_COLOR,
                          color=FG_COLOR),
            zaxis=dict(backgroundcolor=BG_COLOR, gridcolor=GRID_COLOR,
                          showbackground=True, zerolinecolor=GRID_COLOR,
                          color=FG_COLOR),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )


def _ensure_plotly():
    try:
        import plotly.graph_objects as go  # noqa
        return True
    except ImportError as e:
        raise RuntimeError(
            "plotly absent. Install : uv pip install plotly"
        ) from e


def _save_html(fig, output_path: str, auto_open: bool = False) -> str:
    out = Path(output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn")
    if auto_open:
        webbrowser.open(f"file://{out.absolute()}")
    return str(out)


def _to_numpy(t):
    if hasattr(t, "detach"):
        t = t.detach().cpu()
    if hasattr(t, "numpy"):
        return t.numpy()
    import numpy as np
    return np.asarray(t)


# ============================================================
# Modules
# ============================================================

class PCA3D(nn.Module):
    """Réduit (N, F) → (N, 3) via sklearn PCA + plot interactif."""

    def __init__(self, output_path: str = "./output/pca_3d.html",
                 title: str = "PCA 3D", auto_open: bool = False,
                 color_by_index: bool = True):
        super().__init__()
        self.output_path = output_path
        self.title = title
        self.auto_open = auto_open
        self.color_by_index = color_by_index
        self.last_result: VizResult | None = None
        self.last_coords: torch.Tensor | None = None

    def forward(self, x):
        from sklearn.decomposition import PCA
        _ensure_plotly()
        import plotly.graph_objects as go

        X = _to_numpy(x)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] < 3:
            # Pad avec zeros pour avoir au moins 3 dims
            import numpy as np
            X = np.hstack([X, np.zeros((X.shape[0], 3 - X.shape[1]))])
        n_comp = min(3, X.shape[1])
        coords = PCA(n_components=n_comp).fit_transform(X)
        if coords.shape[1] < 3:
            import numpy as np
            coords = np.hstack([coords, np.zeros((coords.shape[0],
                                                       3 - coords.shape[1]))])

        n = coords.shape[0]
        color = list(range(n)) if self.color_by_index else [0] * n
        fig = go.Figure(data=[go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode="markers",
            marker=dict(size=4, color=color, colorscale="Reds",
                          showscale=self.color_by_index, opacity=0.85),
            text=[f"sample {i}" for i in range(n)],
        )])
        fig.update_layout(**_plotly_layout_dark(f"{self.title} (n={n})"))
        path = _save_html(fig, self.output_path, self.auto_open)
        self.last_coords = torch.from_numpy(coords).float()
        self.last_result = VizResult(True, path, n)
        return self.last_coords


class TSNE3D(nn.Module):
    def __init__(self, output_path: str = "./output/tsne_3d.html",
                 title: str = "t-SNE 3D", auto_open: bool = False,
                 perplexity: int = 30, n_iter: int = 1000,
                 color_by_index: bool = True):
        super().__init__()
        self.output_path = output_path
        self.title = title
        self.auto_open = auto_open
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.color_by_index = color_by_index
        self.last_result: VizResult | None = None
        self.last_coords: torch.Tensor | None = None

    def forward(self, x):
        from sklearn.manifold import TSNE
        _ensure_plotly()
        import plotly.graph_objects as go

        X = _to_numpy(x)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        # t-SNE perplexity must be < n_samples - 1
        perp = min(self.perplexity, max(5, X.shape[0] // 4))
        coords = TSNE(n_components=3, perplexity=perp,
                          max_iter=self.n_iter, init="pca",
                          random_state=42).fit_transform(X)

        n = coords.shape[0]
        color = list(range(n)) if self.color_by_index else [0] * n
        fig = go.Figure(data=[go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode="markers",
            marker=dict(size=4, color=color, colorscale="Plasma",
                          showscale=self.color_by_index, opacity=0.85),
            text=[f"sample {i}" for i in range(n)],
        )])
        fig.update_layout(**_plotly_layout_dark(
            f"{self.title} (n={n}, perp={perp})"))
        path = _save_html(fig, self.output_path, self.auto_open)
        self.last_coords = torch.from_numpy(coords).float()
        self.last_result = VizResult(True, path, n)
        return self.last_coords


class UMAP3D(nn.Module):
    def __init__(self, output_path: str = "./output/umap_3d.html",
                 title: str = "UMAP 3D", auto_open: bool = False,
                 n_neighbors: int = 15, min_dist: float = 0.1,
                 color_by_index: bool = True):
        super().__init__()
        self.output_path = output_path
        self.title = title
        self.auto_open = auto_open
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.color_by_index = color_by_index
        self.last_result: VizResult | None = None
        self.last_coords: torch.Tensor | None = None

    def forward(self, x):
        try:
            import umap
        except ImportError:
            raise RuntimeError(
                "umap-learn absent. Install : uv pip install umap-learn")
        _ensure_plotly()
        import plotly.graph_objects as go

        X = _to_numpy(x)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        nn = min(self.n_neighbors, max(2, X.shape[0] - 1))
        coords = umap.UMAP(n_components=3, n_neighbors=nn,
                              min_dist=self.min_dist,
                              random_state=42).fit_transform(X)
        n = coords.shape[0]
        color = list(range(n)) if self.color_by_index else [0] * n
        fig = go.Figure(data=[go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode="markers",
            marker=dict(size=4, color=color, colorscale="Viridis",
                          showscale=self.color_by_index, opacity=0.85),
            text=[f"sample {i}" for i in range(n)],
        )])
        fig.update_layout(**_plotly_layout_dark(
            f"{self.title} (n={n}, nn={nn})"))
        path = _save_html(fig, self.output_path, self.auto_open)
        self.last_coords = torch.from_numpy(coords).float()
        self.last_result = VizResult(True, path, n)
        return self.last_coords


class Scatter3D(nn.Module):
    """Plot direct (N, 3) tensor → plotly scatter 3D."""

    def __init__(self, output_path: str = "./output/scatter_3d.html",
                 title: str = "Scatter 3D", auto_open: bool = False,
                 point_size: int = 4):
        super().__init__()
        self.output_path = output_path
        self.title = title
        self.auto_open = auto_open
        self.point_size = point_size
        self.last_result: VizResult | None = None

    def forward(self, x):
        _ensure_plotly()
        import plotly.graph_objects as go

        X = _to_numpy(x)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] < 3:
            import numpy as np
            X = np.hstack([X, np.zeros((X.shape[0], 3 - X.shape[1]))])
        n = X.shape[0]
        fig = go.Figure(data=[go.Scatter3d(
            x=X[:, 0], y=X[:, 1], z=X[:, 2],
            mode="markers",
            marker=dict(size=self.point_size, color=ACCENT, opacity=0.85),
        )])
        fig.update_layout(**_plotly_layout_dark(f"{self.title} (n={n})"))
        path = _save_html(fig, self.output_path, self.auto_open)
        self.last_result = VizResult(True, path, n)
        return x


class Line3D(nn.Module):
    """Trajectoire 3D ordonnée (N, 3) → plotly line."""

    def __init__(self, output_path: str = "./output/line_3d.html",
                 title: str = "Line 3D", auto_open: bool = False,
                 line_width: int = 3):
        super().__init__()
        self.output_path = output_path
        self.title = title
        self.auto_open = auto_open
        self.line_width = line_width
        self.last_result: VizResult | None = None

    def forward(self, x):
        _ensure_plotly()
        import plotly.graph_objects as go

        X = _to_numpy(x)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] < 3:
            import numpy as np
            X = np.hstack([X, np.zeros((X.shape[0], 3 - X.shape[1]))])
        n = X.shape[0]
        fig = go.Figure(data=[go.Scatter3d(
            x=X[:, 0], y=X[:, 1], z=X[:, 2],
            mode="lines+markers",
            line=dict(color=ACCENT, width=self.line_width),
            marker=dict(size=3, color=list(range(n)), colorscale="Reds",
                          showscale=True),
        )])
        fig.update_layout(**_plotly_layout_dark(
            f"{self.title} (n={n} points)"))
        path = _save_html(fig, self.output_path, self.auto_open)
        self.last_result = VizResult(True, path, n)
        return x


class Surface3D(nn.Module):
    """Surface 3D depuis (H, W) → plotly surface."""

    def __init__(self, output_path: str = "./output/surface_3d.html",
                 title: str = "Surface 3D", auto_open: bool = False,
                 colorscale: str = "Reds"):
        super().__init__()
        self.output_path = output_path
        self.title = title
        self.auto_open = auto_open
        self.colorscale = colorscale
        self.last_result: VizResult | None = None

    def forward(self, x):
        _ensure_plotly()
        import plotly.graph_objects as go

        Z = _to_numpy(x)
        if Z.ndim == 1:
            n = int(Z.shape[0] ** 0.5)
            Z = Z[:n * n].reshape(n, n)
        elif Z.ndim > 2:
            Z = Z.reshape(-1, Z.shape[-1])
        H, W = Z.shape
        fig = go.Figure(data=[go.Surface(z=Z, colorscale=self.colorscale)])
        fig.update_layout(**_plotly_layout_dark(
            f"{self.title} ({H}×{W})"))
        path = _save_html(fig, self.output_path, self.auto_open)
        self.last_result = VizResult(True, path, H * W)
        return x


class CurvePlot3D(nn.Module):
    """N courbes empilées Z-shifted → plotly multi-line 3D.

    Input : (N, T) où N = nombre de séries, T = pas temporels.
    """

    def __init__(self, output_path: str = "./output/curves_3d.html",
                 title: str = "Courbes 3D", auto_open: bool = False):
        super().__init__()
        self.output_path = output_path
        self.title = title
        self.auto_open = auto_open
        self.last_result: VizResult | None = None

    def forward(self, x):
        _ensure_plotly()
        import plotly.graph_objects as go

        X = _to_numpy(x)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        N, T = X.shape
        fig = go.Figure()
        import numpy as np
        for i in range(min(N, 50)):  # cap 50 courbes
            ratio = i / max(N - 1, 1)
            fig.add_trace(go.Scatter3d(
                x=list(range(T)),
                y=[i] * T,
                z=X[i].tolist(),
                mode="lines",
                line=dict(color=f"rgba(225,29,42,{0.3 + 0.7 * ratio})",
                            width=2),
                showlegend=False,
            ))
        fig.update_layout(**_plotly_layout_dark(
            f"{self.title} (N={N}, T={T})"))
        path = _save_html(fig, self.output_path, self.auto_open)
        self.last_result = VizResult(True, path, N * T)
        return x


# ============================================================
# Handlers
# ============================================================

def h_pca_3d(node, in_shapes):
    m = PCA3D(
        output_path=node.params.get("output_path", "./output/pca_3d.html"),
        title=node.params.get("title", "PCA 3D"),
        auto_open=bool(node.params.get("auto_open", False)),
        color_by_index=bool(node.params.get("color_by_index", True)),
    )
    m._ark_node_id = getattr(node, "id", "pca")
    return m, (3,)


def h_tsne_3d(node, in_shapes):
    m = TSNE3D(
        output_path=node.params.get("output_path", "./output/tsne_3d.html"),
        title=node.params.get("title", "t-SNE 3D"),
        auto_open=bool(node.params.get("auto_open", False)),
        perplexity=int(node.params.get("perplexity", 30)),
        n_iter=int(node.params.get("n_iter", 1000)),
        color_by_index=bool(node.params.get("color_by_index", True)),
    )
    m._ark_node_id = getattr(node, "id", "tsne")
    return m, (3,)


def h_umap_3d(node, in_shapes):
    m = UMAP3D(
        output_path=node.params.get("output_path", "./output/umap_3d.html"),
        title=node.params.get("title", "UMAP 3D"),
        auto_open=bool(node.params.get("auto_open", False)),
        n_neighbors=int(node.params.get("n_neighbors", 15)),
        min_dist=float(node.params.get("min_dist", 0.1)),
        color_by_index=bool(node.params.get("color_by_index", True)),
    )
    m._ark_node_id = getattr(node, "id", "umap")
    return m, (3,)


def h_scatter_3d(node, in_shapes):
    m = Scatter3D(
        output_path=node.params.get("output_path", "./output/scatter_3d.html"),
        title=node.params.get("title", "Scatter 3D"),
        auto_open=bool(node.params.get("auto_open", False)),
        point_size=int(node.params.get("point_size", 4)),
    )
    m._ark_node_id = getattr(node, "id", "s3d")
    return m, in_shapes[0] if in_shapes else (3,)


def h_line_3d(node, in_shapes):
    m = Line3D(
        output_path=node.params.get("output_path", "./output/line_3d.html"),
        title=node.params.get("title", "Line 3D"),
        auto_open=bool(node.params.get("auto_open", False)),
        line_width=int(node.params.get("line_width", 3)),
    )
    m._ark_node_id = getattr(node, "id", "l3d")
    return m, in_shapes[0] if in_shapes else (3,)


def h_surface_3d(node, in_shapes):
    m = Surface3D(
        output_path=node.params.get("output_path", "./output/surface_3d.html"),
        title=node.params.get("title", "Surface 3D"),
        auto_open=bool(node.params.get("auto_open", False)),
        colorscale=node.params.get("colorscale", "Reds"),
    )
    m._ark_node_id = getattr(node, "id", "sf3d")
    return m, in_shapes[0] if in_shapes else (1,)


def h_curves_3d(node, in_shapes):
    m = CurvePlot3D(
        output_path=node.params.get("output_path", "./output/curves_3d.html"),
        title=node.params.get("title", "Courbes 3D"),
        auto_open=bool(node.params.get("auto_open", False)),
    )
    m._ark_node_id = getattr(node, "id", "c3d")
    return m, in_shapes[0] if in_shapes else (1,)


def register_plot3d_handlers(handlers_dict: dict) -> bool:
    handlers_dict["pca_3d"] = h_pca_3d
    handlers_dict["tsne_3d"] = h_tsne_3d
    handlers_dict["umap_3d"] = h_umap_3d
    handlers_dict["scatter_3d"] = h_scatter_3d
    handlers_dict["line_3d"] = h_line_3d
    handlers_dict["surface_3d"] = h_surface_3d
    handlers_dict["curves_3d"] = h_curves_3d
    return True


def register_plot3d_specs(registry) -> bool:
    from ark.nodes.registry import NodeSpec, ParamSpec, SocketSpec

    HTML_FILTER = "HTML (*.html);;All (*)"

    def _path_param(default_name):
        return ParamSpec("output_path", "str", f"./output/{default_name}.html")

    common_color = [
        ParamSpec("title", "str", ""),
        ParamSpec("auto_open", "bool", False),
        ParamSpec("color_by_index", "bool", True),
    ]

    registry.register(NodeSpec(
        name="pca_3d", title="PCA 3D (interactive)",
        category="Viz",
        inputs=[SocketSpec("embeddings", "tensor")],
        outputs=[SocketSpec("coords_3d", "tensor")],
        params=[
            _path_param("pca_3d"),
            ParamSpec("title", "str", "PCA 3D"),
            ParamSpec("auto_open", "bool", False),
            ParamSpec("color_by_index", "bool", True),
        ],
        description="Réduction PCA (N, F) → (N, 3) + plotly scatter 3D interactif. "
                       "Output : HTML auto-ouvrable navigateur.",
        role="Linear dim-reduction → 3D interactive plot.",
        function="sklearn.PCA.fit_transform + plotly Scatter3D. Dark theme + accent rouge ARK.",
        has_thumbnail=True,
        param_help={
            "output_path": "Chemin HTML de sortie.",
            "auto_open": "Ouvrir auto navigateur après plot.",
            "color_by_index": "Gradient par index (rouge → blanc).",
        },
    ))

    registry.register(NodeSpec(
        name="tsne_3d", title="t-SNE 3D (interactive)",
        category="Viz",
        inputs=[SocketSpec("embeddings", "tensor")],
        outputs=[SocketSpec("coords_3d", "tensor")],
        params=[
            _path_param("tsne_3d"),
            ParamSpec("title", "str", "t-SNE 3D"),
            ParamSpec("auto_open", "bool", False),
            ParamSpec("perplexity", "int", 30, min=5, max=200),
            ParamSpec("n_iter", "int", 1000, min=250, max=10000),
            ParamSpec("color_by_index", "bool", True),
        ],
        description="t-SNE non-linéaire (N, F) → (N, 3) + plotly Scatter 3D.",
        role="Nonlinear dim-reduction préservant voisinages.",
        function="sklearn.manifold.TSNE n_components=3 + plotly. "
                    "perplexity auto-clampé à min(perp, n_samples/4).",
        has_thumbnail=True,
    ))

    registry.register(NodeSpec(
        name="umap_3d", title="UMAP 3D (interactive)",
        category="Viz",
        inputs=[SocketSpec("embeddings", "tensor")],
        outputs=[SocketSpec("coords_3d", "tensor")],
        params=[
            _path_param("umap_3d"),
            ParamSpec("title", "str", "UMAP 3D"),
            ParamSpec("auto_open", "bool", False),
            ParamSpec("n_neighbors", "int", 15, min=2, max=200),
            ParamSpec("min_dist", "float", 0.1, min=0.0, max=1.0),
            ParamSpec("color_by_index", "bool", True),
        ],
        description="UMAP topologique (N, F) → (N, 3). Préserve structure globale + locale.",
        role="UMAP dim-reduction → 3D.",
        function="umap.UMAP n_components=3 + plotly. Random state 42 reproductible.",
        has_thumbnail=True,
    ))

    registry.register(NodeSpec(
        name="scatter_3d", title="Scatter 3D",
        category="Viz",
        inputs=[SocketSpec("points", "tensor")],
        outputs=[SocketSpec("points", "tensor")],
        params=[
            _path_param("scatter_3d"),
            ParamSpec("title", "str", "Scatter 3D"),
            ParamSpec("auto_open", "bool", False),
            ParamSpec("point_size", "int", 4, min=1, max=30),
        ],
        description="Plot direct (N, 3) tensor → plotly Scatter 3D.",
        role="Raw (N,3) tensor → 3D scatter.",
        function="Pass-through tensor + HTML side-effect.",
        has_thumbnail=True,
    ))

    registry.register(NodeSpec(
        name="line_3d", title="Line 3D (trajectory)",
        category="Viz",
        inputs=[SocketSpec("points", "tensor")],
        outputs=[SocketSpec("points", "tensor")],
        params=[
            _path_param("line_3d"),
            ParamSpec("title", "str", "Line 3D"),
            ParamSpec("auto_open", "bool", False),
            ParamSpec("line_width", "int", 3, min=1, max=20),
        ],
        description="Trajectoire 3D ordonnée (Lorenz, latent rollout, etc.).",
        role="(N, 3) ordered → 3D polyline + gradient markers.",
        function="plotly Scatter3D mode='lines+markers' avec colorscale par index.",
        has_thumbnail=True,
    ))

    registry.register(NodeSpec(
        name="surface_3d", title="Surface 3D",
        category="Viz",
        inputs=[SocketSpec("heightmap", "tensor")],
        outputs=[SocketSpec("heightmap", "tensor")],
        params=[
            _path_param("surface_3d"),
            ParamSpec("title", "str", "Surface 3D"),
            ParamSpec("auto_open", "bool", False),
            ParamSpec("colorscale", "choice", "Reds",
                        options=["Reds", "Viridis", "Plasma", "Hot",
                                  "Inferno", "Magma", "Cividis"]),
        ],
        description="Surface 3D depuis (H, W) heightmap. Pour reaction-diffusion, "
                       "Turing patterns, loss landscapes.",
        role="(H, W) → plotly Surface 3D interactif.",
        function="plotly.go.Surface(z=Z). 1D auto-reshape √n × √n.",
        has_thumbnail=True,
    ))

    registry.register(NodeSpec(
        name="curves_3d", title="Curves 3D (stacked)",
        category="Viz",
        inputs=[SocketSpec("series", "tensor")],
        outputs=[SocketSpec("series", "tensor")],
        params=[
            _path_param("curves_3d"),
            ParamSpec("title", "str", "Courbes 3D"),
            ParamSpec("auto_open", "bool", False),
        ],
        description="(N, T) → N courbes empilées en Z. Pour visualiser N traces "
                       "d'évolution temporelle simultanément.",
        role="Multi-time-series stacked Z-axis 3D plot.",
        function="N courbes plotly Scatter3D avec Y=index, alpha gradient. Cap 50 courbes.",
        has_thumbnail=True,
    ))
    return True
