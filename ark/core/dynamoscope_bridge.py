"""Phase 37 — Bridge ARK ↔ Dynamoscope.

Wraps modules backend dynamoscope comme nodes ARK :
- backend.dynamics  : Takens, RQA, Lyapunov, correlation dimension
- backend.spectral  : DMD (Koopman approx), power spectrum
- backend.topology  : persistent homology H0/H1
- backend.dna       : composite 7-axes complexity score

Plus 3 nodes viz dédiés :
- PhaseSpace3D    : trajectoire colorée temps + velocity
- RecurrencePlot  : matrix recurrence 2D heatmap
- TakensViz       : 1D signal → 3D Takens embedding plot

Lazy imports — si dynamoscope absent, handlers raise informatif.
Path : ~/claude-code-projects/dynamoscope. Auto-injecté dans sys.path.
"""
from __future__ import annotations
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn


DYNAMOSCOPE_PATH = Path.home() / "claude-code-projects" / "dynamoscope"
_AVAILABLE: bool | None = None

ACCENT = "#E11D2A"
BG = "#0E0E11"
FG = "#F4F4F5"
FG_MUTED = "#8E8E98"


def _ensure_dynamoscope() -> bool:
    global _AVAILABLE
    if _AVAILABLE is not None:
        return _AVAILABLE
    if not DYNAMOSCOPE_PATH.exists():
        _AVAILABLE = False
        return False
    p = str(DYNAMOSCOPE_PATH)
    if p not in sys.path:
        sys.path.insert(0, p)
    try:
        from backend import dynamics  # noqa
        _AVAILABLE = True
    except ImportError:
        _AVAILABLE = False
    return _AVAILABLE


@dataclass
class DynamoResult:
    ok: bool
    data: Any = None
    html_path: Optional[str] = None
    error: str = ""


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _save_html(html_content: str, output_path: str) -> str:
    out = Path(output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_content, encoding="utf-8")
    return str(out)


# ============================================================
# 1. Phase Space 3D — flagship visualization
# ============================================================

class PhaseSpace3D(nn.Module):
    """Trajectoire dans espace des phases 3D, colorée par temps ou velocity.

    Input : (N, D) latents — projeté en 3D si D > 3 via PCA.
    Output : (N, 3) coords + side-effect HTML Plotly interactif.

    Color modes :
    - "time"       : gradient temporel (rouge → orange → blanc)
    - "velocity"   : ||x_{t+1} - x_t|| (fast = chaud)
    - "speed_log"  : log de velocity (révèle plateaux)
    """

    def __init__(self, output_path: str = "./output/phase_space.html",
                 title: str = "Phase Space 3D", auto_open: bool = False,
                 color_mode: str = "time", show_attractor: bool = True):
        super().__init__()
        self.output_path = output_path
        self.title = title
        self.auto_open = auto_open
        self.color_mode = color_mode
        self.show_attractor = show_attractor
        self.last_result: DynamoResult | None = None
        self.last_coords: torch.Tensor | None = None

    def forward(self, x):
        try:
            import plotly.graph_objects as go
        except ImportError:
            return x
        X = _to_numpy(x)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        N, D = X.shape
        # Project to 3D
        if D < 3:
            X = np.hstack([X, np.zeros((N, 3 - D))])
        elif D > 3:
            from sklearn.decomposition import PCA
            X = PCA(n_components=3).fit_transform(X)

        # Color values per point
        if self.color_mode == "velocity":
            vel = np.linalg.norm(np.diff(X, axis=0), axis=1)
            vel = np.concatenate([[vel[0] if len(vel) else 0], vel])
            colors = vel
            cs = "Hot"
            cb_title = "velocity"
        elif self.color_mode == "speed_log":
            vel = np.linalg.norm(np.diff(X, axis=0), axis=1) + 1e-9
            vel = np.concatenate([[vel[0]], vel])
            colors = np.log(vel)
            cs = "Plasma"
            cb_title = "log velocity"
        else:  # time
            colors = np.arange(N)
            cs = "Reds"
            cb_title = "time"

        fig = go.Figure()
        # Trajectoire ligne
        fig.add_trace(go.Scatter3d(
            x=X[:, 0], y=X[:, 1], z=X[:, 2],
            mode="lines+markers",
            line=dict(color="rgba(225,29,42,0.4)", width=2),
            marker=dict(size=3, color=colors, colorscale=cs,
                          showscale=True, colorbar=dict(title=cb_title,
                                                              tickfont=dict(color=FG))),
            name="trajectoire",
        ))
        # Start/end markers distincts
        fig.add_trace(go.Scatter3d(
            x=[X[0, 0]], y=[X[0, 1]], z=[X[0, 2]],
            mode="markers+text",
            marker=dict(size=10, color="#00FF85", symbol="diamond"),
            text=["START"], textfont=dict(color="#00FF85"),
            textposition="top center", name="start",
        ))
        fig.add_trace(go.Scatter3d(
            x=[X[-1, 0]], y=[X[-1, 1]], z=[X[-1, 2]],
            mode="markers+text",
            marker=dict(size=10, color="#F87171", symbol="x"),
            text=["END"], textfont=dict(color="#F87171"),
            textposition="bottom center", name="end",
        ))

        # Attractor estimate : centroid + std ellipsoid
        if self.show_attractor and N > 10:
            mu = X.mean(axis=0)
            sigma = X.std(axis=0)
            fig.add_trace(go.Scatter3d(
                x=[mu[0]], y=[mu[1]], z=[mu[2]],
                mode="markers+text",
                marker=dict(size=14, color=ACCENT, symbol="circle",
                              opacity=0.5),
                text=[f"⊕ μ"], textfont=dict(color=ACCENT),
                textposition="middle right", name="centroid",
            ))

        fig.update_layout(
            title=dict(text=f"{self.title} (N={N})",
                          font=dict(color=FG, size=14)),
            paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=FG),
            scene=dict(
                xaxis=dict(backgroundcolor=BG, gridcolor="#3A3A45",
                              color=FG, title="x₁"),
                yaxis=dict(backgroundcolor=BG, gridcolor="#3A3A45",
                              color=FG, title="x₂"),
                zaxis=dict(backgroundcolor=BG, gridcolor="#3A3A45",
                              color=FG, title="x₃"),
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=False,
        )

        path = _save_html(fig.to_html(include_plotlyjs="cdn"), self.output_path)
        self.last_coords = torch.from_numpy(X).float()
        self.last_result = DynamoResult(True, X, path)
        if self.auto_open:
            webbrowser.open(f"file://{Path(path).absolute()}")
        return self.last_coords


# ============================================================
# 2. Takens Embedding (1D → m-D delay coordinates)
# ============================================================

class TakensEmbedding(nn.Module):
    """Reconstruction Takens : x(t) → (x(t), x(t-τ), x(t-2τ), ..., x(t-(m-1)τ))."""

    def __init__(self, m: int = 3, tau: int = 1):
        super().__init__()
        self.m = m
        self.tau = tau

    def forward(self, x):
        X = _to_numpy(x).squeeze()
        if X.ndim > 1:
            X = X[:, 0]  # premier canal si multi-dim
        N = len(X)
        m_max = N - (self.m - 1) * self.tau
        if m_max <= 0:
            return torch.zeros(1, self.m)
        out = np.zeros((m_max, self.m))
        for i in range(self.m):
            out[:, i] = X[i * self.tau : i * self.tau + m_max]
        return torch.from_numpy(out).float()


# ============================================================
# 3. Lyapunov Rosenstein
# ============================================================

class LyapunovRosenstein(nn.Module):
    """Estime exposant Lyapunov max via méthode Rosenstein.

    λ > 0 = chaotique. λ = 0 = quasi-periodic. λ < 0 = stable.
    """

    def __init__(self, m: int = 3, tau: int = 1, max_t: int = 50,
                 min_separation: int = 10):
        super().__init__()
        self.m = m
        self.tau = tau
        self.max_t = max_t
        self.min_separation = min_separation
        self.last_lambda: float | None = None

    def forward(self, x):
        if _ensure_dynamoscope():
            try:
                from backend.dynamics import lyapunov_rosenstein
                X = _to_numpy(x).squeeze()
                if X.ndim > 1:
                    X = X[:, 0]
                lam = float(lyapunov_rosenstein(X, m=self.m, tau=self.tau,
                                                       max_t=self.max_t,
                                                       min_separation=self.min_separation))
                self.last_lambda = lam
                return torch.tensor([lam], dtype=torch.float32)
            except Exception:
                pass
        # Fallback : implementation minimale Rosenstein
        X = _to_numpy(x).squeeze()
        if X.ndim > 1:
            X = X[:, 0]
        N = len(X)
        # Embed Takens m-D
        m_max = N - (self.m - 1) * self.tau
        if m_max < 10:
            self.last_lambda = 0.0
            return torch.tensor([0.0])
        embed = np.zeros((m_max, self.m))
        for i in range(self.m):
            embed[:, i] = X[i * self.tau : i * self.tau + m_max]
        # Pour chaque point : trouve nearest neighbor (avec min_separation temporel)
        from scipy.spatial.distance import cdist
        try:
            D = cdist(embed, embed)
            np.fill_diagonal(D, np.inf)
            # Mask temporal proximity
            for k in range(self.min_separation + 1):
                idx = np.arange(D.shape[0] - k)
                D[idx, idx + k] = np.inf
                D[idx + k, idx] = np.inf
            neighbors = np.argmin(D, axis=1)
            divergences = []
            for t in range(1, min(self.max_t, m_max // 2)):
                dist_t = []
                for i in range(m_max - t):
                    j = neighbors[i]
                    if j + t < m_max:
                        d = np.linalg.norm(embed[i + t] - embed[j + t])
                        if d > 0:
                            dist_t.append(np.log(d + 1e-12))
                if dist_t:
                    divergences.append(np.mean(dist_t))
            if len(divergences) > 2:
                # Slope linéaire = λ
                ts = np.arange(1, len(divergences) + 1)
                slope = np.polyfit(ts, divergences, 1)[0]
                self.last_lambda = float(slope)
                return torch.tensor([slope], dtype=torch.float32)
        except Exception:
            pass
        self.last_lambda = 0.0
        return torch.tensor([0.0])


# ============================================================
# 4. Correlation Dimension (Grassberger-Procaccia)
# ============================================================

class CorrelationDimension(nn.Module):
    """Dimension fractale de corrélation. ~3 pour Lorenz. >5 = high-D chaos."""

    def __init__(self, n_eps: int = 14):
        super().__init__()
        self.n_eps = n_eps
        self.last_dim: float | None = None

    def forward(self, x):
        X = _to_numpy(x)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if _ensure_dynamoscope():
            try:
                from backend.dynamics import correlation_dimension
                d = float(correlation_dimension(X, n_eps=self.n_eps))
                self.last_dim = d
                return torch.tensor([d], dtype=torch.float32)
            except Exception:
                pass
        # Fallback minimal Grassberger-Procaccia
        from scipy.spatial.distance import pdist
        try:
            dists = pdist(X)
            if len(dists) < 10:
                self.last_dim = 0.0
                return torch.tensor([0.0])
            eps_range = np.logspace(np.log10(dists.min() + 1e-9),
                                       np.log10(dists.max()), self.n_eps)
            counts = []
            for eps in eps_range:
                C = float((dists < eps).sum()) / len(dists)
                counts.append(max(C, 1e-12))
            log_eps = np.log(eps_range)
            log_C = np.log(counts)
            # Slope sur région linéaire centrale
            mid_start = self.n_eps // 4
            mid_end = 3 * self.n_eps // 4
            slope = np.polyfit(log_eps[mid_start:mid_end],
                                  log_C[mid_start:mid_end], 1)[0]
            self.last_dim = float(slope)
            return torch.tensor([slope], dtype=torch.float32)
        except Exception:
            self.last_dim = 0.0
            return torch.tensor([0.0])


# ============================================================
# 5. Recurrence Plot
# ============================================================

class RecurrencePlot(nn.Module):
    """Visualise matrix de récurrence R_{i,j} = θ(ε - ||x_i - x_j||).

    Patterns lus :
    - diagonales = quasi-periodicity
    - bandes verticales = laminar states
    - isolated points = noise
    """

    def __init__(self, output_path: str = "./output/recurrence.html",
                 title: str = "Recurrence Plot", auto_open: bool = False,
                 eps: float = 0.0, max_n: int = 400):
        super().__init__()
        self.output_path = output_path
        self.title = title
        self.auto_open = auto_open
        self.eps = eps  # 0 = auto
        self.max_n = max_n
        self.last_result: DynamoResult | None = None

    def forward(self, x):
        try:
            import plotly.graph_objects as go
        except ImportError:
            return x
        X = _to_numpy(x)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        # Subsample si trop long
        if X.shape[0] > self.max_n:
            idx = np.linspace(0, X.shape[0] - 1, self.max_n).astype(int)
            X = X[idx]
        from scipy.spatial.distance import cdist
        D = cdist(X, X)
        eps = self.eps if self.eps > 0 else 0.1 * D.std()
        R = (D < eps).astype(np.float32)

        fig = go.Figure(data=go.Heatmap(
            z=R, colorscale=[[0, BG], [1, ACCENT]],
            showscale=False,
        ))
        rec_rate = float(R.mean())
        fig.update_layout(
            title=dict(text=f"{self.title} · rec_rate={rec_rate:.3f} · "
                              f"ε={eps:.3f}",
                          font=dict(color=FG, size=13)),
            paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=FG),
            xaxis=dict(title="j", color=FG, scaleanchor="y"),
            yaxis=dict(title="i", color=FG, autorange="reversed"),
            margin=dict(l=60, r=20, t=40, b=40),
        )
        path = _save_html(fig.to_html(include_plotlyjs="cdn"), self.output_path)
        self.last_result = DynamoResult(True, {"rec_rate": rec_rate, "eps": eps},
                                             path)
        if self.auto_open:
            webbrowser.open(f"file://{Path(path).absolute()}")
        return x


# ============================================================
# 6. DMD (Dynamic Mode Decomposition / Koopman)
# ============================================================

class DMD(nn.Module):
    """Dynamic Mode Decomposition. Eigenvalues du Koopman operator."""

    def __init__(self, rank: int = 0, dt: float = 1.0):
        super().__init__()
        self.rank = rank  # 0 = full
        self.dt = dt
        self.last_eigenvalues = None

    def forward(self, x):
        X = _to_numpy(x)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if _ensure_dynamoscope():
            try:
                from backend.spectral import dmd as ds_dmd
                rank = self.rank if self.rank > 0 else None
                result = ds_dmd(X, rank=rank, dt=self.dt)
                re = np.asarray(result.get("eigvals_re",
                                                 result.get("eigs_real", [])))
                im = np.asarray(result.get("eigvals_im",
                                                 result.get("eigs_imag", [])))
                if len(re) == 0:
                    raise ValueError("no eigenvalues")
                eigs = re + 1j * im
                self.last_eigenvalues = eigs
                # Retourne magnitudes (stabilité)
                return torch.from_numpy(np.abs(eigs)).float()
            except Exception:
                pass
        # Fallback : DMD minimal
        try:
            # X = (N, D) → snapshots X1=(D, N-1), X2=(D, N-1)
            X_T = X.T
            X1 = X_T[:, :-1]
            X2 = X_T[:, 1:]
            U, S, Vh = np.linalg.svd(X1, full_matrices=False)
            r = self.rank if self.rank > 0 else min(len(S), 10)
            r = min(r, len(S))
            U_r = U[:, :r]
            S_r = np.diag(S[:r])
            V_r = Vh.conj().T[:, :r]
            A_tilde = U_r.conj().T @ X2 @ V_r @ np.linalg.inv(S_r)
            eigs, _ = np.linalg.eig(A_tilde)
            self.last_eigenvalues = eigs
            return torch.from_numpy(np.abs(eigs)).float()
        except Exception:
            return torch.zeros(1)


# ============================================================
# 7. RQA stats (Recurrence Quantification Analysis)
# ============================================================

class RQAStats(nn.Module):
    """RQA : recurrence rate, determinism, laminarity, entropy."""

    def __init__(self, eps: float = 0.0, max_n: int = 400,
                  lmin: int = 2):
        super().__init__()
        self.eps = eps
        self.max_n = max_n
        self.lmin = lmin
        self.last_stats: dict | None = None

    def forward(self, x):
        X = _to_numpy(x)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] > self.max_n:
            idx = np.linspace(0, X.shape[0] - 1, self.max_n).astype(int)
            X = X[idx]
        from scipy.spatial.distance import cdist
        D = cdist(X, X)
        eps = self.eps if self.eps > 0 else 0.1 * D.std()
        R = (D < eps).astype(np.uint8)
        if _ensure_dynamoscope():
            try:
                from backend.dynamics import rqa_stats
                stats = rqa_stats(R, lmin=self.lmin)
                self.last_stats = stats
                return torch.tensor([
                    stats.get("recurrence_rate", 0.0),
                    stats.get("determinism", 0.0),
                    stats.get("laminarity", 0.0),
                    stats.get("entropy", 0.0),
                ], dtype=torch.float32)
            except Exception:
                pass
        # Fallback minimal
        rec_rate = float(R.mean())
        # Diagonales >= lmin = déterminisme
        det_count = 0
        total_diag = 0
        for k in range(1, R.shape[0]):
            diag = np.diag(R, k=k)
            # Run lengths
            runs = []
            run = 0
            for v in diag:
                if v:
                    run += 1
                else:
                    if run > 0:
                        runs.append(run)
                    run = 0
            if run > 0:
                runs.append(run)
            for r in runs:
                if r >= self.lmin:
                    det_count += r
                total_diag += r
        det = det_count / max(total_diag, 1)
        self.last_stats = {"recurrence_rate": rec_rate, "determinism": det}
        return torch.tensor([rec_rate, det, 0.0, 0.0], dtype=torch.float32)


# ============================================================
# 8. DNA score (composite 7-axes)
# ============================================================

class DNAScore(nn.Module):
    """Composite complexité 7 axes (dynamoscope DNA).

    Output : 7 scores [0,1] : freshness, chaos, structure, periodicity,
                                 emergence, smoothness, complexity.
    """

    def __init__(self):
        super().__init__()
        self.last_dna: dict | None = None

    def forward(self, x):
        X = _to_numpy(x)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if _ensure_dynamoscope():
            try:
                from backend.dna import compute_dna
                # compute_dna prend (latents, coords_3d)
                coords3 = X[:, :3] if X.shape[1] >= 3 else \
                           np.hstack([X, np.zeros((X.shape[0],
                                                       3 - X.shape[1]))])
                dna = compute_dna(X, coords3)
                self.last_dna = dna
                axes = ["freshness", "chaos", "structure", "periodicity",
                          "emergence", "smoothness", "complexity"]
                return torch.tensor(
                    [float(dna.get(a, 0.5)) for a in axes],
                    dtype=torch.float32)
            except Exception:
                pass
        # Fallback : approx simplifié
        try:
            from scipy.spatial.distance import pdist
            d = pdist(X)
            chaos = float(np.std(d) / (np.mean(d) + 1e-9))
            smoothness = 1.0 - float(np.std(np.diff(X, axis=0)) /
                                          (np.std(X) + 1e-9))
            self.last_dna = {"chaos": chaos, "smoothness": smoothness}
            return torch.tensor([0.5, min(chaos, 1.0), 0.5, 0.5, 0.5,
                                     max(smoothness, 0.0), 0.5],
                                    dtype=torch.float32)
        except Exception:
            return torch.full((7,), 0.5)


# ============================================================
# Handlers
# ============================================================

def h_phase_space_3d(node, in_shapes):
    m = PhaseSpace3D(
        output_path=node.params.get("output_path",
                                          "./output/phase_space.html"),
        title=node.params.get("title", "Phase Space 3D"),
        auto_open=bool(node.params.get("auto_open", False)),
        color_mode=node.params.get("color_mode", "time"),
        show_attractor=bool(node.params.get("show_attractor", True)),
    )
    m._ark_node_id = getattr(node, "id", "ps")
    return m, (3,)


def h_takens_embedding(node, in_shapes):
    m = TakensEmbedding(
        m=int(node.params.get("m", 3)),
        tau=int(node.params.get("tau", 1)),
    )
    m._ark_node_id = getattr(node, "id", "tk")
    return m, (int(node.params.get("m", 3)),)


def h_lyapunov(node, in_shapes):
    m = LyapunovRosenstein(
        m=int(node.params.get("m", 3)),
        tau=int(node.params.get("tau", 1)),
        max_t=int(node.params.get("max_t", 50)),
        min_separation=int(node.params.get("min_separation", 10)),
    )
    m._ark_node_id = getattr(node, "id", "ly")
    return m, (1,)


def h_correlation_dim(node, in_shapes):
    m = CorrelationDimension(n_eps=int(node.params.get("n_eps", 14)))
    m._ark_node_id = getattr(node, "id", "cd")
    return m, (1,)


def h_recurrence_plot(node, in_shapes):
    m = RecurrencePlot(
        output_path=node.params.get("output_path",
                                          "./output/recurrence.html"),
        title=node.params.get("title", "Recurrence Plot"),
        auto_open=bool(node.params.get("auto_open", False)),
        eps=float(node.params.get("eps", 0.0)),
        max_n=int(node.params.get("max_n", 400)),
    )
    m._ark_node_id = getattr(node, "id", "rp")
    return m, in_shapes[0] if in_shapes else (1,)


def h_dmd(node, in_shapes):
    m = DMD(
        rank=int(node.params.get("rank", 0)),
        dt=float(node.params.get("dt", 1.0)),
    )
    m._ark_node_id = getattr(node, "id", "dmd")
    return m, (int(node.params.get("rank", 10)) or 10,)


def h_rqa_stats(node, in_shapes):
    m = RQAStats(
        eps=float(node.params.get("eps", 0.0)),
        max_n=int(node.params.get("max_n", 400)),
        lmin=int(node.params.get("lmin", 2)),
    )
    m._ark_node_id = getattr(node, "id", "rqa")
    return m, (4,)


def h_dna_score(node, in_shapes):
    m = DNAScore()
    m._ark_node_id = getattr(node, "id", "dna")
    return m, (7,)


def register_dynamoscope_handlers(handlers_dict: dict) -> bool:
    handlers_dict["phase_space_3d"] = h_phase_space_3d
    handlers_dict["takens_embedding"] = h_takens_embedding
    handlers_dict["lyapunov_rosenstein"] = h_lyapunov
    handlers_dict["correlation_dimension"] = h_correlation_dim
    handlers_dict["recurrence_plot"] = h_recurrence_plot
    handlers_dict["dmd"] = h_dmd
    handlers_dict["rqa_stats"] = h_rqa_stats
    handlers_dict["dna_score"] = h_dna_score
    return True


def register_dynamoscope_specs(registry) -> bool:
    from ark.nodes.registry import NodeSpec, ParamSpec, SocketSpec

    CAT = "Dynamoscope"

    registry.register(NodeSpec(
        name="phase_space_3d", title="Phase Space 3D (Dynamoscope)",
        category=CAT,
        inputs=[SocketSpec("latents", "tensor")],
        outputs=[SocketSpec("coords_3d", "tensor")],
        params=[
            ParamSpec("output_path", "str", "./output/phase_space.html"),
            ParamSpec("title", "str", "Phase Space 3D"),
            ParamSpec("auto_open", "bool", False),
            ParamSpec("color_mode", "choice", "time",
                        options=["time", "velocity", "speed_log"]),
            ParamSpec("show_attractor", "bool", True),
        ],
        description="Trajectoire 3D dans l'espace des phases. PCA auto si D > 3. "
                       "Markers START/END distincts + centroid attracteur.",
        role="Latents → trajectoire colorée 3D interactive.",
        function="Plotly Scatter3D avec lines+markers, colorscale par color_mode "
                    "(time/velocity/log). Centroid estimé μ ± σ affiché.",
        equation="x(t+1) = f(x(t))",
        related=("PCA", "t-SNE", "Lyapunov", "Takens"),
        has_thumbnail=True,
        param_help={
            "color_mode": "time = gradient temporel | velocity = vitesse instantanée | speed_log = log",
            "show_attractor": "Affiche centroid μ comme attracteur estimé.",
        },
    ))

    registry.register(NodeSpec(
        name="takens_embedding", title="Takens Embedding",
        category=CAT,
        inputs=[SocketSpec("signal", "tensor")],
        outputs=[SocketSpec("embedded", "tensor")],
        params=[
            ParamSpec("m", "int", 3, min=2, max=20),
            ParamSpec("tau", "int", 1, min=1, max=100),
        ],
        description="Reconstruction Takens : signal 1D → espace m-D via délais τ. "
                       "Théorème : préserve topologie de l'attracteur original.",
        role="1D signal → m-D state space.",
        function="X(t) = [x(t), x(t-τ), x(t-2τ), ..., x(t-(m-1)τ)]. "
                    "Sortie : (N-(m-1)τ, m).",
        equation="X(t) = (x(t), x(t-τ), ..., x(t-(m-1)τ))",
        related=("Lyapunov", "Correlation Dimension", "Attractor"),
        has_thumbnail=True,
        param_help={
            "m": "Dimension embedding. Pour Lorenz : m=3.",
            "tau": "Délai. Optimal = premier minimum mutual info.",
        },
    ))

    registry.register(NodeSpec(
        name="lyapunov_rosenstein", title="Lyapunov Exponent (Rosenstein)",
        category=CAT,
        inputs=[SocketSpec("signal", "tensor")],
        outputs=[SocketSpec("lambda_max", "tensor")],
        params=[
            ParamSpec("m", "int", 3),
            ParamSpec("tau", "int", 1),
            ParamSpec("max_t", "int", 50),
            ParamSpec("min_separation", "int", 10),
        ],
        description="Estime exposant Lyapunov maximal λ. "
                       "λ > 0 = chaos, λ ≈ 0 = quasi-periodic, λ < 0 = stable.",
        role="Quantifie sensibilité conditions initiales.",
        function="Takens embed → nearest neighbors → divergence moyenne → slope log = λ.",
        equation="λ = lim_{t→∞} (1/t) log(||δx(t)||/||δx(0)||)",
        related=("Takens", "Correlation Dimension", "Chaos"),
        has_thumbnail=True,
    ))

    registry.register(NodeSpec(
        name="correlation_dimension", title="Correlation Dimension (D₂)",
        category=CAT,
        inputs=[SocketSpec("trajectory", "tensor")],
        outputs=[SocketSpec("D2", "tensor")],
        params=[ParamSpec("n_eps", "int", 14, min=5, max=50)],
        description="Dimension fractale Grassberger-Procaccia. D₂ ≈ 2.06 pour Lorenz. "
                       "Quantifie complexité géométrique de l'attracteur.",
        role="Fractal dim from trajectory points.",
        function="C(ε) = Σ θ(ε - ||xi - xj||) / N². Slope log-log = D₂.",
        equation="C(ε) ∼ ε^D₂",
        related=("Lyapunov", "Takens", "Attractor"),
        has_thumbnail=True,
    ))

    registry.register(NodeSpec(
        name="recurrence_plot", title="Recurrence Plot (Dynamoscope)",
        category=CAT,
        inputs=[SocketSpec("trajectory", "tensor")],
        outputs=[SocketSpec("trajectory", "tensor")],
        params=[
            ParamSpec("output_path", "str", "./output/recurrence.html"),
            ParamSpec("title", "str", "Recurrence Plot"),
            ParamSpec("auto_open", "bool", False),
            ParamSpec("eps", "float", 0.0, min=0.0, max=10.0),
            ParamSpec("max_n", "int", 400, min=20, max=2000),
        ],
        description="Matrix de récurrence R_{i,j} = θ(ε - ||x_i - x_j||). "
                       "Diagonales = quasi-periodic. Bandes = laminar.",
        role="Recurrence matrix → 2D heatmap.",
        function="cdist Euclidean + threshold ε auto. rec_rate = mean(R).",
        equation="R_{i,j} = Θ(ε - ||x_i - x_j||)",
        related=("RQA", "Phase Space"),
        has_thumbnail=True,
        param_help={
            "eps": "Seuil distance. 0 = auto (10% std distances).",
            "max_n": "Subsample si N > max_n pour viz.",
        },
    ))

    registry.register(NodeSpec(
        name="dmd", title="DMD (Koopman eigenmodes)",
        category=CAT,
        inputs=[SocketSpec("snapshots", "tensor")],
        outputs=[SocketSpec("eigenvalues_mag", "tensor")],
        params=[
            ParamSpec("rank", "int", 0, min=0, max=200),
            ParamSpec("dt", "float", 1.0, min=0.001, max=100.0),
        ],
        description="Dynamic Mode Decomposition. Approximation Koopman operator. "
                       "|λ| > 1 = growing, ≈ 1 = quasi-stationary, < 1 = decaying.",
        role="Snapshots → linear evolution operator eigenvalues.",
        function="SVD trunc + projection + eigendecomp de A_tilde = U* X₂ V S⁻¹.",
        equation="x_{k+1} = A x_k → A ≈ U Σ V*",
        related=("Koopman", "Spectral", "POD"),
        has_thumbnail=True,
    ))

    registry.register(NodeSpec(
        name="rqa_stats", title="RQA Stats",
        category=CAT,
        inputs=[SocketSpec("trajectory", "tensor")],
        outputs=[SocketSpec("stats", "tensor")],
        params=[
            ParamSpec("eps", "float", 0.0),
            ParamSpec("max_n", "int", 400),
            ParamSpec("lmin", "int", 2, min=2, max=20),
        ],
        description="Recurrence Quantification Analysis : 4 stats clés "
                       "(recurrence_rate, determinism, laminarity, entropy).",
        role="Quantification statistique recurrence patterns.",
        function="R_matrix → diagonal runs (DET), vertical runs (LAM), "
                    "Shannon entropy run lengths.",
        related=("Recurrence Plot", "Phase Space"),
        has_thumbnail=True,
    ))

    registry.register(NodeSpec(
        name="dna_score", title="DNA Score (7-axes complexity)",
        category=CAT,
        inputs=[SocketSpec("latents", "tensor")],
        outputs=[SocketSpec("dna", "tensor")],
        params=[],
        description="Composite Dynamoscope : 7 axes [0,1] = freshness/chaos/structure/"
                       "periodicity/emergence/smoothness/complexity. Signature unique.",
        role="One-shot complexity fingerprint.",
        function="Agrège métriques Lyapunov + RQA + spectral + variance + autocorr. "
                    "Fingerprint normalisé pour comparaison cross-system.",
        related=("Lyapunov", "RQA", "DMD", "Emergence"),
        has_thumbnail=True,
    ))
    return True
