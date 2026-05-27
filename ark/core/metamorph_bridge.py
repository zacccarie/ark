"""Bridge ARK ↔ metamorph. Importe les modèles M1/M2/M5 comme handlers ARK.

Demande que `metamorph/` soit accessible (ajouté au sys.path via import).
Si absent, les handlers ne sont pas enregistrés.

Modèles intégrés :
- M1 CriticalGrowNet (criticality_loss + mutation policy SOC β=-2)
- M2 MetaController GNN (graph_snapshot → action_type + node_logits)
- M5 DyadicMirror (2 SelfRefAgent + cross-prediction Mead-Vygotsky)
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn as nn


METAMORPH_PATH = Path.home() / "claude-code-projects" / "metamorph"
_AVAILABLE = False


def _ensure_metamorph_path():
    global _AVAILABLE
    if _AVAILABLE:
        return True
    if not METAMORPH_PATH.exists():
        return False
    p = str(METAMORPH_PATH)
    if p not in sys.path:
        sys.path.insert(0, p)
    try:
        import metamorph  # noqa : F401
        _AVAILABLE = True
        return True
    except ImportError:
        return False


# ============================================================
# Wrappers nn.Module : compatibles avec graph_compiler
# ============================================================

class CriticalGrowNetWrap(nn.Module):
    """Wrap metamorph.core.critical_grow_net.CriticalGrowNet."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 target_beta: float = -2.0, buffer_size: int = 256,
                 activation: str = "gelu"):
        super().__init__()
        if not _ensure_metamorph_path():
            raise RuntimeError("metamorph package not found at " + str(METAMORPH_PATH))
        from metamorph.core.critical_grow_net import CriticalGrowNet
        self.inner = CriticalGrowNet(
            in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
            activation=activation, buffer_size=buffer_size,
            target_beta=target_beta,
        )

    def forward(self, x):
        return self.inner(x)

    @property
    def hidden_dim(self) -> int:
        return self.inner.hidden_dim

    def widen(self, delta: int):
        return self.inner.widen(delta)

    def prune(self, delta: int):
        return self.inner.prune(delta)

    def measure_beta(self) -> float:
        return self.inner.measure_beta()


class DyadicMirrorWrap(nn.Module):
    """Wrap metamorph.core.dyadic_mirror.DyadicMirror.

    Forward attend (x_A, x_B). En mode 'classifier' on retourne juste logits_A.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 state_dim: int = 16, coupled: bool = False):
        super().__init__()
        if not _ensure_metamorph_path():
            raise RuntimeError("metamorph package not found")
        if coupled:
            from metamorph.core.dyadic_mirror import DyadicMirrorCoupled as DM
        else:
            from metamorph.core.dyadic_mirror import DyadicMirror as DM
        self.inner = DM(in_dim=in_dim, hidden_dim=hidden_dim,
                          out_dim=out_dim, state_dim=state_dim)

    def forward(self, x):
        # x peut être un tensor (B, ...) qu'on split en moitiés A/B
        if isinstance(x, tuple) and len(x) == 2:
            x_A, x_B = x
        else:
            B = x.shape[0]
            half = max(B // 2, 1)
            x_A = x[:half]
            x_B = x[half:] if B > half else x[:half]
        out = self.inner(x_A, x_B)
        return out["logits_A"]


class MetaControllerWrap(nn.Module):
    """Wrap metamorph.core.meta_controller_gnn.MetaController.

    Pas un vrai forward classif. Pour intégration ARK, on l'utilise comme
    observateur : reçoit un tensor d'activations + dynamiques metrics,
    retourne action_type_logits.
    """

    def __init__(self, node_feat_dim: int = 8, dynamics_dim: int = 5,
                 hidden: int = 32):
        super().__init__()
        if not _ensure_metamorph_path():
            raise RuntimeError("metamorph package not found")
        from metamorph.core.meta_controller_gnn import MetaController
        self.inner = MetaController(node_feat_dim=node_feat_dim,
                                       dynamics_dim=dynamics_dim, hidden=hidden)
        self.last_action: dict | None = None

    def forward(self, x):
        # x : (B, node_feat_dim) interprété comme N=B nodes
        from metamorph.core.meta_controller_gnn import GraphSnapshot
        N = x.shape[0]
        # Edge index chain 0→1→2→...→N-1
        if N >= 2:
            ei = torch.stack([torch.arange(N - 1), torch.arange(1, N)], dim=0)
        else:
            ei = torch.zeros(2, 0, dtype=torch.long)
        snap = GraphSnapshot(node_features=x.float(), edge_index=ei,
                              node_ids=[f"n_{i}" for i in range(N)],
                              dynamics={})
        out = self.inner(snap)
        # Retourne juste action_type_logits broadcast comme tenseur pass
        # (pour ne pas casser le pipeline) — l'usage réel est via .sample_action
        self.last_action = self.inner.sample_action(snap, deterministic=True)
        return x  # pass-through


# ============================================================
# Handlers ARK
# ============================================================

def h_critical_grow_net(node, in_shapes):
    if len(in_shapes) != 1 or len(in_shapes[0]) != 1:
        raise RuntimeError("critical_grow_net needs 1D input")
    in_dim = in_shapes[0][0]
    hidden = int(node.params.get("hidden_dim", 32))
    out_dim = int(node.params.get("out_dim", 32))
    target_beta = float(node.params.get("target_beta", -2.0))
    m = CriticalGrowNetWrap(in_dim, hidden, out_dim, target_beta=target_beta)
    m._ark_node_id = node.id
    return m, (out_dim,)


def h_dyadic_mirror(node, in_shapes):
    if len(in_shapes) != 1:
        raise RuntimeError("dyadic_mirror needs single input (split internally)")
    in_shape = in_shapes[0]
    flat = 1
    for s in in_shape:
        flat *= int(s)
    out_dim = int(node.params.get("out_dim", 10))
    hidden = int(node.params.get("hidden_dim", 64))
    state_dim = int(node.params.get("state_dim", 16))
    coupled = bool(node.params.get("coupled", False))
    m = DyadicMirrorWrap(flat, hidden, out_dim, state_dim, coupled)
    m._ark_node_id = node.id
    return m, (out_dim,)


def h_meta_controller(node, in_shapes):
    if len(in_shapes) != 1 or len(in_shapes[0]) != 1:
        raise RuntimeError("meta_controller needs 1D input")
    feat_dim = in_shapes[0][0]
    hidden = int(node.params.get("hidden", 32))
    m = MetaControllerWrap(node_feat_dim=feat_dim, hidden=hidden)
    m._ark_node_id = node.id
    return m, in_shapes[0]


def register_metamorph_handlers(handlers_dict: dict) -> bool:
    """Enregistre les handlers metamorph si le package est disponible."""
    if not _ensure_metamorph_path():
        return False
    handlers_dict["critical_grow_net"] = h_critical_grow_net
    handlers_dict["dyadic_mirror"] = h_dyadic_mirror
    handlers_dict["meta_controller"] = h_meta_controller
    return True


def register_metamorph_specs(registry) -> bool:
    """Enregistre les NodeSpec metamorph dans le registry ARK."""
    if not _ensure_metamorph_path():
        return False
    from ark.nodes.registry import NodeSpec, ParamSpec, SocketSpec

    registry.register(NodeSpec(
        name="critical_grow_net", title="Critical Grow Net (M1)",
        category="Metamorph",
        inputs=[SocketSpec("in", "tensor")],
        outputs=[SocketSpec("out", "tensor")],
        params=[
            ParamSpec("hidden_dim", "int", 32, min=2, max=4096),
            ParamSpec("out_dim", "int", 32, min=1, max=4096),
            ParamSpec("target_beta", "float", -2.0, min=-5.0, max=0.0),
            ParamSpec("buffer_size", "int", 256, min=8, max=4096),
            ParamSpec("activation", "choice", "gelu",
                        options=["relu", "gelu", "tanh"]),
        ],
        description="MLP mutable avec criticality_loss SOC (target spectral β=-2).",
        has_thumbnail=True,
    ))
    registry.register(NodeSpec(
        name="dyadic_mirror", title="Dyadic Mirror (M5)",
        category="Metamorph",
        inputs=[SocketSpec("in", "tensor")],
        outputs=[SocketSpec("logits_A", "tensor")],
        params=[
            ParamSpec("hidden_dim", "int", 64, min=4, max=4096),
            ParamSpec("out_dim", "int", 10, min=1, max=1000),
            ParamSpec("state_dim", "int", 16, min=4, max=512),
            ParamSpec("coupled", "bool", False),
        ],
        description="2 SelfRefAgent + cross-prediction Mead-Vygotsky.",
    ))
    registry.register(NodeSpec(
        name="meta_controller", title="Meta Controller GNN (M2)",
        category="Metamorph",
        inputs=[SocketSpec("in", "tensor")],
        outputs=[SocketSpec("out", "tensor")],
        params=[
            ParamSpec("hidden", "int", 32, min=8, max=512),
            ParamSpec("dynamics_dim", "int", 5, min=1, max=32),
        ],
        description="GNN observateur graphe → policy mutation logits.",
    ))
    return True
