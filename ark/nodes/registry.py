"""Registry de types de nodes.

Chaque type possède :
- name (id), title (display), category (Input / Math / Neural / Code / Viz / ...)
- inputs : list[(label, socket_type)]
- outputs : list[(label, socket_type)]
- params : list[(name, kind, default, options?)] avec kind ∈
  {'int', 'float', 'str', 'bool', 'choice', 'multiline'}.

Socket types pilotent la couleur des sockets + validation connexions.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParamSpec:
    name: str
    kind: str  # 'int' | 'float' | 'str' | 'bool' | 'choice' | 'multiline'
    default: Any = None
    options: list = field(default_factory=list)
    min: float | None = None
    max: float | None = None


@dataclass
class SocketSpec:
    label: str
    socket_type: str  # 'tensor' | 'number' | 'string' | 'image' | 'model' | 'exec' | 'any'


@dataclass
class NodeSpec:
    name: str
    title: str
    category: str
    inputs: list[SocketSpec] = field(default_factory=list)
    outputs: list[SocketSpec] = field(default_factory=list)
    params: list[ParamSpec] = field(default_factory=list)
    description: str = ""
    has_thumbnail: bool = False
    is_multi_input: bool = False  # si True, chaque connexion ajoute un connecteur


class NodeRegistry:
    def __init__(self):
        self._specs: dict[str, NodeSpec] = {}

    def register(self, spec: NodeSpec) -> None:
        self._specs[spec.name] = spec

    def get(self, name: str) -> NodeSpec | None:
        return self._specs.get(name)

    def all(self) -> list[NodeSpec]:
        return list(self._specs.values())

    def by_category(self) -> dict[str, list[NodeSpec]]:
        out: dict[str, list[NodeSpec]] = {}
        for s in self._specs.values():
            out.setdefault(s.category, []).append(s)
        return {k: sorted(v, key=lambda s: s.title) for k, v in sorted(out.items())}


def default_registry() -> NodeRegistry:
    """Registry pré-rempli avec les nodes de base ARK.

    Phase 0 : un échantillon pour valider l'UI. Phase 1+ ajoute le reste
    (CGN, transformer, attention, MoE, VG, observer, trigger, ComfyUI, etc.).
    """
    r = NodeRegistry()

    # ---------- Input ----------
    r.register(NodeSpec(
        name="image_input", title="Image Input", category="Input",
        outputs=[SocketSpec("out", "tensor")],
        params=[ParamSpec("shape", "str", "[1, 28, 28]")],
        description="Tenseur image (C, H, W).",
    ))
    r.register(NodeSpec(
        name="tabular_input", title="Tabular Input", category="Input",
        outputs=[SocketSpec("out", "tensor")],
        params=[ParamSpec("features", "int", 32, min=1, max=10000)],
        description="Vecteur de features (N,).",
    ))
    r.register(NodeSpec(
        name="text_input", title="Text Input", category="Input",
        outputs=[SocketSpec("out", "tensor")],
        params=[ParamSpec("max_length", "int", 64),
                 ParamSpec("embed_dim", "int", 128)],
        description="Séquence (T, D) après embedding.",
    ))

    # ---------- Neural ----------
    r.register(NodeSpec(
        name="dense", title="Dense", category="Neural",
        inputs=[SocketSpec("in", "tensor")],
        outputs=[SocketSpec("out", "tensor")],
        params=[
            ParamSpec("neurons", "int", 128, min=1, max=100000),
            ParamSpec("activation", "choice", "relu",
                       options=["none", "relu", "gelu", "tanh", "sigmoid", "softmax"]),
            ParamSpec("dropout", "float", 0.0, min=0.0, max=1.0),
        ],
        description="Linear(in, out) + activation + dropout optionnel.",
    ))
    r.register(NodeSpec(
        name="conv2d", title="Conv2D", category="Neural",
        inputs=[SocketSpec("in", "tensor")],
        outputs=[SocketSpec("out", "tensor")],
        params=[
            ParamSpec("filters", "int", 32, min=1),
            ParamSpec("kernel_size", "int", 3, min=1, max=15),
            ParamSpec("activation", "choice", "relu",
                       options=["none", "relu", "gelu", "tanh"]),
        ],
    ))
    r.register(NodeSpec(
        name="normalize", title="Normalize", category="Neural",
        inputs=[SocketSpec("in", "tensor")],
        outputs=[SocketSpec("out", "tensor")],
        params=[ParamSpec("kind", "choice", "batchnorm",
                            options=["batchnorm", "layernorm", "groupnorm"])],
    ))
    r.register(NodeSpec(
        name="attention", title="Multi-Head Attention", category="Neural",
        inputs=[SocketSpec("in", "tensor")],
        outputs=[SocketSpec("out", "tensor")],
        params=[ParamSpec("n_heads", "int", 4, min=1, max=32)],
    ))
    r.register(NodeSpec(
        name="transformer_block", title="Transformer Block", category="Neural",
        inputs=[SocketSpec("in", "tensor")],
        outputs=[SocketSpec("out", "tensor")],
        params=[ParamSpec("n_heads", "int", 4),
                 ParamSpec("d_ff", "int", 256)],
    ))
    r.register(NodeSpec(
        name="residual_add", title="Residual Add", category="Neural",
        inputs=[SocketSpec("a", "tensor"), SocketSpec("b", "tensor")],
        outputs=[SocketSpec("out", "tensor")],
        is_multi_input=True,
    ))
    r.register(NodeSpec(
        name="moe_switch", title="MoE Switch", category="Neural",
        inputs=[SocketSpec("in", "tensor")],
        outputs=[SocketSpec("out", "tensor")],
        params=[ParamSpec("n_experts", "int", 4, min=2, max=64),
                 ParamSpec("d_out", "int", 128)],
    ))
    r.register(NodeSpec(
        name="vg_block", title="VG Block (mutable)", category="Neural",
        inputs=[SocketSpec("in", "tensor")],
        outputs=[SocketSpec("out", "tensor")],
        params=[ParamSpec("start_hidden", "int", 32),
                 ParamSpec("out_dim", "int", 64),
                 ParamSpec("activation", "choice", "gelu",
                             options=["relu", "gelu", "tanh"])],
        description="Bloc MLP mutable runtime (widen/prune).",
    ))

    # ---------- Math ----------
    r.register(NodeSpec(
        name="math_op", title="Math Op", category="Math",
        inputs=[SocketSpec("a", "number"), SocketSpec("b", "number")],
        outputs=[SocketSpec("out", "number")],
        params=[ParamSpec("op", "choice", "add",
                            options=["add", "sub", "mul", "div", "pow", "min", "max", "mod"])],
        description="Opération arithmétique élémentaire. Ordre des entrées sensible.",
    ))
    r.register(NodeSpec(
        name="constant", title="Constant", category="Math",
        outputs=[SocketSpec("out", "number")],
        params=[ParamSpec("value", "float", 1.0)],
    ))

    # ---------- Code ----------
    r.register(NodeSpec(
        name="code_line", title="Code Line", category="Code",
        inputs=[SocketSpec("in", "any")],
        outputs=[SocketSpec("out", "any")],
        params=[ParamSpec("code", "multiline",
                            "# Python expression — `x` est l'input\nresult = x")],
        description="Exécute une ligne (ou bloc) Python sur l'input.",
    ))

    # ---------- Observe / Act ----------
    r.register(NodeSpec(
        name="observer", title="Observer", category="Observe",
        inputs=[SocketSpec("in", "tensor")],
        outputs=[SocketSpec("out", "tensor")],
        params=[ParamSpec("metrics", "str",
                            "eff_rank,intrinsic_dim,power_slope,mean_abs"),
                 ParamSpec("buffer_size", "int", 256)],
        has_thumbnail=True,
    ))
    r.register(NodeSpec(
        name="trigger_rule", title="Trigger Rule", category="Observe",
        inputs=[SocketSpec("in", "tensor")],
        outputs=[SocketSpec("out", "tensor")],
        params=[ParamSpec("metric", "str", "eff_rank"),
                 ParamSpec("op", "choice", ">",
                             options=["<", ">", "<=", ">=", "==", "!="]),
                 ParamSpec("threshold", "float", 0.0),
                 ParamSpec("action_op", "choice", "widen",
                             options=["widen", "prune"]),
                 ParamSpec("target_node", "str", ""),
                 ParamSpec("delta", "int", 8)],
    ))

    # ---------- Viz ----------
    r.register(NodeSpec(
        name="plot_scatter", title="Plot Scatter", category="Viz",
        inputs=[SocketSpec("x", "number"), SocketSpec("y", "number"),
                 SocketSpec("color", "any")],
        outputs=[],
        has_thumbnail=True,
    ))
    r.register(NodeSpec(
        name="plot_line", title="Plot Line", category="Viz",
        inputs=[SocketSpec("series", "tensor")],
        outputs=[],
        has_thumbnail=True,
    ))

    # ---------- Optim/Loss ----------
    r.register(NodeSpec(
        name="adam", title="Adam Optimizer", category="Training",
        outputs=[SocketSpec("optim", "any")],
        params=[ParamSpec("learning_rate", "float", 1e-3, min=1e-6, max=1.0),
                 ParamSpec("beta1", "float", 0.9),
                 ParamSpec("beta2", "float", 0.999)],
    ))
    r.register(NodeSpec(
        name="cross_entropy", title="Cross-Entropy Loss", category="Training",
        inputs=[SocketSpec("logits", "tensor"), SocketSpec("targets", "tensor")],
        outputs=[SocketSpec("loss", "number")],
    ))

    return r
