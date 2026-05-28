"""Phase 35 — Architecture diagram + NN schematic visualizations.

Deux générateurs HTML/SVG standalone :

1. ArchDiagram : organigramme du graphe ARK
   - Topo sort + Sugiyama-style layered layout
   - Box par node avec title + type + I/O sockets
   - Arrows directionnels avec curved bezier
   - Style dark + accent rouge ARK

2. NNSchematic : schéma classique réseau de neurones
   - Couches = colonnes verticales de cercles
   - Connexions = lignes entre neurones (sampled si > sample_threshold)
   - Pour Linear / MLP / sequential dense
   - Annotations par couche : in_dim → out_dim, activation

Tous output : HTML standalone ouvrable navigateur.
"""
from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


ACCENT = "#E11D2A"
ACCENT_DIM = "#421016"
BG = "#0A0A0A"
BG_PANEL = "#16161D"
BORDER = "#2E2E37"
FG = "#F2F2F2"
FG_MUTED = "#7E7E88"


@dataclass
class VizResult:
    ok: bool
    html_path: Optional[str] = None
    n_nodes: int = 0
    error: str = ""


def _save_html(content: str, output_path: str, title: str) -> str:
    out = Path(output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
  body {{ background: {BG}; color: {FG}; font-family: Arial, sans-serif;
          margin: 0; padding: 20px; }}
  h1 {{ color: {ACCENT}; margin: 0 0 10px 0; font-size: 16px;
        text-transform: uppercase; letter-spacing: 2px; }}
  .sub {{ color: {FG_MUTED}; font-size: 11px; margin-bottom: 20px; }}
  .panel {{ background: {BG_PANEL}; border: 1px solid {BORDER};
              padding: 20px; border-radius: 4px; overflow: auto; }}
  .legend {{ margin-top: 16px; font-size: 10px; color: {FG_MUTED}; }}
  .legend span {{ display: inline-block; padding: 2px 8px;
                    border: 1px solid {BORDER}; margin-right: 6px; }}
</style></head>
<body>
<h1>{title}</h1>
{content}
</body></html>"""
    out.write_text(html, encoding="utf-8")
    return str(out)


# ============================================================
# 1. ArchDiagram : scene → SVG flowchart
# ============================================================

class ArchDiagram(nn.Module):
    """Génère SVG flowchart du graphe ARK depuis scene.

    Doit recevoir la scene en input (pass via param `scene_ref` ou via
    set_scene). Pour usage node : forward retourne x mais render scene
    courante (récupérée via _scene_ref).
    """

    NODE_W = 180
    NODE_H = 70
    GAP_X = 80
    GAP_Y = 40

    def __init__(self, output_path: str = "./output/arch_diagram.html",
                 title: str = "Architecture", auto_open: bool = False):
        super().__init__()
        self.output_path = output_path
        self.title = title
        self.auto_open = auto_open
        self._scene_ref = None
        self.last_result: VizResult | None = None

    def set_scene(self, scene):
        self._scene_ref = scene
        return self

    def _topo_layers(self, nodes_list, conns_list):
        """Sugiyama-style : couches = profondeur depuis sources."""
        in_deg = {n.id: 0 for n in nodes_list}
        outgoing: dict = defaultdict(list)
        for c in conns_list:
            if c.start_socket is None or c.end_socket is None:
                continue
            src = c.start_socket.parent_node.id
            dst = c.end_socket.parent_node.id
            if src in in_deg and dst in in_deg:
                outgoing[src].append(dst)
                in_deg[dst] += 1

        layer_of: dict[str, int] = {}
        queue = deque([n.id for n in nodes_list if in_deg[n.id] == 0])
        for nid in queue:
            layer_of[nid] = 0
        while queue:
            nid = queue.popleft()
            for tgt in outgoing[nid]:
                proposed = layer_of[nid] + 1
                if tgt not in layer_of or layer_of[tgt] < proposed:
                    layer_of[tgt] = proposed
                    queue.append(tgt)
        # Nodes restants (cycles ou disconnected)
        for n in nodes_list:
            if n.id not in layer_of:
                layer_of[n.id] = 0
        # Group par layer
        layers: dict[int, list] = defaultdict(list)
        for n in nodes_list:
            layers[layer_of[n.id]].append(n)
        return layers, layer_of, outgoing

    def render_scene(self, scene) -> str:
        nodes_list = list(scene.nodes())
        conns_list = list(scene.connections())
        if not nodes_list:
            return "<div class='panel'>Scène vide.</div>"

        layers, layer_of, outgoing = self._topo_layers(nodes_list, conns_list)
        max_per_layer = max((len(layers[k]) for k in layers), default=1)
        n_layers = max(layers.keys(), default=0) + 1

        # Layout : (x, y) per node
        pos: dict[str, tuple[float, float]] = {}
        for layer_idx in sorted(layers.keys()):
            ns = layers[layer_idx]
            x = 40 + layer_idx * (self.NODE_W + self.GAP_X)
            total_h = len(ns) * (self.NODE_H + self.GAP_Y) - self.GAP_Y
            y0 = (max_per_layer * (self.NODE_H + self.GAP_Y) - total_h) / 2 + 40
            for i, n in enumerate(ns):
                pos[n.id] = (x, y0 + i * (self.NODE_H + self.GAP_Y))

        W = 40 + n_layers * (self.NODE_W + self.GAP_X)
        H = max_per_layer * (self.NODE_H + self.GAP_Y) + 80

        # SVG
        parts = [f"<svg width='{W}' height='{H}' "
                  f"xmlns='http://www.w3.org/2000/svg' style='background:{BG}'>"]

        # Defs : arrowhead
        parts.append(
            "<defs>"
            f"<marker id='arrow' markerWidth='10' markerHeight='10' "
            f"refX='9' refY='3' orient='auto'>"
            f"<path d='M0,0 L0,6 L9,3 z' fill='{ACCENT}'/>"
            "</marker>"
            "</defs>"
        )

        # Connexions (sous nodes pour z-order)
        for c in conns_list:
            if c.start_socket is None or c.end_socket is None:
                continue
            src = c.start_socket.parent_node.id
            dst = c.end_socket.parent_node.id
            if src not in pos or dst not in pos:
                continue
            sx, sy = pos[src]
            dx, dy = pos[dst]
            x1 = sx + self.NODE_W
            y1 = sy + self.NODE_H / 2
            x2 = dx
            y2 = dy + self.NODE_H / 2
            ctrl = (x1 + x2) / 2
            path = (f"M {x1:.1f} {y1:.1f} "
                       f"C {ctrl:.1f} {y1:.1f}, {ctrl:.1f} {y2:.1f}, "
                       f"{x2 - 8:.1f} {y2:.1f}")
            parts.append(
                f"<path d='{path}' stroke='{ACCENT}' stroke-width='1.5' "
                f"fill='none' opacity='0.7' marker-end='url(#arrow)'/>")

        # Boxes nodes
        for n in nodes_list:
            x, y = pos[n.id]
            # Box background
            parts.append(
                f"<rect x='{x}' y='{y}' width='{self.NODE_W}' "
                f"height='{self.NODE_H}' rx='3' ry='3' "
                f"fill='{BG_PANEL}' stroke='{BORDER}' stroke-width='1'/>")
            # Header band
            parts.append(
                f"<rect x='{x}' y='{y}' width='{self.NODE_W}' "
                f"height='20' fill='{ACCENT_DIM}'/>")
            # Title
            title = n.spec.title[:24]
            parts.append(
                f"<text x='{x + 8}' y='{y + 14}' fill='{FG}' "
                f"font-family='Arial' font-size='11' "
                f"font-weight='bold'>{title}</text>")
            # Category
            cat = n.spec.category[:20]
            parts.append(
                f"<text x='{x + 8}' y='{y + 34}' fill='{FG_MUTED}' "
                f"font-family='Menlo, monospace' font-size='9'>{cat}</text>")
            # I/O summary
            n_in = len(n.spec.inputs)
            n_out = len(n.spec.outputs)
            io_text = f"in:{n_in} → out:{n_out}"
            parts.append(
                f"<text x='{x + 8}' y='{y + 50}' fill='{FG_MUTED}' "
                f"font-family='Menlo, monospace' font-size='9'>{io_text}</text>")
            # Params count
            n_params = len(n.params)
            parts.append(
                f"<text x='{x + 8}' y='{y + 63}' fill='{FG_MUTED}' "
                f"font-family='Menlo, monospace' font-size='9'>"
                f"params: {n_params}</text>")
            # Input/Output sockets dots
            if n_in > 0:
                parts.append(
                    f"<circle cx='{x}' cy='{y + self.NODE_H / 2}' r='4' "
                    f"fill='{ACCENT}'/>")
            if n_out > 0:
                parts.append(
                    f"<circle cx='{x + self.NODE_W}' "
                    f"cy='{y + self.NODE_H / 2}' r='4' "
                    f"fill='{ACCENT}'/>")

        parts.append("</svg>")
        svg = "\n".join(parts)

        legend = (
            f"<div class='legend'>"
            f"<span style='border-color:{ACCENT}'>node</span>"
            f"<span style='color:{ACCENT}'>→ flow</span>"
            f"&nbsp; {len(nodes_list)} nodes · {len(conns_list)} connexions · "
            f"{n_layers} couches"
            f"</div>"
        )
        return f"<div class='panel' style='overflow-x:auto'>{svg}</div>{legend}"

    def forward(self, x=None):
        scene = self._scene_ref
        if scene is None:
            # Empty diagram si pas de scene
            content = "<div class='panel'>scene_ref non défini.</div>"
        else:
            content = self.render_scene(scene)
        n_nodes = len(scene.nodes()) if scene else 0
        path = _save_html(content, self.output_path, self.title)
        self.last_result = VizResult(True, path, n_nodes)
        if self.auto_open:
            import webbrowser
            webbrowser.open(f"file://{Path(path).absolute()}")
        if x is None:
            return torch.zeros(1)
        return x


# ============================================================
# 2. NNSchematic : classic neural net diagram
# ============================================================

class NNSchematic(nn.Module):
    """Schéma classique réseau de neurones avec couches + connexions.

    Param sizes = liste de tailles : [in_dim, hidden1, hidden2, ..., out_dim]
    Max neurones rendus par couche : 16 (sample sinon).
    """

    NEURON_R = 10
    LAYER_GAP = 140
    NEURON_GAP = 28
    MAX_NEURONS = 16

    def __init__(self, output_path: str = "./output/nn_schematic.html",
                 title: str = "Neural Network", auto_open: bool = False,
                 sizes: str = "784,128,64,10",
                 activations: str = "relu,relu,softmax"):
        super().__init__()
        self.output_path = output_path
        self.title = title
        self.auto_open = auto_open
        self.sizes = sizes
        self.activations = activations
        self.last_result: VizResult | None = None

    def _parse_sizes(self) -> list[int]:
        try:
            return [int(s.strip()) for s in self.sizes.split(",") if s.strip()]
        except ValueError:
            return [10, 8, 4]

    def _parse_activations(self) -> list[str]:
        return [a.strip() for a in self.activations.split(",") if a.strip()]

    def forward(self, x=None):
        sizes = self._parse_sizes()
        acts = self._parse_activations()
        L = len(sizes)
        n_render = [min(s, self.MAX_NEURONS) for s in sizes]

        max_n = max(n_render)
        H = max_n * self.NEURON_GAP + 100
        W = (L - 1) * self.LAYER_GAP + 200

        parts = [f"<svg width='{W}' height='{H}' "
                  f"xmlns='http://www.w3.org/2000/svg' style='background:{BG}'>"]

        # Compute neuron positions per layer
        pos: list[list[tuple[float, float]]] = []
        for li, n_rend in enumerate(n_render):
            x = 60 + li * self.LAYER_GAP
            total_h = n_rend * self.NEURON_GAP
            y0 = (H - total_h) / 2
            pos.append([
                (x, y0 + i * self.NEURON_GAP + self.NEURON_GAP / 2)
                for i in range(n_rend)
            ])

        # Connexions (drawn first, behind neurons)
        for li in range(L - 1):
            for i, (x1, y1) in enumerate(pos[li]):
                for j, (x2, y2) in enumerate(pos[li + 1]):
                    # opacity faible pour clarté
                    parts.append(
                        f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' "
                        f"stroke='{ACCENT}' stroke-width='0.5' opacity='0.15'/>")

        # Neurones
        for li, layer_pos in enumerate(pos):
            for x, y in layer_pos:
                parts.append(
                    f"<circle cx='{x}' cy='{y}' r='{self.NEURON_R}' "
                    f"fill='{BG_PANEL}' stroke='{ACCENT}' stroke-width='1.5'/>")
            # Si layer plus grande que MAX → indicateur "…"
            if sizes[li] > self.MAX_NEURONS:
                last_x, last_y = layer_pos[-1]
                parts.append(
                    f"<text x='{last_x}' y='{last_y + 28}' "
                    f"fill='{FG_MUTED}' font-family='Arial' font-size='10' "
                    f"text-anchor='middle'>… ({sizes[li]} total)</text>")

        # Labels couches
        for li in range(L):
            x_layer = 60 + li * self.LAYER_GAP
            if li == 0:
                label = f"INPUT ({sizes[0]})"
            elif li == L - 1:
                label = f"OUTPUT ({sizes[-1]})"
            else:
                label = f"HIDDEN {li} ({sizes[li]})"
            parts.append(
                f"<text x='{x_layer}' y='30' fill='{ACCENT}' "
                f"font-family='Arial' font-size='11' font-weight='bold' "
                f"text-anchor='middle'>{label}</text>")
            # Activation entre couches
            if li < L - 1 and li < len(acts):
                act = acts[li]
                mid_x = x_layer + self.LAYER_GAP / 2
                parts.append(
                    f"<text x='{mid_x}' y='{H - 20}' fill='{FG_MUTED}' "
                    f"font-family='Menlo, monospace' font-size='10' "
                    f"text-anchor='middle' "
                    f"font-style='italic'>{act}()</text>")

        # Stats globales
        total_params = sum(sizes[i] * sizes[i + 1] + sizes[i + 1]
                              for i in range(L - 1))
        parts.append(
            f"<text x='10' y='{H - 5}' fill='{FG_MUTED}' "
            f"font-family='Menlo, monospace' font-size='10'>"
            f"Total params: {total_params:,}</text>")

        parts.append("</svg>")
        svg = "\n".join(parts)

        legend = (
            f"<div class='legend'>"
            f"<span style='border-color:{ACCENT}'>neurone</span>"
            f"<span style='color:{ACCENT}'>— synapse</span>"
            f"&nbsp; {L} couches · {sum(sizes)} neurones total · "
            f"{total_params:,} params"
            f"</div>"
        )
        content = f"<div class='panel' style='overflow-x:auto'>{svg}</div>{legend}"
        path = _save_html(content, self.output_path, self.title)
        self.last_result = VizResult(True, path, L)
        if self.auto_open:
            import webbrowser
            webbrowser.open(f"file://{Path(path).absolute()}")
        if x is None:
            return torch.zeros(1)
        return x


# ============================================================
# Handlers + Specs
# ============================================================

def h_arch_diagram(node, in_shapes):
    m = ArchDiagram(
        output_path=node.params.get("output_path",
                                          "./output/arch_diagram.html"),
        title=node.params.get("title", "Architecture"),
        auto_open=bool(node.params.get("auto_open", False)),
    )
    m._ark_node_id = getattr(node, "id", "arch")
    # Scene ref injectée plus tard via set_scene (depuis main_window action)
    return m, in_shapes[0] if in_shapes else (1,)


def h_nn_schematic(node, in_shapes):
    m = NNSchematic(
        output_path=node.params.get("output_path",
                                          "./output/nn_schematic.html"),
        title=node.params.get("title", "Neural Network"),
        auto_open=bool(node.params.get("auto_open", False)),
        sizes=node.params.get("sizes", "784,128,64,10"),
        activations=node.params.get("activations", "relu,relu,softmax"),
    )
    m._ark_node_id = getattr(node, "id", "nn")
    return m, in_shapes[0] if in_shapes else (1,)


def register_arch_viz_handlers(handlers_dict: dict) -> bool:
    handlers_dict["arch_diagram"] = h_arch_diagram
    handlers_dict["nn_schematic"] = h_nn_schematic
    return True


def register_arch_viz_specs(registry) -> bool:
    from ark.nodes.registry import NodeSpec, ParamSpec, SocketSpec

    registry.register(NodeSpec(
        name="arch_diagram", title="Architecture Diagram",
        category="Viz",
        inputs=[SocketSpec("trigger", "any")],
        outputs=[SocketSpec("html", "any")],
        params=[
            ParamSpec("output_path", "str", "./output/arch_diagram.html"),
            ParamSpec("title", "str", "Architecture ARK"),
            ParamSpec("auto_open", "bool", False),
        ],
        description="Génère organigramme SVG du graphe ARK : boxes + arrows. "
                       "Topo sort Sugiyama-style. HTML standalone.",
        role="Scene topologique → flowchart SVG.",
        function="Topo sort Kahn → layered layout → SVG boxes + bezier curves. "
                    "Scene injectée via set_scene() avant forward.",
        has_thumbnail=True,
        param_help={
            "output_path": "HTML output.",
            "auto_open": "Ouvrir auto dans navigateur.",
        },
    ))

    registry.register(NodeSpec(
        name="nn_schematic", title="Neural Network Schematic",
        category="Viz",
        inputs=[SocketSpec("trigger", "any")],
        outputs=[SocketSpec("html", "any")],
        params=[
            ParamSpec("output_path", "str", "./output/nn_schematic.html"),
            ParamSpec("title", "str", "Neural Network"),
            ParamSpec("auto_open", "bool", False),
            ParamSpec("sizes", "str", "784,128,64,10"),
            ParamSpec("activations", "str", "relu,relu,softmax"),
        ],
        description="Schéma classique neurones empilés + connexions denses. "
                       "Sample 16 neurones max par couche si layer plus grande.",
        role="Sizes list → classic NN diagram (circles + edges).",
        function="Pour chaque couche : N cercles + lignes denses vers couche "
                    "suivante. Stats params calculés depuis sizes pour Σ(sizes[i] × "
                    "sizes[i+1] + sizes[i+1]).",
        has_thumbnail=True,
        param_help={
            "sizes": "Tailles couches (CSV ints) ex. 784,128,64,10.",
            "activations": "Liste activations entre couches (CSV).",
        },
    ))
    return True
