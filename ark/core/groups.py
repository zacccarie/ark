"""Phase 31 — Node Groups : sélection → composite node réutilisable.

Format JSON `ark-group-v1` sauvé dans `~/.ark/groups/<name>.json`.

Workflow :
1. User sélectionne plusieurs nodes connectés dans la scene.
2. Ctrl+G ou menu → dialog (name/title/category/description).
3. save_group() écrit JSON.
4. Au prochain start (ou rescan), register_group_specs() charge chaque fichier
   et expose un NodeSpec custom dans la palette (category 'User Groups' par défaut).
5. Lors du compile, handler instancie un GroupModule(nn.Module) qui orchestre
   les sous-modules selon les connexions internes.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


ARK_GROUP_VERSION = "ark-group-v1"
GROUPS_DIR = Path.home() / ".ark" / "groups"


@dataclass
class ExposedSocket:
    """Socket interne exposé comme I/O externe du group."""
    node_id: str
    port: int
    label: str


# ============================================================
# Serialization
# ============================================================

def _slug(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "group"


def serialize_group(scene, selected_nodes: list, name: str, title: str,
                     category: str = "User Groups",
                     description: str = "") -> dict:
    """Construit payload group depuis sélection.

    Détecte automatiquement les sockets exposés :
    - inputs exposés = sockets input non-connectés EN INTERNE (signal vient du dehors)
    - outputs exposés = sockets output non-connectés EN INTERNE (signal va dehors)
    """
    if not selected_nodes:
        raise ValueError("sélection vide")
    selected_ids = {n.id for n in selected_nodes}

    # Nodes payload
    nodes_payload = []
    for n in selected_nodes:
        pos = n.pos()
        nodes_payload.append({
            "id": n.id,
            "type": n.spec.name,
            "params": dict(n.params),
            "bypass": bool(n.bypass),
            "width": int(getattr(n, "width", 200)),
            "pos": [float(pos.x()), float(pos.y())],
        })

    # Connexions INTERNES seulement (les 2 endpoints sont dans la sélection)
    internal_conns = []
    crossing_in: dict[tuple[str, int], int] = {}  # (node_id, port) → count externe
    crossing_out: dict[tuple[str, int], int] = {}
    for c in scene.connections():
        if c.start_socket is None or c.end_socket is None:
            continue
        from_nid = c.start_socket.parent_node.id
        to_nid = c.end_socket.parent_node.id
        from_in_sel = from_nid in selected_ids
        to_in_sel = to_nid in selected_ids
        if from_in_sel and to_in_sel:
            internal_conns.append({
                "from_node": from_nid,
                "from_socket": c.start_socket.index,
                "to_node": to_nid,
                "to_socket": c.end_socket.index,
            })
        elif from_in_sel and not to_in_sel:
            key = (from_nid, c.start_socket.index)
            crossing_out[key] = crossing_out.get(key, 0) + 1
        elif to_in_sel and not from_in_sel:
            key = (to_nid, c.end_socket.index)
            crossing_in[key] = crossing_in.get(key, 0) + 1

    # Auto-détecte sockets exposés
    # input exposé = socket input sans connexion interne entrante
    nodes_by_id = {n.id: n for n in selected_nodes}
    connected_internal_inputs: set[tuple[str, int]] = set()
    connected_internal_outputs: set[tuple[str, int]] = set()
    for ic in internal_conns:
        connected_internal_inputs.add((ic["to_node"], ic["to_socket"]))
        connected_internal_outputs.add((ic["from_node"], ic["from_socket"]))

    exposed_inputs: list[dict] = []
    exposed_outputs: list[dict] = []
    for n in selected_nodes:
        for i, sock in enumerate(n.spec.inputs):
            if (n.id, i) not in connected_internal_inputs:
                exposed_inputs.append({
                    "node_id": n.id, "port": i,
                    "label": f"{n.spec.title}.{sock.label}",
                    "socket_type": sock.socket_type,
                })
        for i, sock in enumerate(n.spec.outputs):
            if (n.id, i) not in connected_internal_outputs:
                exposed_outputs.append({
                    "node_id": n.id, "port": i,
                    "label": f"{n.spec.title}.{sock.label}",
                    "socket_type": sock.socket_type,
                })

    return {
        "version": ARK_GROUP_VERSION,
        "name": _slug(name),
        "title": title or name,
        "category": category,
        "description": description,
        "nodes": nodes_payload,
        "connections": internal_conns,
        "exposed_inputs": exposed_inputs,
        "exposed_outputs": exposed_outputs,
    }


def save_group(payload: dict, path: Path | None = None) -> Path:
    name = payload.get("name", "group")
    if path is None:
        GROUPS_DIR.mkdir(parents=True, exist_ok=True)
        path = GROUPS_DIR / f"{name}.json"
    else:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def load_group(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def list_group_files() -> list[Path]:
    if not GROUPS_DIR.exists():
        return []
    return sorted(GROUPS_DIR.glob("*.json"))


# ============================================================
# Compile : group payload → nn.Module
# ============================================================

class GroupModule(nn.Module):
    """Wrap un sous-graphe ARK. Forward orchestre topologiquement.

    Pour simplicité v1 : sequential des nodes selon ordre topologique des
    connexions internes. Inputs externes injectés via exposed_inputs, outputs
    récupérés via exposed_outputs.
    """

    def __init__(self, payload: dict, handlers: dict):
        super().__init__()
        self.payload = payload
        self.handlers = handlers
        self.exposed_inputs = payload.get("exposed_inputs", [])
        self.exposed_outputs = payload.get("exposed_outputs", [])
        self._build()

    def _topo_order(self) -> list[str]:
        nodes = self.payload.get("nodes", [])
        conns = self.payload.get("connections", [])
        # Adjacency
        graph: dict[str, list[str]] = {n["id"]: [] for n in nodes}
        in_deg: dict[str, int] = {n["id"]: 0 for n in nodes}
        for c in conns:
            if c["to_node"] in graph and c["from_node"] in graph:
                graph[c["from_node"]].append(c["to_node"])
                in_deg[c["to_node"]] += 1
        # Kahn
        queue = [nid for nid, d in in_deg.items() if d == 0]
        order: list[str] = []
        while queue:
            n = queue.pop(0)
            order.append(n)
            for tgt in graph[n]:
                in_deg[tgt] -= 1
                if in_deg[tgt] == 0:
                    queue.append(tgt)
        # Ajoute les nodes restants (cycles) en fin
        for nid in in_deg:
            if nid not in order:
                order.append(nid)
        return order

    def _build(self):
        from ark.nodes.registry import default_registry
        self._registry = default_registry()
        self._modules_by_id: dict[str, nn.Module] = {}
        self._order = self._topo_order()
        self._shapes: dict[str, Any] = {}
        # in_shape par défaut (heuristique) — sera infered lors du forward réel
        default_shape = (1,)
        from types import SimpleNamespace
        for nid in self._order:
            ndata = next((n for n in self.payload["nodes"] if n["id"] == nid), None)
            if ndata is None:
                continue
            ntype = ndata["type"]
            spec = self._registry.get(ntype)
            handler = self.handlers.get(ntype)
            if handler is None:
                # No handler : skip (pass-through Identity)
                self._modules_by_id[nid] = nn.Identity()
                continue
            fake_node = SimpleNamespace(id=nid, params=ndata.get("params", {}))
            try:
                m, out_shape = handler(fake_node, [default_shape])
                self._modules_by_id[nid] = m
                self._shapes[nid] = out_shape
                self.add_module(f"sub_{nid[:8]}", m)
            except Exception:
                self._modules_by_id[nid] = nn.Identity()

    def forward(self, *args):
        """Forward sequential par ordre topo.

        Pour v1 : on assume 1 input principal (premier exposed input).
        Le tensor circule : exposed_input → sequentially modules → exposed_output.
        """
        if not args:
            return None
        x = args[0]
        # Map node_id → output current
        outputs: dict[str, torch.Tensor] = {}
        # Premier input exposé reçoit x
        if self.exposed_inputs:
            outputs[self.exposed_inputs[0]["node_id"]] = x

        conns = self.payload.get("connections", [])
        adj_in: dict[str, list[str]] = {}
        for c in conns:
            adj_in.setdefault(c["to_node"], []).append(c["from_node"])

        for nid in self._order:
            m = self._modules_by_id.get(nid)
            if m is None:
                continue
            # Input = output du prédécesseur (s'il y en a)
            if nid in outputs:
                inp = outputs[nid]
            elif nid in adj_in and adj_in[nid]:
                pred = adj_in[nid][0]
                inp = outputs.get(pred, x)
            else:
                inp = x
            try:
                out = m(inp)
                if isinstance(out, tuple):
                    out = out[0]
                outputs[nid] = out
            except Exception:
                outputs[nid] = inp  # pass-through si erreur

        # Output = exposed output
        if self.exposed_outputs:
            return outputs.get(self.exposed_outputs[0]["node_id"], x)
        # Fallback : dernier output
        return outputs.get(self._order[-1], x) if self._order else x


# ============================================================
# Auto-registration depuis ~/.ark/groups/
# ============================================================

def _make_group_handler(payload: dict):
    """Factory handler. Capture payload par closure."""
    def handler(node, in_shapes):
        from ark.core.graph_compiler import HANDLERS
        m = GroupModule(payload, HANDLERS)
        m._ark_node_id = getattr(node, "id", "group")
        out_shape = in_shapes[0] if in_shapes else (1,)
        return m, out_shape
    return handler


def register_groups(registry, handlers_dict: dict) -> int:
    """Scan ~/.ark/groups/*.json et enregistre comme NodeSpec + handler."""
    from ark.nodes.registry import NodeSpec, SocketSpec

    count = 0
    for path in list_group_files():
        try:
            payload = load_group(path)
        except Exception:
            continue
        if payload.get("version") != ARK_GROUP_VERSION:
            continue
        name = payload["name"]
        if registry.get(name) is not None:
            # Déjà enregistré (sans doute reload)
            continue
        title = payload.get("title", name)
        category = payload.get("category", "User Groups")
        description = payload.get("description", "")
        # Build sockets depuis exposed
        inputs = [SocketSpec(s.get("label", f"in_{i}"),
                                s.get("socket_type", "tensor"))
                    for i, s in enumerate(payload.get("exposed_inputs", []))]
        outputs = [SocketSpec(s.get("label", f"out_{i}"),
                                 s.get("socket_type", "tensor"))
                     for i, s in enumerate(payload.get("exposed_outputs", []))]
        if not inputs:
            inputs = [SocketSpec("in", "tensor")]
        if not outputs:
            outputs = [SocketSpec("out", "tensor")]

        n_internal = len(payload.get("nodes", []))
        spec = NodeSpec(
            name=name, title=title, category=category,
            inputs=inputs, outputs=outputs, params=[],
            description=description or f"Composite : {n_internal} nodes internes.",
            role=description or f"Composite Group ({n_internal} nodes).",
            function=f"Sous-graphe sauvé contenant {n_internal} nodes "
                       f"et {len(payload.get('connections', []))} connexions internes.",
        )
        try:
            registry.register(spec)
            handlers_dict[name] = _make_group_handler(payload)
            count += 1
        except Exception:
            pass
    return count
