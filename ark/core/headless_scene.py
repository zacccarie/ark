"""Phase 68 — Headless scene reconstruction (no Qt).

Permet compile_scene() côté serveur (Modal) sans PySide6. Reconstruit
une scène mock depuis un payload ark-arch-v1 JSON, avec objets plain Python
exposant l'API minimale attendue par graph_compiler :

  scene.nodes() → list[NodeProxy(id, spec, params, set_error, in/out sockets)]
  scene.connections() → list[ConnProxy(start_socket, end_socket)]
  socket.parent_node.id, socket.index

Réutilise default_registry() pour les NodeSpec.
"""
from __future__ import annotations
from typing import Any, Optional


class SocketProxy:
    def __init__(self, parent_node, index: int, is_input: bool):
        self.parent_node = parent_node
        self.index = index
        self.is_input = is_input


class NodeProxy:
    def __init__(self, node_id: str, spec, params: dict):
        self.id = node_id
        self.spec = spec
        self.params = dict(params or {})
        self.error: Optional[str] = None
        self.input_sockets = [
            SocketProxy(self, i, True) for i in range(len(spec.inputs))
        ]
        self.output_sockets = [
            SocketProxy(self, i, False) for i in range(len(spec.outputs))
        ]

    def set_error(self, msg):
        self.error = msg

    def set_param(self, name, value):
        self.params[name] = value


class ConnProxy:
    def __init__(self, start_socket, end_socket):
        self.start_socket = start_socket
        self.end_socket = end_socket


class HeadlessScene:
    def __init__(self, payload: dict):
        from ark.nodes.registry import default_registry
        reg = default_registry()
        self._nodes: list[NodeProxy] = []
        self._conns: list[ConnProxy] = []
        by_id: dict[str, NodeProxy] = {}

        for nd in payload.get("nodes", []):
            spec = reg.get(nd["type"])
            if spec is None:
                continue
            node = NodeProxy(nd["id"], spec, nd.get("params", {}))
            self._nodes.append(node)
            by_id[node.id] = node

        for c in payload.get("connections", []):
            src = by_id.get(c.get("from_node"))
            dst = by_id.get(c.get("to_node"))
            if src is None or dst is None:
                continue
            fs = int(c.get("from_socket", 0))
            ts = int(c.get("to_socket", 0))
            if fs >= len(src.output_sockets) or ts >= len(dst.input_sockets):
                continue
            self._conns.append(ConnProxy(
                src.output_sockets[fs], dst.input_sockets[ts]))

    def nodes(self):
        return list(self._nodes)

    def connections(self):
        return list(self._conns)

    def items(self):
        return list(self._nodes)


def compile_payload(payload: dict, num_classes: Optional[int] = None):
    """arch JSON payload → CompiledGraph (sans Qt)."""
    from ark.core.graph_compiler import compile_scene
    scene = HeadlessScene(payload)
    return compile_scene(scene, num_classes=num_classes)
