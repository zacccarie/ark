"""ARK save/load — format `ark-arch-v1` (compat avec metamorph-arch-v1)."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path


ARK_FORMAT_VERSION = "ark-arch-v1"


def scene_to_payload(scene, name: str = "ark_graph",
                       num_classes: int | None = None,
                       tags: list[str] | None = None) -> dict:
    """Sérialise une NodeScene en dict ark-arch-v1."""
    nodes = []
    for n in scene.nodes():
        pos = n.pos()
        nodes.append({
            "id": n.id,
            "type": n.spec.name,
            "params": dict(n.params),
            "bypass": bool(n.bypass),
            "width": int(n.width),
            "height_extra": int(getattr(n, "user_height_extra", 0)),
            "pos": [float(pos.x()), float(pos.y())],
        })
    connections = []
    for c in scene.connections():
        if c.start_socket is None or c.end_socket is None:
            continue
        connections.append({
            "from_node": c.start_socket.parent_node.id,
            "from_socket": c.start_socket.index,
            "to_node": c.end_socket.parent_node.id,
            "to_socket": c.end_socket.index,
        })
    return {
        "version": ARK_FORMAT_VERSION,
        "name": name,
        "nodes": nodes,
        "connections": connections,
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "num_classes": num_classes,
            "tags": list(tags) if tags else [],
        },
    }


def payload_to_scene(scene, payload: dict) -> None:
    """Reconstruit la scène depuis un payload ark-arch-v1 (clear puis rebuild)."""
    if payload.get("version") not in (ARK_FORMAT_VERSION, "metamorph-arch-v1"):
        raise ValueError(f"Unsupported format version: {payload.get('version')}")
    scene.clear_graph()
    # Map ancien id (du JSON) → nouveau NodeItem
    id_map: dict[str, str] = {}
    nodes_data = payload.get("nodes", [])
    # Note : si format metamorph-arch-v1, les pos sont peut-être 'x'/'y'
    for nd in nodes_data:
        ntype = nd.get("type")
        pos = nd.get("pos") or [nd.get("x", 0), nd.get("y", 0)]
        node = scene._raw_create_node(ntype, float(pos[0]), float(pos[1]))
        if node is None:
            continue
        # Restaure params
        for k, v in (nd.get("params") or {}).items():
            if k in node.params:
                node.params[k] = v
        node.bypass = bool(nd.get("bypass", False))
        if "width" in nd:
            try:
                node.width = int(nd["width"])
                node._layout_sockets()
            except Exception:
                pass
        if "height_extra" in nd:
            try:
                node.user_height_extra = int(nd["height_extra"])
                node.prepareGeometryChange()
            except Exception:
                pass
        id_map[nd["id"]] = node.id

    # Remap params référençant des node ids (e.g. trigger_rule.target_node)
    for node in scene.nodes():
        for k, v in list(node.params.items()):
            if isinstance(v, str) and v in id_map:
                node.params[k] = id_map[v]

    # Rebuild connections
    from ark.ui.node_editor.connection_item import ConnectionItem
    nodes_by_id = {n.id: n for n in scene.nodes()}
    for cd in payload.get("connections", []):
        if "from_node" in cd:
            from_nid = id_map.get(cd["from_node"])
            to_nid = id_map.get(cd["to_node"])
            from_sock_idx = int(cd.get("from_socket", 0))
            to_sock_idx = int(cd.get("to_socket", 0))
        else:
            # Fallback format metamorph (from/to = id)
            from_nid = id_map.get(cd.get("from"))
            to_nid = id_map.get(cd.get("to"))
            from_sock_idx = 0
            to_sock_idx = 0
        if not from_nid or not to_nid:
            continue
        a = nodes_by_id.get(from_nid)
        b = nodes_by_id.get(to_nid)
        if not a or not b:
            continue
        if from_sock_idx >= len(a.output_sockets) or to_sock_idx >= len(b.input_sockets):
            continue
        conn = ConnectionItem(start_socket=a.output_sockets[from_sock_idx],
                                end_socket=b.input_sockets[to_sock_idx])
        a.output_sockets[from_sock_idx].add_connection(conn)
        b.input_sockets[to_sock_idx].add_connection(conn)
        scene._raw_add_connection(conn)


def save_scene(scene, path: str | Path, name: str = "ark_graph",
                num_classes: int | None = None,
                tags: list[str] | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = scene_to_payload(scene, name=name, num_classes=num_classes, tags=tags)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return p


def load_scene(scene, path: str | Path) -> dict:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    payload_to_scene(scene, payload)
    return payload
