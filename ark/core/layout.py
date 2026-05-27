"""Auto-layout algo — topological levels horizontaux + stagger vertical."""
from __future__ import annotations
from collections import defaultdict, deque


def compute_layout(scene, h_spacing: float = 260, v_spacing: float = 160,
                     x_origin: float = 80, y_origin: float = 80) -> dict:
    """Renvoie {node_id: (x, y)} via topo levels.

    Algo :
    1. Topo sort + assign chaque node à un level (longest path depuis source).
    2. Pour chaque level, stagger Y autour de y_origin.
    3. x = x_origin + level * h_spacing
    """
    nodes = scene.nodes()
    if not nodes:
        return {}
    by_id = {n.id: n for n in nodes}
    incoming = defaultdict(list)
    outgoing = defaultdict(list)
    for c in scene.connections():
        if c.start_socket and c.end_socket:
            src = c.start_socket.parent_node.id
            dst = c.end_socket.parent_node.id
            incoming[dst].append(src)
            outgoing[src].append(dst)
    indeg = {nid: len(incoming[nid]) for nid in by_id}

    # Longest path from sources → level
    level: dict = {}
    queue = deque([nid for nid in by_id if indeg[nid] == 0])
    for nid in queue:
        level[nid] = 0
    while queue:
        nid = queue.popleft()
        for nxt in outgoing[nid]:
            new_l = level[nid] + 1
            level[nxt] = max(level.get(nxt, 0), new_l)
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    # Nodes en cycle / déconnectés : level 0 par défaut
    for nid in by_id:
        level.setdefault(nid, 0)

    # Group par level
    by_level: dict = defaultdict(list)
    for nid, l in level.items():
        by_level[l].append(nid)

    out: dict = {}
    for l, nids in by_level.items():
        # Stagger vertical : centré sur y_origin
        n = len(nids)
        offset = -(n - 1) / 2
        for i, nid in enumerate(sorted(nids)):
            x = x_origin + l * h_spacing
            y = y_origin + (offset + i) * v_spacing
            out[nid] = (x, y)
    return out


def apply_layout(scene, positions: dict) -> None:
    """Applique positions à la scene."""
    from PySide6.QtCore import QPointF
    for n in scene.nodes():
        if n.id in positions:
            x, y = positions[n.id]
            n.setPos(QPointF(x, y))
            for sock in n.input_sockets + n.output_sockets:
                for conn in sock.connections:
                    conn.update_path()


def auto_layout_scene(scene, h_spacing: float = 260, v_spacing: float = 160) -> None:
    pos = compute_layout(scene, h_spacing=h_spacing, v_spacing=v_spacing)
    apply_layout(scene, pos)
