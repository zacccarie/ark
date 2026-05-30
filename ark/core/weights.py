"""Phase 80 — Persistance des poids + état d'entraînement du modèle.

Problème résolu : compile_scene reconstruit le modèle avec poids aléatoires
à chaque compile. Ici on sauvegarde/recharge le state_dict pour qu'un
modèle entraîné garde ses poids.

Signature : hash stable de l'architecture (types de nodes + connexions +
params déterminant la forme). Si l'archi change, la signature change → on
ne recharge pas des poids incompatibles.

Stockage : ~/.ark/scene_weights/<signature>.pt + <signature>.json (méta).
"""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import torch

WEIGHTS_DIR = Path.home() / ".ark" / "scene_weights"

# Params qui changent la FORME (donc invalident les poids)
_SHAPE_PARAMS = {
    "neurons", "channels", "out_features", "hidden_dim", "out_dim",
    "features", "d_model", "n_blocks", "n_heads", "num_layers", "num_heads",
    "kernel", "stride", "padding", "base_ch", "cond_dim", "in_ch", "vocab_size",
    "embed_dim", "max_len", "block_size", "num_classes",
}


def scene_signature(nodes, connections) -> str:
    """Hash stable de l'archi. nodes : list avec .id/.spec.name/.params.
    connections : list de (from_id, from_sock, to_id, to_sock)."""
    parts = []
    # nodes triés par id pour stabilité
    id_order = {}
    for i, n in enumerate(sorted(nodes, key=lambda x: x.id)):
        id_order[n.id] = i
        shape_p = {k: v for k, v in n.params.items() if k in _SHAPE_PARAMS}
        parts.append(f"{i}:{n.spec.name}:{sorted(shape_p.items())}")
    # connexions normalisées via id_order
    conn_sig = []
    for (fid, fs, tid, ts) in connections:
        if fid in id_order and tid in id_order:
            conn_sig.append((id_order[fid], fs, id_order[tid], ts))
    conn_sig.sort()
    parts.append(f"conns:{conn_sig}")
    blob = "|".join(parts)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def signature_from_scene(scene) -> str:
    conns = []
    for c in scene.connections():
        if c.start_socket is None or c.end_socket is None:
            continue
        conns.append((c.start_socket.parent_node.id,
                       getattr(c.start_socket, "index", 0),
                       c.end_socket.parent_node.id,
                       getattr(c.end_socket, "index", 0)))
    return scene_signature(scene.nodes(), conns)


def save_weights(model, signature: str, meta: Optional[dict] = None) -> Path:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    pt = WEIGHTS_DIR / f"{signature}.pt"
    torch.save(model.state_dict(), pt)
    m = {"signature": signature, "saved_at": time.time(),
         "n_params": sum(p.numel() for p in model.parameters())}
    if meta:
        m.update(meta)
    (WEIGHTS_DIR / f"{signature}.json").write_text(
        json.dumps(m, indent=2))
    return pt


def load_weights(model, signature: str) -> bool:
    pt = WEIGHTS_DIR / f"{signature}.pt"
    if not pt.exists():
        return False
    try:
        sd = torch.load(pt, map_location="cpu")
        model.load_state_dict(sd, strict=True)
        return True
    except Exception:
        # Forme incompatible → ignore
        return False


def weights_info(signature: str) -> dict:
    """{exists, trained_at, val_acc, n_params, operational}."""
    jp = WEIGHTS_DIR / f"{signature}.json"
    pt = WEIGHTS_DIR / f"{signature}.pt"
    if not pt.exists():
        return {"exists": False}
    out = {"exists": True}
    if jp.exists():
        try:
            out.update(json.loads(jp.read_text()))
        except Exception:
            pass
    out["operational"] = bool(out.get("val_acc", 0) and
                                  out.get("val_acc", 0) > 0)
    return out


def mark_operational(signature: str, val_acc: float, extra: Optional[dict] = None):
    jp = WEIGHTS_DIR / f"{signature}.json"
    m = {}
    if jp.exists():
        try:
            m = json.loads(jp.read_text())
        except Exception:
            pass
    m["val_acc"] = float(val_acc)
    m["operational"] = True
    m["validated_at"] = time.time()
    if extra:
        m.update(extra)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(m, indent=2))


def clear_weights(signature: str):
    for ext in (".pt", ".json"):
        p = WEIGHTS_DIR / f"{signature}{ext}"
        if p.exists():
            p.unlink()
