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


def _load_meta(signature: str) -> dict:
    jp = WEIGHTS_DIR / f"{signature}.json"
    if jp.exists():
        try:
            return json.loads(jp.read_text())
        except Exception:
            return {}
    return {}


def _write_meta(signature: str, m: dict):
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    (WEIGHTS_DIR / f"{signature}.json").write_text(json.dumps(m, indent=2))


def weights_info(signature: str) -> dict:
    """{exists, saved_at, val_acc, n_params, kind, trained, validated,
    operational, test_metric, test_metric_name, test_gap}."""
    pt = WEIGHTS_DIR / f"{signature}.pt"
    if not pt.exists():
        return {"exists": False}
    out = {"exists": True}
    out.update(_load_meta(signature))
    # États lus tels quels (plus dérivés de val_acc > 0 — Phase 81)
    out["validated"] = bool(out.get("validated", False))
    out["operational"] = bool(out.get("operational", False))
    return out


def model_state_from_info(info: dict) -> str:
    """blank → trained → validated → operational (Phase 81)."""
    if not info.get("exists"):
        return "blank"
    if info.get("operational"):
        return "operational"
    if info.get("validated"):
        return "validated"
    return "trained"


def mark_trained(signature: str, train_acc: float = 0.0, val_acc: float = 0.0,
                 kind: str = "classifier", extra: Optional[dict] = None):
    """Post-entraînement : poids persistés, PAS encore validés (held-out)."""
    m = _load_meta(signature)
    m["trained_at"] = time.time()
    m["train_acc"] = float(train_acc)
    m["val_acc"] = float(val_acc)
    m["kind"] = kind
    m["validated"] = False
    m["operational"] = False
    if extra:
        m.update(extra)
    _write_meta(signature, m)


def mark_validated(signature: str, test_metric: float, metric_name: str = "accuracy",
                   gap: float = 0.0, kind: str = "classifier",
                   extra: Optional[dict] = None):
    """Test held-out passé → modèle VALIDÉ (généralise)."""
    m = _load_meta(signature)
    m["validated"] = True
    m["validated_at"] = time.time()
    m["test_metric"] = float(test_metric)
    m["test_metric_name"] = metric_name
    m["test_gap"] = float(gap)
    m["kind"] = kind
    if extra:
        m.update(extra)
    _write_meta(signature, m)


def mark_operational(signature: str, val_acc: float, extra: Optional[dict] = None):
    """Mis en production (Phase 83 ship). Conserve val_acc pour compat."""
    m = _load_meta(signature)
    m["val_acc"] = float(val_acc)
    m["operational"] = True
    m["shipped_at"] = time.time()
    if extra:
        m.update(extra)
    _write_meta(signature, m)


def clear_weights(signature: str):
    for ext in (".pt", ".json"):
        p = WEIGHTS_DIR / f"{signature}{ext}"
        if p.exists():
            p.unlink()
