"""Import depuis ai-arch-builder (Electron) → ARK format.

ai-arch-builder JSON :
- nodes : [{id, type, params: {camelCase}, x, y}]
- connections : [{from: node_id, to: node_id}]
- type mapping :
  * conv2d (kernelSize, filters, activation) → conv2d (kernel_size, filters, activation)
  * dense (neurons, activation, dropout) → dense (idem)
  * normalize → normalize (kind='batchnorm')
  * residual-add → residual_add
  * transformer-block → transformer_block
  * moe-switch → moe_switch (d_out, n_experts)
  * vg-block → vg_block (start_hidden, out_dim, activation)
  * observer → observer (metrics, buffer_size)
  * trigger-rule → trigger_rule (metric, op, threshold, action_op, target_node, delta)
  * image-input → image_input
  * tabular-input → tabular_input
  * text-input → text_input
  * adam → adam (learningRate → learning_rate)
  * cross-entropy → cross_entropy

Format metamorph-arch-v1 (déjà géré par format.payload_to_scene) — pas
besoin de mapping particulier.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from datetime import datetime, timezone


# Mapping type renderer Electron → type ARK (snake_case)
TYPE_MAP = {
    "dense": "dense",
    "conv2d": "conv2d",
    "normalize": "normalize",
    "residual-add": "residual_add",
    "concat": "concat",  # pas encore implémenté ARK, mais reconnu
    "attention": "attention",
    "transformer-block": "transformer_block",
    "moe-switch": "moe_switch",
    "vg-block": "vg_block",
    "observer": "observer",
    "trigger-rule": "trigger_rule",
    "image-input": "image_input",
    "tabular-input": "tabular_input",
    "text-input": "text_input",
    "adam": "adam",
    "cross-entropy": "cross_entropy",
    "constant": "constant",
    "math-op": "math_op",
    "plot-scatter": "plot_scatter",
    "plot-histogram": "plot_line",  # ARK n'a pas histogram, fallback line
    "plot-confusion": "plot_scatter",
}

# Mapping nom param camelCase → snake_case
PARAM_RENAMES = {
    "kernelSize": "kernel_size",
    "learningRate": "learning_rate",
    "embedDim": "embed_dim",
    "maxLength": "max_length",
    "n_heads": "n_heads",      # déjà bon
    "d_ff": "d_ff",
    "n_experts": "n_experts",
    "d_out": "d_out",
    "out_dim": "out_dim",
    "start_hidden": "start_hidden",
    "buffer_size": "buffer_size",
    "action_op": "action_op",
    "target_node": "target_node",
    "n_components": "n_components",
}


def _camel_to_snake(s: str) -> str:
    """Fallback générique : camelCase → snake_case."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()


def _convert_params(type_name: str, params: dict) -> dict:
    out = {}
    for k, v in params.items():
        if k in PARAM_RENAMES:
            out[PARAM_RENAMES[k]] = v
        else:
            new_k = _camel_to_snake(k)
            out[new_k] = v
    return out


def detect_format(payload: dict) -> str:
    """Auto-détecte le format. Returns 'ark' | 'metamorph' | 'electron' | 'unknown'."""
    ver = payload.get("version")
    if ver == "ark-arch-v1":
        return "ark"
    if ver == "metamorph-arch-v1":
        return "metamorph"
    # ai-arch-builder n'a pas de champ version
    nodes = payload.get("nodes", [])
    conns = payload.get("connections", [])
    if nodes and not ver:
        # Heuristique : si pos format renderer (x, y direct sur node) OR
        # types avec tirets (kebab-case)
        if any("-" in (n.get("type") or "") for n in nodes):
            return "electron"
        if any(("x" in n or "y" in n) and "pos" not in n for n in nodes):
            return "electron"
    return "unknown"


def convert_electron_to_ark(payload: dict) -> dict:
    """ai-arch-builder JSON → ark-arch-v1 payload."""
    out_nodes = []
    unknown = []
    for n in payload.get("nodes", []):
        elec_type = n.get("type", "")
        ark_type = TYPE_MAP.get(elec_type)
        if ark_type is None:
            unknown.append(elec_type)
            continue
        params = _convert_params(elec_type, n.get("params", {}))
        out_nodes.append({
            "id": n["id"],
            "type": ark_type,
            "params": params,
            "pos": [float(n.get("x", 200)), float(n.get("y", 200))],
            "width": int(n.get("width", 220)),
            "bypass": bool(n.get("bypass", False)),
        })
    out_conns = []
    for c in payload.get("connections", []):
        out_conns.append({
            "from_node": c.get("from") or c.get("from_node"),
            "from_socket": 0,
            "to_node": c.get("to") or c.get("to_node"),
            "to_socket": 0,
        })
    return {
        "version": "ark-arch-v1",
        "name": payload.get("name", "imported_from_electron"),
        "nodes": out_nodes,
        "connections": out_conns,
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "imported_from": "ai-arch-builder",
            "unknown_types_skipped": unknown,
        },
    }


def load_any(path: str | Path) -> tuple[dict, str]:
    """Charge un fichier JSON (n'importe quel format) et retourne (payload_ark, fmt_origine)."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    fmt = detect_format(raw)
    if fmt == "ark":
        return raw, "ark"
    if fmt == "metamorph":
        # format.payload_to_scene gère déjà la compat metamorph
        return raw, "metamorph"
    if fmt == "electron":
        return convert_electron_to_ark(raw), "electron"
    raise ValueError(f"Format JSON non reconnu : {path}")
