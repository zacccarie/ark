"""ARK agent CLI — pilote ARK depuis terminal sans GUI.

Permet à Claude / scripts / automatisations de construire et compiler des
architectures sans l'interface PySide6. Utile pour CI, benchmark, batch
processing.

CLI :
    python -m ark.agent_api list-templates
    python -m ark.agent_api list-nodes [--category Neural]
    python -m ark.agent_api inspect <template_name>
    python -m ark.agent_api compile <arch.json> [--num-classes 10]
    python -m ark.agent_api export python <arch.json> <out.py>
    python -m ark.agent_api export onnx <arch.json> <out.onnx>
    python -m ark.agent_api convert <electron.json> <out_ark.json>
    python -m ark.agent_api quick-mlp --layers 128,64,10 --out arch.json
    python -m ark.agent_api quick-cnn --filters 16,32 --out arch.json
    python -m ark.agent_api quick-resnet --filters 16 --out arch.json
    python -m ark.agent_api quick-transformer --d-model 64 --n-blocks 2 --out arch.json

Lib (import depuis Python) :
    from ark.agent_api import build_quick_mlp, compile_payload, simulate_forward
    g = build_quick_mlp(layers=[64, 32, 10])
    sim = simulate_forward(g, input_shape=(1, 28, 28), num_classes=10)
"""
from __future__ import annotations
import argparse
import json
import sys
import uuid
from pathlib import Path

import torch

from ark.nodes.registry import default_registry
from ark.core.templates import TEMPLATES_LIST, get_template
from ark.core.import_legacy import load_any, convert_electron_to_ark


# ============================================================
# Quick builders (sans Qt)
# ============================================================

def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:6]}"


def _node(type_name: str, x: float, y: float, params: dict | None = None) -> dict:
    return {
        "id": _uid(type_name),
        "type": type_name,
        "params": params or {},
        "pos": [x, y],
        "width": 220,
    }


def _conn(a: dict, b: dict, fs: int = 0, ts: int = 0) -> dict:
    return {"from_node": a["id"], "from_socket": fs, "to_node": b["id"], "to_socket": ts}


def _wrap(name: str, nodes: list, connections: list,
            num_classes: int | None = None, tags: list[str] | None = None) -> dict:
    from datetime import datetime, timezone
    return {
        "version": "ark-arch-v1",
        "name": name,
        "nodes": nodes,
        "connections": connections,
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "num_classes": num_classes,
            "tags": tags or [],
        },
    }


def build_quick_mlp(layers: list[int] = [128, 64, 10],
                      input_features: int = 784,
                      activation: str = "gelu",
                      dropout: float = 0.0) -> dict:
    nodes = [_node("tabular_input", 0, 200, {"features": input_features})]
    prev = nodes[0]
    for i, n in enumerate(layers):
        params = {"neurons": n, "activation": "none" if i == len(layers) - 1 else activation}
        if dropout > 0 and i < len(layers) - 1:
            params["dropout"] = dropout
        node = _node("dense", 220 * (i + 1), 200, params)
        nodes.append(node)
    conns = [_conn(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
    return _wrap("quick_mlp", nodes, conns,
                  num_classes=layers[-1], tags=["agent", "mlp"])


def build_quick_cnn(filters: list[int] = [16, 32],
                      dense_layers: list[int] = [128, 10],
                      input_shape: list = [1, 28, 28],
                      kernel: int = 3,
                      activation: str = "relu") -> dict:
    nodes = [_node("image_input", 0, 200, {"shape": str(input_shape).replace(' ', '')})]
    prev = nodes[0]
    for i, f in enumerate(filters):
        n = _node("conv2d", 220 * (i + 1), 200,
                    {"filters": f, "kernel_size": kernel, "activation": activation})
        nodes.append(n)
        prev = n
    x_off = 220 * (len(filters) + 1)
    for i, neurons in enumerate(dense_layers):
        n = _node("dense", x_off + 220 * i, 200,
                    {"neurons": neurons,
                     "activation": "none" if i == len(dense_layers) - 1 else activation})
        nodes.append(n)
    conns = [_conn(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
    return _wrap("quick_cnn", nodes, conns,
                  num_classes=dense_layers[-1], tags=["agent", "cnn"])


def build_quick_resnet(filters: int = 16,
                          input_shape: list = [3, 32, 32],
                          kernel: int = 3) -> dict:
    inp = _node("image_input", 0, 200, {"shape": str(input_shape).replace(' ', '')})
    c1 = _node("conv2d", 220, 200,
                 {"filters": filters, "kernel_size": kernel, "activation": "relu"})
    bn1 = _node("normalize", 440, 200, {"kind": "batchnorm"})
    c2 = _node("conv2d", 660, 200,
                 {"filters": filters, "kernel_size": kernel, "activation": "none"})
    bn2 = _node("normalize", 880, 200, {"kind": "batchnorm"})
    add = _node("residual_add", 1100, 200)
    head = _node("dense", 1320, 200, {"neurons": 10, "activation": "none"})
    conns = [
        _conn(inp, c1), _conn(c1, bn1), _conn(bn1, c2), _conn(c2, bn2),
        _conn(c1, add, ts=0), _conn(bn2, add, ts=1),
        _conn(add, head),
    ]
    return _wrap("quick_resnet", [inp, c1, bn1, c2, bn2, add, head], conns,
                  num_classes=10, tags=["agent", "resnet"])


def build_quick_transformer(n_blocks: int = 2, d_model: int = 64,
                              n_heads: int = 4, d_ff: int = 128,
                              max_len: int = 16) -> dict:
    nodes = [_node("text_input", 0, 200,
                     {"max_length": max_len, "embed_dim": d_model})]
    for i in range(n_blocks):
        nodes.append(_node("transformer_block", 220 * (i + 1), 200,
                              {"n_heads": n_heads, "d_ff": d_ff}))
    head = _node("dense", 220 * (n_blocks + 1), 200,
                   {"neurons": 10, "activation": "none"})
    nodes.append(head)
    conns = [_conn(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
    return _wrap("quick_transformer", nodes, conns,
                  num_classes=10, tags=["agent", "transformer"])


# ============================================================
# Compile / simulate / export (sans Qt)
# ============================================================

def _build_headless_scene(payload: dict):
    """Crée NodeScene Qt-headless avec QApplication minimale en mémoire."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from ark.app import create_app
    from ark.ui.node_editor.scene import NodeScene
    from ark.core.format import payload_to_scene
    create_app([])
    scene = NodeScene(registry=default_registry())
    payload_to_scene(scene, payload)
    return scene


def compile_payload(payload: dict, num_classes: int | None = None) -> dict:
    """Compile + retourne summary dict (n_params, shapes, modules)."""
    from ark.core.graph_compiler import compile_scene
    scene = _build_headless_scene(payload)
    cg = compile_scene(scene, num_classes=num_classes)
    n_params = sum(p.numel() for p in cg.model.parameters())
    return {
        "name": payload.get("name"),
        "input_shape": list(cg.input_shape),
        "output_shape": list(cg.output_shape),
        "num_classes": cg.num_classes,
        "n_params": n_params,
        "n_modules": len(cg.model.modules_dict),
        "modules": list(cg.model.modules_dict.keys()),
    }


def simulate_forward(payload: dict, input_shape: tuple | None = None,
                       num_classes: int | None = None, batch_size: int = 2) -> dict:
    """Compile + forward dummy → output shape."""
    from ark.core.graph_compiler import compile_scene
    scene = _build_headless_scene(payload)
    cg = compile_scene(scene, num_classes=num_classes)
    in_shape = input_shape or cg.input_shape
    x = torch.randn(batch_size, *in_shape)
    with torch.no_grad():
        y = cg.model(x)
    return {
        "input_shape": [batch_size, *list(in_shape)],
        "output_shape": list(y.shape),
        "n_params": sum(p.numel() for p in cg.model.parameters()),
    }


def export_payload(payload: dict, kind: str, out_path: str,
                    num_classes: int | None = None) -> str:
    from ark.core.export import export_python, export_onnx
    scene = _build_headless_scene(payload)
    if kind == "python":
        p = export_python(scene, out_path, num_classes=num_classes,
                            name=payload.get("name", "ark_model"))
    elif kind == "onnx":
        p = export_onnx(scene, out_path, num_classes=num_classes)
    else:
        raise ValueError(f"unknown export kind {kind!r}")
    return str(p)


def train_payload(payload: dict, epochs: int = 1, batch_size: int = 128,
                    n_train: int = 3000, n_val: int = 1000,
                    auto_mutate: bool = False, num_classes: int | None = None,
                    verbose: bool = True) -> dict:
    """Train headless une arch sur MNIST. Retourne dict résultats + history."""
    from ark.core.graph_compiler import compile_scene
    from ark.core.trainer import mnist_loaders, pick_device
    import time
    import torch
    import torch.nn as nn

    scene = _build_headless_scene(payload)
    cg = compile_scene(scene, num_classes=num_classes)
    device = pick_device()
    model = cg.model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    train_loader, val_loader = mnist_loaders(
        n_train=n_train, n_val=n_val, batch_size=batch_size
    )
    history = {"epochs": [], "auto_mutations": []}
    t0_total = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        t0 = time.perf_counter()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optim.step()
            running_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += x.size(0)
        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                vc += (model(x).argmax(dim=-1) == y).sum().item()
                vt += x.size(0)
        ep_info = {
            "epoch": ep, "train_loss": running_loss / total,
            "train_acc": correct / total, "val_acc": vc / vt,
            "elapsed": time.perf_counter() - t0,
        }
        history["epochs"].append(ep_info)
        if verbose:
            print(f"  ep{ep} train_loss={ep_info['train_loss']:.4f} "
                  f"train_acc={ep_info['train_acc']*100:.2f}% "
                  f"val_acc={ep_info['val_acc']*100:.2f}% "
                  f"({ep_info['elapsed']:.1f}s)")
        # Auto-mutate (vague C : observer + trigger_rule)
        if auto_mutate and hasattr(model, "modules_dict"):
            metrics = {}
            for nid, m in model.modules_dict.items():
                if hasattr(m, "last_metrics") and isinstance(m.last_metrics, dict):
                    metrics.update(m.last_metrics)
            # ark_id → module map pour résolution target
            ark_to_mod = {getattr(m, "_ark_node_id", None): m
                            for m in model.modules_dict.values()
                            if getattr(m, "_ark_node_id", None)}
            for nid, m in model.modules_dict.items():
                if not (hasattr(m, "evaluate") and callable(m.evaluate)):
                    continue
                a = m.evaluate(metrics)
                if not a:
                    continue
                target = ark_to_mod.get(a["node_id"])
                if target and hasattr(target, a["op"]):
                    try:
                        getattr(target, a["op"])(a.get("delta", 8))
                        a["applied"] = True
                        if hasattr(target, "hidden_dim"):
                            a["new_hidden_dim"] = target.hidden_dim
                        history["auto_mutations"].append({"epoch": ep, **a})
                        if verbose:
                            print(f"    ⟳ mutate {a['node_id'][:14]} "
                                  f"{a['op']}+{a.get('delta', 8)} → "
                                  f"h={a.get('new_hidden_dim', '?')}")
                        model.to(device)
                        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
                    except Exception as e:
                        a["error"] = str(e)
    history["total_elapsed"] = time.perf_counter() - t0_total
    history["final_n_params"] = sum(p.numel() for p in model.parameters())
    return history


# ============================================================
# CLI
# ============================================================

def _cmd_list_templates(args):
    for name, label in TEMPLATES_LIST:
        print(f"  {name:<22} {label}")


def _cmd_list_nodes(args):
    r = default_registry()
    cats = r.by_category()
    if args.category:
        cats = {k: v for k, v in cats.items() if k.lower() == args.category.lower()}
    for cat, specs in cats.items():
        print(f"\n=== {cat} ===")
        for s in specs:
            inputs = "+".join(f"{i.label}:{i.socket_type}" for i in s.inputs) or "—"
            outputs = "+".join(f"{o.label}:{o.socket_type}" for o in s.outputs) or "—"
            print(f"  {s.name:<26} in={inputs:<28} out={outputs}")


def _cmd_inspect(args):
    payload = get_template(args.name)
    if payload is None:
        p = Path(args.name)
        if p.exists():
            payload, fmt = load_any(p)
            print(f"# loaded as format: {fmt}", file=sys.stderr)
        else:
            print(f"unknown template or file: {args.name}", file=sys.stderr)
            sys.exit(1)
    nodes = payload.get("nodes", [])
    conns = payload.get("connections", [])
    print(json.dumps({
        "name": payload.get("name"),
        "n_nodes": len(nodes),
        "n_connections": len(conns),
        "node_types": sorted(set(n["type"] for n in nodes)),
        "tags": payload.get("metadata", {}).get("tags", []),
    }, indent=2))
    try:
        summary = compile_payload(payload, num_classes=args.num_classes)
        print("\n# compile:")
        print(json.dumps(summary, indent=2))
    except Exception as e:
        print(f"\n# compile skipped : {e}", file=sys.stderr)


def _cmd_compile(args):
    p = Path(args.spec)
    payload, fmt = load_any(p)
    summary = compile_payload(payload, num_classes=args.num_classes)
    print(json.dumps({"format_origin": fmt, **summary}, indent=2))


def _cmd_export(args):
    payload, _ = load_any(Path(args.spec))
    out = export_payload(payload, args.kind, args.out, num_classes=args.num_classes)
    print(f"saved → {out}")


def _cmd_train(args):
    payload = get_template(args.spec)
    if payload is None:
        payload, _fmt = load_any(Path(args.spec))
    print(f"# training: {payload.get('name', args.spec)}")
    nc = args.num_classes or payload.get("metadata", {}).get("num_classes")
    hist = train_payload(payload, epochs=args.epochs, batch_size=args.batch_size,
                            n_train=args.n_train, n_val=args.n_val,
                            auto_mutate=args.auto_mutate, num_classes=nc,
                            verbose=True)
    print(f"\n[done] total {hist['total_elapsed']:.1f}s · "
          f"final params {hist['final_n_params']:,} · "
          f"mutations {len(hist['auto_mutations'])}")
    if args.out:
        Path(args.out).write_text(json.dumps(hist, indent=2, default=str))
        print(f"[saved] {args.out}")


def _cmd_convert(args):
    src = Path(args.src)
    with src.open() as f:
        raw = json.load(f)
    out_payload = convert_electron_to_ark(raw)
    dst = Path(args.dst)
    dst.write_text(json.dumps(out_payload, indent=2))
    print(f"converted {src} → {dst}")
    skipped = out_payload.get("metadata", {}).get("unknown_types_skipped", [])
    if skipped:
        print(f"skipped types: {set(skipped)}")


def _save_or_print(payload: dict, out: str | None):
    if out:
        Path(out).write_text(json.dumps(payload, indent=2))
        print(f"saved → {out}")
    else:
        print(json.dumps(payload, indent=2))


def _cmd_quick_mlp(args):
    layers = [int(x) for x in args.layers.split(",")]
    g = build_quick_mlp(layers=layers, input_features=args.input_features,
                          dropout=args.dropout)
    _save_or_print(g, args.out)
    if args.simulate:
        print(json.dumps(simulate_forward(g, num_classes=layers[-1]), indent=2))


def _cmd_quick_cnn(args):
    filters = [int(x) for x in args.filters.split(",")]
    dense = [int(x) for x in args.dense.split(",")] if args.dense else [10]
    shape = [int(x) for x in args.input_shape.split(",")]
    g = build_quick_cnn(filters=filters, dense_layers=dense, input_shape=shape)
    _save_or_print(g, args.out)
    if args.simulate:
        print(json.dumps(simulate_forward(g, num_classes=dense[-1]), indent=2))


def _cmd_quick_resnet(args):
    shape = [int(x) for x in args.input_shape.split(",")]
    g = build_quick_resnet(filters=args.filters, input_shape=shape)
    _save_or_print(g, args.out)
    if args.simulate:
        print(json.dumps(simulate_forward(g, num_classes=10), indent=2))


def _cmd_quick_transformer(args):
    g = build_quick_transformer(n_blocks=args.n_blocks, d_model=args.d_model,
                                  n_heads=args.n_heads, d_ff=args.d_ff,
                                  max_len=args.max_len)
    _save_or_print(g, args.out)
    if args.simulate:
        print(json.dumps(simulate_forward(g, num_classes=10), indent=2))


def main():
    ap = argparse.ArgumentParser(prog="ark.agent_api",
                                   description="CLI ARK headless pour agents.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-templates").set_defaults(func=_cmd_list_templates)

    p = sub.add_parser("list-nodes")
    p.add_argument("--category", default=None)
    p.set_defaults(func=_cmd_list_nodes)

    p = sub.add_parser("inspect")
    p.add_argument("name", help="nom de template OU chemin fichier JSON")
    p.add_argument("--num-classes", type=int, default=None)
    p.set_defaults(func=_cmd_inspect)

    p = sub.add_parser("compile")
    p.add_argument("spec")
    p.add_argument("--num-classes", type=int, default=None)
    p.set_defaults(func=_cmd_compile)

    p = sub.add_parser("export")
    p.add_argument("kind", choices=["python", "onnx"])
    p.add_argument("spec")
    p.add_argument("out")
    p.add_argument("--num-classes", type=int, default=None)
    p.set_defaults(func=_cmd_export)

    p = sub.add_parser("train", help="MNIST headless train")
    p.add_argument("spec", help="template name OR .json file")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--n-train", type=int, default=3000)
    p.add_argument("--n-val", type=int, default=1000)
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--auto-mutate", action="store_true")
    p.add_argument("--out", default=None, help="save history JSON")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("convert", help="ai-arch-builder JSON → ark-arch-v1")
    p.add_argument("src")
    p.add_argument("dst")
    p.set_defaults(func=_cmd_convert)

    p = sub.add_parser("quick-mlp")
    p.add_argument("--layers", default="128,64,10")
    p.add_argument("--input-features", type=int, default=784)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--out", default=None)
    p.add_argument("--simulate", action="store_true")
    p.set_defaults(func=_cmd_quick_mlp)

    p = sub.add_parser("quick-cnn")
    p.add_argument("--filters", default="16,32")
    p.add_argument("--dense", default="128,10")
    p.add_argument("--input-shape", default="1,28,28")
    p.add_argument("--out", default=None)
    p.add_argument("--simulate", action="store_true")
    p.set_defaults(func=_cmd_quick_cnn)

    p = sub.add_parser("quick-resnet")
    p.add_argument("--filters", type=int, default=16)
    p.add_argument("--input-shape", default="3,32,32")
    p.add_argument("--out", default=None)
    p.add_argument("--simulate", action="store_true")
    p.set_defaults(func=_cmd_quick_resnet)

    p = sub.add_parser("quick-transformer")
    p.add_argument("--n-blocks", type=int, default=2)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--max-len", type=int, default=16)
    p.add_argument("--out", default=None)
    p.add_argument("--simulate", action="store_true")
    p.set_defaults(func=_cmd_quick_transformer)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
