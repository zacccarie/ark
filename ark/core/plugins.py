"""ARK plugin system — load user-defined NodeSpec + handlers depuis ~/.ark/plugins.

Convention plugin file (~/.ark/plugins/my_plugin.py) :

    from ark.nodes.registry import NodeSpec, ParamSpec, SocketSpec

    NODE_SPEC = NodeSpec(
        name="my_custom_op",
        title="My Custom Op",
        category="Custom",
        inputs=[SocketSpec("in", "tensor")],
        outputs=[SocketSpec("out", "tensor")],
        params=[ParamSpec("scale", "float", 1.0)],
    )

    def handler(node, in_shapes):
        import torch.nn as nn
        class Op(nn.Module):
            def __init__(self, s): super().__init__(); self.s = s
            def forward(self, x): return x * self.s
        return Op(float(node.params.get("scale", 1.0))), in_shapes[0]

Au boot ARK :
- scan ~/.ark/plugins/*.py
- import via importlib + extrait NODE_SPEC + handler
- register dans registry + HANDLERS

Sécurité : exec arbitrary Python depuis files user. À utiliser sur sa propre
machine, pas en serveur multi-user.
"""
from __future__ import annotations
import importlib.util
import sys
import traceback
from pathlib import Path
from typing import Callable


PLUGINS_DIR = Path.home() / ".ark" / "plugins"


def _load_plugin_file(path: Path) -> dict | None:
    """Charge un fichier .py et extrait NODE_SPEC + handler.

    Retourne dict {spec, handler, path} ou None si invalide.
    """
    try:
        mod_name = f"ark_plugin_{path.stem}"
        spec = importlib.util.spec_from_file_location(mod_name, str(path))
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        node_spec = getattr(mod, "NODE_SPEC", None)
        handler = getattr(mod, "handler", None)
        if node_spec is None or handler is None:
            return None
        if not callable(handler):
            return None
        return {"spec": node_spec, "handler": handler, "path": path}
    except Exception:
        print(f"[ark.plugins] failed to load {path}:")
        traceback.print_exc()
        return None


def discover_plugins(plugins_dir: Path | None = None) -> list[dict]:
    """Scan plugins_dir → liste de plugins chargés."""
    d = plugins_dir or PLUGINS_DIR
    if not d.exists():
        return []
    found = []
    for f in sorted(d.glob("*.py")):
        if f.name.startswith("_"):
            continue
        loaded = _load_plugin_file(f)
        if loaded:
            found.append(loaded)
    return found


def register_plugins(registry, handlers_dict: dict,
                       plugins_dir: Path | None = None,
                       verbose: bool = False) -> int:
    """Découvre plugins + register dans registry + handlers_dict."""
    plugins = discover_plugins(plugins_dir)
    for p in plugins:
        spec = p["spec"]
        registry.register(spec)
        handlers_dict[spec.name] = p["handler"]
        if verbose:
            print(f"[ark.plugins] registered {spec.name} (category={spec.category})")
    return len(plugins)


def example_plugin_text() -> str:
    """Retourne un exemple plugin à coller pour démarrer."""
    return '''"""Example ARK plugin — sauve dans ~/.ark/plugins/scale.py"""
from ark.nodes.registry import NodeSpec, ParamSpec, SocketSpec
import torch.nn as nn


NODE_SPEC = NodeSpec(
    name="scale",
    title="Scale",
    category="Custom",
    inputs=[SocketSpec("in", "tensor")],
    outputs=[SocketSpec("out", "tensor")],
    params=[ParamSpec("factor", "float", 2.0, min=-100.0, max=100.0)],
    description="Multiplie tensor par factor.",
)


class _ScaleOp(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = float(factor)

    def forward(self, x):
        return x * self.factor


def handler(node, in_shapes):
    if len(in_shapes) != 1:
        raise RuntimeError("scale needs 1 input")
    return _ScaleOp(node.params.get("factor", 2.0)), in_shapes[0]
'''
