"""Bench compile ARK : timing par template + memory params.

Lance via :
    cd ~/claude-code-projects/ark
    QT_QPA_PLATFORM=offscreen .venv/bin/python bench/compile_benchmark.py
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
sys.path.insert(0, "/Users/zac/claude-code-projects/metamorph")

import json
import time
import statistics
from pathlib import Path

import torch

from ark.app import create_app
from ark.nodes.registry import default_registry
from ark.ui.node_editor.scene import NodeScene
from ark.core.format import payload_to_scene
from ark.core.templates import TEMPLATES_LIST, get_template
from ark.core.graph_compiler import compile_scene, GraphCompileError


def bench_template(name: str, n_iter: int = 5) -> dict:
    payload = get_template(name)
    if payload is None:
        return {"name": name, "error": "template not found"}
    timings_compile = []
    timings_forward = []
    n_params = None
    n_modules = None
    input_shape = None
    output_shape = None
    error = None
    for i in range(n_iter):
        scene = NodeScene(registry=default_registry())
        t0 = time.perf_counter()
        try:
            payload_to_scene(scene, payload)
            cg = compile_scene(scene, num_classes=10)
            t1 = time.perf_counter()
            timings_compile.append((t1 - t0) * 1000)
            n_params = sum(p.numel() for p in cg.model.parameters())
            n_modules = len(cg.model.modules_dict)
            input_shape = list(cg.input_shape)
            output_shape = list(cg.output_shape)
            # Forward dummy
            cg.model.eval()
            x = torch.randn(2, *cg.input_shape)
            t2 = time.perf_counter()
            with torch.no_grad():
                _ = cg.model(x)
            t3 = time.perf_counter()
            timings_forward.append((t3 - t2) * 1000)
        except GraphCompileError as e:
            error = str(e)
            break
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            break
    return {
        "name": name,
        "n_iter": len(timings_compile),
        "error": error,
        "n_params": n_params,
        "n_modules": n_modules,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "compile_ms_mean": statistics.mean(timings_compile) if timings_compile else None,
        "compile_ms_std": statistics.stdev(timings_compile) if len(timings_compile) > 1 else 0.0,
        "compile_ms_min": min(timings_compile) if timings_compile else None,
        "forward_ms_mean": statistics.mean(timings_forward) if timings_forward else None,
    }


def main():
    create_app([])
    print("=" * 90)
    print(f"{'template':<22} {'n_params':>12} {'modules':>8} "
          f"{'compile (ms)':>16} {'forward (ms)':>14} {'shape':>22}")
    print("=" * 90)
    results = []
    for name, label in TEMPLATES_LIST:
        r = bench_template(name, n_iter=5)
        results.append(r)
        if r.get("error"):
            print(f"{name:<22} ERROR : {r['error']}")
            continue
        compile_str = f"{r['compile_ms_mean']:.1f} ± {r['compile_ms_std']:.1f}"
        forward_str = f"{r['forward_ms_mean']:.2f}" if r['forward_ms_mean'] else "—"
        shape_str = f"{r['input_shape']}→{r['output_shape']}"
        print(f"{name:<22} {r['n_params'] or 0:>12,} {r['n_modules'] or 0:>8} "
              f"{compile_str:>16} {forward_str:>14} {shape_str:>22}")
    print("=" * 90)

    out = Path(__file__).parent / "benchmark_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
