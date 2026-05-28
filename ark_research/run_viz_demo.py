"""Demo SciVizWrap : génère 3 visualisations depuis bench data ARK existant.

Lance via : python run_viz_demo.py
Output : 3 PNG dans ./output/
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARK_ROOT = HERE.parent
sys.path.insert(0, str(ARK_ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ark.core.research_bridge import SciVizWrap  # noqa


# Bench réel ARK (extrait ROADMAP.md)
bench_training = {
    "mlp_mnist_static": [0.0, 0.62, 0.84],          # val_acc 2 epochs
    "vg_mutable_automut": [0.0, 0.45, 0.661],
    "metamorph_m1_automut": [0.0, 0.58, 0.79],
}

bench_compile_ms = {
    "mlp_mnist": [5.7],
    "cnn_cifar": [25.4],
    "resnet_block": [1.5],
    "transformer_mini": [1.0],
    "vg_mutable": [0.8],
    "metamorph_m1": [1.3],
    "metamorph_m5": [1.1],
}

out_dir = HERE / "output"
out_dir.mkdir(exist_ok=True)


# 1. Training curves
w1 = SciVizWrap(
    plot_kind="line",
    title="ARK auto-mutate val_acc MNIST (2 epochs)",
    output_path=str(out_dir / "training_curves.png"),
    metric_keys="mlp_mnist_static,vg_mutable_automut,metamorph_m1_automut",
)
r1 = w1.run(data=bench_training)
print("[viz1]", "OK" if r1.ok else "FAIL", r1.output_path)

# 2. Compile time bar chart
w2 = SciVizWrap(
    plot_kind="bar",
    title="ARK compile time per template (ms, MPS)",
    output_path=str(out_dir / "compile_times.png"),
    metric_keys=",".join(bench_compile_ms.keys()),
)
r2 = w2.run(data=bench_compile_ms)
print("[viz2]", "OK" if r2.ok else "FAIL", r2.output_path)

# 3. Ideas score scatter (depuis ark_research/ideas.json)
ideas_path = HERE / "ideas.json"
ideas_data = json.loads(ideas_path.read_text()) if ideas_path.exists() else []
scores = {
    "interestingness": [i.get("Interestingness", 0) for i in ideas_data],
    "feasibility": [i.get("Feasibility", 0) for i in ideas_data],
    "novelty": [i.get("Novelty", 0) for i in ideas_data],
}
w3 = SciVizWrap(
    plot_kind="bar",
    title=f"ARK research ideas scoring (n={len(ideas_data)})",
    output_path=str(out_dir / "ideas_scoring.png"),
    metric_keys="interestingness,feasibility,novelty",
)
r3 = w3.run(data=scores)
print("[viz3]", "OK" if r3.ok else "FAIL", r3.output_path)
