"""Bench training réel ARK — MNIST 2 epochs sur templates clés.

Lance via :
    cd ~/claude-code-projects/ark
    QT_QPA_PLATFORM=offscreen .venv/bin/python bench/training_benchmark.py
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
sys.path.insert(0, "/Users/zac/claude-code-projects/metamorph")

import json
from pathlib import Path

from ark.app import create_app
from ark.core.templates import get_template
from ark.agent_api import train_payload


CASES = [
    ("mlp_mnist", False),
    ("cnn_cifar", False),  # note : MNIST input mismatch, juste compile test
    ("vg_mutable", True),
    ("metamorph_m1", True),
]


def main():
    create_app([])
    print("=" * 100)
    print(f"{'template':<22} {'auto_mut':>10} {'epochs':>7} {'final_acc':>10} "
          f"{'final_loss':>12} {'time(s)':>9} {'params':>10} {'muts':>5}")
    print("=" * 100)
    results = []
    for name, auto_mut in CASES:
        payload = get_template(name)
        if payload is None:
            print(f"{name:<22} ERROR: template missing")
            continue
        # cnn_cifar attend (3, 32, 32) → MNIST (1, 28, 28) → skip
        if name == "cnn_cifar":
            print(f"{name:<22} {'skip MNIST (CIFAR shape)':>40}")
            continue
        nc = payload.get("metadata", {}).get("num_classes", 10)
        try:
            hist = train_payload(payload, epochs=2, batch_size=128,
                                    n_train=3000, n_val=1000,
                                    auto_mutate=auto_mut, num_classes=nc,
                                    verbose=False)
        except Exception as e:
            print(f"{name:<22} ERROR: {type(e).__name__}: {e}")
            continue
        last = hist["epochs"][-1]
        print(f"{name:<22} {str(auto_mut):>10} {len(hist['epochs']):>7} "
              f"{last['val_acc']*100:>9.2f}% {last['train_loss']:>12.4f} "
              f"{hist['total_elapsed']:>9.1f} "
              f"{hist['final_n_params']:>10,} {len(hist['auto_mutations']):>5}")
        results.append({"name": name, "auto_mutate": auto_mut, **hist})
    print("=" * 100)
    out = Path(__file__).parent / "training_results.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
