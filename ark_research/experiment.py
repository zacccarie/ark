"""Baseline experiment ARK : compile + train metamorph_m1 template sur MNIST.

Lance via : python experiment.py --out_dir run_baseline

Sortie attendue : run_baseline/final_info.json avec val_acc, params, time.
Sert de base aux mutations/variations proposées par /ai-scientist ideas.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ajout du repo ARK au path
HERE = Path(__file__).resolve().parent
ARK_ROOT = HERE.parent
sys.path.insert(0, str(ARK_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ark.app import create_app  # noqa
from ark.core.templates import TEMPLATES, build_scene_from_template  # noqa
from ark.core.graph_compiler import compile_scene  # noqa
from ark.ui.node_editor.scene import NodeScene  # noqa


def run_experiment(template_name: str = "metamorph_m1",
                     epochs: int = 2, batch_size: int = 64,
                     auto_mutate: bool = True,
                     out_dir: str = "run_baseline") -> dict:
    create_app()  # nécessaire pour QGraphicsScene
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    scene = NodeScene()
    build_scene_from_template(scene, template_name)

    t0 = time.perf_counter()
    model, info = compile_scene(scene)
    compile_time = time.perf_counter() - t0
    params = sum(p.numel() for p in model.parameters())

    # Training réel (MNIST minimal)
    import torch
    import torch.nn.functional as F
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root=str(ARK_ROOT / ".data"), train=True,
                                download=True, transform=tfm)
    val_ds = datasets.MNIST(root=str(ARK_ROOT / ".data"), train=False,
                              download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    val_accs = []
    t_start = time.perf_counter()
    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            out_logits = model(xb)
            if isinstance(out_logits, tuple):
                out_logits = out_logits[0]
            loss = F.cross_entropy(out_logits, yb)
            loss.backward()
            optim.step()
            losses.append(loss.item())
        # Eval
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                if isinstance(logits, tuple):
                    logits = logits[0]
                pred = logits.argmax(-1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
        val_accs.append(correct / total)
        print(f"[epoch {ep+1}] val_acc={val_accs[-1]:.4f}")

    train_time = time.perf_counter() - t_start

    result = {
        "template": template_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "auto_mutate": auto_mutate,
        "params": params,
        "compile_time_s": compile_time,
        "train_time_s": train_time,
        "device": device,
        "val_acc_per_epoch": val_accs,
        "final_val_acc": val_accs[-1] if val_accs else 0.0,
        "loss_curve": losses,
    }
    (out / "final_info.json").write_text(json.dumps(result, indent=2))
    print("[done] →", out / "final_info.json")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="metamorph_m1")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--out_dir", default="run_baseline")
    args = ap.parse_args()
    run_experiment(args.template, args.epochs, args.batch_size,
                     True, args.out_dir)


if __name__ == "__main__":
    main()
