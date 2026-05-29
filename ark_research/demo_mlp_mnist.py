"""Phase 72 — Demo perceptron multi-couche MNIST (preuve de concept ARK).

Construit le MLP depuis le template mlp_mnist_demo (nodes élémentaires
image_input + 3× dense), compile headless (sans Qt), entraîne sur vraies
données MNIST, évalue, affiche prédictions sur échantillons réels.

Usage : python ark_research/demo_mlp_mnist.py
"""
from __future__ import annotations
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


def main():
    from ark.core.templates import get_template
    from ark.core.headless_scene import compile_payload

    print("=" * 60)
    print("ARK · MLP MNIST — perceptron multi-couche (nodes élémentaires)")
    print("=" * 60)

    # 1. Build architecture depuis template (image_input → dense×3)
    tpl = get_template("mlp_mnist_demo")
    layers = [n["type"] for n in tpl["nodes"]]
    print(f"\nArchitecture ({len(tpl['nodes'])} nodes) : {' → '.join(layers)}")

    cg = compile_payload(tpl, num_classes=10)
    model = cg.model
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Compilé : in={cg.input_shape} → out={cg.output_shape} · "
          f"{n_params:,} paramètres")
    print("Couches denses :")
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            print(f"  {name}: {mod.in_features} → {mod.out_features}")

    # 2. Vraies données MNIST
    from ark.core.trainer import mnist_loaders, pick_device
    device = pick_device()
    print(f"\nDevice : {device}")
    print("Téléchargement / chargement MNIST réel…")
    train_loader, val_loader = mnist_loaders(
        n_train=20000, n_val=2000, batch_size=128)

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def run_forward(x):
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        if out.dim() > 2:
            out = out.flatten(1)
        return out

    # 3. Entraînement réel
    EPOCHS = 4
    print(f"\nEntraînement {EPOCHS} epochs sur 20k images…")
    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        loss_sum, nb = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = run_forward(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            nb += 1
        # éval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = run_forward(x).argmax(-1)
                correct += (pred == y).sum().item()
                total += y.numel()
        acc = correct / total
        print(f"  epoch {ep+1}/{EPOCHS} · loss {loss_sum/nb:.4f} · "
              f"val acc {acc*100:.2f}% · {time.time()-t0:.1f}s")

    # 4. Prédictions sur échantillons réels
    print("\nPrédictions sur 12 chiffres manuscrits réels :")
    model.eval()
    x, y = next(iter(val_loader))
    x, y = x[:12].to(device), y[:12]
    with torch.no_grad():
        probs = F.softmax(run_forward(x), dim=-1)
    pred = probs.argmax(-1).cpu()
    conf = probs.max(-1).values.cpu()
    line_true = "  vrai  : " + " ".join(f"{int(v):>2}" for v in y)
    line_pred = "  prédit: " + " ".join(f"{int(v):>2}" for v in pred)
    line_ok = "          " + " ".join(
        " ✓" if int(p) == int(t) else " ✗" for p, t in zip(pred, y))
    print(line_true)
    print(line_pred)
    print(line_ok)
    print("  conf  : " + " ".join(f"{c*100:.0f}" for c in conf))

    # 5. ASCII render d'un échantillon
    print("\nÉchantillon #0 (ASCII, vrai={}, prédit={}) :".format(
        int(y[0]), int(pred[0])))
    img = x[0, 0].cpu()
    chars = " .:-=+*#%@"
    for row in range(0, 28, 1):
        line = ""
        for col in range(28):
            v = float(img[row, col])
            line += chars[min(len(chars) - 1, int(v * len(chars)))]
        if line.strip():
            print("  " + line)

    # 6. Sauve checkpoint
    from pathlib import Path
    out = Path.home() / ".ark" / "demo_checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "mlp_mnist_demo.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"\nCheckpoint sauvé : {ckpt}")
    print("Template ARK : Fichier → Templates → 'MLP MNIST Demo ★'")
    print("=" * 60)


if __name__ == "__main__":
    main()
