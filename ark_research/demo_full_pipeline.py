"""Demo pipeline complet ARK :
1. Build scene programmatique : image_input → conv2d → flatten → dense → softmax
2. Compile via graph_compiler
3. Train MNIST 2 epochs avec eval
4. SciViz : loss curve + accuracy + confusion matrix
5. Save group (Phase 31) + save JSON architecture (Phase 3)

Lance : python demo_full_pipeline.py
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARK_ROOT = HERE.parent
sys.path.insert(0, str(ARK_ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ark.app import create_app
from ark.nodes.registry import default_registry
from ark.ui.node_editor.scene import NodeScene
from ark.ui.node_editor.connection_item import ConnectionItem
from ark.core.graph_compiler import compile_scene, HANDLERS
from ark.core.research_bridge import SciVizWrap
from ark.core.format import scene_to_payload, save_scene
from ark.core.groups import serialize_group, save_group


OUT = HERE / "demo_output"
OUT.mkdir(exist_ok=True)


def section(title):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


# ============================================================
# 1. Setup app + scene
# ============================================================

section("1. Setup ARK")
app = create_app()
registry = default_registry()
scene = NodeScene(registry=registry)
print(f"Registry : {len(registry.all())} NodeSpec disponibles")
print(f"HANDLERS  : {len(HANDLERS)} compilables")


# ============================================================
# 2. Build architecture : MNIST CNN classifier
# ============================================================

section("2. Build scene programmatique")

# Helper local pour créer nodes + connexions
def create_node(ntype, x, y, **params):
    n = scene._raw_create_node(ntype, x=x, y=y)
    for k, v in params.items():
        if k in n.params:
            n.params[k] = v
    return n


def connect(a, b, out_port=0, in_port=0):
    conn = ConnectionItem(
        start_socket=a.output_sockets[out_port],
        end_socket=b.input_sockets[in_port],
    )
    a.output_sockets[out_port].add_connection(conn)
    b.input_sockets[in_port].add_connection(conn)
    scene._raw_add_connection(conn)
    return conn


# Architecture : MNIST CNN — Input(1,28,28) → conv2d(32) → conv2d(64) → dense(128) → dense(10)
# Note : ARK conv2d auto-padding K//2 + ReLU intégré. Dense handle flatten interne.
inp = create_node("image_input", 50, 100, shape="[1, 28, 28]")
c1 = create_node("conv2d", 250, 100, filters=32, kernel_size=3, activation="relu")
c2 = create_node("conv2d", 450, 100, filters=64, kernel_size=3, activation="relu")
d1 = create_node("dense", 650, 100, neurons=128, activation="gelu", dropout=0.2)
d2 = create_node("dense", 850, 100, neurons=10, activation="none")

# Connexions chaîne
chain = [inp, c1, c2, d1, d2]
for a, b in zip(chain, chain[1:]):
    connect(a, b)

print(f"Architecture : {len(chain)} nodes en chaîne")
print(f"  {' → '.join(n.spec.title for n in chain)}")


# ============================================================
# 3. Compile
# ============================================================

section("3. Compile scene → nn.Module")
t0 = time.perf_counter()
cg = compile_scene(scene)
model = cg.model
print(f"Input shape  : {cg.input_shape}")
print(f"Output shape : {cg.output_shape}")
compile_time = time.perf_counter() - t0
n_params = sum(p.numel() for p in model.parameters())
print(f"Compile : {compile_time * 1000:.1f}ms")
print(f"Params  : {n_params:,}")
print(f"Modules :")
for name, m in model.named_modules():
    if name and "." not in name:
        print(f"  {name} : {type(m).__name__}")


# ============================================================
# 4. Train MNIST
# ============================================================

section("4. Training MNIST")
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device : {device}")
model = model.to(device)

tfm = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST(root=str(ARK_ROOT / ".data"), train=True,
                              download=True, transform=tfm)
val_ds = datasets.MNIST(root=str(ARK_ROOT / ".data"), train=False,
                            download=True, transform=tfm)

# Sub-sample pour démo rapide
N_TRAIN = 6000
N_VAL = 1000
train_subset = torch.utils.data.Subset(train_ds, list(range(N_TRAIN)))
val_subset = torch.utils.data.Subset(val_ds, list(range(N_VAL)))

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=256)

optim = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_history = []
val_acc_history = []

EPOCHS = 2
print(f"Train sur {N_TRAIN} samples × {EPOCHS} epochs (val {N_VAL})")

t_start = time.perf_counter()
for ep in range(EPOCHS):
    model.train()
    epoch_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optim.zero_grad()
        logits = model(xb)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        optim.step()
        epoch_losses.append(loss.item())
    loss_history.extend(epoch_losses)

    # Eval
    model.eval()
    correct = total = 0
    confusion = torch.zeros(10, 10, dtype=torch.long)
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            if isinstance(logits, tuple):
                logits = logits[0]
            pred = logits.argmax(-1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
            for t, p in zip(yb.cpu(), pred.cpu()):
                confusion[t, p] += 1
    val_acc = correct / total
    val_acc_history.append(val_acc)
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    print(f"  Epoch {ep + 1}/{EPOCHS} : avg_loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

train_time = time.perf_counter() - t_start


# ============================================================
# 5. Eval final + analyses
# ============================================================

section("5. Résultats finaux")
final_acc = val_acc_history[-1]
print(f"Train time : {train_time:.1f}s")
print(f"Final val_acc : {final_acc:.4f} ({final_acc * 100:.2f}%)")
print(f"Loss : {loss_history[0]:.4f} → {loss_history[-1]:.4f}")
print("\nConfusion matrix (top 3 classes par diagonale) :")
diag = confusion.diag().tolist()
top3 = sorted(range(10), key=lambda i: -diag[i])[:3]
for cls in top3:
    print(f"  classe {cls} : {diag[cls]} corrects sur {confusion[cls].sum().item()}")


# ============================================================
# 6. SciViz : visualisations
# ============================================================

section("6. Visualisations (SciViz)")

# Loss curve
viz1 = SciVizWrap(
    plot_kind="line",
    title=f"ARK demo MNIST — Loss training ({EPOCHS} epochs)",
    output_path=str(OUT / "loss_curve.png"),
    metric_keys="loss",
)
viz1.run(data={"loss": loss_history})
print(f"  → {OUT / 'loss_curve.png'}")

# Accuracy per epoch
viz2 = SciVizWrap(
    plot_kind="bar",
    title="ARK demo — Val accuracy per epoch",
    output_path=str(OUT / "val_acc.png"),
    metric_keys="val_acc",
)
viz2.run(data={"val_acc": [a * 100 for a in val_acc_history]})
print(f"  → {OUT / 'val_acc.png'}")

# Confusion diagonal
viz3 = SciVizWrap(
    plot_kind="bar",
    title="ARK demo — Predictions correctes par classe",
    output_path=str(OUT / "confusion_diag.png"),
    metric_keys="correct_per_class",
)
viz3.run(data={"correct_per_class": diag})
print(f"  → {OUT / 'confusion_diag.png'}")


# ============================================================
# 7. Save architecture + group
# ============================================================

section("7. Persistance")

# JSON scene
arch_path = OUT / "demo_arch.json"
save_scene(scene, arch_path, name="mnist_cnn_demo",
              num_classes=10, tags=["demo", "mnist", "cnn"])
print(f"Architecture JSON → {arch_path}")

# Group (composite)
try:
    selected = chain  # tous les nodes
    payload = serialize_group(scene, selected,
                                  name="mnist_cnn_block",
                                  title="MNIST CNN Block (Demo)",
                                  category="User Groups",
                                  description=f"CNN MNIST entraîné — "
                                               f"val_acc={final_acc:.2%}, "
                                               f"{n_params:,} params")
    grp_path = save_group(payload)
    print(f"Group composite → {grp_path}")
except Exception as e:
    print(f"Group save error: {e}")

# Save metrics JSON pour analyse
metrics_path = OUT / "metrics.json"
metrics = {
    "architecture": {
        "n_nodes": len(chain),
        "node_types": [n.spec.name for n in chain],
        "n_params": n_params,
        "compile_time_ms": compile_time * 1000,
    },
    "training": {
        "device": device,
        "epochs": EPOCHS,
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "train_time_s": train_time,
        "final_val_acc": final_acc,
        "val_acc_per_epoch": val_acc_history,
        "first_loss": loss_history[0],
        "last_loss": loss_history[-1],
    },
    "confusion_diag": diag,
}
metrics_path.write_text(json.dumps(metrics, indent=2))
print(f"Metrics → {metrics_path}")


# ============================================================
# 8. Test sanity : ré-instancie via JSON + check identique
# ============================================================

section("8. Sanity : reload architecture + forward identique")
from ark.core.format import load_scene
scene2 = NodeScene(registry=registry)
load_scene(scene2, arch_path)
cg2 = compile_scene(scene2)
model2 = cg2.model.to(device)
model.eval()
model2.eval()

with torch.no_grad():
    x = torch.randn(2, 1, 28, 28, device=device)
    y1 = model(x)
    y2 = model2(x)
    if isinstance(y1, tuple): y1 = y1[0]
    if isinstance(y2, tuple): y2 = y2[0]

# Note : IDs uuid regénérés au reload → weights randomisés différemment
# On vérifie juste la cohérence architecturale (shape + params count)
p1 = sum(p.numel() for p in model.parameters())
p2 = sum(p.numel() for p in model2.parameters())
print(f"Original  : output shape {tuple(y1.shape)}, {p1:,} params")
print(f"Reloaded  : output shape {tuple(y2.shape)}, {p2:,} params")
print(f"{'✅ ARCHITECTURE IDENTIQUE' if y1.shape == y2.shape and p1 == p2 else '⚠ MISMATCH'}")


print("\n" + "=" * 60)
print("  DEMO TERMINÉE")
print("=" * 60)
print(f"  Final accuracy : {final_acc * 100:.2f}%")
print(f"  Output dir     : {OUT}")
print(f"  Files          : 3 PNG viz + 1 JSON arch + 1 JSON metrics + 1 group")
