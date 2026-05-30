"""Phase 81 — Évaluation held-out (test) kind-aware.

Valider ≠ entraîner. On mesure la généralisation sur un set JAMAIS vu
pendant l'entraînement. La métrique dépend du type de modèle (voir
model_kind.py).

- classifier  : accuracy + confusion matrix + loss CE sur held-out
- lm          : perplexity = exp(mean cross-entropy)
- generative  : FID/CLIP → nécessite un set de référence (stub honnête)
- world_model : rollout error → nécessite trajectoires (stub honnête)

`evaluate_compiled` renvoie toujours un dict JSON-sérialisable.
"""
from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ark.core.model_kind import KindSpec


# ============================================================
# Held-out loaders (disjoints du val utilisé en entraînement)
# ============================================================

def mnist_test_loader(n_test: int = 2000, offset: int = 2000,
                      batch_size: int = 128, data_dir: str = "./.data"):
    """Slice du test set MNIST DISJOINTE du val (val = test[0:1000]).
    → vraie estimation held-out."""
    import torchvision
    from torchvision import transforms
    from torch.utils.data import DataLoader, Subset
    tf = transforms.Compose([transforms.ToTensor()])
    va = torchvision.datasets.MNIST(root=data_dir, train=False,
                                    download=True, transform=tf)
    n = len(va)
    idx = list(range(offset, min(offset + n_test, n)))
    sub = Subset(va, idx)
    return DataLoader(sub, batch_size=batch_size, shuffle=False, num_workers=0)


# ============================================================
# Classifier
# ============================================================

def evaluate_classifier(model, loader, device=None, num_classes: int = 10,
                        train_acc: Optional[float] = None) -> dict:
    device = device or torch.device("cpu")
    model = model.to(device).eval()
    loss_fn = nn.CrossEntropyLoss()
    correct, total, loss_sum = 0, 0, 0.0
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            pred = logits.argmax(dim=-1)
            loss_sum += loss_fn(logits, y).item() * x.size(0)
            correct += (pred == y).sum().item()
            total += x.size(0)
            for t, p in zip(y.view(-1).cpu(), pred.view(-1).cpu()):
                if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
                    conf[int(t), int(p)] += 1
    acc = correct / max(total, 1)
    loss = loss_sum / max(total, 1)
    # accuracy par classe (diagonale / total ligne)
    per_class = []
    for c in range(num_classes):
        row = int(conf[c].sum())
        per_class.append(round(float(conf[c, c]) / row, 4) if row else None)
    gap = (float(train_acc) - acc) if train_acc is not None else None
    return {
        "kind": "classifier",
        "metric_name": "accuracy",
        "metric": round(acc, 4),
        "accuracy": round(acc, 4),
        "loss": round(loss, 4),
        "n": total,
        "confusion": conf.tolist(),
        "per_class_acc": per_class,
        "gap": round(gap, 4) if gap is not None else None,
        "status": "ok",
    }


# ============================================================
# LM
# ============================================================

def evaluate_lm(model, loader, device=None) -> dict:
    """Perplexity = exp(mean CE) sur séquences held-out. loader → (x, y)
    de tokens (y = x décalé d'un cran)."""
    device = device or torch.device("cpu")
    model = model.to(device).eval()
    total_nll, total_tok = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            V = logits.shape[-1]
            nll = F.cross_entropy(logits.view(-1, V), y.view(-1),
                                  reduction="sum")
            total_nll += float(nll)
            total_tok += y.numel()
    mean_ce = total_nll / max(total_tok, 1)
    ppl = math.exp(min(mean_ce, 20))  # clamp overflow
    bpb = mean_ce / math.log(2)
    return {
        "kind": "lm",
        "metric_name": "perplexity",
        "metric": round(ppl, 3),
        "perplexity": round(ppl, 3),
        "bits_per_byte": round(bpb, 4),
        "cross_entropy": round(mean_ce, 4),
        "n": total_tok,
        "status": "ok",
    }


# ============================================================
# Dispatcher
# ============================================================

def evaluate_compiled(compiled, spec: KindSpec, loader, device=None,
                      train_acc: Optional[float] = None,
                      num_classes: int = 10) -> dict:
    """Évalue le modèle compilé selon son type. Toujours JSON-sérialisable."""
    model = getattr(compiled, "model", compiled)
    kind = spec.kind
    if loader is None and kind in ("classifier", "lm", "regressor"):
        return {"kind": kind, "metric_name": spec.primary_metric,
                "metric": None, "status": "needs_reference_set",
                "note": "Aucun test set held-out branché pour ce type."}
    try:
        if kind == "classifier":
            return evaluate_classifier(model, loader, device, num_classes,
                                       train_acc)
        if kind == "lm":
            return evaluate_lm(model, loader, device)
    except Exception as e:
        return {"kind": kind, "metric_name": spec.primary_metric,
                "metric": None, "status": "error", "error": str(e)}
    # generative / world_model / regressor : harnais dédié requis
    return {
        "kind": kind,
        "metric_name": spec.primary_metric,
        "metric": None,
        "status": "needs_reference_set",
        "note": {
            "generative": "FID/CLIP nécessitent un set de référence réel + "
                          "échantillons générés. Harnais Phase 82.",
            "world_model": "Rollout error nécessite des trajectoires "
                           "(state, futures frames). Harnais dédié.",
            "regressor": "Brancher un test set (input, valeur) labellisé.",
        }.get(kind, "Harnais d'évaluation dédié requis pour ce type."),
    }


def passes(spec: KindSpec, result: dict) -> bool:
    """Le modèle passe-t-il le seuil de validation ?"""
    if result.get("status") != "ok":
        return False
    m = result.get("metric")
    if m is None:
        return False
    thr = spec.pass_threshold
    if thr is None:
        return True  # pas de seuil dur → mesure réussie = validé
    return (m >= thr) if spec.metric_direction == "higher" else (m <= thr)
