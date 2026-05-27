"""ARK training thread — QThread + signals pour UI."""
from __future__ import annotations
import time
import torch
import torch.nn as nn
from PySide6.QtCore import QObject, QThread, Signal


class TrainerWorker(QObject):
    status_changed = Signal(str)
    epoch_finished = Signal(dict)
    batch_progressed = Signal(dict)
    observations = Signal(dict)   # {ark_node_id: {metric: value}}
    auto_mutated = Signal(dict)    # {epoch, triggered: [...], n_params}
    error = Signal(str)
    done = Signal()

    def __init__(self, compiled, train_loader, val_loader=None,
                 epochs: int = 1, lr: float = 1e-3,
                 loss_kind: str = "cross_entropy",
                 device: torch.device | None = None,
                 auto_mutate: bool = False):
        super().__init__()
        self.compiled = compiled
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.lr = lr
        self.loss_kind = loss_kind
        self.device = device or torch.device("cpu")
        self.auto_mutate = auto_mutate
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            model = self.compiled.model.to(self.device)
            optim = torch.optim.Adam(model.parameters(), lr=self.lr)
            loss_fn = nn.CrossEntropyLoss() if self.loss_kind == "cross_entropy" else nn.MSELoss()
            self.status_changed.emit(f"Training started — {sum(p.numel() for p in model.parameters())} params · device {self.device}")
            for ep in range(1, self.epochs + 1):
                if self._stop:
                    self.status_changed.emit("stopped")
                    self.done.emit()
                    return
                model.train()
                t0 = time.time()
                running_loss, correct, total = 0.0, 0, 0
                n_b = len(self.train_loader)
                for i, (x, y) in enumerate(self.train_loader):
                    if self._stop:
                        self.status_changed.emit("stopped")
                        self.done.emit()
                        return
                    x = x.to(self.device); y = y.to(self.device)
                    optim.zero_grad()
                    logits = model(x)
                    loss = loss_fn(logits, y)
                    loss.backward()
                    optim.step()
                    running_loss += loss.item() * x.size(0)
                    correct += (logits.argmax(dim=-1) == y).sum().item()
                    total += x.size(0)
                    if i % max(1, n_b // 20) == 0:
                        self.batch_progressed.emit({
                            "epoch": ep, "batch": i + 1, "total_batches": n_b,
                            "loss": running_loss / max(total, 1),
                            "acc": correct / max(total, 1),
                        })
                val_loss, val_acc = None, None
                if self.val_loader is not None:
                    model.eval()
                    vl, vc, vt = 0.0, 0, 0
                    with torch.no_grad():
                        for x, y in self.val_loader:
                            x = x.to(self.device); y = y.to(self.device)
                            logits = model(x)
                            vl += loss_fn(logits, y).item() * x.size(0)
                            vc += (logits.argmax(dim=-1) == y).sum().item()
                            vt += x.size(0)
                    val_loss = vl / max(vt, 1)
                    val_acc = vc / max(vt, 1)
                self.epoch_finished.emit({
                    "epoch": ep, "epochs": self.epochs,
                    "train_loss": running_loss / max(total, 1),
                    "train_acc": correct / max(total, 1),
                    "val_loss": val_loss, "val_acc": val_acc,
                    "elapsed": time.time() - t0,
                })
                # Vague C — emit observations + apply triggers
                obs_dict, all_metrics = self._collect_observers(model)
                if obs_dict:
                    self.observations.emit({"epoch": ep, "observers": obs_dict})
                if self.auto_mutate:
                    triggered = self._evaluate_and_apply(model, all_metrics)
                    if triggered:
                        any_applied = any(a.get("applied") for a in triggered)
                        if any_applied:
                            model.to(self.device)
                            optim = torch.optim.Adam(model.parameters(), lr=self.lr)
                        self.auto_mutated.emit({
                            "epoch": ep, "triggered": triggered,
                            "n_params": sum(p.numel() for p in model.parameters()),
                        })
            self.status_changed.emit("done")
            self.done.emit()
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")

    def _collect_observers(self, model):
        """Récupère last_metrics par node_id ARK + dict agrégé."""
        per_node: dict = {}
        all_metrics: dict = {}
        for nid, m in model.modules_dict.items():
            ark_id = getattr(m, "_ark_node_id", None)
            if ark_id and hasattr(m, "last_metrics") and isinstance(m.last_metrics, dict):
                per_node[ark_id] = dict(m.last_metrics)
                all_metrics.update(m.last_metrics)
        return per_node, all_metrics

    def _evaluate_and_apply(self, model, metrics: dict) -> list[dict]:
        triggered = []
        # Map ark_node_id → module pour résolution target
        ark_to_module = {}
        for nid, mod in model.modules_dict.items():
            aid = getattr(mod, "_ark_node_id", None)
            if aid:
                ark_to_module[aid] = mod
        for nid, m in model.modules_dict.items():
            if hasattr(m, "evaluate") and callable(m.evaluate):
                action = m.evaluate(metrics)
                if not action:
                    continue
                action["from_rule"] = getattr(m, "_ark_node_id", nid)
                target_id = action["node_id"]
                target = ark_to_module.get(target_id)
                if target is None or not hasattr(target, action["op"]):
                    action["applied"] = False
                    action["error"] = f"node {target_id!r} cannot {action['op']}"
                else:
                    try:
                        getattr(target, action["op"])(action.get("delta", 8))
                        action["applied"] = True
                        if hasattr(target, "hidden_dim"):
                            action["new_hidden_dim"] = target.hidden_dim
                    except Exception as e:
                        action["applied"] = False
                        action["error"] = str(e)
                triggered.append(action)
        return triggered


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mnist_loaders(n_train: int = 5000, n_val: int = 1000, batch_size: int = 64,
                    data_dir: str = "./.data"):
    import torchvision
    from torchvision import transforms
    from torch.utils.data import DataLoader, Subset
    tf = transforms.Compose([transforms.ToTensor()])
    tr = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=tf)
    va = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=tf)
    tr_sub = Subset(tr, list(range(n_train)))
    va_sub = Subset(va, list(range(n_val)))
    return (DataLoader(tr_sub, batch_size=batch_size, shuffle=True, num_workers=0),
            DataLoader(va_sub, batch_size=batch_size, shuffle=False, num_workers=0))
