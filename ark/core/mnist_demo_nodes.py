"""Phase 74 — Concrete MNIST recognition demo nodes.

Preuve end-to-end dans l'UI :
  mnist_sample (vrai glyphe chiffre) → mnist_recognizer (MLP entraîné)
    → top_k_predictions (chiffre reconnu)
  mnist_sample → image_viewer (affiche le glyphe)

Compile → forward → le modèle sort le chiffre reconnu, sans setup.

mnist_recognizer entraîne un petit MLP au 1er usage (~10s) et cache les
poids ~/.ark/mnist_recognizer.pt. Réutilisé ensuite instantanément.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


_MNIST_CACHE: dict = {}
_CKPT = Path.home() / ".ark" / "mnist_recognizer.pt"


def _load_mnist_test():
    """Charge (lazy) le test set MNIST réel."""
    if "test" in _MNIST_CACHE:
        return _MNIST_CACHE["test"]
    import torchvision
    from torchvision import transforms
    tf = transforms.Compose([transforms.ToTensor()])
    ds = torchvision.datasets.MNIST(root="./.data", train=False,
                                        download=True, transform=tf)
    _MNIST_CACHE["test"] = ds
    return ds


# ============================================================
# MNIST sample — vrai glyphe en entrée
# ============================================================

class MnistSample(nn.Module):
    """Émet un vrai chiffre manuscrit MNIST (1,1,28,28). Source node."""

    def __init__(self, index: int = 0):
        super().__init__()
        self.index = index
        self.true_label: Optional[int] = None
        self.last_image_tensor: Optional[torch.Tensor] = None
        self.last_text: str = ""
        self._ark_node_ref = None

    def forward(self, x=None):
        try:
            ds = _load_mnist_test()
            idx = int(self.index) % len(ds)
            img, label = ds[idx]            # img: (1,28,28) in [0,1]
            self.true_label = int(label)
            batch = img.unsqueeze(0)         # (1,1,28,28)
            self.last_image_tensor = img
            self.last_text = f"Glyphe MNIST #{idx} · vrai chiffre = {label}"
            # Push preview to NodeItem si lié
            if self._ark_node_ref is not None:
                try:
                    self._ark_node_ref.set_preview_image(img)
                except Exception:
                    pass
            return batch
        except Exception as e:
            self.last_text = f"err: {e}"
            return torch.zeros(1, 1, 28, 28)


# ============================================================
# MNIST recognizer — MLP entraîné (lazy)
# ============================================================

class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


class MnistRecognizer(nn.Module):
    """Reconnaît un chiffre depuis image (1,1,28,28). MLP pré-entraîné.

    1er usage : entraîne 3 epochs sur MNIST (~10s) + cache poids.
    Sinon : charge le checkpoint. Sortie = logits (1,10).
    """

    def __init__(self, auto_train: bool = True, epochs: int = 3):
        super().__init__()
        self.auto_train = auto_train
        self.epochs = epochs
        self._mlp: Optional[_MLP] = None
        self.last_text: str = ""
        self.last_value: str = ""
        self.last_predictions: list = []

    def _ensure_model(self):
        if self._mlp is not None:
            return self._mlp
        mlp = _MLP()
        if _CKPT.exists():
            try:
                mlp.load_state_dict(torch.load(_CKPT, map_location="cpu"))
                mlp.eval()
                self._mlp = mlp
                return mlp
            except Exception:
                pass
        if self.auto_train:
            self._train(mlp)
        self._mlp = mlp
        return mlp

    def _train(self, mlp: _MLP):
        import torchvision
        from torchvision import transforms
        from torch.utils.data import DataLoader, Subset
        tf = transforms.Compose([transforms.ToTensor()])
        tr = torchvision.datasets.MNIST(root="./.data", train=True,
                                            download=True, transform=tf)
        tr = Subset(tr, list(range(20000)))
        loader = DataLoader(tr, batch_size=128, shuffle=True)
        dev = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
        mlp = mlp.to(dev)
        opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3)
        mlp.train()
        for ep in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(dev), yb.to(dev)
                loss = F.cross_entropy(mlp(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()
        mlp = mlp.cpu().eval()
        try:
            _CKPT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(mlp.state_dict(), _CKPT)
        except Exception:
            pass

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            return x
        try:
            mlp = self._ensure_model()
            xi = x
            if xi.dim() == 3:
                xi = xi.unsqueeze(0)
            if xi.dim() == 2:
                xi = xi.view(1, 1, 28, 28)
            with torch.no_grad():
                logits = mlp(xi.float())
            probs = F.softmax(logits, dim=-1)[0]
            digit = int(probs.argmax().item())
            conf = float(probs.max().item())
            self.last_value = str(digit)
            self.last_text = f"Chiffre reconnu : {digit}  ({conf*100:.1f}%)"
            top = torch.topk(probs, 3)
            self.last_predictions = [
                {"rank": i + 1, "index": int(idx), "label": str(int(idx)),
                 "prob": float(p)}
                for i, (p, idx) in enumerate(zip(top.values.tolist(),
                                                  top.indices.tolist()))
            ]
            return logits
        except Exception as e:
            self.last_text = f"err: {e}"
            return x


# ============================================================
# Handlers + specs
# ============================================================

def h_mnist_sample(node, in_shapes):
    m = MnistSample(index=int(node.params.get("index", 0)))
    m._ark_node_ref = node
    return m, (1, 1, 28, 28)


def h_mnist_recognizer(node, in_shapes):
    m = MnistRecognizer(
        auto_train=bool(node.params.get("auto_train", True)),
        epochs=int(node.params.get("epochs", 3)),
    )
    return m, (1, 10)


def register_mnist_demo_handlers(handlers_dict: dict) -> bool:
    handlers_dict["mnist_sample"] = h_mnist_sample
    handlers_dict["mnist_recognizer"] = h_mnist_recognizer
    return True


def register_mnist_demo_specs(registry) -> bool:
    from ark.nodes.registry import NodeSpec, ParamSpec, SocketSpec
    CAT = "Demo"

    registry.register(NodeSpec(
        name="mnist_sample", title="MNIST Sample (vrai glyphe)",
        category=CAT,
        inputs=[],
        outputs=[SocketSpec("image", "tensor")],
        params=[ParamSpec("index", "int", 0, min=0, max=9999)],
        description="Émet un vrai chiffre manuscrit du test set MNIST "
                       "(1,1,28,28). Change `index` pour un autre glyphe. "
                       "Affiche le vrai label + miniature.",
        role="Source : image réelle d'un chiffre manuscrit.",
        function="MNIST test[index] → tensor (1,1,28,28) + preview.",
        has_thumbnail=True,
    ))

    registry.register(NodeSpec(
        name="mnist_recognizer", title="MNIST Recognizer (MLP entraîné)",
        category=CAT,
        inputs=[SocketSpec("image", "tensor")],
        outputs=[SocketSpec("logits", "tensor")],
        params=[
            ParamSpec("auto_train", "bool", True),
            ParamSpec("epochs", "int", 3, min=1, max=20),
        ],
        description="Reconnaît le chiffre via MLP 784→128→64→10. "
                       "S'entraîne au 1er usage (~10s, cache poids), puis "
                       "instantané. Sortie = logits + chiffre reconnu.",
        role="Classifie image → chiffre 0-9.",
        function="MLP pré-entraîné MNIST · softmax → argmax. "
                    "last_text = 'Chiffre reconnu : N (xx%)'.",
        has_thumbnail=True,
    ))
    return True
