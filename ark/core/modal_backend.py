"""Phase 68 — Modal native remote-compute backend.

Entraîne une architecture ARK sur GPU distant (Modal) sans quitter l'app.

Flow :
  ARK local : scene → payload JSON → submit_training()
  Modal GPU : compile_payload(payload) → train N epochs → stream metrics
  ARK local : reçoit metrics live → download state_dict final

Auth : token Modal stocké ~/.ark/secrets.json["modal"] =
       {"token_id": "ak-...", "token_secret": "as-..."}
       OU ~/.modal.toml standard. Env MODAL_TOKEN_ID/SECRET set au runtime.

Le module est importable même sans `modal` installé (lazy). Vérifier
modal_available() avant usage.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Iterable, Optional

ARK_ROOT = str(Path(__file__).parent.parent.parent)  # .../ark repo root


def modal_available() -> bool:
    try:
        import modal  # noqa
        return True
    except Exception:
        return False


def _apply_token_env():
    """Pousse le token Modal depuis ARK secrets vers env vars."""
    from ark.core.llm_providers import load_secrets
    cfg = load_secrets().get("modal", {})
    tid = cfg.get("token_id") or os.environ.get("MODAL_TOKEN_ID")
    tsec = cfg.get("token_secret") or os.environ.get("MODAL_TOKEN_SECRET")
    if tid:
        os.environ["MODAL_TOKEN_ID"] = tid
    if tsec:
        os.environ["MODAL_TOKEN_SECRET"] = tsec
    return bool(tid and tsec)


def token_configured() -> bool:
    from ark.core.llm_providers import load_secrets
    cfg = load_secrets().get("modal", {})
    if cfg.get("token_id") and cfg.get("token_secret"):
        return True
    if os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"):
        return True
    # ~/.modal.toml present?
    return (Path.home() / ".modal.toml").exists()


def save_token(token_id: str, token_secret: str):
    from ark.core.llm_providers import load_secrets, save_secrets
    s = load_secrets()
    s["modal"] = {"token_id": token_id, "token_secret": token_secret}
    save_secrets(s)


# ============================================================
# Modal app builder (lazy — only when modal installed)
# ============================================================

def _build_app(gpu: str = "A10G"):
    import modal

    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("torch", "torchvision", "numpy", "scipy")
        .add_local_dir(ARK_ROOT, remote_path="/root/ark_repo",
                       ignore=["*.pyc", "__pycache__", ".git", "ui_web/node_modules",
                               "ui_web/dist", "*.pt", "output", "demo_output"])
    )
    app = modal.App("ark-train")

    @app.function(gpu=gpu, image=image, timeout=3600)
    def train_remote(payload_json: str, epochs: int = 3, lr: float = 1e-3,
                     batch_size: int = 64, dataset: str = "mnist"):
        """Rebuild + train. Returns dict with metrics history + state_dict b64."""
        import sys, io, base64, json as _json
        sys.path.insert(0, "/root/ark_repo")
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from ark.core.headless_scene import compile_payload

        payload = _json.loads(payload_json)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_classes = payload.get("metadata", {}).get("num_classes", 10)
        cg = compile_payload(payload, num_classes=num_classes)
        model = cg.model.to(device)

        # Dataset
        from torchvision import datasets, transforms
        tf = transforms.Compose([transforms.ToTensor()])
        if dataset == "cifar10":
            ds = datasets.CIFAR10("/tmp/data", train=True, download=True,
                                  transform=tf)
        else:
            ds = datasets.MNIST("/tmp/data", train=True, download=True,
                                transform=tf)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                             shuffle=True, num_workers=2)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        history = []
        model.train()
        for ep in range(epochs):
            tot, correct, loss_sum, nb = 0, 0, 0.0, 0
            for i, (x, y) in enumerate(loader):
                x, y = x.to(device), y.to(device)
                out = model(x)
                if isinstance(out, tuple):
                    out = out[0]
                if out.dim() > 2:
                    out = out.flatten(1)
                loss = F.cross_entropy(out, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_sum += loss.item()
                nb += 1
                pred = out.argmax(-1)
                correct += (pred == y).sum().item()
                tot += y.numel()
                if i >= 200:  # cap per epoch for speed
                    break
            metric = {"epoch": ep + 1, "loss": loss_sum / max(nb, 1),
                      "acc": correct / max(tot, 1)}
            history.append(metric)
            print(f"[modal] epoch {ep+1}/{epochs} loss={metric['loss']:.4f} "
                  f"acc={metric['acc']:.4f}", flush=True)

        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        sd_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return {"history": history, "state_dict_b64": sd_b64,
                "device": device, "params": sum(p.numel()
                                                 for p in model.parameters())}

    return app, train_remote


# ============================================================
# Local launcher
# ============================================================

def submit_training(payload: dict, epochs: int = 3, lr: float = 1e-3,
                    batch_size: int = 64, gpu: str = "A10G",
                    dataset: str = "mnist",
                    on_log=None) -> dict:
    """Bloquant. Lance training Modal, retourne dict résultat.

    on_log(str) : callback pour chaque ligne de log/metric.
    Retourne {"ok": bool, "history": [...], "checkpoint": path|None,
              "error": str|None}.
    """
    def log(msg):
        if on_log:
            on_log(msg)

    if not modal_available():
        return {"ok": False, "error": "modal package not installed "
                "(pip install modal)", "history": []}
    if not _apply_token_env():
        if not (Path.home() / ".modal.toml").exists():
            return {"ok": False, "error": "Modal token not configured. "
                    "Settings → Modal → token.", "history": []}

    log(f"[modal] building app (gpu={gpu})…")
    try:
        import modal
        app, train_remote = _build_app(gpu=gpu)
        payload_json = json.dumps(payload)
        log(f"[modal] submitting training ({epochs} epochs, {dataset})…")
        with app.run():
            result = train_remote.remote(
                payload_json, epochs=epochs, lr=lr,
                batch_size=batch_size, dataset=dataset)
        for m in result.get("history", []):
            log(f"[modal] epoch {m['epoch']} loss={m['loss']:.4f} "
                f"acc={m['acc']:.4f}")
        # Save checkpoint locally
        ckpt_path = None
        sd_b64 = result.get("state_dict_b64")
        if sd_b64:
            import base64
            out_dir = Path.home() / ".ark" / "modal_checkpoints"
            out_dir.mkdir(parents=True, exist_ok=True)
            name = payload.get("name", "model").replace(" ", "_")
            ckpt_path = str(out_dir / f"{name}.pt")
            Path(ckpt_path).write_bytes(base64.b64decode(sd_b64))
            log(f"[modal] checkpoint saved : {ckpt_path}")
        log(f"[modal] done · {result.get('params', 0):,} params · "
            f"device={result.get('device')}")
        return {"ok": True, "history": result.get("history", []),
                "checkpoint": ckpt_path, "error": None}
    except Exception as e:
        log(f"[modal] error : {e}")
        return {"ok": False, "error": str(e), "history": []}


GPU_CHOICES = ["T4", "A10G", "L4", "A100", "A100-80GB", "H100"]
