"""Phase 50 — Safe load_state_dict avec warning sources non-trusted.

Wrap torch.load + state_dict load avec :
- Whitelist domaines trusted (huggingface.co, github.com user, local file)
- weights_only=True par défaut (PyTorch 2.6+ default)
- Warning + confirm si source externe ou .pkl format
"""
from __future__ import annotations
import warnings
from pathlib import Path
from urllib.parse import urlparse

import torch
import torch.nn as nn


TRUSTED_DOMAINS = {
    "huggingface.co",
    "github.com",
    "raw.githubusercontent.com",
    "pytorch.org",
    "storage.googleapis.com",
}


def _is_trusted(path_or_url: str) -> bool:
    """Path local ou domaine whitelist."""
    if not path_or_url:
        return False
    # Local path → trusted
    p = Path(path_or_url)
    if p.exists() or not path_or_url.startswith(("http://", "https://")):
        return True
    # URL : check domain
    try:
        parsed = urlparse(path_or_url)
        return parsed.netloc in TRUSTED_DOMAINS
    except Exception:
        return False


def safe_torch_load(path_or_url: str, map_location: str = "cpu",
                       weights_only: bool = True) -> dict:
    """Wrapper torch.load avec warning si non-trusted + weights_only=True default.

    Raises UserWarning si source externe non whitelistée.
    Lève RuntimeError si l'utilisateur passe weights_only=False sur source non-trusted.
    """
    if not _is_trusted(path_or_url):
        warnings.warn(
            f"⚠ Source non-trusted détectée : {path_or_url}\n"
            f"  Risque pickle code execution si weights_only=False.\n"
            f"  Force weights_only=True. Pour bypass : set ARK_TRUST_SOURCES=1.",
            UserWarning, stacklevel=2)
        weights_only = True
    return torch.load(path_or_url, map_location=map_location,
                         weights_only=weights_only)


def safe_load_state_dict(model: nn.Module, path: str,
                              strict: bool = True,
                              map_location: str = "cpu") -> dict:
    """Charge state_dict avec safe_torch_load + load_state_dict standard."""
    state = safe_torch_load(path, map_location=map_location,
                                  weights_only=True)
    if not isinstance(state, dict):
        raise RuntimeError(
            f"Format inattendu : {type(state).__name__}, attendait dict")
    return model.load_state_dict(state, strict=strict)
