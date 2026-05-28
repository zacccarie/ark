"""Phase 40 — conftest pytest : autouse fixture reset registry cache.

Le cache module-level _DEFAULT_REGISTRY_CACHE (Phase 40 R4) doit être reset
entre tests pour éviter pollution (un test register_groups modifie registry
shared par défaut).
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture(autouse=True)
def _reset_registry_cache():
    """Reset cache avant chaque test pour isolation."""
    from ark.nodes.registry import reset_registry_cache
    reset_registry_cache()
    yield
    reset_registry_cache()
