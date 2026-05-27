"""Tests ARK Phases 12 (multi-projets) + 14 (templates browser) + 15 (import legacy)."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import tempfile
from pathlib import Path
import pytest

from ark.app import create_app
from ark.nodes.registry import default_registry
from ark.ui.project_tabs import ProjectTabs
from ark.core.import_legacy import detect_format, convert_electron_to_ark, load_any
from ark.core.templates import template_mlp_mnist


@pytest.fixture(scope="module")
def app():
    return create_app()


# ============ Phase 12 multi-projets ============


def test_project_tabs_initial(app):
    pt = ProjectTabs(registry=default_registry())
    assert pt.count() == 0
    pt.new_project("Untitled 1")
    assert pt.count() == 1
    scene, view, undo = pt.active()
    assert scene is not None
    assert undo is not None


def test_project_tabs_multi(app):
    pt = ProjectTabs(registry=default_registry())
    pt.new_project("Proj A")
    pt.new_project("Proj B")
    pt.new_project("Proj C")
    assert pt.count() == 3
    # Scènes indépendantes
    pt.setCurrentIndex(0)
    sA, _, _ = pt.active()
    sA.create_node("dense")
    pt.setCurrentIndex(1)
    sB, _, _ = pt.active()
    assert len(sA.nodes()) == 1
    assert len(sB.nodes()) == 0
    pt.setCurrentIndex(2)
    sC, _, _ = pt.active()
    sC.create_node("conv2d")
    sC.create_node("dense")
    assert len(sC.nodes()) == 2
    assert len(sA.nodes()) == 1  # toujours indépendant


def test_project_tabs_close_and_reopen(app):
    pt = ProjectTabs(registry=default_registry())
    pt.new_project("A")
    pt.new_project("B")
    pt._on_close(0)
    assert pt.count() == 1
    # Close last → auto-recréation "Sans titre"
    pt._on_close(0)
    assert pt.count() == 1
    assert pt.tabText(0) == "Sans titre"


def test_project_tabs_duplicate(app):
    pt = ProjectTabs(registry=default_registry())
    pt.new_project("Original")
    s, _, _ = pt.active()
    s.create_node("dense")
    s.create_node("conv2d")
    pt.duplicate_current()
    assert pt.count() == 2
    s2, _, _ = pt.active()
    assert len(s2.nodes()) == 2


# ============ Phase 15 import legacy ============


def test_detect_format_ark():
    payload = {"version": "ark-arch-v1", "nodes": [], "connections": []}
    assert detect_format(payload) == "ark"


def test_detect_format_metamorph():
    payload = {"version": "metamorph-arch-v1", "nodes": [], "connections": []}
    assert detect_format(payload) == "metamorph"


def test_detect_format_electron():
    payload = {
        "nodes": [{"id": "n1", "type": "conv2d", "x": 100, "y": 200,
                    "params": {"filters": 32, "kernelSize": 3}}],
        "connections": [],
    }
    assert detect_format(payload) == "electron"


def test_detect_format_unknown():
    assert detect_format({}) == "unknown"


def test_convert_electron_basic():
    payload = {
        "name": "test",
        "nodes": [
            {"id": "n1", "type": "image-input", "x": 0, "y": 0,
              "params": {"shape": "[1,28,28]"}},
            {"id": "n2", "type": "conv2d", "x": 200, "y": 0,
              "params": {"filters": 16, "kernelSize": 3, "activation": "relu"}},
            {"id": "n3", "type": "vg-block", "x": 400, "y": 0,
              "params": {"start_hidden": 16, "out_dim": 32}},
        ],
        "connections": [
            {"from": "n1", "to": "n2"},
            {"from": "n2", "to": "n3"},
        ],
    }
    out = convert_electron_to_ark(payload)
    assert out["version"] == "ark-arch-v1"
    assert len(out["nodes"]) == 3
    # Type kebab → snake
    types = [n["type"] for n in out["nodes"]]
    assert "image_input" in types
    assert "conv2d" in types
    assert "vg_block" in types
    # Params renamed : kernelSize → kernel_size
    conv = next(n for n in out["nodes"] if n["type"] == "conv2d")
    assert "kernel_size" in conv["params"]
    assert "kernelSize" not in conv["params"]
    assert conv["params"]["kernel_size"] == 3
    # Connections format ARK
    assert all("from_node" in c and "to_node" in c for c in out["connections"])


def test_convert_electron_unknown_type_skipped():
    payload = {
        "nodes": [
            {"id": "n1", "type": "image-input", "x": 0, "y": 0, "params": {}},
            {"id": "x", "type": "fake-node-type", "x": 100, "y": 0, "params": {}},
        ],
        "connections": [],
    }
    out = convert_electron_to_ark(payload)
    assert len(out["nodes"]) == 1
    assert "fake-node-type" in out["metadata"]["unknown_types_skipped"]


def test_load_any_dispatches(tmp_path):
    # ARK format
    p = tmp_path / "ark.json"
    p.write_text(json.dumps({"version": "ark-arch-v1", "nodes": [], "connections": []}))
    payload, fmt = load_any(p)
    assert fmt == "ark"
    # Electron format
    p2 = tmp_path / "electron.json"
    p2.write_text(json.dumps({
        "nodes": [{"id": "n1", "type": "dense", "x": 0, "y": 0, "params": {"neurons": 64}}],
        "connections": [],
    }))
    payload2, fmt2 = load_any(p2)
    assert fmt2 == "electron"
    assert payload2["version"] == "ark-arch-v1"


# ============ Phase 14 templates browser ============


def test_templates_browser_construct(app):
    from ark.ui.dialogs.templates_browser import TemplatesBrowser
    dlg = TemplatesBrowser()
    # Au moins 6 items dans la liste
    assert dlg.list.count() >= 6


def test_templates_browser_filter(app):
    from ark.ui.dialogs.templates_browser import TemplatesBrowser
    dlg = TemplatesBrowser()
    dlg._refilter("mnist")
    # MLP MNIST devrait matcher
    found_mnist = False
    for i in range(dlg.list.count()):
        if "MNIST" in dlg.list.item(i).text():
            found_mnist = True
    assert found_mnist
    dlg._refilter("zzzzz_nope")
    assert dlg.list.count() == 0


def test_templates_browser_fav_persist(app, tmp_path, monkeypatch):
    from ark.ui.dialogs import templates_browser as tb
    monkeypatch.setattr(tb, "FAV_FILE", tmp_path / "favs.json")
    dlg = tb.TemplatesBrowser()
    dlg.favs.add("mlp_mnist")
    tb._save_favs(dlg.favs)
    assert (tmp_path / "favs.json").exists()
    loaded = tb._load_favs()
    assert "mlp_mnist" in loaded
