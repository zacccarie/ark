"""Tests Phase 33 — Resize 2D + Load CSV/JSON."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import csv
from pathlib import Path
import pytest
import torch
from PySide6.QtCore import QPointF
from PySide6.QtGui import QMouseEvent
from PySide6.QtCore import QEvent, Qt

from ark.app import create_app
from ark.nodes.registry import default_registry
from ark.core.media_loaders import LoadCSV, LoadJSON
from ark.core.graph_compiler import HANDLERS
from ark.core.format import scene_to_payload, payload_to_scene
from ark.ui.node_editor.node_item import NodeItem
from ark.ui.node_editor.scene import NodeScene


@pytest.fixture(scope="module")
def app():
    return create_app()


# ============ Resize 2D ============

def test_node_natural_height_unchanged(app):
    r = default_registry()
    spec = r.get("dense")
    node = NodeItem(spec)
    h_natural = node.natural_height
    assert h_natural > 0
    assert node.height == h_natural  # extra=0 par défaut
    assert node.user_height_extra == 0


def test_node_height_extra_adds(app):
    r = default_registry()
    spec = r.get("dense")
    node = NodeItem(spec)
    h0 = node.height
    node.user_height_extra = 80
    assert node.height == h0 + 80


def test_resize_format_persistence(app, tmp_path):
    r = default_registry()
    scene = NodeScene(registry=r)
    spec = r.get("dense")
    n = scene._raw_create_node("dense", x=10, y=20)
    n.width = 300
    n.user_height_extra = 60

    payload = scene_to_payload(scene, name="t")
    assert payload["nodes"][0]["width"] == 300
    assert payload["nodes"][0]["height_extra"] == 60

    # Round-trip via payload_to_scene
    scene2 = NodeScene(registry=r)
    payload_to_scene(scene2, payload)
    nodes2 = scene2.nodes()
    assert len(nodes2) == 1
    assert nodes2[0].width == 300
    assert nodes2[0].user_height_extra == 60


# ============ LoadCSV ============

def test_load_csv_fallback(app):
    m = LoadCSV(path="/nonexistent.csv")
    out = m()
    assert out.shape == (32, 10)


def test_load_csv_real(tmp_path, app):
    """CSV réel avec header → tenseur (N, F)."""
    p = tmp_path / "data.csv"
    with p.open("w") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "z"])
        w.writerow([1.0, 2.0, 3.0])
        w.writerow([4.0, 5.0, 6.0])
        w.writerow([7.0, 8.0, 9.0])
    m = LoadCSV(path=str(p), has_header=True, separator=",")
    out = m()
    assert out.shape == (3, 3)
    assert out[0, 0].item() == pytest.approx(1.0)
    assert out[2, 2].item() == pytest.approx(9.0)


def test_load_csv_max_rows(tmp_path, app):
    p = tmp_path / "big.csv"
    with p.open("w") as f:
        w = csv.writer(f)
        w.writerow(["a", "b"])
        for i in range(100):
            w.writerow([i, i * 2])
    m = LoadCSV(path=str(p), has_header=True, max_rows=5)
    out = m()
    assert out.shape == (5, 2)


def test_load_csv_no_header(tmp_path, app):
    p = tmp_path / "noheader.csv"
    with p.open("w") as f:
        w = csv.writer(f)
        for i in range(4):
            w.writerow([i, i + 0.5])
    m = LoadCSV(path=str(p), has_header=False, separator=",")
    out = m()
    assert out.shape == (4, 2)


# ============ LoadJSON ============

def test_load_json_fallback(app):
    m = LoadJSON(path="/nonexistent.json")
    out = m()
    assert out.shape == (1,)


def test_load_json_list(tmp_path, app):
    p = tmp_path / "vec.json"
    p.write_text("[1.0, 2.5, 3.7, 4.0]")
    m = LoadJSON(path=str(p))
    out = m()
    assert out.shape == (4,)
    assert out[1].item() == pytest.approx(2.5)


def test_load_json_nested(tmp_path, app):
    """Drill via path_in_json."""
    p = tmp_path / "n.json"
    p.write_text(json.dumps({
        "metadata": {"name": "test"},
        "samples": {"values": [1, 2, 3, 4, 5]},
    }))
    m = LoadJSON(path=str(p), path_in_json="samples.values")
    out = m()
    assert out.shape == (5,)
    assert out[-1].item() == pytest.approx(5.0)


def test_load_json_data_attr(tmp_path, app):
    """Le raw dict reste accessible via module.data."""
    p = tmp_path / "raw.json"
    p.write_text(json.dumps({"k": "v", "n": [1, 2]}))
    m = LoadJSON(path=str(p))
    _ = m()
    assert m.data is not None
    assert m.data["k"] == "v"


# ============ Registration ============

def test_csv_json_handlers_registered(app):
    assert "load_csv" in HANDLERS
    assert "load_json" in HANDLERS


def test_csv_json_specs_registered(app):
    r = default_registry()
    csv_spec = r.get("load_csv")
    assert csv_spec is not None
    assert csv_spec.category == "Input"
    assert any(p.kind == "filepath" for p in csv_spec.params)

    json_spec = r.get("load_json")
    assert json_spec is not None
    assert json_spec.category == "Input"


def test_csv_handler_dispatch(app):
    from types import SimpleNamespace
    node = SimpleNamespace(id="x", params={
        "path": "/tmp/none.csv", "has_header": True, "separator": ",",
        "max_rows": 0, "numeric_only": True,
    })
    m, shape = HANDLERS["load_csv"](node, [])
    assert m is not None
    out = m()
    assert isinstance(out, torch.Tensor)
