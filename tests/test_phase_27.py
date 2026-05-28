"""Tests ARK Phase 27 — Research nodes (ai-scientist + autoresearch + sci-viz)."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
from pathlib import Path
import pytest

from ark.app import create_app
from ark.nodes.registry import default_registry
from ark.core.research_bridge import (
    IdeaGeneratorWrap, AutoResearchLoopWrap, SciVizWrap,
    register_research_handlers, register_research_specs,
)
from ark.core.graph_compiler import HANDLERS


@pytest.fixture(scope="module")
def app():
    return create_app()


def test_specs_registered(app):
    r = default_registry()
    for name in ("idea_generator", "autoresearch_loop", "sci_viz"):
        spec = r.get(name)
        assert spec is not None, f"NodeSpec {name} non enregistrée"
        assert spec.category == "Research"


def test_handlers_registered():
    for name in ("idea_generator", "autoresearch_loop", "sci_viz"):
        assert name in HANDLERS, f"handler {name} absent de HANDLERS"


def test_idea_generator_invalid_dir():
    w = IdeaGeneratorWrap(project_dir="/nonexistent/path", n_ideas=3)
    res = w.run()
    assert not res.ok
    assert "introuvable" in res.stderr or "vide" in res.stderr


def test_autoresearch_loop_invalid_dir():
    w = AutoResearchLoopWrap(project_dir="", max_iters=1)
    res = w.run()
    assert not res.ok


def test_sci_viz_dict_data(tmp_path):
    out = tmp_path / "viz.png"
    w = SciVizWrap(plot_kind="line", title="test",
                     output_path=str(out), metric_keys="a,b")
    res = w.run(data={"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
    assert res.ok, res.stderr
    assert out.exists()
    assert out.stat().st_size > 1000  # PNG non vide


def test_sci_viz_list_data(tmp_path):
    out = tmp_path / "viz2.png"
    w = SciVizWrap(plot_kind="line", title="single",
                     output_path=str(out), metric_keys="loss")
    res = w.run(data=[0.5, 0.3, 0.2, 0.1])
    assert res.ok
    assert out.exists()


def test_sci_viz_from_json(tmp_path):
    src = tmp_path / "metrics.json"
    src.write_text(json.dumps({"loss": [1.0, 0.5, 0.25]}))
    out = tmp_path / "viz3.png"
    w = SciVizWrap(plot_kind="line", output_path=str(out),
                     metric_keys="loss")
    res = w.run(source_json=str(src))
    assert res.ok
    assert out.exists()


def test_sci_viz_bar_scatter(tmp_path):
    for kind in ("bar", "scatter", "hist"):
        out = tmp_path / f"viz_{kind}.png"
        w = SciVizWrap(plot_kind=kind, output_path=str(out),
                         metric_keys="x")
        res = w.run(data={"x": [1, 2, 3, 4, 5]})
        assert res.ok, f"{kind}: {res.stderr}"
        assert out.exists()


def test_compile_node_research_nodes_passthrough(app):
    """Vérifie que les handlers retournent un (module, shape) valide."""
    from ark.core.graph_compiler import HANDLERS
    from types import SimpleNamespace

    node = SimpleNamespace(id="test", params={
        "project_dir": "", "n_ideas": 5,
        "model": "claude-3-5-sonnet-20241022",
    })
    m, shape = HANDLERS["idea_generator"](node, [(10,)])
    assert m is not None
    assert shape == (10,)

    node2 = SimpleNamespace(id="t2", params={
        "plot_kind": "line", "title": "t",
        "output_path": "/tmp/x.png", "metric_keys": "a",
    })
    m2, shape2 = HANDLERS["sci_viz"](node2, [(5,)])
    assert m2 is not None
    assert shape2 == (5,)
