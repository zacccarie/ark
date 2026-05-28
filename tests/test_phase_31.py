"""Tests Phase 31 — Node Groups."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import tempfile
from pathlib import Path

import pytest
import torch

from ark.app import create_app
from ark.nodes.registry import default_registry, NodeRegistry
from ark.core.groups import (
    serialize_group, save_group, load_group, register_groups,
    GroupModule, ARK_GROUP_VERSION, _slug,
)
from ark.core.graph_compiler import HANDLERS
from ark.ui.node_editor.scene import NodeScene
from ark.ui.node_editor.connection_item import ConnectionItem


@pytest.fixture(scope="module")
def app():
    return create_app()


def _build_scene_chain(registry, types: list[str]):
    """Crée scene + chaîne linéaire de nodes connectés."""
    scene = NodeScene(registry=registry)
    nodes = []
    for i, t in enumerate(types):
        spec = registry.get(t)
        assert spec is not None, f"spec {t} absent"
        node = scene._raw_create_node(t, x=i * 200, y=0)
        nodes.append(node)
    # Connect séquentiellement output[0] → input[0]
    for a, b in zip(nodes, nodes[1:]):
        if a.output_sockets and b.input_sockets:
            conn = ConnectionItem(start_socket=a.output_sockets[0],
                                       end_socket=b.input_sockets[0])
            a.output_sockets[0].add_connection(conn)
            b.input_sockets[0].add_connection(conn)
            scene._raw_add_connection(conn)
    return scene, nodes


def test_slug_normalize():
    assert _slug("My ResNet Block") == "my_resnet_block"
    assert _slug("HELLO-World!") == "hello_world"
    assert _slug("") == "group"


def test_serialize_group_basic(app):
    r = default_registry()
    scene, nodes = _build_scene_chain(r, ["act_relu", "act_tanh", "act_sigmoid"])
    payload = serialize_group(scene, nodes, name="test_chain", title="Test Chain",
                                  description="test")
    assert payload["version"] == ARK_GROUP_VERSION
    assert payload["name"] == "test_chain"
    assert payload["title"] == "Test Chain"
    assert len(payload["nodes"]) == 3
    assert len(payload["connections"]) == 2
    # Exposed inputs : 1 (relu input)
    assert len(payload["exposed_inputs"]) >= 1
    # Exposed outputs : 1 (sigmoid output)
    assert len(payload["exposed_outputs"]) >= 1


def test_serialize_group_empty_raises(app):
    r = default_registry()
    scene = NodeScene(registry=r)
    with pytest.raises(ValueError, match="vide"):
        serialize_group(scene, [], name="x", title="x")


def test_save_load_group_roundtrip(app, tmp_path):
    r = default_registry()
    scene, nodes = _build_scene_chain(r, ["act_relu", "act_tanh"])
    payload = serialize_group(scene, nodes, name="rt", title="RT")
    p = tmp_path / "rt.json"
    save_group(payload, path=p)
    assert p.exists()
    loaded = load_group(p)
    assert loaded["name"] == "rt"
    assert len(loaded["nodes"]) == 2


def test_group_module_forward(app):
    """GroupModule compose les sous-modules → forward fonctionne."""
    r = default_registry()
    scene, nodes = _build_scene_chain(r, ["act_relu", "act_tanh"])
    payload = serialize_group(scene, nodes, name="rt", title="RT")
    m = GroupModule(payload, HANDLERS)
    x = torch.randn(4, 8)
    out = m(x)
    assert out.shape == x.shape
    # tanh borné [-1, 1]
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_register_groups_creates_spec(app, tmp_path, monkeypatch):
    """register_groups depuis dir tmp → NodeSpec + handler."""
    from ark.core import groups as G
    monkeypatch.setattr(G, "GROUPS_DIR", tmp_path)

    # Sauve un group bidon
    r = default_registry()
    scene, nodes = _build_scene_chain(r, ["act_relu", "act_tanh"])
    payload = serialize_group(scene, nodes, name="test_grp_reg",
                                  title="Test Grp Reg", category="User Groups")
    save_group(payload, path=tmp_path / "test_grp_reg.json")

    # Register
    reg2 = NodeRegistry()
    handlers = {}
    n = register_groups(reg2, handlers)
    assert n >= 1
    spec = reg2.get("test_grp_reg")
    assert spec is not None
    assert spec.title == "Test Grp Reg"
    assert spec.category == "User Groups"
    assert "test_grp_reg" in handlers


def test_group_handler_compile_chain(app, tmp_path, monkeypatch):
    """Group enregistré utilisable via HANDLERS."""
    from ark.core import groups as G
    from ark.core.graph_compiler import HANDLERS as H
    monkeypatch.setattr(G, "GROUPS_DIR", tmp_path)
    r = default_registry()
    scene, nodes = _build_scene_chain(r, ["act_relu", "act_tanh"])
    payload = serialize_group(scene, nodes, name="grp_compile",
                                  title="Grp Compile")
    save_group(payload, path=tmp_path / "grp_compile.json")
    G.register_groups(r, H)

    assert "grp_compile" in H
    from types import SimpleNamespace
    fake = SimpleNamespace(id="g", params={})
    m, shape = H["grp_compile"](fake, [(8,)])
    out = m(torch.randn(2, 8))
    assert out.shape == (2, 8)


def test_save_group_dialog_validation(app):
    """Le dialog refuse names invalides."""
    from ark.ui.dialogs.save_group_dialog import SaveGroupDialog
    dlg = SaveGroupDialog(n_selected=3)
    # Champ vide → reject ne ferme pas (validation interne)
    dlg.name_edit.setText("")
    dlg._on_save()  # devrait NOT accept
    assert dlg.result() != dlg.DialogCode.Accepted

    dlg.name_edit.setText("123invalid")
    dlg._on_save()
    assert dlg.result() != dlg.DialogCode.Accepted

    dlg.name_edit.setText("valid_name")
    dlg.title_edit.setText("Valid Name")
    dlg._on_save()
    assert dlg.result() == dlg.DialogCode.Accepted
    payload = dlg.get_payload()
    assert payload["name"] == "valid_name"
    assert payload["title"] == "Valid Name"
    assert payload["category"] == "User Groups"
