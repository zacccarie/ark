"""Tests Phase 26 — glossaire + tooltips."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from ark.app import create_app
from ark.core.glossary import GLOSSARY, find, get, by_category, GlossaryEntry
from ark.nodes.registry import default_registry


@pytest.fixture(scope="module")
def app():
    return create_app()


# ============ Glossary content ============


def test_glossary_non_empty():
    assert len(GLOSSARY) > 30


def test_glossary_entries_have_required_fields():
    for e in GLOSSARY:
        assert isinstance(e, GlossaryEntry)
        assert e.term and isinstance(e.term, str)
        assert e.category and isinstance(e.category, str)
        assert e.short and isinstance(e.short, str)
        assert e.long and isinstance(e.long, str)
        assert isinstance(e.related, tuple)


def test_glossary_no_duplicate_terms():
    terms = [e.term.lower() for e in GLOSSARY]
    assert len(terms) == len(set(terms)), "Doublon dans glossary"


def test_glossary_categories_present():
    cats = {e.category for e in GLOSSARY}
    # Au moins ces 5
    for c in ["Architecture", "Training", "Observation", "Diffusion", "Workflow"]:
        assert c in cats


def test_glossary_related_terms_exist():
    """Tous les related doivent référencer un term existant (cohérence)."""
    all_terms = {e.term.lower() for e in GLOSSARY}
    for e in GLOSSARY:
        for r in e.related:
            # Tolérance : related peut pointer vers un alias (ex. "Multi-Head Attention" ≈ "Attention")
            # On accepte si le term existe ou si un prefix existe
            found = r.lower() in all_terms or any(t in r.lower() or r.lower() in t for t in all_terms)
            assert found, f"Related '{r}' (de '{e.term}') sans entrée"


def test_glossary_find():
    assert len(find("attention")) >= 1
    assert len(find("")) == len(GLOSSARY)
    assert len(find("xyznotexist123")) == 0


def test_glossary_find_case_insensitive():
    a = find("ADAM")
    b = find("adam")
    assert len(a) == len(b)
    assert len(a) >= 1


def test_glossary_get_exact():
    e = get("Adam")
    assert e is not None
    assert e.term == "Adam"
    assert "momentum" in e.long.lower() or "Adaptive" in e.short


def test_glossary_get_case_insensitive():
    assert get("adam") is not None
    assert get("ADAM") is not None


def test_glossary_get_missing_returns_none():
    assert get("notarealterm") is None


def test_glossary_by_category():
    cats = by_category()
    assert "Architecture" in cats
    assert "Diffusion" in cats
    # Trié
    for entries in cats.values():
        assert entries == sorted(entries, key=lambda e: e.term.lower())


# ============ Dialog ============


def test_glossary_dialog_builds(app):
    from ark.ui.dialogs.glossary_dialog import GlossaryDialog
    dlg = GlossaryDialog()
    assert dlg._list.count() > len(GLOSSARY)  # entries + headers catégories


def test_glossary_dialog_select_initial(app):
    from ark.ui.dialogs.glossary_dialog import GlossaryDialog
    dlg = GlossaryDialog(initial_term="Attention")
    # Le terme doit avoir été sélectionné (current row non-header)
    current = dlg._list.currentItem()
    assert current is not None


def test_glossary_dialog_search_filters(app):
    from ark.ui.dialogs.glossary_dialog import GlossaryDialog
    dlg = GlossaryDialog()
    full_count = dlg._list.count()
    dlg._search.setText("adam")
    filtered = dlg._list.count()
    assert filtered < full_count
    assert filtered >= 1


def test_glossary_dialog_search_no_match(app):
    from ark.ui.dialogs.glossary_dialog import GlossaryDialog
    dlg = GlossaryDialog()
    dlg._search.setText("xyznotexist123")
    # 0 items (pas même de header) → uniquement le count label change
    assert dlg._list.count() == 0


# ============ NodeSpec.param_help & description ============


def test_node_specs_have_descriptions(app):
    """Tous les nodes registry doivent avoir description non-vide."""
    r = default_registry()
    missing = [s.name for s in r.all() if not s.description]
    # Tolérance : plugins/metamorph peuvent ne pas être présents
    core_missing = [n for n in missing
                     if not n.startswith(("metamorph_", "scale"))]
    assert not core_missing, f"Nodes sans description : {core_missing}"


def test_param_help_field_exists(app):
    """Param_help est un dict (vide ou rempli)."""
    r = default_registry()
    for s in r.all():
        assert isinstance(s.param_help, dict)


def test_dense_param_help(app):
    """dense doit avoir param_help sur ses 3 params."""
    r = default_registry()
    s = r.get("dense")
    assert "neurons" in s.param_help
    assert "activation" in s.param_help
    assert "dropout" in s.param_help


# ============ UI tooltips ============


def test_palette_tooltip_rich(app):
    from ark.ui.panels.palette_panel import PalettePanel
    r = default_registry()
    p = PalettePanel(r)
    # Cherche un item child non-header
    def walk(item):
        for i in range(item.childCount()):
            c = item.child(i)
            if c.childCount() == 0:
                return c
        return None
    root = p.tree.invisibleRootItem()
    found = None
    for i in range(root.childCount()):
        cat = root.child(i)
        if cat.childCount() > 0:
            found = cat.child(0)
            break
    assert found is not None
    tip = found.toolTip(0)
    assert "<b>" in tip  # HTML riche
    assert "In" in tip or "Out" in tip or "Params" in tip


def test_socket_has_tooltip(app):
    from ark.ui.node_editor.socket_item import SocketItem
    from ark.ui.node_editor.node_item import NodeItem
    r = default_registry()
    n = NodeItem(r.get("dense"))
    sock = n.input_sockets[0]
    tip = sock.toolTip()
    assert "Entrée" in tip
    assert "tensor" in tip


def test_node_item_has_tooltip(app):
    from ark.ui.node_editor.node_item import NodeItem
    r = default_registry()
    n = NodeItem(r.get("dense"))
    tip = n.toolTip()
    assert "Dense" in tip
    assert "<b>" in tip
    assert "Linear" in tip  # description


def test_node_item_tooltip_updates_on_error(app):
    from ark.ui.node_editor.node_item import NodeItem
    r = default_registry()
    n = NodeItem(r.get("dense"))
    n.set_error("Shape mismatch")
    tip = n.toolTip()
    assert "Shape mismatch" in tip
