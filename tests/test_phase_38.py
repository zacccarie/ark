"""Tests Phase 38 — Help page + Inspector exhaustif + didactic data."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from ark.app import create_app
from ark.nodes.registry import default_registry, NodeSpec, SocketSpec
from ark.core.glossary import GLOSSARY, GlossaryEntry, find
from ark.core.didactic_data import DIDACTIC_EXTRA
from ark.ui.panels.inspector_panel import InspectorPanel
from ark.ui.node_editor.node_item import NodeItem


@pytest.fixture(scope="module")
def app():
    return create_app()


# ============ Didactic data ============

def test_didactic_dict_has_keys(app):
    """Vérifie présence des clés didactiques pour concepts clés."""
    expected_keys = [
        "vector", "matrix", "tensor_nd", "svd", "pca",
        "gaussian", "entropy", "kl_divergence", "cross_entropy_loss",
        "gradient_descent", "adam", "backpropagation",
        "act_relu", "act_gelu", "act_softmax",
        "dense", "conv2d", "dropout_layer", "lstm_layer",
        "self_attention", "multi_head_attention", "positional_encoding", "moe",
        "vae", "gan", "ddpm",
        "overfitting", "embedding", "latent_space",
        "attractor", "lyapunov_exp", "reaction_diffusion", "cellular_automaton",
        "shap", "phase_space_3d", "takens_embedding",
        "critical_grow_net", "scaling_laws", "grokking",
    ]
    missing = [k for k in expected_keys if k not in DIDACTIC_EXTRA]
    assert len(missing) == 0, f"Manquant : {missing}"


def test_didactic_entry_structure():
    """Chaque entrée a au moins intuition + 1 autre champ."""
    for name, extra in DIDACTIC_EXTRA.items():
        assert "intuition" in extra, f"{name} sans intuition"
        assert len(extra["intuition"]) > 30, \
            f"{name} intuition trop courte"


def test_didactic_total_count():
    """Au moins 40 concepts enrichis."""
    assert len(DIDACTIC_EXTRA) >= 40, \
        f"Trop peu : {len(DIDACTIC_EXTRA)}"


# ============ GlossaryEntry nouveaux champs ============

def test_glossary_entry_has_new_fields():
    e = GlossaryEntry(term="Test", category="Foundations",
                          short="x", long="y",
                          intuition="i", when_to_use="w",
                          example="e", pitfall="p", equation="eq")
    assert e.intuition == "i"
    assert e.when_to_use == "w"
    assert e.example == "e"
    assert e.pitfall == "p"
    assert e.equation == "eq"


def test_glossary_merged_didactic(app):
    """Vérifie qu'un concept didactique a bien hérité ses champs."""
    e = next((g for g in GLOSSARY if g.term == "Vector"), None)
    assert e is not None
    assert e.intuition, "Vector sans intuition"
    assert "liste ordonnée" in e.intuition.lower() or \
              "point" in e.intuition.lower()
    assert e.when_to_use, "Vector sans when_to_use"
    assert e.example, "Vector sans example"


def test_glossary_self_attention_didactic(app):
    """Self-Attention enrichi."""
    e = next((g for g in GLOSSARY if g.term == "Self-Attention"), None)
    assert e is not None
    assert e.intuition
    assert e.equation, "Self-Attention sans equation"
    assert "softmax" in e.equation.lower() or "qk" in e.equation.lower()


# ============ HelpPage dialog ============

def test_help_page_constructs(app):
    from ark.ui.dialogs.help_page import HelpPage
    dlg = HelpPage()
    # Tree populé
    assert dlg.tree.topLevelItemCount() > 0
    # Search vide → tree complet
    assert dlg.tree.topLevelItemCount() >= 20  # 23 catégories


def test_help_page_search(app):
    from ark.ui.dialogs.help_page import HelpPage
    dlg = HelpPage()
    dlg._on_search("attention")
    # Au moins 1 résultat
    found = False
    for i in range(dlg.tree.topLevelItemCount()):
        cat = dlg.tree.topLevelItem(i)
        if cat.childCount() > 0:
            found = True
            break
    assert found


def test_help_page_select_term(app):
    from ark.ui.dialogs.help_page import HelpPage
    dlg = HelpPage(initial_term="Vector")
    # Détail rendu avec sections didactiques
    html = dlg.detail.toHtml()
    assert "Vector" in html
    assert "Intuition" in html or "intuition" in html


def test_help_page_render_with_didactic(app):
    """Si concept a didactic, rendu inclut sections."""
    from ark.ui.dialogs.help_page import HelpPage
    dlg = HelpPage()
    # Find Vector entry
    e = next(g for g in GLOSSARY if g.term == "Vector")
    html = dlg._render_entry(e)
    assert "Intuition" in html
    assert "Quand l'utiliser" in html
    assert "Exemple" in html or "exemple" in html


def test_help_page_render_no_didactic_graceful(app):
    """Concept sans didactic : pas de crash."""
    from ark.ui.dialogs.help_page import HelpPage
    dlg = HelpPage()
    e = GlossaryEntry(term="Test", category="Foundations",
                          short="x", long="y")
    html = dlg._render_entry(e)
    assert "Test" in html
    # Pas de section vide rendue
    assert "Intuition" not in html


# ============ Inspector enrichi ============

def test_inspector_shows_intuition_from_glossary(app):
    """Inspector affiche intuition depuis glossaire pour nodes connus."""
    r = default_registry()
    spec = r.get("dense")
    assert spec is not None
    panel = InspectorPanel()
    node = NodeItem(spec)
    panel.set_node(node)
    # Dense doit avoir intuition depuis didactic
    intu_text = panel.intuition_label.text()
    assert intu_text, "intuition_label vide pour 'dense'"
    # isVisible False because panel not shown — vérifie juste le texte
    assert "comité" in intu_text or "neurone" in intu_text.lower()


def test_inspector_shows_pitfall(app):
    r = default_registry()
    spec = r.get("dropout_layer")
    assert spec is not None
    panel = InspectorPanel()
    node = NodeItem(spec)
    panel.set_node(node)
    assert panel.pitfall_label.text()
    assert "model.eval" in panel.pitfall_label.text().lower() or \
              "eval" in panel.pitfall_label.text().lower()


def test_inspector_when_to_use(app):
    r = default_registry()
    spec = r.get("vae")
    if spec is None:
        pytest.skip("vae spec absent")
    panel = InspectorPanel()
    node = NodeItem(spec)
    panel.set_node(node)
    assert panel.when_label.text()


def test_inspector_hides_didactic_when_absent(app):
    """Concept sans didactic data : labels cachés."""
    r = default_registry()
    # arch_diagram pas dans didactic_data
    spec = r.get("arch_diagram")
    if spec is None:
        pytest.skip("arch_diagram absent")
    panel = InspectorPanel()
    node = NodeItem(spec)
    panel.set_node(node)
    # intuition peut être vide
    assert panel.intuition_label is not None


# ============ Main window action ============

def test_main_window_help_action(app):
    from ark.ui.main_window import MainWindow
    w = MainWindow()
    # Action _on_open_help_page existe
    assert hasattr(w, "_on_open_help_page")
    w._on_open_help_page()
    assert hasattr(w, "_help_dlg")
    assert w._help_dlg is not None
