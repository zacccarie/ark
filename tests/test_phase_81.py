"""Tests ARK Phase 81 — model-kind taxonomy + held-out eval + état validé."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Taxonomie
# ============================================================

def test_detect_kind_classifier():
    from ark.core.model_kind import detect_kind
    s = detect_kind(["mnist_recognizer", "dense", "softmax"])
    assert s.kind == "classifier"
    assert s.primary_metric == "accuracy"
    assert s.metric_direction == "higher"
    assert s.needs_labels is True


def test_detect_kind_lm():
    from ark.core.model_kind import detect_kind
    s = detect_kind(["token_embedding", "causal_self_attention", "tiny_llm"])
    assert s.kind == "lm"
    assert s.primary_metric == "perplexity"
    assert s.metric_direction == "lower"


def test_detect_kind_generative():
    from ark.core.model_kind import detect_kind
    assert detect_kind(["unet_mini", "ddim_sampler"]).kind == "generative"
    assert detect_kind(["diffusion"]).has_ground_truth is False


def test_detect_kind_world_model():
    from ark.core.model_kind import detect_kind
    s = detect_kind(["rssm", "rollout", "dynamics_model"])
    assert s.kind == "world_model"
    assert s.primary_metric == "rollout_mse"


def test_detect_kind_default_dense():
    from ark.core.model_kind import detect_kind
    assert detect_kind(["dense", "relu", "linear"]).kind == "classifier"
    assert detect_kind(["mystery_node"]).kind == "unknown"


# ============================================================
# Évaluation classifier
# ============================================================

def _fake_mnist_loader(n=64):
    x = torch.rand(n, 1, 28, 28)
    y = torch.randint(0, 10, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=16)


def test_evaluate_classifier_structure():
    from ark.core.evaluate import evaluate_classifier
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    res = evaluate_classifier(model, _fake_mnist_loader(), train_acc=0.99)
    assert res["status"] == "ok"
    assert 0.0 <= res["accuracy"] <= 1.0
    assert res["n"] == 64
    assert len(res["confusion"]) == 10 and len(res["confusion"][0]) == 10
    # somme confusion == n
    assert sum(sum(r) for r in res["confusion"]) == 64
    assert len(res["per_class_acc"]) == 10
    assert res["gap"] is not None  # train_acc fourni


def test_evaluate_lm_perplexity():
    from ark.core.evaluate import evaluate_lm
    V, T, N = 16, 8, 4
    model = nn.Embedding(V, V)  # (N,T) -> (N,T,V) logits jouet
    x = torch.randint(0, V, (N, T))
    y = torch.randint(0, V, (N, T))
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    res = evaluate_lm(model, loader)
    assert res["status"] == "ok"
    assert res["perplexity"] > 1.0
    assert res["bits_per_byte"] > 0.0


def test_evaluate_compiled_stub_without_loader():
    from ark.core.evaluate import evaluate_compiled
    from ark.core.model_kind import kind_spec
    res = evaluate_compiled(object(), kind_spec("generative"), None)
    assert res["status"] == "needs_reference_set"
    assert res["metric"] is None


def test_passes_threshold():
    from ark.core.evaluate import passes
    from ark.core.model_kind import kind_spec
    cls = kind_spec("classifier")  # threshold 0.85, higher
    assert passes(cls, {"status": "ok", "metric": 0.94}) is True
    assert passes(cls, {"status": "ok", "metric": 0.50}) is False
    lm = kind_spec("lm")  # pas de seuil dur → mesure ok = validé
    assert passes(lm, {"status": "ok", "metric": 12.3}) is True
    assert passes(cls, {"status": "needs_reference_set", "metric": None}) is False


# ============================================================
# Machine à états des poids
# ============================================================

def test_state_machine_trained_validated_operational(tmp_path, monkeypatch):
    import ark.core.weights as W
    monkeypatch.setattr(W, "WEIGHTS_DIR", tmp_path / "w")
    from ark.core.templates import get_template
    from ark.core.headless_scene import HeadlessScene, compile_payload
    t = get_template("mlp_mnist_demo")
    sig = W.signature_from_scene(HeadlessScene(t))
    cg = compile_payload(t, num_classes=10)

    # vierge
    assert W.model_state_from_info(W.weights_info(sig)) == "blank"

    # entraîné
    W.save_weights(cg.model, sig)
    W.mark_trained(sig, train_acc=0.98, val_acc=0.93, kind="classifier")
    info = W.weights_info(sig)
    assert W.model_state_from_info(info) == "trained"
    assert info["validated"] is False
    assert info["operational"] is False

    # validé (held-out passé)
    W.mark_validated(sig, 0.945, metric_name="accuracy", gap=0.035,
                     kind="classifier")
    info = W.weights_info(sig)
    assert W.model_state_from_info(info) == "validated"
    assert info["validated"] is True
    assert abs(info["test_metric"] - 0.945) < 1e-6
    assert info["test_metric_name"] == "accuracy"

    # opérationnel (shippé)
    W.mark_operational(sig, 0.945)
    assert W.model_state_from_info(W.weights_info(sig)) == "operational"


# ============================================================
# Bridge
# ============================================================

def test_bridge_model_kind_and_state():
    import json
    from ark.ui.web_bridge import ArkBridge
    b = ArkBridge()
    k = json.loads(b.model_kind())
    assert "kind" in k
    st = json.loads(b.model_state())
    assert "state" in st
