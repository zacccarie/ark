"""Tests ARK Phase 80 — weight persistence + model state."""
import torch


def test_signature_stable():
    from ark.core.templates import get_template
    from ark.core.headless_scene import HeadlessScene
    from ark.core.weights import signature_from_scene
    t = get_template("mlp_mnist_demo")
    s1 = signature_from_scene(HeadlessScene(t))
    s2 = signature_from_scene(HeadlessScene(t))
    assert s1 == s2
    assert len(s1) == 16


def test_save_load_roundtrip(tmp_path, monkeypatch):
    import ark.core.weights as W
    monkeypatch.setattr(W, "WEIGHTS_DIR", tmp_path / "w")
    from ark.core.templates import get_template
    from ark.core.headless_scene import HeadlessScene, compile_payload
    t = get_template("mlp_mnist_demo")
    sig = W.signature_from_scene(HeadlessScene(t))
    cg = compile_payload(t, num_classes=10)
    assert W.weights_info(sig).get("exists") is False
    W.save_weights(cg.model, sig, meta={"val_acc": 0.9})
    info = W.weights_info(sig)
    assert info["exists"] and info["n_params"] > 0
    cg2 = compile_payload(t, num_classes=10)
    assert W.load_weights(cg2.model, sig)
    k = list(cg.model.state_dict().keys())[0]
    assert torch.equal(cg.model.state_dict()[k], cg2.model.state_dict()[k])


def test_mark_operational(tmp_path, monkeypatch):
    import ark.core.weights as W
    monkeypatch.setattr(W, "WEIGHTS_DIR", tmp_path / "w")
    from ark.core.templates import get_template
    from ark.core.headless_scene import HeadlessScene, compile_payload
    t = get_template("mlp_mnist_demo")
    sig = W.signature_from_scene(HeadlessScene(t))
    cg = compile_payload(t, num_classes=10)
    W.save_weights(cg.model, sig)
    W.mark_operational(sig, 0.945)
    info = W.weights_info(sig)
    assert info["operational"] is True
    assert abs(info["val_acc"] - 0.945) < 1e-6


def test_compile_exposes_signature():
    from ark.core.templates import get_template
    from ark.core.headless_scene import compile_payload
    t = get_template("mlp_mnist_demo")
    cg = compile_payload(t, num_classes=10)
    assert getattr(cg, "signature", None) is not None
    assert hasattr(cg, "loaded_trained")


def test_bridge_model_state_json():
    import json
    from ark.ui.web_bridge import ArkBridge
    b = ArkBridge()
    st = json.loads(b.model_state())
    assert "state" in st  # empty (no mw)
