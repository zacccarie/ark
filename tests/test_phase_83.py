"""Tests ARK Phase 83 — ship/production + run inference + gate."""


def test_can_ship_gate():
    from ark.core.weights import can_ship
    assert can_ship({"exists": True, "validated": True}) is True
    assert can_ship({"exists": True, "validated": False}) is False
    assert can_ship({"exists": False, "validated": True}) is False
    assert can_ship({}) is False


def test_ship_unship_transitions(tmp_path, monkeypatch):
    import ark.core.weights as W
    monkeypatch.setattr(W, "WEIGHTS_DIR", tmp_path / "w")
    from ark.core.templates import get_template
    from ark.core.headless_scene import HeadlessScene, compile_payload
    t = get_template("mlp_mnist_demo")
    sig = W.signature_from_scene(HeadlessScene(t))
    cg = compile_payload(t, num_classes=10)
    W.save_weights(cg.model, sig)

    # entraîné → ne peut pas shipper
    W.mark_trained(sig, train_acc=0.97, val_acc=0.93, kind="classifier")
    assert W.can_ship(W.weights_info(sig)) is False

    # validé → peut shipper
    W.mark_validated(sig, 0.95, metric_name="accuracy", gap=0.02,
                     kind="classifier")
    assert W.can_ship(W.weights_info(sig)) is True
    assert W.model_state_from_info(W.weights_info(sig)) == "validated"

    # ship → operational
    W.mark_operational(sig, 0.95)
    assert W.model_state_from_info(W.weights_info(sig)) == "operational"

    # unship → revient validé (validated reste True)
    W.unmark_operational(sig)
    info = W.weights_info(sig)
    assert info["operational"] is False
    assert info["validated"] is True
    assert W.model_state_from_info(info) == "validated"


def test_bridge_ship_run_no_window():
    import json
    from ark.ui.web_bridge import ArkBridge
    b = ArkBridge()
    assert json.loads(b.ship_model("x"))["status"] == "error"
    assert json.loads(b.unship_model())["status"] == "error"
    assert json.loads(b.run_inference())["status"] == "error"
