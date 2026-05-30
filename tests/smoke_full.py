"""Smoke harness — compile+forward chaque template, exerce bridge + nodes.
Run: ./.venv/bin/python tests/smoke_full.py
Collecte les crashes, ne s'arrête pas au 1er."""
import sys
import traceback

import torch

PASS, FAIL, KNOWN = [], [], []

# Templates "showcase" multimodaux/diffusion : forward exige des poids réels
# ou un conditionnement (labels/prompt) → échec attendu avec un dummy.
# Non bloquant en app (forward try-wrapped). Tracés à part des vrais bugs.
KNOWN_INCOMPLETE = {
    "text2image_sd", "img2img_sd", "img2img_style", "text2audio_musicgen",
    "text2video_diffusion", "text2video_tiny", "text2audio_tiny",
    "img2text_resnet", "tiny_llm_demo", "data_convert_demo",
    "tiny_diffusion_mnist",
}


def rec(ok, label, err="", known=False):
    if not ok and known:
        KNOWN.append((label, err))
        print(f".. {label}  (known-incomplete)")
        return
    (PASS if ok else FAIL).append((label, err))
    mark = "ok " if ok else "XX "
    print(f"{mark}{label}" + (f"  → {err.splitlines()[-1][:120]}" if err else ""))


# ============================================================
# A. Templates : compile + forward
# ============================================================
def smoke_templates():
    from ark.core.templates import TEMPLATES_LIST, get_template
    from ark.core.headless_scene import compile_payload
    for name, _ in TEMPLATES_LIST:
        try:
            t = get_template(name)
        except Exception:
            rec(False, f"tpl/get/{name}", traceback.format_exc())
            continue
        try:
            cg = compile_payload(t, num_classes=10)
        except Exception:
            rec(False, f"tpl/compile/{name}", traceback.format_exc())
            continue
        # forward avec dummy dérivé de input_shape
        ish = getattr(cg, "input_shape", None)
        try:
            if isinstance(ish, tuple) and ish and all(
                    isinstance(d, int) and d > 0 for d in ish):
                dummy = torch.zeros(1, *ish)
            else:
                dummy = torch.zeros(1, 4, 64, 64)
            out = cg.model(dummy)
            rec(True, f"tpl/fwd/{name}  in={ish}")
        except Exception as e:
            # 2e essai en long (LLM token ids)
            try:
                dummy = torch.zeros(1, 16, dtype=torch.long)
                cg.model(dummy)
                rec(True, f"tpl/fwd/{name}  (long ids)")
            except Exception:
                rec(False, f"tpl/fwd/{name}", traceback.format_exc(),
                    known=(name in KNOWN_INCOMPLETE))


# ============================================================
# B. Bridge : appelle chaque Slot (mw=None → exerce les gardes)
# ============================================================
def smoke_bridge():
    from ark.ui.web_bridge import ArkBridge
    b = ArkBridge()
    # signatures simples : args bidons par type
    trivial = {
        "list_nodes_in_category": ("Core",),
        "get_node_schema": ("dense",),
        "set_node_param": ("n1", "neurons", "64"),
        "spawn_node": ("dense",),
        "compile": (1,), "train": (1,),
        "load_template": ("mlp_mnist_demo",),
        "set_zoom": (1.0,), "toggle_minimap": (True,),
        "toggle_snap": (True,), "play_audio": ("",),
        "download_audio": ("",), "architect_send": ("hi",),
        "architect_set_provider": ("openai", "k", "m"),
        "modal_connect": ("tok",),
    }
    import inspect
    for nm in dir(b):
        if nm.startswith("_"):
            continue
        fn = getattr(b, nm)
        if not callable(fn):
            continue
        # uniquement les Slots (ont __name__ + sont méthodes)
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            continue
        nreq = sum(1 for p in sig.parameters.values()
                   if p.default is inspect._empty
                   and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD))
        try:
            if nm in trivial:
                fn(*trivial[nm])
            elif nreq == 0:
                fn()
            else:
                continue  # skip ceux qu'on sait pas appeler
            rec(True, f"bridge/{nm}")
        except Exception:
            rec(False, f"bridge/{nm}", traceback.format_exc())


# ============================================================
# C. Imports critiques
# ============================================================
def smoke_imports():
    mods = [
        "ark.core.graph_compiler", "ark.core.headless_scene",
        "ark.core.weights", "ark.core.model_kind", "ark.core.evaluate",
        "ark.core.trainer", "ark.core.templates", "ark.core.data_nodes",
        "ark.core.dataset_nodes", "ark.core.mnist_demo_nodes",
        "ark.core.media_loaders", "ark.ui.web_bridge",
    ]
    for m in mods:
        try:
            __import__(m)
            rec(True, f"import/{m}")
        except Exception:
            rec(False, f"import/{m}", traceback.format_exc())


if __name__ == "__main__":
    print("=== IMPORTS ===");   smoke_imports()
    print("=== TEMPLATES ==="); smoke_templates()
    print("=== BRIDGE ===");    smoke_bridge()
    print(f"\n=== RESULT : {len(PASS)} ok · {len(FAIL)} FAIL · "
          f"{len(KNOWN)} known-incomplete ===")
    for label, _ in FAIL:
        print("  FAIL", label)
    sys.exit(1 if FAIL else 0)
