"""Workflows templates prêts à charger via menu Fichier."""
from __future__ import annotations
import uuid


TEMPLATES_LIST = [
    ("mlp_mnist", "MLP MNIST (image → 2× dense → 10)"),
    ("cnn_cifar", "CNN CIFAR-10 (conv×2 → dense → 10)"),
    ("resnet_block", "ResNet block (conv+bn+conv+bn+residual)"),
    ("transformer_mini", "Transformer mini (text → 2× transformer block)"),
    ("vg_mutable", "VG mutable + observer + trigger_rule (vagues B+C)"),
    ("diffusion_basic", "Diffusion basic (loader+CLIP+empty_latent+KSampler+VAE+save)"),
    ("metamorph_m1", "Metamorph M1 CriticalGrowNet (target β=-2 SOC)"),
    ("metamorph_m5", "Metamorph M5 DyadicMirror (Mead-Vygotsky)"),
]


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:6]}"


def _node(type_name: str, x: float, y: float, params: dict | None = None) -> dict:
    return {
        "id": _uid(type_name),
        "type": type_name,
        "params": params or {},
        "pos": [x, y],
        "width": 220,
    }


def _conn(a: dict, b: dict, fs: int = 0, ts: int = 0) -> dict:
    return {"from_node": a["id"], "from_socket": fs, "to_node": b["id"], "to_socket": ts}


def _wrap(name: str, nodes: list, connections: list, num_classes: int | None = None,
            tags: list[str] | None = None) -> dict:
    from datetime import datetime, timezone
    return {
        "version": "ark-arch-v1",
        "name": name,
        "nodes": nodes,
        "connections": connections,
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "num_classes": num_classes,
            "tags": tags or [],
        },
    }


def template_mlp_mnist() -> dict:
    inp = _node("image_input", 0, 200, {"shape": "[1, 28, 28]"})
    d1 = _node("dense", 260, 200, {"neurons": 128, "activation": "gelu", "dropout": 0.1})
    d2 = _node("dense", 520, 200, {"neurons": 10, "activation": "none"})
    return _wrap("MLP MNIST", [inp, d1, d2],
                  [_conn(inp, d1), _conn(d1, d2)],
                  num_classes=10, tags=["template", "vision", "mnist"])


def template_cnn_cifar() -> dict:
    inp = _node("image_input", 0, 200, {"shape": "[3, 32, 32]"})
    c1 = _node("conv2d", 260, 200, {"filters": 32, "kernel_size": 3, "activation": "relu"})
    c2 = _node("conv2d", 520, 200, {"filters": 64, "kernel_size": 3, "activation": "relu"})
    d1 = _node("dense", 780, 200, {"neurons": 128, "activation": "gelu", "dropout": 0.2})
    d2 = _node("dense", 1040, 200, {"neurons": 10, "activation": "none"})
    return _wrap("CNN CIFAR-10", [inp, c1, c2, d1, d2],
                  [_conn(inp, c1), _conn(c1, c2), _conn(c2, d1), _conn(d1, d2)],
                  num_classes=10, tags=["template", "vision", "cifar"])


def template_resnet_block() -> dict:
    inp = _node("image_input", 0, 200, {"shape": "[3, 32, 32]"})
    c1 = _node("conv2d", 260, 200, {"filters": 16, "kernel_size": 3, "activation": "relu"})
    bn1 = _node("normalize", 520, 200, {"kind": "batchnorm"})
    c2 = _node("conv2d", 780, 200, {"filters": 16, "kernel_size": 3, "activation": "none"})
    bn2 = _node("normalize", 1040, 200, {"kind": "batchnorm"})
    add = _node("residual_add", 1300, 200)
    d1 = _node("dense", 1560, 200, {"neurons": 10, "activation": "none"})
    conns = [
        _conn(inp, c1), _conn(c1, bn1), _conn(bn1, c2), _conn(c2, bn2),
        _conn(c1, add, ts=0),       # skip from c1
        _conn(bn2, add, ts=1),
        _conn(add, d1),
    ]
    return _wrap("ResNet block", [inp, c1, bn1, c2, bn2, add, d1], conns,
                  num_classes=10, tags=["template", "vision", "resnet"])


def template_transformer_mini() -> dict:
    inp = _node("text_input", 0, 200, {"max_length": 16, "embed_dim": 64})
    t1 = _node("transformer_block", 280, 200, {"n_heads": 4, "d_ff": 128})
    t2 = _node("transformer_block", 560, 200, {"n_heads": 4, "d_ff": 128})
    d1 = _node("dense", 840, 200, {"neurons": 10, "activation": "none"})
    return _wrap("Transformer mini", [inp, t1, t2, d1],
                  [_conn(inp, t1), _conn(t1, t2), _conn(t2, d1)],
                  num_classes=10, tags=["template", "transformer"])


def template_vg_mutable() -> dict:
    inp = _node("image_input", 0, 200, {"shape": "[1, 28, 28]"})
    d0 = _node("dense", 260, 200, {"neurons": 64, "activation": "gelu"})
    vg = _node("vg_block", 520, 200, {"start_hidden": 16, "out_dim": 32, "activation": "gelu"})
    obs = _node("observer", 780, 200,
                  {"metrics": "eff_rank,intrinsic_dim,power_slope,mean_abs",
                   "buffer_size": 128})
    rule = _node("trigger_rule", 1040, 200,
                   {"metric": "eff_rank", "op": "<", "threshold": 100.0,
                    "action_op": "widen", "target_node": vg["id"], "delta": 8})
    d2 = _node("dense", 1300, 200, {"neurons": 10, "activation": "none"})
    conns = [_conn(inp, d0), _conn(d0, vg), _conn(vg, obs), _conn(obs, rule),
              _conn(rule, d2)]
    return _wrap("VG mutable + observer + trigger",
                  [inp, d0, vg, obs, rule, d2], conns,
                  num_classes=10, tags=["template", "vague_b", "vague_c"])


def template_diffusion_basic() -> dict:
    loader = _node("diffusion_model_loader", 0, 200,
                     {"checkpoint": "model.safetensors", "device": "auto"})
    pos = _node("clip_text_encode", 280, 50,
                  {"text": "a serene cyberpunk cityscape, neon, rain"})
    neg = _node("clip_text_encode", 280, 250, {"text": "blurry, ugly, deformed"})
    latent = _node("empty_latent_image", 280, 450,
                     {"width": 512, "height": 512, "batch_size": 1})
    sampler = _node("k_sampler", 560, 200,
                       {"seed": 42, "steps": 20, "cfg": 7.0,
                        "sampler": "euler", "scheduler": "normal", "denoise": 1.0})
    vae_dec = _node("vae_decode", 820, 200)
    save = _node("save_image", 1080, 200, {"prefix": "ark_", "dir": "./output"})
    conns = [
        _conn(loader, pos, fs=1, ts=0),    # clip → CLIP text encode
        _conn(loader, neg, fs=1, ts=0),
        _conn(loader, sampler, fs=0, ts=0),  # model
        _conn(pos, sampler, fs=0, ts=1),     # positive
        _conn(neg, sampler, fs=0, ts=2),     # negative
        _conn(latent, sampler, fs=0, ts=3),  # latent
        _conn(loader, vae_dec, fs=2, ts=0),  # vae
        _conn(sampler, vae_dec, fs=0, ts=1), # latent → vae decode
        _conn(vae_dec, save),
    ]
    return _wrap("Diffusion basic",
                  [loader, pos, neg, latent, sampler, vae_dec, save], conns,
                  tags=["template", "diffusion", "comfyui-like"])


def template_metamorph_m1() -> dict:
    """Metamorph M1 CriticalGrowNet : MLP mutable avec criticality target β=-2."""
    inp = _node("image_input", 0, 200, {"shape": "[1, 28, 28]"})
    d0 = _node("dense", 260, 200, {"neurons": 128, "activation": "gelu"})
    cgn = _node("critical_grow_net", 520, 200,
                  {"hidden_dim": 32, "out_dim": 64, "target_beta": -2.0,
                   "buffer_size": 256, "activation": "gelu"})
    obs = _node("observer", 780, 200,
                  {"metrics": "eff_rank,intrinsic_dim,power_slope,mean_abs",
                   "buffer_size": 128})
    rule = _node("trigger_rule", 1040, 200,
                   {"metric": "power_slope", "op": ">", "threshold": -1.7,
                    "action_op": "widen", "target_node": cgn["id"], "delta": 4})
    out = _node("dense", 1300, 200, {"neurons": 10, "activation": "none"})
    conns = [_conn(inp, d0), _conn(d0, cgn), _conn(cgn, obs),
              _conn(obs, rule), _conn(rule, out)]
    return _wrap("Metamorph M1 CriticalGrowNet",
                  [inp, d0, cgn, obs, rule, out], conns,
                  num_classes=10, tags=["template", "metamorph", "m1", "soc"])


def template_metamorph_m5() -> dict:
    """Metamorph M5 DyadicMirror : 2 SelfRefAgent en miroir."""
    inp = _node("image_input", 0, 200, {"shape": "[1, 28, 28]"})
    d0 = _node("dense", 260, 200, {"neurons": 64, "activation": "gelu"})
    dm = _node("dyadic_mirror", 520, 200,
                 {"hidden_dim": 64, "out_dim": 10,
                  "state_dim": 16, "coupled": True})
    conns = [_conn(inp, d0), _conn(d0, dm)]
    return _wrap("Metamorph M5 DyadicMirror",
                  [inp, d0, dm], conns,
                  num_classes=10, tags=["template", "metamorph", "m5", "mead-vygotsky"])


_BUILDERS = {
    "mlp_mnist": template_mlp_mnist,
    "cnn_cifar": template_cnn_cifar,
    "resnet_block": template_resnet_block,
    "transformer_mini": template_transformer_mini,
    "vg_mutable": template_vg_mutable,
    "diffusion_basic": template_diffusion_basic,
    "metamorph_m1": template_metamorph_m1,
    "metamorph_m5": template_metamorph_m5,
}


def get_template(name: str) -> dict | None:
    builder = _BUILDERS.get(name)
    return builder() if builder else None
