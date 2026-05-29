"""Workflows templates prêts à charger via menu Fichier."""
from __future__ import annotations
import uuid


TEMPLATES_LIST = [
    ("mlp_mnist", "MLP MNIST (image → 2× dense → 10)"),
    ("mlp_mnist_demo",
     "MLP MNIST Demo ★ (perceptron 784→128→64→10 + top-K + probas)"),
    ("cnn_cifar", "CNN CIFAR-10 (conv×2 → dense → 10)"),
    ("resnet_block", "ResNet block (conv+bn+conv+bn+residual)"),
    ("transformer_mini", "Transformer mini (text → 2× transformer block)"),
    ("vg_mutable", "VG mutable + observer + trigger_rule (vagues B+C)"),
    ("diffusion_basic", "Diffusion basic (loader+CLIP+empty_latent+KSampler+VAE+save)"),
    ("text2image_sd", "Text2Image SD (real_backend=True, OFA small SD ~400MB)"),
    ("img2img_sd", "Img2Img SD (load image + VAE encode + prompt + KSampler)"),
    ("tiny_llm_demo", "TinyLLM (corpus → train → generate, 100k params char-level)"),
    ("metamorph_m1", "Metamorph M1 CriticalGrowNet (target β=-2 SOC)"),
    ("metamorph_m5", "Metamorph M5 DyadicMirror (Mead-Vygotsky)"),
    # Phase 56 — multimodal procedural
    ("text2video_tiny", "Text2Video Tiny (prompt → procedural frames → .gif)"),
    ("text2audio_tiny", "Text2Audio Tiny (prompt → melody → ADSR synth → .wav)"),
    ("img2text_resnet", "Img2Text ResNet18 (image → top-K → caption)"),
    ("img2img_style", "Img2Img Style (conv pipeline neuronal léger)"),
    # Phase 59 — REAL neural (slow, downloads models)
    ("text2audio_musicgen",
     "Text2Audio MusicGen ★ NEURAL (~1.5GB download, ~30s gen)"),
    ("text2video_diffusion",
     "Text2Video Diffusion ★ NEURAL (~4GB download, ~2min gen)"),
    # Phase 60 — from-scratch diffusion (briques ARK)
    ("tiny_diffusion_mnist",
     "Tiny Diffusion MNIST ★ FROM-SCRATCH (DDPM + cross-attn UNet + DDIM)"),
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


def template_mlp_mnist_demo() -> dict:
    """Phase 72 : MLP MNIST primitif — nodes les plus élémentaires.

    Perceptron multi-couche complet :
    image_input (1×28×28 = 784) → dense(128, relu) → dense(64, relu)
      → dense(10, none = logits) → top_k_predictions (chiffre prédit)
                                  ↘ prob_distribution (10 classes bar chart)

    Pédagogique : chaque dense = 1 couche fully-connected. Pas de conv,
    pas d'astuce — juste W·x + b + activation. Reconnaît chiffres
    manuscrits ~96% après entraînement.
    """
    inp = _node("image_input", 0, 220, {"shape": "[1, 28, 28]"})
    h1 = _node("dense", 260, 220,
                 {"neurons": 128, "activation": "relu", "dropout": 0.0})
    h2 = _node("dense", 520, 220,
                 {"neurons": 64, "activation": "relu", "dropout": 0.0})
    out = _node("dense", 780, 220,
                  {"neurons": 10, "activation": "none"})
    topk = _node("top_k_predictions", 1040, 140,
                   {"title": "Chiffre prédit", "k": 3,
                    "labels": "0,1,2,3,4,5,6,7,8,9"})
    dist = _node("prob_distribution", 1040, 340,
                   {"output_path": "./output/mnist_probs.html",
                    "title": "Probabilités 0-9", "max_bars": 10})
    conns = [
        _conn(inp, h1), _conn(h1, h2), _conn(h2, out),
        _conn(out, topk), _conn(out, dist),
    ]
    return _wrap("MLP MNIST Demo (perceptron)",
                  [inp, h1, h2, out, topk, dist], conns,
                  num_classes=10,
                  tags=["template", "mlp", "perceptron", "mnist",
                        "from-scratch", "didactique"])


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


def template_text2image_sd() -> dict:
    """Phase 47 : Text2Image SD réel.

    Workflow validé Phase 46 : 88.9s génération 512×512 sur MPS avec
    OFA-Sys/small-stable-diffusion-v0 (~400MB).
    real_backend=True sur loader/clip/sampler/vae.
    """
    loader = _node("diffusion_model_loader", 0, 250,
                     {"checkpoint": "OFA-Sys/small-stable-diffusion-v0",
                      "device": "auto", "real_backend": True})
    pos = _node("clip_text_encode", 320, 80,
                  {"text": "a cute cyberpunk cat sitting on a neon sign at night, "
                              "vivid colors, masterpiece, 4k",
                   "real_backend": True})
    neg = _node("clip_text_encode", 320, 280,
                  {"text": "blurry, low quality, distorted, bad anatomy",
                   "real_backend": True})
    latent = _node("empty_latent_image", 320, 480,
                     {"width": 512, "height": 512, "batch_size": 1})
    sampler = _node("k_sampler", 640, 250,
                       {"seed": 42, "steps": 20, "cfg": 7.5,
                        "sampler": "euler", "scheduler": "normal", "denoise": 1.0,
                        "real_backend": True})
    vae_dec = _node("vae_decode", 960, 250, {"real_backend": True})
    save = _node("save_image", 1280, 250,
                   {"prefix": "ark_t2i_", "dir": "./output"})
    conns = [
        _conn(loader, sampler, fs=0, ts=0),  # model
        _conn(loader, pos, fs=1, ts=0),       # clip → positive
        _conn(loader, neg, fs=1, ts=0),       # clip → negative
        _conn(loader, vae_dec, fs=2, ts=0),   # vae
        _conn(pos, sampler, fs=0, ts=1),      # positive cond
        _conn(neg, sampler, fs=0, ts=2),      # negative cond
        _conn(latent, sampler, fs=0, ts=3),   # initial latent
        _conn(sampler, vae_dec, fs=0, ts=1),  # sampled latent → vae
        _conn(vae_dec, save),                  # image → save
    ]
    return _wrap("Text2Image SD (real)",
                  [loader, pos, neg, latent, sampler, vae_dec, save], conns,
                  tags=["template", "diffusion", "text2image", "stable-diffusion",
                          "real_backend"])


def template_img2img_sd() -> dict:
    """Phase 50 : img2img pipeline. load_image → VAE encode → KSampler avec denoise<1."""
    loader = _node("diffusion_model_loader", 0, 250,
                     {"checkpoint": "OFA-Sys/small-stable-diffusion-v0",
                      "device": "auto", "real_backend": True})
    src_img = _node("load_image", 0, 50,
                     {"path": "input.png"})
    vae_enc = _node("vae_encode", 280, 50, {"real_backend": True})
    pos = _node("clip_text_encode", 320, 200,
                  {"text": "stylized version, vivid colors, illustration",
                   "real_backend": True})
    neg = _node("clip_text_encode", 320, 380,
                  {"text": "blurry, low quality", "real_backend": True})
    sampler = _node("k_sampler", 640, 200,
                       {"seed": 42, "steps": 20, "cfg": 7.0,
                        "sampler": "euler", "scheduler": "normal",
                        "denoise": 0.55,
                        "real_backend": True})
    vae_dec = _node("vae_decode", 960, 200, {"real_backend": True})
    save = _node("save_image", 1280, 200,
                   {"prefix": "ark_i2i_", "dir": "./output"})
    conns = [
        _conn(loader, vae_enc, fs=2, ts=0),  # vae
        _conn(src_img, vae_enc, fs=0, ts=1), # image
        _conn(loader, sampler, fs=0, ts=0),  # model
        _conn(loader, pos, fs=1, ts=0),       # clip → pos
        _conn(loader, neg, fs=1, ts=0),       # clip → neg
        _conn(loader, vae_dec, fs=2, ts=0),   # vae
        _conn(pos, sampler, fs=0, ts=1),
        _conn(neg, sampler, fs=0, ts=2),
        _conn(vae_enc, sampler, fs=0, ts=3),  # latent from encode
        _conn(sampler, vae_dec, fs=0, ts=1),
        _conn(vae_dec, save),
    ]
    return _wrap("Img2Img SD",
                  [loader, src_img, vae_enc, pos, neg, sampler, vae_dec, save], conns,
                  tags=["template", "diffusion", "img2img", "stable-diffusion"])


def template_tiny_llm_demo() -> dict:
    """Phase 54 : TinyLLM corpus → train → generate."""
    corpus = _node("tiny_llm_corpus", 0, 200,
                     {"n_sentences": 2000, "seed": 42})
    model = _node("tiny_llm_model", 300, 200,
                    {"d_model": 64, "n_blocks": 2, "n_heads": 4,
                     "ffn_dim": 256, "max_len": 128,
                     "train_steps": 800, "lr": 3e-3,
                     "batch_size": 32, "block_size": 64,
                     "save_name": "tiny_llm_demo"})
    gen = _node("tiny_llm_generate", 600, 200,
                  {"load_name": "tiny_llm_demo",
                   "prompt": "the cat ",
                   "max_new": 100, "temperature": 0.8, "top_k": 10})
    tv = _node("text_viewer", 900, 200,
                 {"title": "Generated Text", "max_chars": 4000})
    conns = [
        _conn(corpus, model),
        _conn(model, gen),
        _conn(gen, tv),
    ]
    return _wrap("TinyLLM Demo",
                  [corpus, model, gen, tv], conns,
                  tags=["template", "llm", "transformer", "tiny"])


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


def template_text2video_tiny() -> dict:
    """Phase 56 : prompt → animated frame seq → .gif/.mp4.

    Le prompt est édité sur text_input (Inspector). text_to_frame_seq
    le reçoit en string downstream.
    """
    txt = _node("text_input", 0, 200,
                  {"prompt": "ocean sunset waves crashing"})
    tfs = _node("text_to_frame_seq", 280, 200,
                  {"n_frames": 32, "height": 128, "width": 128})
    vw = _node("video_writer", 580, 200,
                 {"output_path": "./output/text2video.gif", "fps": 12})
    iv = _node("image_viewer", 580, 360,
                 {"title": "Last Frame"})
    conns = [
        _conn(txt, tfs),
        _conn(tfs, vw),
        _conn(tfs, iv),
    ]
    return _wrap("Text2Video Tiny",
                  [txt, tfs, vw, iv], conns,
                  tags=["template", "video", "multimodal", "text2video"])


def template_text2audio_tiny() -> dict:
    """Phase 56 : prompt → melody → synth → .wav.

    Édite le prompt sur text_input dans Inspector.
    text_to_melody le reçoit downstream.
    """
    txt = _node("text_input", 0, 200, {"prompt": "hello world music"})
    mel = _node("text_to_melody", 280, 200,
                  {"note_dur": 0.25, "base_octave": 4})
    syn = _node("audio_synth", 580, 200,
                  {"sample_rate": 22050, "envelope": "adsr", "wave": "sine"})
    ap = _node("audio_player", 880, 200,
                 {"title": "Generated Audio", "sample_rate": 22050})
    aw = _node("audio_writer", 1180, 200,
                 {"output_path": "./output/text2audio.wav",
                  "sample_rate": 22050})
    nd = _node("number_display", 880, 380,
                 {"title": "Samples", "fmt": ".0f", "unit": "samples"})
    conns = [
        _conn(txt, mel),
        _conn(mel, syn),
        _conn(syn, ap),
        _conn(ap, aw),
        _conn(syn, nd),
    ]
    return _wrap("Text2Audio Tiny",
                  [txt, mel, syn, ap, aw, nd], conns,
                  tags=["template", "audio", "multimodal", "text2audio"])


def template_img2text_resnet() -> dict:
    """Phase 56 : image file → ResNet18 → top-K → caption."""
    li = _node("load_image_file", 0, 200,
                 {"file_path": "", "size": 224})
    clf = _node("imagenet_classifier", 280, 200,
                  {"backbone": "resnet18"})
    cap = _node("caption_formatter", 580, 200,
                  {"k": 3, "template": "default",
                   "use_imagenet_labels": True})
    tv = _node("text_viewer", 880, 200,
                 {"title": "Caption", "max_chars": 1000})
    tk = _node("top_k_predictions", 580, 380,
                 {"title": "Top-5", "k": 5})
    conns = [
        _conn(li, clf),
        _conn(clf, cap),
        _conn(cap, tv),
        _conn(clf, tk),
    ]
    return _wrap("Img2Text ResNet18",
                  [li, clf, cap, tv, tk], conns,
                  tags=["template", "vision", "multimodal", "img2text",
                        "classification"])


def template_text2audio_musicgen() -> dict:
    """Phase 59 : REAL neural text2audio via MusicGen 300M.

    ~1.5GB download au 1er run. ~30s génération MPS pour 5s d'audio.
    """
    txt = _node("text_input", 0, 200,
                  {"prompt": "uplifting electronic music with synth"})
    mg = _node("musicgen_audio", 280, 200,
                 {"model_id": "facebook/musicgen-small",
                  "duration_s": 5.0, "guidance": 3.0,
                  "sampling_rate": 32000})
    ap = _node("audio_player", 580, 200,
                 {"title": "MusicGen Output", "sample_rate": 32000})
    aw = _node("audio_writer", 880, 200,
                 {"output_path": "./output/musicgen.wav",
                  "sample_rate": 32000})
    conns = [_conn(txt, mg), _conn(mg, ap), _conn(ap, aw)]
    return _wrap("Text2Audio MusicGen (Neural)",
                  [txt, mg, ap, aw], conns,
                  tags=["template", "audio", "neural", "musicgen"])


def template_text2video_diffusion() -> dict:
    """Phase 59 : REAL neural text2video via ModelScope 1.7B.

    ~4GB download au 1er run. ~2min génération MPS pour 16 frames 256x256.
    """
    txt = _node("text_input", 0, 200,
                  {"prompt": "A panda is eating bamboo on a rock"})
    t2v = _node("text2video_diffusion", 280, 200,
                  {"num_frames": 16, "height": 256, "width": 256,
                   "steps": 25, "guidance": 9.0})
    vw = _node("video_writer", 580, 200,
                 {"output_path": "./output/t2v_neural.gif", "fps": 8})
    iv = _node("image_viewer", 580, 380, {"title": "Last Frame"})
    conns = [_conn(txt, t2v), _conn(t2v, vw), _conn(t2v, iv)]
    return _wrap("Text2Video Diffusion (Neural)",
                  [txt, t2v, vw, iv], conns,
                  tags=["template", "video", "neural", "diffusion"])


def template_tiny_diffusion_mnist() -> dict:
    """Phase 60 : class-conditional DDPM from scratch sur MNIST.

    Chain training:
    torchvision_dataset (MNIST) → x_0 → diffusion_scheduler (forward)
                                  ↘ ground truth class → class_embedding
                                                       → unet_mini → ε_pred
    MSE(ε_pred, scheduler.noise) → adam

    Phase d'inférence : ddim_sampler(class_embed) → image generated.

    Démo ARK : pipeline complet diffusion construit depuis briques basiques.
    Train ~5min CPU pour résultats reconnaissables.
    """
    ds = _node("torchvision_dataset", 0, 200,
                 {"dataset": "MNIST", "train": True,
                  "batch_size": 64, "image_size": 28})
    sch = _node("diffusion_scheduler", 280, 200,
                  {"num_timesteps": 200, "beta_start": 1e-4,
                   "beta_end": 0.02, "mode": "forward"})
    ce = _node("class_embedding", 280, 380,
                 {"num_classes": 10, "dim": 128})
    unet = _node("unet_mini", 560, 280,
                   {"in_ch": 1, "base_ch": 32, "cond_dim": 128,
                    "use_attn": True})
    loss = _node("mse_loss", 840, 280, {})
    opt = _node("adam", 1080, 280, {"lr": 2e-4})
    # Inference branch (parallel)
    samp = _node("ddim_sampler", 560, 500,
                   {"n_steps": 50, "channels": 1, "height": 28,
                    "width": 28, "batch_size": 8})
    iv = _node("image_viewer", 840, 500, {"title": "Generated Digits"})
    conns = [
        _conn(ds, sch),
        _conn(sch, unet, fs=0, ts=0),       # x_t → unet
        _conn(ce, unet, fs=0, ts=1),        # cond → unet
        _conn(unet, loss),
        _conn(loss, opt),
        _conn(ce, samp),                     # cond → sampler
        _conn(samp, iv),
    ]
    return _wrap("Tiny Diffusion MNIST (from scratch)",
                  [ds, sch, ce, unet, loss, opt, samp, iv], conns,
                  num_classes=10,
                  tags=["template", "diffusion", "from-scratch", "mnist",
                        "ddpm"])


def template_img2img_style() -> dict:
    """Phase 56 : load_image → conv2d transform pipeline → save_image.

    Léger img2img neuronal (sans SD). Conv apprenable + activation +
    pooling → upsample, preview.
    """
    li = _node("load_image_file", 0, 200,
                 {"file_path": "", "size": 256})
    c1 = _node("conv2d", 280, 200,
                 {"channels": 32, "kernel": 3, "stride": 1,
                  "padding": 1, "activation": "gelu"})
    c2 = _node("conv2d", 480, 200,
                 {"channels": 3, "kernel": 3, "stride": 1,
                  "padding": 1, "activation": "sigmoid"})
    iv = _node("image_viewer", 680, 200, {"title": "Stylized"})
    si = _node("save_image", 680, 380,
                 {"output_path": "./output/img2img_style.png"})
    conns = [
        _conn(li, c1),
        _conn(c1, c2),
        _conn(c2, iv),
        _conn(c2, si),
    ]
    return _wrap("Img2Img Style (Conv)",
                  [li, c1, c2, iv, si], conns,
                  tags=["template", "vision", "img2img", "style", "conv"])


_BUILDERS = {
    "mlp_mnist": template_mlp_mnist,
    "mlp_mnist_demo": template_mlp_mnist_demo,
    "cnn_cifar": template_cnn_cifar,
    "resnet_block": template_resnet_block,
    "transformer_mini": template_transformer_mini,
    "vg_mutable": template_vg_mutable,
    "diffusion_basic": template_diffusion_basic,
    "text2image_sd": template_text2image_sd,
    "img2img_sd": template_img2img_sd,
    "tiny_llm_demo": template_tiny_llm_demo,
    "metamorph_m1": template_metamorph_m1,
    "metamorph_m5": template_metamorph_m5,
    # Phase 56
    "text2video_tiny": template_text2video_tiny,
    "text2audio_tiny": template_text2audio_tiny,
    "img2text_resnet": template_img2text_resnet,
    "img2img_style": template_img2img_style,
    # Phase 59 — REAL neural
    "text2audio_musicgen": template_text2audio_musicgen,
    "text2video_diffusion": template_text2video_diffusion,
    # Phase 60 — from-scratch diffusion
    "tiny_diffusion_mnist": template_tiny_diffusion_mnist,
}


def get_template(name: str) -> dict | None:
    builder = _BUILDERS.get(name)
    return builder() if builder else None
