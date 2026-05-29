"""Registry de types de nodes.

Chaque type possède :
- name (id), title (display), category (Input / Math / Neural / Code / Viz / ...)
- inputs : list[(label, socket_type)]
- outputs : list[(label, socket_type)]
- params : list[(name, kind, default, options?)] avec kind ∈
  {'int', 'float', 'str', 'bool', 'choice', 'multiline'}.

Socket types pilotent la couleur des sockets + validation connexions.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParamSpec:
    name: str
    kind: str  # 'int' | 'float' | 'str' | 'bool' | 'choice' | 'multiline' | 'filepath' | 'dirpath'
    default: Any = None
    options: list = field(default_factory=list)  # pour 'choice' OU pour 'filepath' = liste de filtres Qt
    min: float | None = None
    max: float | None = None
    file_filter: str = ""  # ex. "Images (*.png *.jpg);;All (*)"


@dataclass
class SocketSpec:
    label: str
    socket_type: str  # 'tensor' | 'number' | 'string' | 'image' | 'model' | 'exec' | 'any'


@dataclass
class NodeSpec:
    name: str
    title: str
    category: str
    inputs: list[SocketSpec] = field(default_factory=list)
    outputs: list[SocketSpec] = field(default_factory=list)
    params: list[ParamSpec] = field(default_factory=list)
    description: str = ""
    has_thumbnail: bool = False
    is_multi_input: bool = False  # si True, chaque connexion ajoute un connecteur
    param_help: dict = field(default_factory=dict)  # name → tooltip explication
    short: str = ""  # 1-line summary (above description)
    equation: str = ""  # formula (LaTeX-ish string)
    related: tuple = ()  # noms de concepts liés
    role: str = ""  # ce que fait le node (visible Inspector)
    function: str = ""  # comment il fonctionne en interne (visible Inspector)


class NodeRegistry:
    def __init__(self):
        self._specs: dict[str, NodeSpec] = {}

    def register(self, spec: NodeSpec) -> None:
        self._specs[spec.name] = spec

    def get(self, name: str) -> NodeSpec | None:
        return self._specs.get(name)

    def all(self) -> list[NodeSpec]:
        return list(self._specs.values())

    def by_category(self) -> dict[str, list[NodeSpec]]:
        out: dict[str, list[NodeSpec]] = {}
        for s in self._specs.values():
            out.setdefault(s.category, []).append(s)
        return {k: sorted(v, key=lambda s: s.title) for k, v in sorted(out.items())}


_DEFAULT_REGISTRY_CACHE: NodeRegistry | None = None


def reset_registry_cache() -> None:
    """Phase 40 : reset cache (utile pour tests + après reload groups/plugins)."""
    global _DEFAULT_REGISTRY_CACHE
    _DEFAULT_REGISTRY_CACHE = None


def default_registry(force_rebuild: bool = False) -> NodeRegistry:
    """Registry pré-rempli avec les nodes de base ARK.

    Phase 40 : cache module-level (singleton). Premier appel construit
    (175ms). Suivants retournent cache (~0ms).
    Force_rebuild=True pour rescan plugins/groups (live reload).
    """
    global _DEFAULT_REGISTRY_CACHE
    if _DEFAULT_REGISTRY_CACHE is not None and not force_rebuild:
        return _DEFAULT_REGISTRY_CACHE
    r = NodeRegistry()

    # ---------- Input ----------
    r.register(NodeSpec(
        name="image_input", title="Image Input", category="Input",
        outputs=[SocketSpec("out", "tensor")],
        params=[ParamSpec("shape", "str", "[1, 28, 28]")],
        description="Tenseur image (C, H, W). Source du graphe.",
        param_help={"shape": "Forme du tenseur sans le batch — JSON list [C, H, W]. Ex. [3, 224, 224] pour RGB 224×224."},
    ))
    r.register(NodeSpec(
        name="tabular_input", title="Tabular Input", category="Input",
        outputs=[SocketSpec("out", "tensor")],
        params=[ParamSpec("features", "int", 32, min=1, max=10000)],
        description="Vecteur de features (N,). Pour données tabulaires (CSV, embeddings).",
        param_help={"features": "Dimension N du vecteur d'entrée."},
    ))
    r.register(NodeSpec(
        name="text_input", title="Text Input", category="Input",
        outputs=[SocketSpec("out", "tensor")],
        params=[
            ParamSpec("prompt", "multiline", ""),
            ParamSpec("max_length", "int", 64),
            ParamSpec("embed_dim", "int", 128),
        ],
        description="Saisir prompt texte. Si prompt non vide, renvoie string "
                       "downstream (text_to_melody / text_to_frame_seq / "
                       "tiny_llm_generate). Sinon pure shape hint (T, D).",
        param_help={
            "prompt": "Texte saisi par l'utilisateur. Source des nodes "
                          "multimodales (audio/video/llm).",
            "max_length": "Longueur max séquence T (tokens) — compat embeddings.",
            "embed_dim": "Dimension D embedding — compat embeddings.",
        },
    ))

    # ---------- Neural ----------
    r.register(NodeSpec(
        name="dense", title="Dense", category="Neural",
        inputs=[SocketSpec("in", "tensor")],
        outputs=[SocketSpec("out", "tensor")],
        params=[
            ParamSpec("neurons", "int", 128, min=1, max=100000),
            ParamSpec("activation", "choice", "relu",
                       options=["none", "relu", "gelu", "tanh", "sigmoid", "softmax"]),
            ParamSpec("dropout", "float", 0.0, min=0.0, max=1.0),
        ],
        description="Linear(in, out) + activation + dropout optionnel. Couche fully-connected classique.",
        param_help={
            "neurons": "Nombre d'unités en sortie (dimension cachée).",
            "activation": "Fonction non-linéaire post-Linear. relu = max(0,x), gelu = lisse, softmax = proba.",
            "dropout": "Probabilité de mise à zéro d'une unité durant le training. Régularisation. 0 = pas de dropout.",
        },
    ))
    r.register(NodeSpec(
        name="conv2d", title="Conv2D", category="Neural",
        inputs=[SocketSpec("in", "tensor")],
        outputs=[SocketSpec("out", "tensor")],
        params=[
            ParamSpec("filters", "int", 32, min=1),
            ParamSpec("kernel_size", "int", 3, min=1, max=15),
            ParamSpec("activation", "choice", "relu",
                       options=["none", "relu", "gelu", "tanh"]),
        ],
        description="Convolution 2D — extraction features spatiales. Cœur des CNN.",
        param_help={
            "filters": "Nombre de canaux de sortie (feature maps).",
            "kernel_size": "Taille du noyau carré (3 = 3×3). Impair conseillé.",
            "activation": "Non-linéarité post-convolution.",
        },
    ))
    r.register(NodeSpec(
        name="normalize", title="Normalize", category="Neural",
        inputs=[SocketSpec("in", "tensor")],
        outputs=[SocketSpec("out", "tensor")],
        params=[ParamSpec("kind", "choice", "batchnorm",
                            options=["batchnorm", "layernorm", "groupnorm"])],
        description="Normalisation des activations. Stabilise et accélère le training.",
        param_help={"kind": "batchnorm = sur batch (CNN), layernorm = par échantillon (Transformer), groupnorm = par groupes de canaux."},
    ))
    r.register(NodeSpec(
        name="attention", title="Multi-Head Attention", category="Neural",
        inputs=[SocketSpec("in", "tensor")],
        outputs=[SocketSpec("out", "tensor")],
        params=[ParamSpec("n_heads", "int", 4, min=1, max=32)],
        description="Self-attention multi-têtes. Q·Kᵀ/√d → softmax → ·V. Cœur des Transformers.",
        param_help={"n_heads": "Nombre de têtes parallèles. Doit diviser d_model."},
    ))
    r.register(NodeSpec(
        name="transformer_block", title="Transformer Block", category="Neural",
        inputs=[SocketSpec("in", "tensor")],
        outputs=[SocketSpec("out", "tensor")],
        params=[ParamSpec("n_heads", "int", 4),
                 ParamSpec("d_ff", "int", 256)],
        description="Bloc Transformer : MHA + FFN + résiduels + LayerNorm. Architecture standard GPT/BERT.",
        param_help={
            "n_heads": "Nombre de têtes de l'attention.",
            "d_ff": "Dimension cachée du feed-forward interne (typiquement 4× d_model).",
        },
    ))
    r.register(NodeSpec(
        name="residual_add", title="Residual Add", category="Neural",
        inputs=[SocketSpec("a", "tensor"), SocketSpec("b", "tensor")],
        outputs=[SocketSpec("out", "tensor")],
        is_multi_input=True,
        description="Addition résiduelle out = a + b (ResNet, Transformer). Évite le vanishing gradient.",
    ))
    r.register(NodeSpec(
        name="moe_switch", title="MoE Switch", category="Neural",
        inputs=[SocketSpec("in", "tensor")],
        outputs=[SocketSpec("out", "tensor")],
        params=[ParamSpec("n_experts", "int", 4, min=2, max=64),
                 ParamSpec("d_out", "int", 128)],
        description="Mixture of Experts — Switch Transformer. Router top-1 → un expert activé par token. Sparse, scaling efficient.",
        param_help={
            "n_experts": "Nombre d'experts FFN en parallèle.",
            "d_out": "Dimension de sortie de chaque expert.",
        },
    ))
    r.register(NodeSpec(
        name="vg_block", title="VG Block (mutable)", category="Neural",
        inputs=[SocketSpec("in", "tensor")],
        outputs=[SocketSpec("out", "tensor")],
        params=[ParamSpec("start_hidden", "int", 32),
                 ParamSpec("out_dim", "int", 64),
                 ParamSpec("activation", "choice", "gelu",
                             options=["relu", "gelu", "tanh"])],
        description="Bloc MLP mutable runtime (widen/prune). Compatible Net2Net function-preserving. Cœur des architectures auto-modifiantes.",
        param_help={
            "start_hidden": "Largeur initiale cachée (croît via widen runtime).",
            "out_dim": "Dimension de sortie fixe.",
            "activation": "Non-linéarité interne.",
        },
    ))

    # ---------- Math ----------
    r.register(NodeSpec(
        name="math_op", title="Math Op", category="Math",
        inputs=[SocketSpec("a", "number"), SocketSpec("b", "number")],
        outputs=[SocketSpec("out", "number")],
        params=[ParamSpec("op", "choice", "add",
                            options=["add", "sub", "mul", "div", "pow", "min", "max", "mod"])],
        description="Opération arithmétique élémentaire. Ordre des entrées sensible (a op b).",
        param_help={"op": "add: a+b, sub: a-b, mul: a*b, div: a/b, pow: a^b, mod: a%b."},
    ))
    r.register(NodeSpec(
        name="constant", title="Constant", category="Math",
        outputs=[SocketSpec("out", "number")],
        params=[ParamSpec("value", "float", 1.0)],
        description="Valeur scalaire constante. Source du graphe.",
        param_help={"value": "Valeur numérique constante émise."},
    ))

    # ---------- Code ----------
    r.register(NodeSpec(
        name="code_line", title="Code Line", category="Code",
        inputs=[SocketSpec("in", "any")],
        outputs=[SocketSpec("out", "any")],
        params=[ParamSpec("code", "multiline",
                            "# Python expression — `x` est l'input\nresult = x")],
        description="Exécute une ligne (ou bloc) Python sur l'input. Variable `x` = entrée, `result` = sortie.",
        param_help={"code": "Bloc Python. `x` est l'input. Assignez `result` à la valeur de sortie."},
    ))

    # ---------- Observe / Act ----------
    r.register(NodeSpec(
        name="observer", title="Observer", category="Observe",
        inputs=[SocketSpec("in", "tensor")],
        outputs=[SocketSpec("out", "tensor")],
        params=[ParamSpec("metrics", "str",
                            "eff_rank,intrinsic_dim,power_slope,mean_abs"),
                 ParamSpec("buffer_size", "int", 256)],
        has_thumbnail=True,
        description="Mesure non-intrusive sur les activations (passthrough). Émet metrics → trigger.",
        param_help={
            "metrics": "Liste CSV de métriques : eff_rank (entropie spectrale), intrinsic_dim (Levina-Bickel), power_slope (FFT log-log), mean_abs (magnitude moyenne).",
            "buffer_size": "Taille du buffer circulaire (nombre d'observations conservées).",
        },
    ))
    r.register(NodeSpec(
        name="trigger_rule", title="Trigger Rule", category="Observe",
        inputs=[SocketSpec("in", "tensor")],
        outputs=[SocketSpec("out", "tensor")],
        params=[ParamSpec("metric", "str", "eff_rank"),
                 ParamSpec("op", "choice", ">",
                             options=["<", ">", "<=", ">=", "==", "!="]),
                 ParamSpec("threshold", "float", 0.0),
                 ParamSpec("action_op", "choice", "widen",
                             options=["widen", "prune"]),
                 ParamSpec("target_node", "str", ""),
                 ParamSpec("delta", "int", 8)],
        description="Règle conditionnelle : si metric op threshold → mutate target_node. Cœur de l'architecture auto-modifiante.",
        param_help={
            "metric": "Nom de la métrique observer à surveiller.",
            "op": "Opérateur de comparaison.",
            "threshold": "Seuil déclencheur.",
            "action_op": "widen = élargir (Net2Net), prune = élaguer.",
            "target_node": "ID du node cible (typiquement VG Block).",
            "delta": "Amplitude de la mutation (nombre d'unités à ajouter/retirer).",
        },
    ))

    # ---------- Viz ----------
    r.register(NodeSpec(
        name="plot_scatter", title="Plot Scatter", category="Viz",
        inputs=[SocketSpec("x", "number"), SocketSpec("y", "number"),
                 SocketSpec("color", "any")],
        outputs=[],
        has_thumbnail=True,
        description="Nuage de points 2D. x/y obligatoires, color optionnel.",
    ))
    r.register(NodeSpec(
        name="plot_line", title="Plot Line", category="Viz",
        inputs=[SocketSpec("series", "tensor")],
        outputs=[],
        has_thumbnail=True,
        description="Courbe 1D temporelle. Affiche le tenseur d'entrée comme série.",
    ))

    # ---------- Optim/Loss ----------
    r.register(NodeSpec(
        name="adam", title="Adam Optimizer", category="Training",
        outputs=[SocketSpec("optim", "any")],
        params=[ParamSpec("learning_rate", "float", 1e-3, min=1e-6, max=1.0),
                 ParamSpec("beta1", "float", 0.9),
                 ParamSpec("beta2", "float", 0.999)],
        description="Optimiseur Adam (Adaptive Moment Estimation). Combine momentum + RMSprop. Default robuste.",
        param_help={
            "learning_rate": "Taux d'apprentissage (lr). Typique 1e-3 pour Adam.",
            "beta1": "Décroissance du moment 1 (gradient). Typique 0.9.",
            "beta2": "Décroissance du moment 2 (gradient²). Typique 0.999.",
        },
    ))
    r.register(NodeSpec(
        name="cross_entropy", title="Cross-Entropy Loss", category="Training",
        inputs=[SocketSpec("logits", "tensor"), SocketSpec("targets", "tensor")],
        outputs=[SocketSpec("loss", "number")],
        description="Loss classification multi-classes. -log(softmax(logits)[target]). Stable numérique.",
    ))

    # ---------- Diffusion (ComfyUI-like, Phase 4 — specs + stubs) ----------
    r.register(NodeSpec(
        name="diffusion_model_loader", title="Diffusion Model Loader",
        category="Diffusion",
        outputs=[SocketSpec("model", "model"), SocketSpec("clip", "model"),
                  SocketSpec("vae", "model")],
        params=[ParamSpec("checkpoint", "str", "runwayml/stable-diffusion-v1-5"),
                 ParamSpec("device", "choice", "auto",
                             options=["auto", "cuda", "mps", "cpu"]),
                 ParamSpec("real_backend", "bool", False)],
        description="Charge un checkpoint diffusion. Stub par défaut. real_backend=True → diffusers HF réel (download ~4GB SD1.5).",
        param_help={
            "checkpoint": "Repo HF (ex. runwayml/stable-diffusion-v1-5) ou chemin .safetensors.",
            "device": "auto = cuda > mps > cpu.",
            "real_backend": "False = stub déterministe rapide. True = diffusers réel.",
        },
    ))
    r.register(NodeSpec(
        name="clip_text_encode", title="CLIP Text Encode",
        category="Diffusion",
        inputs=[SocketSpec("clip", "model")],
        outputs=[SocketSpec("conditioning", "tensor")],
        params=[ParamSpec("text", "multiline", "a serene cyberpunk cityscape, neon, rain"),
                 ParamSpec("real_backend", "bool", False)],
        description="Encode prompt texte via CLIP → conditioning tensor.",
    ))
    r.register(NodeSpec(
        name="empty_latent_image", title="Empty Latent Image",
        category="Diffusion",
        outputs=[SocketSpec("latent", "tensor")],
        params=[ParamSpec("width", "int", 512, min=64, max=4096),
                 ParamSpec("height", "int", 512, min=64, max=4096),
                 ParamSpec("batch_size", "int", 1, min=1, max=64)],
        description="Tenseur latent vide (bruit) — point de départ du sampling diffusion. Résolution latente = 1/8 image.",
        param_help={
            "width": "Largeur image en pixels (latent = w/8).",
            "height": "Hauteur image en pixels (latent = h/8).",
            "batch_size": "Nombre d'images générées en parallèle.",
        },
    ))
    r.register(NodeSpec(
        name="k_sampler", title="KSampler",
        category="Diffusion",
        inputs=[SocketSpec("model", "model"),
                 SocketSpec("positive", "tensor"),
                 SocketSpec("negative", "tensor"),
                 SocketSpec("latent", "tensor")],
        outputs=[SocketSpec("latent", "tensor")],
        params=[ParamSpec("seed", "int", 42),
                 ParamSpec("steps", "int", 20, min=1, max=200),
                 ParamSpec("cfg", "float", 7.0, min=0.0, max=30.0),
                 ParamSpec("sampler", "choice", "euler",
                             options=["euler", "euler_ancestral", "heun", "dpm_2", "dpmpp_2m",
                                       "dpmpp_2m_sde", "ddim", "uni_pc"]),
                 ParamSpec("scheduler", "choice", "normal",
                             options=["normal", "karras", "exponential", "simple", "ddim_uniform"]),
                 ParamSpec("denoise", "float", 1.0, min=0.0, max=1.0),
                 ParamSpec("real_backend", "bool", False)],
        description="Sampling diffusion classique. Inspiré ComfyUI KSampler. Itère sur le latent en débruitant pas à pas.",
        param_help={
            "seed": "Graine PRNG. Même seed = même image.",
            "steps": "Nombre d'itérations de débruitage. Plus = mieux mais plus lent.",
            "cfg": "Classifier-Free Guidance — force du conditioning. 7-12 typique.",
            "sampler": "Algorithme de sampling (euler, dpm++, ddim...).",
            "scheduler": "Stratégie de bruit/timesteps (karras = qualité, normal = standard).",
            "denoise": "Quantité de bruit à enlever (1.0 = full denoise, <1 = img2img).",
        },
    ))
    r.register(NodeSpec(
        name="vae_encode", title="VAE Encode",
        category="Diffusion",
        inputs=[SocketSpec("vae", "model"), SocketSpec("image", "image")],
        outputs=[SocketSpec("latent", "tensor")],
        params=[ParamSpec("real_backend", "bool", False)],
        description="Image → latent (1/8 résolution). Pour img2img/inpainting.",
    ))
    r.register(NodeSpec(
        name="vae_decode", title="VAE Decode",
        category="Diffusion",
        inputs=[SocketSpec("vae", "model"), SocketSpec("latent", "tensor")],
        outputs=[SocketSpec("image", "image")],
        params=[ParamSpec("real_backend", "bool", False)],
        description="Latent (post-KSampler) → image RGB pixel-space (×8 upsample).",
    ))
    r.register(NodeSpec(
        name="load_image", title="Load Image",
        category="Diffusion",
        outputs=[SocketSpec("image", "image"), SocketSpec("mask", "image")],
        params=[ParamSpec("path", "filepath", "input.png",
                              file_filter="Images (*.png *.jpg *.jpeg *.webp);;All (*)")],
        has_thumbnail=True,
        description="Charge image depuis disque. Mask optionnel pour inpainting.",
        param_help={"path": "Fichier image. Cliquer … pour parcourir."},
    ))
    r.register(NodeSpec(
        name="save_image", title="Save Image",
        category="Diffusion",
        inputs=[SocketSpec("image", "image")],
        params=[ParamSpec("prefix", "str", "ark_"),
                 ParamSpec("dir", "str", "./output")],
        has_thumbnail=True,
        description="Sauvegarde image (PNG). Nom = prefix + timestamp.",
        param_help={
            "prefix": "Préfixe du nom de fichier.",
            "dir": "Dossier de destination (créé si absent).",
        },
    ))
    r.register(NodeSpec(
        name="lora_loader", title="LoRA Loader",
        category="Diffusion",
        inputs=[SocketSpec("model", "model"), SocketSpec("clip", "model")],
        outputs=[SocketSpec("model", "model"), SocketSpec("clip", "model")],
        params=[ParamSpec("lora_name", "str", "lora.safetensors"),
                 ParamSpec("strength_model", "float", 1.0, min=-2.0, max=2.0),
                 ParamSpec("strength_clip", "float", 1.0, min=-2.0, max=2.0)],
        description="Applique LoRA (Low-Rank Adaptation) → patch léger de model+clip. Style/concept finetuning.",
        param_help={
            "lora_name": "Fichier .safetensors LoRA.",
            "strength_model": "Force d'application sur le model (UNet).",
            "strength_clip": "Force d'application sur le CLIP (text encoder).",
        },
    ))
    r.register(NodeSpec(
        name="controlnet_apply", title="ControlNet Apply",
        category="Diffusion",
        inputs=[SocketSpec("conditioning", "tensor"),
                 SocketSpec("control_net", "model"),
                 SocketSpec("image", "image")],
        outputs=[SocketSpec("conditioning", "tensor")],
        params=[ParamSpec("strength", "float", 1.0, min=0.0, max=2.0)],
        description="ControlNet — conditionne la diffusion via une image guide (canny, depth, pose...).",
        param_help={"strength": "Force du conditioning ControlNet (0 = ignoré, 1 = standard)."},
    ))

    # ---------- HuggingFace ----------
    r.register(NodeSpec(
        name="hf_model_loader", title="HuggingFace Loader",
        category="HuggingFace",
        outputs=[SocketSpec("model", "model")],
        params=[ParamSpec("repo_id", "str", "EleutherAI/pythia-410m"),
                 ParamSpec("revision", "str", "main"),
                 ParamSpec("dtype", "choice", "float32",
                             options=["float32", "float16", "bfloat16"])],
        description="Télécharge un modèle HF Hub (transformers). Cache ~/.cache/huggingface/.",
        param_help={
            "repo_id": "Format owner/model (ex. EleutherAI/pythia-410m).",
            "revision": "Branche/tag/commit. 'main' par défaut.",
            "dtype": "Précision : float32 (default), float16 (GPU rapide), bfloat16 (TPU/A100).",
        },
    ))
    r.register(NodeSpec(
        name="hf_tokenizer", title="HuggingFace Tokenizer",
        category="HuggingFace",
        outputs=[SocketSpec("tokenizer", "any")],
        params=[ParamSpec("repo_id", "str", "EleutherAI/pythia-410m")],
        description="Charge tokenizer associé au modèle HF (BPE, WordPiece, SentencePiece...).",
        param_help={"repo_id": "Même repo_id que le modèle pour cohérence vocab."},
    ))
    r.register(NodeSpec(
        name="hf_pipeline", title="HF Pipeline",
        category="HuggingFace",
        inputs=[SocketSpec("model", "model"), SocketSpec("input", "any")],
        outputs=[SocketSpec("output", "any")],
        params=[ParamSpec("task", "choice", "text-generation",
                            options=["text-generation", "fill-mask", "translation",
                                      "summarization", "image-classification",
                                      "object-detection", "feature-extraction"])],
        description="Pipeline haut-niveau HuggingFace : preprocess → model → postprocess.",
        param_help={"task": "Type de tâche. Auto-configure preprocessing/postprocessing."},
    ))

    # ---------- Periph (webcam, micro) ----------
    r.register(NodeSpec(
        name="webcam", title="Webcam",
        category="Devices",
        outputs=[SocketSpec("frame", "image")],
        params=[ParamSpec("device_id", "int", 0, min=0, max=8),
                 ParamSpec("fps", "int", 30, min=1, max=60)],
        has_thumbnail=True,
        description="Capture flux webcam live → tenseur image (H, W, 3).",
        param_help={
            "device_id": "Index de la caméra (0 = défaut).",
            "fps": "Frames par seconde cibles.",
        },
    ))
    r.register(NodeSpec(
        name="microphone", title="Microphone",
        category="Devices",
        outputs=[SocketSpec("audio", "tensor")],
        params=[ParamSpec("sample_rate", "int", 16000),
                 ParamSpec("channels", "int", 1, min=1, max=8)],
        has_thumbnail=True,
        description="Capture audio live → tenseur (T,) ou (C, T).",
        param_help={
            "sample_rate": "Échantillons par seconde (16000 = parole, 44100 = musique).",
            "channels": "1 = mono, 2 = stéréo.",
        },
    ))

    # Metamorph (Phase 17) — registered si dispo
    try:
        from ark.core.metamorph_bridge import register_metamorph_specs
        register_metamorph_specs(r)
    except Exception:
        pass

    # Research (Phase 27) — ai-scientist + autoresearch + sci-viz
    try:
        from ark.core.research_bridge import register_research_specs
        register_research_specs(r)
    except Exception:
        pass

    # Glossary massive (Phase 29) — 400+ concepts, 150+ nodes auto-enregistrés
    try:
        from ark.core.auto_register import register_glossary_specs
        register_glossary_specs(r)
    except Exception as e:
        import sys
        print(f"[ARK] auto_register specs failed: {e}", file=sys.stderr)

    # Media loaders (Phase 32) — image/video/sound/multi-image
    try:
        from ark.core.media_loaders import register_media_loader_specs
        register_media_loader_specs(r)
    except Exception as e:
        import sys
        print(f"[ARK] media loaders specs failed: {e}", file=sys.stderr)

    # 3D plots Plotly (Phase 34)
    try:
        from ark.core.plot3d_nodes import register_plot3d_specs
        register_plot3d_specs(r)
    except Exception as e:
        import sys
        print(f"[ARK] plot3d specs failed: {e}", file=sys.stderr)

    # BertViz attention viz (Phase 34)
    try:
        from ark.core.bertviz_nodes import register_bertviz_specs
        register_bertviz_specs(r)
    except Exception as e:
        import sys
        print(f"[ARK] bertviz specs failed: {e}", file=sys.stderr)

    # Architecture viz + NN schematic (Phase 35)
    try:
        from ark.core.arch_viz_nodes import register_arch_viz_specs
        register_arch_viz_specs(r)
    except Exception as e:
        import sys
        print(f"[ARK] arch_viz specs failed: {e}", file=sys.stderr)

    # Dynamoscope bridge (Phase 37)
    try:
        from ark.core.dynamoscope_bridge import register_dynamoscope_specs
        register_dynamoscope_specs(r)
    except Exception as e:
        import sys
        print(f"[ARK] dynamoscope specs failed: {e}", file=sys.stderr)

    # Benchmark + audit (Phase 39)
    try:
        from ark.core.benchmark_nodes import register_benchmark_specs
        register_benchmark_specs(r)
    except Exception as e:
        import sys
        print(f"[ARK] benchmark specs failed: {e}", file=sys.stderr)

    # Datasets (Phase 45)
    try:
        from ark.core.dataset_nodes import register_dataset_specs
        register_dataset_specs(r)
    except Exception as e:
        import sys
        print(f"[ARK] dataset specs failed: {e}", file=sys.stderr)

    # TinyLLM (Phase 54)
    try:
        from ark.core.tiny_llm_node import register_tiny_llm_specs
        register_tiny_llm_specs(r)
    except Exception as e:
        import sys
        print(f"[ARK] tiny_llm specs failed: {e}", file=sys.stderr)

    # Output Viewers (Phase 55)
    try:
        from ark.core.output_viewer_nodes import register_output_viewer_specs
        register_output_viewer_specs(r)
    except Exception as e:
        import sys
        print(f"[ARK] output viewer specs failed: {e}", file=sys.stderr)

    # Multimodal (Phase 56)
    try:
        from ark.core.multimodal_nodes import register_multimodal_specs
        register_multimodal_specs(r)
    except Exception as e:
        import sys
        print(f"[ARK] multimodal specs failed: {e}", file=sys.stderr)

    # Multimodal Neural (Phase 59) — MusicGen, Bark, Text2Video Diffusion
    try:
        from ark.core.multimodal_real_nodes import register_real_multimodal_specs
        register_real_multimodal_specs(r)
    except Exception as e:
        import sys
        print(f"[ARK] multimodal neural specs failed: {e}", file=sys.stderr)

    # Diffusion primitives (Phase 60) — timestep, scheduler, cross-attn, UNet, DDIM
    try:
        from ark.core.diffusion_nodes import register_diffusion_specs
        register_diffusion_specs(r)
    except Exception as e:
        import sys
        print(f"[ARK] diffusion specs failed: {e}", file=sys.stderr)

    # MNIST demo (Phase 74) — concrete recognition
    try:
        from ark.core.mnist_demo_nodes import register_mnist_demo_specs
        register_mnist_demo_specs(r)
    except Exception as e:
        import sys
        print(f"[ARK] mnist demo specs failed: {e}", file=sys.stderr)

    # Data toolkit (Phase 75) — conversion/adaptation anything↔anything
    try:
        from ark.core.data_nodes import register_data_specs
        register_data_specs(r)
    except Exception as e:
        import sys
        print(f"[ARK] data toolkit specs failed: {e}", file=sys.stderr)

    # User Groups (Phase 31) — scan ~/.ark/groups/*.json
    try:
        from ark.core.groups import register_groups
        from ark.core.graph_compiler import HANDLERS
        register_groups(r, HANDLERS)
    except Exception as e:
        import sys
        print(f"[ARK] groups load failed: {e}", file=sys.stderr)

    # User plugins (Phase 21) — chargés depuis ~/.ark/plugins/
    try:
        from ark.core.plugins import register_plugins
        from ark.core.graph_compiler import HANDLERS
        register_plugins(r, HANDLERS, verbose=False)
    except Exception:
        pass

    # Phase 40 : cache pour évite re-parse à chaque appel
    _DEFAULT_REGISTRY_CACHE = r
    globals()["_DEFAULT_REGISTRY_CACHE"] = r
    return r
