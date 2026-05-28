"""Glossaire ARK — définitions des concepts ML, architecture, et workflow.

Source unique de vérité. Consommé par GlossaryDialog (UI) et help CLI.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class GlossaryEntry:
    term: str
    category: str  # 'Architecture' | 'Training' | 'Observation' | 'Diffusion' | 'Workflow' | 'Math' + glossary_data cats
    short: str  # 1-line summary
    long: str  # paragraph détaillé
    related: tuple[str, ...] = ()
    # Phase 38 : enrichissement didactique
    intuition: str = ""  # analogie/explication vulgarisée
    equation: str = ""   # formule math (LaTeX-friendly)
    when_to_use: str = ""  # cas concret d'utilisation
    example: str = ""    # exemple concret
    pitfall: str = ""    # piège fréquent à éviter


GLOSSARY: list[GlossaryEntry] = [
    # ============ Architecture ============
    GlossaryEntry(
        term="Activation",
        category="Architecture",
        short="Fonction non-linéaire appliquée post-Linear/Conv.",
        long="Sans activation, un empilement de Linear est équivalent à un seul Linear. "
             "L'activation introduit la non-linéarité, condition nécessaire pour approximer "
             "des fonctions complexes (théorème d'approximation universelle). "
             "ReLU = max(0,x), GELU = lisse (Transformer), Softmax = distribution de probabilité.",
        related=("Dense", "Conv2D"),
    ),
    GlossaryEntry(
        term="Attention",
        category="Architecture",
        short="Mécanisme de pondération contextuelle Q·Kᵀ/√d → softmax → ·V.",
        long="L'attention permet à chaque position d'une séquence d'agréger sélectivement "
             "l'information des autres positions. Multi-Head Attention parallélise plusieurs "
             "subspaces d'attention pour capturer différents types de relations. "
             "Cœur des Transformers (Vaswani et al., 2017).",
        related=("Transformer Block", "Multi-Head Attention"),
    ),
    GlossaryEntry(
        term="Backpropagation",
        category="Training",
        short="Rétropropagation du gradient via la règle de la chaîne.",
        long="Algorithme calculant ∂L/∂θ pour tous les paramètres θ du réseau. "
             "Forward : calcul des activations + cache. Backward : application chain rule "
             "depuis la loss vers les couches d'entrée. PyTorch via autograd.",
        related=("Gradient", "Loss"),
    ),
    GlossaryEntry(
        term="Conv2D",
        category="Architecture",
        short="Convolution 2D — extracteur de features spatiales locales.",
        long="Applique un kernel glissant sur une image. Equivariant à la translation. "
             "Paramètres : filters (canaux de sortie), kernel_size (3×3 ou 5×5 typique), "
             "stride, padding. Cœur des CNN (ResNet, VGG, EfficientNet).",
        related=("Activation", "Normalize"),
    ),
    GlossaryEntry(
        term="Dense",
        category="Architecture",
        short="Couche fully-connected : y = activation(W·x + b).",
        long="Chaque sortie est une combinaison linéaire de toutes les entrées. "
             "Paramètres : W de forme (in, out), b de forme (out,). Coûteux en mémoire "
             "pour grandes dimensions. Standard pour têtes de classification (logits).",
        related=("Activation", "MLP"),
    ),
    GlossaryEntry(
        term="MLP",
        category="Architecture",
        short="Multi-Layer Perceptron — empilement de Dense + activations.",
        long="Architecture la plus basique. Approximateur universel mais inefficace pour "
             "structures spatiales/temporelles (préférer Conv2D/Transformer). "
             "Sert souvent de baseline ou de tête de classification.",
        related=("Dense", "VG Block"),
    ),
    GlossaryEntry(
        term="MoE",
        category="Architecture",
        short="Mixture of Experts — N FFN parallèles, routeur top-k.",
        long="Scaling efficient : ajoute des paramètres sans coût compute proportionnel. "
             "Un routeur apprend à dispatcher chaque token vers k experts (k=1 = Switch Transformer). "
             "Sparse activation → modèles 1T+ params (Mixtral, GShard, Switch).",
        related=("Transformer Block", "Attention"),
    ),
    GlossaryEntry(
        term="Normalize",
        category="Architecture",
        short="Centre + scale les activations pour stabiliser le training.",
        long="BatchNorm : normalise sur la dimension batch (CNN). LayerNorm : par échantillon "
             "(Transformer, séquences). GroupNorm : par groupes de canaux (petits batches). "
             "Accélère la convergence, permet learning rate plus élevés, agit comme régularisation.",
        related=("Dense", "Conv2D"),
    ),
    GlossaryEntry(
        term="Residual Connection",
        category="Architecture",
        short="out = f(x) + x — skip connection. Évite vanishing gradient.",
        long="Introduit par ResNet (He et al., 2015). Permet d'entraîner des réseaux très "
             "profonds (1000+ couches). Le gradient se propage directement via la skip "
             "connection. Universellement utilisé dans Transformers + CNN modernes.",
        related=("Conv2D", "Transformer Block"),
    ),
    GlossaryEntry(
        term="Transformer",
        category="Architecture",
        short="Architecture séquentielle basée sur self-attention. Standard NLP+vision moderne.",
        long="Bloc = Multi-Head Attention + FFN + 2 LayerNorm + 2 résiduels. "
             "Empilement de N blocs (12-96+). Family : encoder-only (BERT), decoder-only (GPT), "
             "encoder-decoder (T5). Domine NLP depuis 2018, vision depuis ViT (2020).",
        related=("Attention", "MoE", "Residual Connection"),
    ),

    # ============ Self-modifying / Metamorph ============
    GlossaryEntry(
        term="VG Block",
        category="Architecture",
        short="VerifyGrowNet block — MLP mutable runtime (widen/prune).",
        long="Bloc dont la largeur cachée peut changer pendant le training. "
             "Widen : Net2Net function-preserving (ajout neurones initialisés tels que "
             "f(x) reste identique). Prune : suppression neurones par importance "
             "(magnitude × variance). Cœur des architectures auto-modifiantes ARK/Metamorph.",
        related=("Net2Net", "Trigger Rule", "CriticalGrowNet"),
    ),
    GlossaryEntry(
        term="Net2Net",
        category="Architecture",
        short="Widening function-preserving (Chen et al., 2015).",
        long="Permet d'élargir une couche cachée sans changer le mapping f(x). "
             "Trick : la nouvelle colonne de W₂ est initialisée à 0 → contribution nulle "
             "à f(x). Le training subséquent peut alors la spécialiser. Permet d'éviter "
             "le restart depuis zéro lors de mutations architecturales.",
        related=("VG Block", "Mutation"),
    ),
    GlossaryEntry(
        term="Observer",
        category="Observation",
        short="Node passthrough qui mesure les activations sans intrusion.",
        long="Calcule des métriques sur les activations en aval : eff_rank, intrinsic_dim, "
             "power_slope, mean_abs. Émet sur signal Qt → graphes UI live. Non-différentiable "
             "(detach des activations). Source de feedback pour Trigger Rules.",
        related=("eff_rank", "intrinsic_dim", "power_slope", "Trigger Rule"),
    ),
    GlossaryEntry(
        term="Trigger Rule",
        category="Observation",
        short="Conditionnelle : si metric op threshold → mutate target_node.",
        long="Mécanisme déclencheur de mutations architecturales. Surveille en continu "
             "une métrique observer. Quand le seuil est franchi → applique l'action "
             "(widen N, prune N) sur le node cible. Boucle observation → action = "
             "architecture auto-modifiante.",
        related=("Observer", "VG Block", "Mutation"),
    ),
    GlossaryEntry(
        term="eff_rank",
        category="Observation",
        short="Rang effectif = entropie spectrale exponentiée.",
        long="Mesure la dimensionnalité effective d'un tenseur d'activations. "
             "eff_rank = exp(H(σᵢ²/Σσⱼ²)) où σᵢ sont les valeurs singulières. "
             "≈ N si toutes les directions sont utilisées, ≈ 1 si une seule domine. "
             "Indicateur de collapse représentationnel.",
        related=("Observer", "intrinsic_dim"),
    ),
    GlossaryEntry(
        term="intrinsic_dim",
        category="Observation",
        short="Dimension intrinsèque du manifold (Levina-Bickel).",
        long="Estime la dimensionnalité du sous-variété peuplée par les activations. "
             "Méthode Levina-Bickel basée sur ratios de distances aux k plus proches voisins. "
             "Souvent << dimension ambiante → manifold hypothesis. Mesure de complexité "
             "intrinsèque de la représentation.",
        related=("Observer", "eff_rank"),
    ),
    GlossaryEntry(
        term="power_slope",
        category="Observation",
        short="Pente du spectre de puissance FFT en log-log.",
        long="β tel que P(f) ∝ 1/f^β. β=0 = bruit blanc, β=2 = bruit brownien, "
             "β≈1.8-2 = criticalité auto-organisée (SOC). Cible des modèles critiques. "
             "Calculé via numpy.fft + régression linéaire log P vs log f.",
        related=("Observer", "Criticality"),
    ),
    GlossaryEntry(
        term="Criticality",
        category="Observation",
        short="État critique auto-organisé (Self-Organized Criticality, Bak 1987).",
        long="Régime entre ordre et chaos où les avalanches suivent une loi de puissance. "
             "Spectral slope β ≈ -2 est l'invariant. Conjecture : les réseaux opérant "
             "à criticalité ont maximum d'expressivité, sensibilité optimale aux entrées, "
             "et capacité de calcul universelle (Langton 1990, Beggs 2003).",
        related=("power_slope", "SOC"),
    ),
    GlossaryEntry(
        term="CriticalGrowNet",
        category="Architecture",
        short="Modèle Metamorph M1 — autorégule sa criticalité par mutations.",
        long="Architecture VG mutable + policy de mutation guidée par spectral_slope. "
             "Cible : maintenir β ≈ -2 pendant le training. Quand β s'éloigne → widen/prune. "
             "Falsification : sans cette régulation, β dérive (test M1).",
        related=("Criticality", "VG Block", "power_slope"),
    ),
    GlossaryEntry(
        term="DyadicMirror",
        category="Architecture",
        short="Modèle Metamorph M5 — deux agents qui se modélisent mutuellement.",
        long="Deux SelfRefAgent (state, value, model) où chacun apprend à prédire l'état "
             "interne de l'autre. Convergence vers représentation partagée. Test : "
             "coupling via state_proj partagé. Lien proto-conscience interactive.",
        related=("MetaController", "CriticalGrowNet"),
    ),
    GlossaryEntry(
        term="MetaController",
        category="Architecture",
        short="Modèle Metamorph M2 — contrôleur GNN qui pilote les mutations.",
        long="Réseau de neurones graphique (GNN) prenant en entrée l'état du graphe "
             "(nodes + leurs métriques observer) et émettant des actions de mutation. "
             "Politique apprise par RL ou par évolution. Permet une auto-modification "
             "plus structurée que des Trigger Rules manuelles.",
        related=("Trigger Rule", "Mutation", "DyadicMirror"),
    ),

    # ============ Training ============
    GlossaryEntry(
        term="SGD",
        category="Training",
        short="Stochastic Gradient Descent — descente de gradient stochastique.",
        long="Optimizer le plus simple : θ ← θ - lr·∇L. Avec momentum : "
             "v ← β·v + ∇L ; θ ← θ - lr·v. Plus stable que Adam sur certaines tâches "
             "(vision, CNN). Trouve souvent des minima plus généralisables. Hyperparam : "
             "lr typiquement 0.01-0.1 (10× plus haut qu'Adam).",
        related=("Adam", "Gradient", "Learning Rate"),
    ),
    GlossaryEntry(
        term="SOC",
        category="Observation",
        short="Self-Organized Criticality (Bak, Tang, Wiesenfeld 1987).",
        long="Théorie expliquant l'apparition de comportements critiques dans des systèmes "
             "dissipatifs sans tuning fin (sandpile model). Signature : lois de puissance "
             "dans les avalanches et le spectre de fluctuations (β ≈ 2). Lien conjecturé "
             "avec l'intelligence : opérer à criticalité maximise l'expressivité.",
        related=("Criticality", "power_slope", "CriticalGrowNet"),
    ),
    GlossaryEntry(
        term="Adam",
        category="Training",
        short="Optimizer : Adaptive Moment Estimation. Default robuste.",
        long="Combine momentum (moyenne mobile du gradient) + RMSprop (moyenne mobile du "
             "gradient²). Lr adaptatif par paramètre. Hyperparams : lr=1e-3, β₁=0.9, β₂=0.999. "
             "AdamW = variant avec weight decay découplé (recommandé pour Transformer).",
        related=("SGD", "Learning Rate"),
    ),
    GlossaryEntry(
        term="Cross-Entropy",
        category="Training",
        short="Loss classification : -log(softmax(logits)[target]).",
        long="Mesure la divergence entre la distribution prédite et la one-hot target. "
             "Combinée avec softmax pour stabilité numérique (logsumexp trick). "
             "Multi-classes par défaut. Pour multi-label : BinaryCrossEntropy par classe.",
        related=("Loss", "Softmax"),
    ),
    GlossaryEntry(
        term="Gradient",
        category="Training",
        short="∂L/∂θ — direction de plus forte montée de la loss.",
        long="Vecteur des dérivées partielles de la loss par rapport aux paramètres. "
             "L'optimizer applique θ ← θ - lr·∇L pour descendre. Calculé par "
             "backpropagation (chain rule). Vanishing/exploding = problèmes classiques.",
        related=("Backpropagation", "Adam"),
    ),
    GlossaryEntry(
        term="Learning Rate",
        category="Training",
        short="Taux d'apprentissage — pas de la descente de gradient.",
        long="Hyperparam critique. Trop haut = divergence, trop bas = convergence lente. "
             "Schedules courants : cosine annealing, warmup + decay, OneCycleLR. "
             "Pour Adam : 1e-3 default. Pour Transformers : 5e-4 avec warmup.",
        related=("Adam", "Gradient"),
    ),
    GlossaryEntry(
        term="Loss",
        category="Training",
        short="Fonction scalaire à minimiser. Mesure l'erreur du modèle.",
        long="Le training optimise θ pour réduire E[L(f(x; θ), y)]. Choix dépend de "
             "la tâche : CrossEntropy (classif), MSE (régression), BCE (binaire), "
             "Contrastive (embeddings), Diffusion (MSE sur bruit). Différentiable par "
             "construction.",
        related=("Cross-Entropy", "Gradient"),
    ),

    # ============ Diffusion ============
    GlossaryEntry(
        term="Diffusion",
        category="Diffusion",
        short="Modèle génératif par débruitage itératif.",
        long="Forward process : ajoute du bruit gaussien à une image en T étapes "
             "jusqu'à pur bruit. Reverse process : un réseau (UNet) apprend à inverser "
             "chaque étape. Inférence : sample à partir de bruit pur, débruite T fois → "
             "image. Stable Diffusion = diffusion opérant dans l'espace latent VAE.",
        related=("Stable Diffusion", "KSampler", "VAE"),
    ),
    GlossaryEntry(
        term="Stable Diffusion",
        category="Diffusion",
        short="Modèle text-to-image open-source (Stability AI 2022).",
        long="Architecture : VAE encoder/decoder + UNet diffusion dans l'espace latent + "
             "CLIP text encoder pour conditioning. Versions : SD 1.5, SD 2.x, SDXL, SD3. "
             "Standard pour génération d'image conditionnée par texte. Compatible LoRA, "
             "ControlNet, IP-Adapter, etc.",
        related=("Diffusion", "KSampler", "CLIP", "VAE", "LoRA"),
    ),
    GlossaryEntry(
        term="CLIP",
        category="Diffusion",
        short="Contrastive Language-Image Pretraining — encoder texte/image partagé.",
        long="Modèle OpenAI 2021 : image_encoder + text_encoder partageant un espace "
             "latent. Entraîné contrastivement sur 400M paires (image, caption) web. "
             "Cœur de la guidance dans Stable Diffusion (text → conditioning).",
        related=("Stable Diffusion", "KSampler"),
    ),
    GlossaryEntry(
        term="CFG",
        category="Diffusion",
        short="Classifier-Free Guidance — pondération du conditioning.",
        long="Pendant le sampling : ε_final = ε_uncond + cfg·(ε_cond - ε_uncond). "
             "cfg > 1 amplifie l'effet du prompt. Typique 7-12. Trop haut → artefacts, "
             "trop bas → ignore le prompt.",
        related=("KSampler", "CLIP"),
    ),
    GlossaryEntry(
        term="KSampler",
        category="Diffusion",
        short="Sampler diffusion — itère le débruitage latent.",
        long="Algorithme de résolution numérique de l'ODE/SDE de diffusion inverse. "
             "Variants : Euler, DPM-Solver++, DDIM, UniPC. Trade-off qualité/vitesse. "
             "Schedulers : karras (qualité), normal (standard), exponential.",
        related=("Diffusion", "CFG", "VAE"),
    ),
    GlossaryEntry(
        term="LoRA",
        category="Diffusion",
        short="Low-Rank Adaptation — finetune via matrices low-rank.",
        long="W ← W + α·B·A où B, A sont des matrices low-rank (~kB). Permet de "
             "finetuner un grand modèle (SD, LLM) en 1-100MB au lieu de GB. "
             "Multiples LoRAs combinables par addition. Standard pour personnalisation.",
        related=("Stable Diffusion",),
    ),
    GlossaryEntry(
        term="ControlNet",
        category="Diffusion",
        short="Conditionne la diffusion via une image guide (canny, depth, pose).",
        long="Architecture copie du UNet avec connexions zero-init. Injecte des contraintes "
             "spatiales fortes pendant le sampling. Permet contrôle précis : compose à partir "
             "d'un sketch, transfert de pose, depth-guided gen, etc.",
        related=("KSampler",),
    ),
    GlossaryEntry(
        term="VAE",
        category="Diffusion",
        short="Variational Autoencoder — image ↔ latent (1/8 résolution).",
        long="Compresse l'image pixel-space (3×512×512) en latent (4×64×64). "
             "Stable Diffusion opère dans cet espace latent (×64 plus compact). "
             "VAE.encode pour img2img, VAE.decode après KSampler pour récupérer l'image.",
        related=("Stable Diffusion", "KSampler"),
    ),

    # ============ Workflow ============
    GlossaryEntry(
        term="Mutation",
        category="Workflow",
        short="Modification architecturale runtime (widen, prune).",
        long="ARK supporte la mutation pendant le training via Trigger Rules. "
             "Widen : ajoute des unités (Net2Net function-preserving). "
             "Prune : retire des unités (importance par magnitude). "
             "Préserve l'optimizer state via recreation post-mutation.",
        related=("VG Block", "Trigger Rule", "Net2Net"),
    ),
    GlossaryEntry(
        term="Auto-layout",
        category="Workflow",
        short="Disposition automatique des nodes par niveaux topologiques.",
        long="Algorithme : (1) topological sort, (2) longest-path level assignment "
             "(x croissant), (3) stagger vertical pour nodes du même niveau. "
             "Menu Layout → Auto-Layout (ou Cmd+L).",
        related=("Template",),
    ),
    GlossaryEntry(
        term="Plugin",
        category="Workflow",
        short="Module Python utilisateur chargé depuis ~/.ark/plugins/.",
        long="Chaque .py définit un PLUGIN_SPEC (NodeSpec) + handler(node, in_shapes) "
             "→ (nn.Module, out_shape). Registered au démarrage. Fichiers préfixés _ ignorés. "
             "Crashes silencieusement skip avec traceback console.",
        related=("Template",),
    ),
    GlossaryEntry(
        term="Template",
        category="Workflow",
        short="Architecture préconçue (mlp_mnist, cnn_cifar, transformer_mini...).",
        long="Workflows JSON livrés avec ARK. Disponibles via menu File → Templates "
             "(ou Cmd+T). 8 templates : mlp_mnist, cnn_cifar, resnet_block, "
             "transformer_mini, vg_mutable, diffusion_basic, metamorph_m1, metamorph_m5.",
        related=("Template",),
    ),
    GlossaryEntry(
        term="Format ark-arch-v1",
        category="Workflow",
        short="Format JSON natif ARK. Compatible metamorph-arch-v1.",
        long="Schéma : nodes (id, type, x, y, params), connections (from_node, "
             "from_socket, to_node, to_socket), metadata. Sauvegarde/chargement via "
             "Cmd+S / Cmd+O. Import legacy ai-arch-builder via menu File → Import.",
        related=("Auto-layout",),
    ),
    GlossaryEntry(
        term="ONNX Export",
        category="Workflow",
        short="Export du modèle compilé vers format ONNX interopérable.",
        long="ONNX = Open Neural Network Exchange. Format standard pour déploiement "
             "cross-framework (PyTorch → ONNX Runtime → TensorRT/CoreML/...). "
             "Menu File → Export → ONNX. Requiert un input shape concret.",
        related=("Template",),
    ),
    GlossaryEntry(
        term="DAG",
        category="Workflow",
        short="Directed Acyclic Graph — graphe orienté sans cycle.",
        long="Structure des workflows ARK. Compile_scene applique un topological sort + "
             "routage multi-output via dictionnaire {to_node: {to_socket: source_tensor}}. "
             "Les cycles sont détectés au compile-time → GraphCompileError.",
        related=("Template", "Auto-layout"),
    ),

    # ============ Math ============
    GlossaryEntry(
        term="Softmax",
        category="Math",
        short="Normalise un vecteur en distribution de probabilité.",
        long="softmax(x)ᵢ = exp(xᵢ) / Σⱼ exp(xⱼ). Sortie ∈ [0,1], somme = 1. "
             "Différentiable. Combinée avec CrossEntropy pour classification. "
             "Pour stabilité numérique : softmax(x - max(x)) = softmax(x).",
        related=("Cross-Entropy", "Attention"),
    ),
    GlossaryEntry(
        term="GELU",
        category="Math",
        short="Gaussian Error Linear Unit — activation lisse moderne.",
        long="GELU(x) = x · Φ(x) où Φ est la CDF de N(0,1). Approximation : "
             "0.5·x·(1+tanh(√(2/π)·(x+0.044715·x³))). Standard dans Transformers "
             "(BERT, GPT-2+). Smoother que ReLU autour de 0.",
        related=("Activation", "Transformer"),
    ),
    GlossaryEntry(
        term="ReLU",
        category="Math",
        short="Rectified Linear Unit — max(0, x).",
        long="Activation la plus simple et la plus utilisée. Non-saturante (pas de "
             "vanishing gradient pour x>0). Sparse (≈ 50% d'activations à 0). "
             "Problème : 'dying ReLU' (neurones bloqués à 0). Variantes : LeakyReLU, PReLU.",
        related=("Activation",),
    ),
]


# Auto-merge from glossary_data CONCEPTS (Phase 29)
def _norm_term(t: str) -> str:
    return t.lower().strip().replace("·", "").replace("(", "").replace(")", "")


def _merge_glossary_data() -> None:
    """Importe les 400+ concepts de glossary_data dans GLOSSARY (idempotent)."""
    try:
        from ark.core.glossary_data import CONCEPTS
        from ark.core.didactic_data import DIDACTIC_EXTRA
    except ImportError:
        DIDACTIC_EXTRA = {}
        try:
            from ark.core.glossary_data import CONCEPTS
        except ImportError:
            return
    existing = {_norm_term(e.term) for e in GLOSSARY}
    seen_new: set[str] = set()
    for c in CONCEPTS:
        norm = _norm_term(c["title"])
        if norm in existing or norm in seen_new:
            continue
        seen_new.add(norm)
        short = c.get("short", "") or c["title"]
        long_ = c.get("long", "") or short or c["title"]
        # Recherche enrichissement didactique
        extra = DIDACTIC_EXTRA.get(c["name"], {})
        GLOSSARY.append(GlossaryEntry(
            term=c["title"],
            category=c["category"],
            short=short,
            long=long_,
            related=tuple(c.get("related", ())),
            intuition=extra.get("intuition", ""),
            equation=c.get("equation", "") or extra.get("equation", ""),
            when_to_use=extra.get("when_to_use", ""),
            example=extra.get("example", ""),
            pitfall=extra.get("pitfall", ""),
        ))


_merge_glossary_data()


def by_category() -> dict[str, list[GlossaryEntry]]:
    """Regroupe les entrées par catégorie."""
    out: dict[str, list[GlossaryEntry]] = {}
    for e in GLOSSARY:
        out.setdefault(e.category, []).append(e)
    for k in out:
        out[k].sort(key=lambda e: e.term.lower())
    return dict(sorted(out.items()))


def find(query: str) -> list[GlossaryEntry]:
    """Recherche fuzzy dans term + short + long."""
    q = query.lower().strip()
    if not q:
        return list(GLOSSARY)
    return [e for e in GLOSSARY
            if q in e.term.lower() or q in e.short.lower() or q in e.long.lower()]


def get(term: str) -> GlossaryEntry | None:
    """Lookup exact (case-insensitive)."""
    t = term.lower()
    for e in GLOSSARY:
        if e.term.lower() == t:
            return e
    return None
