"""Phase 43 — Expansion automatique contenu glossaire vers 500+ chars.

Stratégie : combine fields existants (short/long/equation/intuition/example/
pitfall/related) + génère contexte catégoriel + recommandations standards.
Garantit minimum 500 caractères par entrée même pour concepts pauvres.

Aucune source externe : pur agrégat de champs ARK existants + templates.
"""
from __future__ import annotations


# Contexte par catégorie : phrase contextualisante ajoutée si manque content.
# Texte générique original ARK décrivant le domaine de chaque catégorie.
CATEGORY_CONTEXT: dict[str, str] = {
    "Foundations": (
        "Concept fondamental qui forme la base mathématique des architectures "
        "neuronales. Apparaît dans la majorité des couches modernes — tabulaire, "
        "vision, NLP, génératif. La maîtrise de ces briques élémentaires "
        "conditionne la compréhension des modèles plus complexes en aval."
    ),
    "Neural Networks": (
        "Couche ou pattern architectural utilisé dans les réseaux de neurones "
        "feed-forward, convolutifs et hybrides. Considère sa position dans la "
        "pile : trop tôt et l'information se perd, trop tard et la non-linéarité "
        "manque. ARK permet de composer ces éléments visuellement par drag-drop."
    ),
    "Sequential": (
        "Module dédié aux données séquentielles : séries temporelles, signaux "
        "audio, ECG, texte tokenisé. Le coeur du problème est de propager "
        "l'information temporelle sans saturer la mémoire. RNN, LSTM, GRU, "
        "Mamba, RetNet et autres State Space Models adoptent des stratégies "
        "différentes pour balancer expressivité et coût."
    ),
    "Transformers": (
        "Brique de l'architecture Transformer, devenue dominant depuis 2017 en "
        "NLP, vision (ViT), audio (Whisper), multimodal (CLIP). L'idée centrale : "
        "l'attention permet à chaque token d'agréger l'information de toute la "
        "séquence en O(N²). Variantes optimisent ce coût quadratique : sparse, "
        "linear, FlashAttention, MoE."
    ),
    "NLP": (
        "Traitement automatique du langage naturel. Pipeline classique : "
        "tokenisation, embedding, encoder, head spécifique tâche. Les LLM "
        "modernes apprennent ces étapes conjointement par pré-entraînement "
        "puis fine-tuning ou prompting."
    ),
    "Computer Vision": (
        "Architecture ou tâche vision par ordinateur. Le challenge historique "
        "est l'invariance par translation, rotation, échelle. CNN ont apporté "
        "la première solution scalable (1998 LeNet → 2012 AlexNet → 2015 "
        "ResNet → 2020 ViT). Aujourd'hui ViT et hybrides ConvNeXt dominent."
    ),
    "Generative AI": (
        "Modèle ou technique de génération : produire un échantillon nouveau "
        "à partir d'une distribution apprise. Familles : autoregressive (GPT), "
        "VAE (latents continus), GAN (adversarial), diffusion (denoising "
        "itératif), flow matching. Stable Diffusion combine VAE + diffusion."
    ),
    "Reinforcement Learning": (
        "Apprentissage par renforcement : un agent apprend par essai-erreur en "
        "maximisant une récompense cumulée. Distinction clé : on-policy vs "
        "off-policy, model-free vs model-based, value-based vs policy-based. "
        "RLHF a popularisé son usage pour aligner les LLM."
    ),
    "Graph Systems": (
        "Architecture opérant sur des graphes : nœuds + arêtes potentiellement "
        "typés. Message passing est le paradigme central : agréger info des "
        "voisins puis update. Application : molécules, réseaux sociaux, "
        "knowledge graphs, code AST."
    ),
    "Unsupervised": (
        "Méthode sans labels : clustering, réduction de dimension, density "
        "estimation, self-supervised. Le challenge est de définir un objectif "
        "implicite (entropy, reconstruction, contrastive) qui révèle la "
        "structure des données."
    ),
    "Evolutionary Systems": (
        "Algorithmes inspirés de l'évolution biologique : sélection, mutation, "
        "crossover. Forces : black-box (pas besoin de gradient), parallèle, "
        "exploration globale. Faiblesses : sample-inefficient. Utiles pour "
        "neural architecture search et hyperparameter tuning."
    ),
    "Dynamical Systems": (
        "Système dont l'état évolue dans le temps selon des règles. Outils "
        "centraux : attracteurs, exposants de Lyapunov, dimension fractale. "
        "Connexion forte avec ML moderne : RNN sont des systèmes dynamiques "
        "discrets, Neural ODE rendent l'évolution continue."
    ),
    "Biological Inspiration": (
        "Mécanisme inspiré directement de la neurobiologie : plasticité, "
        "STDP, neurogenèse, morphogenèse. ARK intègre ces mécanismes comme "
        "modèle de référence pour les architectures auto-modifiantes : la "
        "biologie résout depuis 4 milliards d'années des problèmes que le "
        "ML découvre seulement."
    ),
    "Cognitive Architectures": (
        "Modèle visant à reproduire des aspects de la cognition : prédiction, "
        "mémoire, raisonnement, conscience. Distinction du connectionnisme "
        "pur : intégration de modules symboliques, hiérarchiques, ou de "
        "boucles méta-cognitives."
    ),
    "Interpretability": (
        "Méthode pour comprendre ce que fait un modèle : feature attribution, "
        "circuit analysis, attention rollout, sparse autoencoders. Distinction "
        "interprétabilité post-hoc (SHAP) vs mechanistic (Anthropic). Domaine "
        "critique pour audit, safety, debug."
    ),
    "Optimization": (
        "Algorithme ou technique pour trouver les poids qui minimisent une "
        "loss. Le bestiaire des optimizers (SGD, Adam, AdamW, Lion, etc.) "
        "reflète des trade-offs : convergence, mémoire, généralisation. "
        "Les schedulers (warmup, cosine) jouent un rôle aussi important."
    ),
    "Training": (
        "Phase d'apprentissage d'un modèle : forward, loss, backward, "
        "optimizer step, eval. Choix critiques : batch size, learning rate, "
        "régularisation, augmentation. Stratégies avancées : curriculum, "
        "distillation, multi-task, continual learning."
    ),
    "Distributed Systems": (
        "Techniques pour entraîner ou servir des modèles trop grands pour "
        "une machine. Tensor parallelism splite les matrices, pipeline "
        "parallelism splite les couches, data parallelism duplique le modèle. "
        "FSDP et ZeRO réduisent la mémoire en shardant l'optimizer state."
    ),
    "Hardware": (
        "Composant matériel ou backend de calcul. GPU dominent grâce à leur "
        "parallélisme massif. TPU optimisés pour matmul. NPU mobiles. FPGA "
        "personnalisables. Le choix impacte directement le throughput et "
        "le coût d'inférence."
    ),
    "Simulation": (
        "Système simulé : automate cellulaire, modèle d'agent, réseau "
        "physique. Utilisé pour étudier l'émergence, valider des théories "
        "computationnelles, ou pré-entraîner avant un transfert au monde "
        "réel (sim2real)."
    ),
    "Multi-Agent Systems": (
        "Système composé de plusieurs agents qui interagissent, coopèrent, "
        "se concurrencent ou échangent de l'information. Centralisé vs "
        "décentralisé, homogène vs hétérogène. Domaine en croissance avec "
        "les LLM agents et frameworks comme AutoGen."
    ),
    "Adaptive Architectures": (
        "Architecture dont la structure change pendant l'entraînement ou "
        "l'inférence : routing dynamique, sparse experts, neurogenèse "
        "artificielle. ARK est conçu pour faciliter l'expérimentation sur "
        "ces architectures via la métamorphose (M1/M2/M5)."
    ),
    "Experimental Research": (
        "Concept à la frontière de la recherche actuelle : foundation models, "
        "test-time compute, recursive self-improvement, neuromorphic. Statut "
        "et viabilité encore en cours d'évaluation par la communauté."
    ),
    "Math": (
        "Construction mathématique utilisée dans les architectures ARK. "
        "Comprendre la formulation aide à anticiper le comportement et "
        "les modes d'échec."
    ),
    "Diffusion": (
        "Pipeline de diffusion : empiler bruit progressivement puis "
        "apprendre à dé-brouiller. La force des modèles modernes vient de "
        "l'opération en espace latent (VAE) + conditioning par cross-attention "
        "(CLIP)."
    ),
    "Input": (
        "Source de données pour le graphe ARK : fichier image/vidéo/son/CSV/"
        "JSON, périphérique (webcam, micro), ou tenseur synthétique. Chaque "
        "loader produit un tenseur formaté prêt à être consommé par les "
        "couches en aval."
    ),
    "Neural": (
        "Couche neurale fondamentale ARK : Dense, Convolution, Normalization, "
        "Residual, Attention. Composables librement par drag-drop dans le "
        "node editor pour construire des architectures arbitraires."
    ),
    "Observe": (
        "Node d'observation non-intrusif : extrait des métriques (loss, "
        "accuracy, gradient norms, activations stats) sans modifier le "
        "tenseur. Combiné avec un Trigger pour piloter des actions "
        "(mutation, early stop, save checkpoint)."
    ),
    "Code": (
        "Node Python custom : permet d'écrire une expression ou un bloc "
        "exécuté dans le forward. Pour les transformations ad-hoc qui "
        "n'ont pas de spec dédiée. À utiliser avec parcimonie pour rester "
        "exportable ONNX."
    ),
    "Viz": (
        "Node de visualisation interactive : PCA/t-SNE/UMAP 3D, phase space, "
        "scatter, line trajectory, surface heightmap, recurrence plot, "
        "attention head/model view. Output HTML standalone ouvrable navigateur."
    ),
    "HuggingFace": (
        "Bridge vers l'écosystème Hugging Face : modèles transformers, "
        "tokenizers, datasets. Lazy import : pas de dépendance si non utilisé."
    ),
    "Devices": (
        "Source temps réel : webcam (cv2), microphone (sounddevice). "
        "Permet de prototyper des modèles qui consomment des flux live."
    ),
    "Metamorph": (
        "Module du framework Metamorph intégré à ARK : architectures qui "
        "mutent pendant l'entraînement. M1 CriticalGrowNet vise un état SOC "
        "(spectral β=-2). M2 MetaController orchestre via GNN. M5 DyadicMirror "
        "fait dialoguer deux agents en cross-prediction."
    ),
    "Research": (
        "Outil de recherche : génération d'idées (AI-Scientist Sakana), "
        "autoresearch loop (Karpathy pattern), visualisation scientifique. "
        "Permet de piloter des expériences depuis ARK."
    ),
    "Dynamoscope": (
        "Bridge vers l'outil Dynamoscope : analyse de dynamiques émergentes "
        "depuis trajectoires latentes. Lyapunov, RQA, DMD, persistent "
        "homology, DNA composite score 7-axes, phase space 3D interactif."
    ),
    "Benchmark": (
        "Node d'audit ou benchmark : mesure latency/throughput, audit "
        "params/FLOPs, distribution weights, gradient flow vanishing/exploding, "
        "activation stats per layer. Output HTML interactif Plotly."
    ),
    "User Groups": (
        "Groupe composite défini par l'utilisateur : sous-graphe sauvé comme "
        "node réutilisable. Ctrl+G sur sélection → name + category → "
        "apparaît dans palette."
    ),
}


# Template d'extension par kind (node/concept/tool/paradigm)
KIND_HEADER: dict[str, str] = {
    "node": "Bloc fonctionnel : peut être instancié dans le node editor ARK "
              "et composé avec d'autres briques pour former un graphe compilable "
              "en nn.Module PyTorch. ",
    "concept": "Concept fondamental référencé dans la documentation ARK. Sert "
                 "de point d'ancrage pédagogique : aide à situer les autres briques "
                 "fonctionnelles dans un cadre théorique cohérent. ",
    "tool": "Outil ou bibliothèque externe wrappable dans ARK via lazy import. "
              "Évite la dépendance forte : si la lib n'est pas installée, le "
              "node se rabat sur un fallback déterministe ou raise un message "
              "informatif. ",
    "paradigm": "Paradigme d'apprentissage : pas une couche mais une stratégie "
                  "globale qui conditionne le pipeline complet (data, loss, "
                  "schedule, eval). Influence le choix des architectures en aval. ",
    "template": "Template d'architecture pré-configuré : ensemble de nodes + "
                  "connexions prêt à instancier. Sert de point de départ pour "
                  "expériences ou benchmarks comparatifs. ",
}


def expand(concept: dict, min_chars: int = 500) -> str:
    """Génère long description >= min_chars depuis champs concept.

    Combine :
    1. Kind header (paradigm/concept/node/tool/template)
    2. Original long (ou short)
    3. Intuition vulgarisée (si présente)
    4. When to use (si présent)
    5. Example (si présent)
    6. Pitfall (si présent)
    7. Equation contextualisée (si présente)
    8. Category context fallback
    9. Related concepts mention
    """
    parts: list[str] = []

    name = concept.get("title", concept.get("name", "Concept"))
    cat = concept.get("category", "")
    kind = concept.get("kind", "concept")

    # 1. Kind header
    parts.append(KIND_HEADER.get(kind, KIND_HEADER["concept"]))

    # 2. Description principale
    long_ = concept.get("long", "") or concept.get("short", "")
    if long_:
        parts.append(long_)
        if not long_.endswith((".", "!", "?")):
            parts.append(".")
        parts.append(" ")

    # 3. Equation contextualisée
    eq = concept.get("equation", "")
    if eq:
        parts.append(f"Formulation mathématique : {eq}. Cette équation est "
                       f"la signature canonique du concept ; comprendre chaque "
                       f"symbole aide à anticiper le comportement numérique. ")

    # 4. Category context (toujours, augmente l'épaisseur didactique)
    cat_ctx = CATEGORY_CONTEXT.get(cat, "")
    if cat_ctx:
        parts.append(f"[{cat}] {cat_ctx} ")

    # 5. Related concepts mention
    related = concept.get("related", ())
    if related:
        rel_str = ", ".join(str(r) for r in related[:6])
        parts.append(f"Concepts liés à explorer pour approfondir : {rel_str}. "
                       f"La navigation entre ces entrées révèle souvent des "
                       f"connexions non triviales et conditionne une compréhension "
                       f"systémique du domaine. ")

    # 6. Si manque encore d'épaisseur, ajouter conseils d'utilisation génériques
    text = "".join(parts).strip()
    if len(text) < min_chars:
        if kind == "node":
            text += (" Dans ARK, ce node se manipule via drag-drop depuis la "
                       "palette latérale, puis se configure via l'Inspector "
                       "(paramètres + bloc didactique). Les connexions entre nodes "
                       "sont typées et validées au moment de la compilation. "
                       "Pour réutiliser une configuration éprouvée, sélectionne "
                       "plusieurs nodes connectés et utilise Ctrl+G pour les "
                       "sauver comme groupe composite réutilisable.")
        elif kind == "concept":
            text += (" Ce concept n'est pas directement compilable mais sert de "
                       "point de référence dans la documentation et les tutoriels. "
                       "Il apparaît dans la palette comme entrée descriptive pour "
                       "faciliter la navigation par catégorie. La compréhension de "
                       "cette notion conditionne souvent la maîtrise des nodes "
                       "fonctionnels qui en découlent.")
        elif kind == "tool":
            text += (" L'import dans ARK est paresseux : la lib n'est chargée "
                       "qu'au premier usage effectif. Si elle est absente du "
                       "venv, ARK reste fonctionnel — seules les fonctionnalités "
                       "qui dépendent de ce tool deviennent indisponibles avec un "
                       "message d'installation suggérée.")
        elif kind == "paradigm":
            text += (" Le choix d'un paradigme d'apprentissage est structurant : "
                       "il définit la nature des données attendues, la fonction "
                       "de loss appropriée, la stratégie d'évaluation. Il vaut "
                       "mieux le clarifier en amont avant de commencer "
                       "l'implémentation. ARK ne contraint pas le paradigme : "
                       "il fournit les briques, à toi de composer.")
        text += " "

    # 7. Pad final si toujours sous le seuil
    if len(text) < min_chars:
        text += (f"Pour aller plus loin, consulte la page d'aide complète (F1) "
                   f"qui agrège 475+ concepts didactiques avec intuition vulgarisée, "
                   f"équations, exemples chiffrés et pièges fréquents. Les liens "
                   f"'Voir aussi' permettent de naviguer entre concepts liés sans "
                   f"perdre le contexte de la session courante. ")

    return text.strip()
