"""Phase 38 — Contenu didactique vulgarisé niveau bac+0/Bac+2.

Pour chaque concept clé, propose :
- intuition : analogie/explication simple, sans jargon
- when_to_use : situation concrète d'utilisation
- example : exemple chiffré ou cas réel
- pitfall : erreur typique à éviter

Clés = `name` des CONCEPTS (glossary_data.py).
Auto-fallback si concept absent du dict : Inspector affiche seulement
description + role + function.

Sources : vulgarisation choisie pour étudiant Bac+ littérature, sans
sacrifier précision technique.
"""

DIDACTIC_EXTRA: dict[str, dict[str, str]] = {

    # ============================================================
    # Foundations — Algèbre
    # ============================================================
    "vector": {
        "intuition": "Une liste ordonnée de nombres. Pense à un point dans l'espace : "
                       "un vecteur de taille 3 = (x, y, z). En IA, chaque image, chaque mot, "
                       "chaque utilisateur peut devenir un vecteur de centaines de nombres.",
        "when_to_use": "Toute donnée que tu veux que la machine 'comprenne' : pixels, "
                          "embeddings de mots, features tabulaires.",
        "example": "Un mot comme 'chat' devient le vecteur [0.21, -0.84, 0.05, ...] de "
                     "300 dimensions dans Word2Vec.",
        "pitfall": "Confondre taille (nombre d'éléments) et dimension (espace dans lequel "
                     "il vit). Un vecteur 2D vit dans un plan, pas dans une ligne.",
    },
    "matrix": {
        "intuition": "Un tableau de nombres en grille. Si un vecteur est un point, "
                       "une matrice transforme tous les points en même temps : rotation, "
                       "zoom, projection.",
        "when_to_use": "Représenter une transformation linéaire (réseau de neurones = "
                          "empilement de matrices). Stocker des données tabulaires.",
        "example": "Tourner une image 2D de 30° = multiplier ses pixels par une matrice "
                     "2×2 spéciale.",
        "pitfall": "Multiplication de matrices PAS commutative : A·B ≠ B·A en général.",
    },
    "tensor_nd": {
        "intuition": "Une matrice généralisée à N dimensions. Une image RGB = tenseur 3D "
                       "(hauteur, largeur, 3 couleurs). Une vidéo = tenseur 4D (temps + image).",
        "when_to_use": "Quand tu as plus que des lignes×colonnes : images batch, séquences, "
                          "volumes 3D.",
        "example": "Un batch de 64 images couleur 224×224 = tenseur shape (64, 3, 224, 224).",
        "pitfall": "L'ordre des dimensions varie selon les frameworks (NCHW vs NHWC).",
    },
    "dot_product": {
        "intuition": "Mesure à quel point deux vecteurs 'pointent dans la même direction'. "
                       "Si positif et grand : alignés. Zéro : perpendiculaires. Négatif : "
                       "opposés.",
        "when_to_use": "Attention dans Transformers (qui regarde qui). Similarité entre "
                          "embeddings (recherche sémantique).",
        "example": "Query 'chat' · Key 'chien' = 0.8 (proches). Query 'chat' · Key 'voiture' "
                     "= 0.1 (lointains).",
        "pitfall": "Sensible à la magnitude. Si tu veux juste la direction → cosine_similarity.",
    },
    "svd": {
        "intuition": "Décompose une matrice en 3 morceaux qui révèlent sa structure cachée. "
                       "Comme analyser un mélange chimique pour voir les ingrédients dominants.",
        "when_to_use": "Compression (LoRA), réduction de dimension, débruitage. Cœur "
                          "mathématique de PCA.",
        "example": "Une matrice 1000×1000 de notes Netflix → SVD → 50 'genres latents' "
                     "qui capturent 95% des préférences.",
        "pitfall": "Coûteux O(N³). Pour N > 10000, utiliser truncated SVD ou randomized SVD.",
    },
    "pca": {
        "intuition": "Trouve les axes le long desquels tes données varient le plus. "
                       "Comme regarder un nuage de points de profil pour voir sa forme principale.",
        "when_to_use": "Visualiser data haute-dim en 2D/3D. Décorréler features. "
                          "Première analyse exploratoire toujours.",
        "example": "Iris dataset 4D (longueur/largeur pétales+sépales) → PCA 2D → 3 "
                     "espèces séparées visuellement.",
        "pitfall": "Linéaire seulement. Pour structures courbes (manifolds) → t-SNE ou UMAP.",
    },
    "cosine_similarity": {
        "intuition": "Angle entre deux flèches dans un espace : 1 = même direction, "
                       "0 = perpendiculaires, -1 = opposées. Ignore les longueurs.",
        "when_to_use": "Comparer embeddings (texte, image, son). Search engines, RAG, "
                          "recommandations.",
        "example": "cos('roi' - 'homme' + 'femme', 'reine') ≈ 0.9 — analogie classique "
                     "Word2Vec.",
        "pitfall": "Pas une vraie distance (ne respecte pas inégalité triangulaire).",
    },

    # ============================================================
    # Probabilités / Stats
    # ============================================================
    "gaussian": {
        "intuition": "La fameuse 'courbe en cloche'. La plupart des phénomènes naturels "
                       "(taille, erreurs de mesure) suivent cette forme.",
        "when_to_use": "Bruit. Initialisation poids NN. Prior dans VAE. Modélisation "
                          "d'incertitude.",
        "example": "Tailles adultes en France : N(μ=170cm, σ=8cm). 68% entre 162 et 178.",
        "pitfall": "Présupposer Gaussien par défaut. Vérifier avec un histogramme avant.",
    },
    "entropy": {
        "intuition": "Mesure du 'désordre' d'une distribution. Pile/face équitable = "
                       "entropie max. Si toujours pile = entropie nulle.",
        "when_to_use": "Loss classification (cross-entropy). Régularisation (max entropy). "
                          "Compression : Shannon limite codage.",
        "example": "Phrase 'le' a entropie faible (prédictible). Phrase 'antidisestablishmentarianism' "
                     "a entropie élevée.",
        "pitfall": "Unités : nats (log nat) vs bits (log₂). Choisir et rester cohérent.",
    },
    "kl_divergence": {
        "intuition": "Distance asymétrique entre 2 distributions. Mesure combien tu "
                       "perds en remplaçant P (vraie) par Q (approximation).",
        "when_to_use": "Loss VAE. RLHF. Knowledge distillation. Variational inference.",
        "example": "KL(N(0,1) || N(0.5, 1.2)) = 0.18. Plus c'est petit, plus Q ressemble à P.",
        "pitfall": "Asymétrique : KL(P||Q) ≠ KL(Q||P). Choisir le bon sens compte.",
    },
    "cross_entropy_loss": {
        "intuition": "Punit le modèle quand il met de la probabilité sur la mauvaise classe. "
                       "Si vérité = 'chat' et modèle dit chat à 90% → loss faible. "
                       "À 10% → loss énorme.",
        "when_to_use": "Loss standard pour classification multi-classes. Toujours combinée "
                          "avec softmax.",
        "example": "Modèle MNIST : sortie softmax [0.7, 0.1, 0.2, ...]. Vérité = classe 0. "
                     "Loss = -log(0.7) ≈ 0.36.",
        "pitfall": "Numériquement instable si appliquée après softmax séparé. "
                     "Utiliser nn.CrossEntropyLoss qui fait les deux fused.",
    },

    # ============================================================
    # Optimisation
    # ============================================================
    "gradient_descent": {
        "intuition": "Skier en aveugle dans une montagne : sentir la pente, descendre dans "
                       "la direction de la plus forte descente, recommencer.",
        "when_to_use": "Toute optimisation de fonction différentiable. Base de tout entraînement NN.",
        "example": "θ ← θ - 0.01 · ∇L. Lr=0.01 = pas de 1cm par étape sur la montagne.",
        "pitfall": "Lr trop grand → overshoot. Lr trop petit → mille ans. Lr scheduler aide.",
    },
    "adam": {
        "intuition": "Comme SGD mais avec mémoire : suit la vitesse moyenne (momentum) ET "
                       "adapte la taille du pas pour chaque paramètre individuellement.",
        "when_to_use": "Default pour 90% des cas. Particulièrement bon pour transformers, GANs.",
        "example": "Adam(lr=1e-3, betas=(0.9, 0.999)). Beta1 = inertie. Beta2 = stabilité.",
        "pitfall": "Worse generalization parfois vs SGD+momentum pour vision. AdamW corrige "
                     "le weight decay.",
    },
    "backpropagation": {
        "intuition": "Comme remonter une chaîne d'événements pour blâmer chaque responsable. "
                       "Chaque poids du réseau reçoit sa part de responsabilité dans l'erreur.",
        "when_to_use": "Cœur de l'entraînement de tout réseau différentiable.",
        "example": "Sortie fausse → propage erreur en arrière → chaque poids w reçoit dL/dw → "
                     "update.",
        "pitfall": "Gradients vanishing (RNN profonds) ou exploding. Solutions : LSTM, "
                     "ResNet skip, gradient clipping.",
    },

    # ============================================================
    # Activations
    # ============================================================
    "act_relu": {
        "intuition": "Filtre 'éteint les négatifs'. Si signal positif → passe. Si négatif → 0. "
                       "Simple comme bonjour, marche miraculeusement bien.",
        "when_to_use": "Default pour CNN, MLP cachées. Pas pour sortie classification (utiliser softmax).",
        "example": "ReLU(-2.3) = 0. ReLU(0.5) = 0.5. ReLU(100) = 100.",
        "pitfall": "Dying ReLU : neurones bloqués à 0 jamais réactivés. Solution : LeakyReLU.",
    },
    "act_gelu": {
        "intuition": "Version lissée de ReLU, comme une rampe avec coin arrondi. "
                       "Statistique : 'éteint le négatif proportionnellement à sa probabilité'.",
        "when_to_use": "Default Transformer modernes (BERT, GPT). Plus performant que ReLU "
                          "pour attention.",
        "example": "GELU(-1) ≈ -0.16, GELU(0) = 0, GELU(1) ≈ 0.84.",
        "pitfall": "Légèrement plus coûteux que ReLU. Pour edge devices, préférer ReLU.",
    },
    "act_softmax": {
        "intuition": "Convertit nombres bruts en probabilités qui somment à 1. "
                       "Comme normaliser des scores en pourcentages.",
        "when_to_use": "Sortie classification multi-classes. Attention scores. Reinforcement "
                          "learning policies.",
        "example": "softmax([1, 2, 3]) ≈ [0.09, 0.24, 0.67]. Le 3 reçoit 67% de la masse.",
        "pitfall": "Sensible aux valeurs extrêmes. Toujours sous-maximum trick : softmax(x - max).",
    },

    # ============================================================
    # Layers NN
    # ============================================================
    "dense": {
        "intuition": "Chaque neurone de la couche écoute TOUS les neurones de la couche "
                       "précédente. Comme un comité où chacun reçoit avis de tout le monde.",
        "when_to_use": "Tabular data, dernière couche classifier. Cœur des MLP.",
        "example": "Dense(in=784, out=128) sur MNIST flatten → 100k params (784·128 + 128).",
        "pitfall": "Trop dense → overfit + lent. Dropout aide. Préférer conv si données spatiales.",
    },
    "conv2d": {
        "intuition": "Filtre qui glisse sur l'image et reconnaît des motifs locaux : "
                       "bords, textures, formes. Comme passer une loupe-détecteur partout.",
        "when_to_use": "Images. Cœur des CNN. Aussi sur audio spectrogrammes, ECG.",
        "example": "Conv2d(3→32, k=3) sur image 224×224 → 32 cartes features. ~864 params "
                     "(3·32·9) — bien moins qu'un Dense équivalent.",
        "pitfall": "Padding/stride mal compris → tailles bizarres. Calcul out_size = "
                     "(in + 2p - k)/s + 1.",
    },
    "dropout_layer": {
        "intuition": "Éteint aléatoirement des neurones à l'entraînement. Comme entraîner "
                       "une équipe où chaque membre peut tomber malade : force les autres à "
                       "apprendre toutes les tâches.",
        "when_to_use": "Régularisation contre overfitting. Surtout dans MLP, peu en CNN.",
        "example": "Dropout(p=0.5) → 50% des activations mises à 0. Au test : 0% (tous actifs).",
        "pitfall": "Oublier model.eval() au test → dropout encore actif → résultats incohérents.",
    },
    "batchnorm2d": {
        "intuition": "Normalise les activations pour qu'elles aient moyenne 0, variance 1 "
                       "par batch. Comme recalibrer un microphone pour chaque chanson.",
        "when_to_use": "Stabilise et accélère training des CNN profonds. Standard depuis 2015.",
        "example": "BN avant ou après activation ? Débat. Original paper : avant ReLU.",
        "pitfall": "Mauvais en small batch (< 16) car stats peu fiables. Utiliser GroupNorm ou "
                     "LayerNorm.",
    },

    # ============================================================
    # Sequential / RNN
    # ============================================================
    "lstm_layer": {
        "intuition": "RNN avec 3 portes (forget, input, output) qui gèrent ce qu'on retient, "
                       "oublie et libère. Comme un cahier avec des post-it qu'on enlève "
                       "ou ajoute selon l'importance.",
        "when_to_use": "Séries temporelles, NLP avant Transformer. Encore utile en edge devices.",
        "example": "LSTM(input_size=64, hidden_size=256) traite séquence de 64-dim → "
                     "états cachés 256-dim.",
        "pitfall": "Lent à entraîner (pas parallélisable). Transformers ont remplacé en NLP.",
    },

    # ============================================================
    # Transformers
    # ============================================================
    "self_attention": {
        "intuition": "Chaque mot de la phrase regarde tous les autres mots et décide "
                       "qui est important pour le comprendre. Comme lire en sautant entre les mots.",
        "when_to_use": "NLP moderne, vision (ViT), audio, multimodal. Tout ce qui ressemble à séquence.",
        "example": "Phrase 'la banque est fermée'. Pour 'banque', attention forte sur 'fermée' → "
                     "désambigue 'institution' vs 'rive'.",
        "pitfall": "Coût O(N²) en séquence. Long contexte → FlashAttention ou sparse.",
    },
    "multi_head_attention": {
        "intuition": "Au lieu d'une seule attention, plusieurs (heads) en parallèle. "
                       "Chaque head se spécialise : une regarde syntaxe, une regarde sémantique...",
        "when_to_use": "Toujours en Transformer. Default 8 ou 12 heads.",
        "example": "MultiHead(768, 12 heads) → chaque head opère sur 64-dim sous-espaces.",
        "pitfall": "Trop de heads → chacune sous-dimensionnée. Tradeoff num_heads vs head_dim.",
    },
    "positional_encoding": {
        "intuition": "Ajoute info de position aux mots car attention ne sait pas l'ordre. "
                       "Comme numéroter chaque mot pour que le modèle sache 'le' vient avant 'chat'.",
        "when_to_use": "Tout Transformer non-autoregressive ou avec besoin de position absolue.",
        "example": "PE(pos, 2i) = sin(pos/10000^(2i/d)). Sinusoïdes de fréquences différentes.",
        "pitfall": "Encodage absolu mal pour long contexte. Préférer RoPE (rotary) pour 4k+.",
    },
    "moe": {
        "intuition": "Au lieu d'un seul gros expert, plusieurs petits experts spécialisés. "
                       "Un router choisit les 2 meilleurs pour chaque token.",
        "when_to_use": "Scale model size sans scale compute. Mixtral 8x7B = 8 experts.",
        "example": "MoE(8 experts, top_k=2) → 8x params, 2x compute par token.",
        "pitfall": "Balance load : sans contrainte, certains experts dominent (collapse).",
    },

    # ============================================================
    # Generative AI
    # ============================================================
    "vae": {
        "intuition": "Encode image en distribution gaussienne (μ, σ) puis sample latent z, "
                       "décode. Apprend espace latent continu et structuré.",
        "when_to_use": "Génération + représentation. Cœur Stable Diffusion (compress espace pixels).",
        "example": "VAE 784→32 (latent)→784 sur MNIST. Sample z=N(0,1)→generate digit aléatoire.",
        "pitfall": "Posterior collapse : z ignoré par decoder. β-VAE ou KL annealing aide.",
    },
    "gan": {
        "intuition": "Faussaire (G) vs détective (D) en compétition. Faussaire fabrique faux, "
                       "détective détecte. Amélioration mutuelle → faux indétectables.",
        "when_to_use": "Génération images haute qualité. Style transfer. Data augmentation.",
        "example": "DCGAN sur faces 64×64 → 60k epochs → visages photoréalistes.",
        "pitfall": "Mode collapse (G produit toujours la même chose). Instable training.",
    },
    "ddpm": {
        "intuition": "Apprends à dé-brouiller. Ajoute du bruit graduellement à une image "
                       "(forward), puis apprends à inverser le processus (reverse).",
        "when_to_use": "Génération haute qualité. Stable Diffusion, DALL-E 2. Plus stable que GAN.",
        "example": "1000 steps forward (image → bruit pur). Reverse 1000 steps → image générée.",
        "pitfall": "Inférence lente (N steps). DDIM/DPM++ accélèrent à 20 steps.",
    },

    # ============================================================
    # Concepts généraux
    # ============================================================
    "overfitting": {
        "intuition": "Le modèle mémorise les données d'entraînement par cœur au lieu de "
                       "comprendre les règles. Comme un élève qui apprend les réponses au QCM "
                       "mais ne saura pas généraliser.",
        "when_to_use": "Diagnostiquer quand train_acc >> val_acc. Symptôme bien connu.",
        "example": "Train accuracy 99%, val accuracy 65% = overfit massif.",
        "pitfall": "Solutions : plus de data, dropout, weight decay, early stopping, "
                     "augmentation.",
    },
    "embedding": {
        "intuition": "Convertit un objet discret (mot, ID utilisateur, catégorie) en vecteur "
                       "dense apprenable. Comme attribuer des coordonnées à chaque mot du dico.",
        "when_to_use": "Inputs catégoriels en NN. NLP. Recommandation. Recherche.",
        "example": "Vocabulaire 50k mots → matrix (50000, 256). Each row = embedding d'un mot.",
        "pitfall": "Dimension trop faible → manque expressivité. Trop grand → overfit.",
    },
    "latent_space": {
        "intuition": "Espace interne caché où le modèle a appris à représenter les données. "
                       "Comme la carte mentale qu'on construit en lisant un livre.",
        "when_to_use": "Compression, génération, similarité, exploration créative.",
        "example": "BERT a un latent space 768-dim. t-SNE de ses embeddings montre clusters "
                     "sémantiques.",
        "pitfall": "Latent space peu interprétable par défaut. β-VAE, ICA, disentanglement "
                     "explorent.",
    },
    "attention_concept": {
        "intuition": "Mécanisme qui décide où regarder dans l'input. Comme un projecteur qu'on "
                       "déplace dans un théâtre pour éclairer la scène importante.",
        "when_to_use": "Quand l'info pertinente est local mais sa position varie. NLP, vision, "
                          "audio.",
        "example": "Traduction 'le chat est noir' → 'the cat is black'. Pour 'cat', attention "
                     "forte sur 'chat'.",
        "pitfall": "Attention ≠ explanation parfaite. Attention rollouts mieux pour interprétation.",
    },
    "emergence_concept": {
        "intuition": "Comportement complexe qui apparaît à partir de règles simples. "
                       "Les boids se déplacent comme une nuée, sans 'chef' central.",
        "when_to_use": "Comprendre LLM capabilities, automates cellulaires, dynamics neurale.",
        "example": "Game of Life : 4 règles → planeurs, oscillateurs, gun.",
        "pitfall": "Emergence ≠ magie. Toujours réductible (en principe) aux règles.",
    },

    # ============================================================
    # Dynamical Systems
    # ============================================================
    "attractor": {
        "intuition": "Zone vers laquelle un système évolue toujours, peu importe d'où on part. "
                       "Comme une bille dans un bol : finit toujours au fond.",
        "when_to_use": "Analyse dynamiques NN, comportements RNN, weather prediction.",
        "example": "Lorenz attractor : papillon. Point fixe stable. Cycle limite (oscillateur).",
        "pitfall": "Attracteur étrange (chaos) : prédictible court terme, chaotique long terme.",
    },
    "lyapunov_exp": {
        "intuition": "Mesure à quel point 2 trajectoires proches divergent. λ>0 = chaos "
                       "(papillon → ouragan). λ=0 = stable. λ<0 = converge.",
        "when_to_use": "Caractériser dynamiques NN, prévisibilité, stabilité d'entraînement.",
        "example": "Lorenz λ_max ≈ 0.9 nats/sec : prédiction perd 1 bit toutes les ~0.7s.",
        "pitfall": "Mesurer demande long trajet + algorithme Rosenstein ou Wolf. Pas trivial.",
    },
    "reaction_diffusion": {
        "intuition": "Deux substances qui réagissent et diffusent en même temps. Turing a montré "
                       "que ça produit des motifs : taches sur léopards, spirales coquillages.",
        "when_to_use": "Modèle morphogenèse. Génère patterns naturels. Comprendre auto-organisation.",
        "example": "Gray-Scott avec D_u=0.16, f=0.035, k=0.065 → spots U et V.",
        "pitfall": "Numériquement sensible. Pas trop grand dt sinon instabilité.",
    },
    "cellular_automaton": {
        "intuition": "Grille de cellules qui s'updatent en regardant juste leurs voisines. "
                       "Règles simples → comportements complexes. Wolfram a classifié 4 classes.",
        "when_to_use": "Étudier emergence, calcul universel, biologie computationnelle, art génératif.",
        "example": "Rule 110 (Wolfram) est Turing-complete. Peut simuler n'importe quel calcul.",
        "pitfall": "Pas tous CA sont intéressants. La plupart convergent en 0 ou bruit aléatoire.",
    },

    # ============================================================
    # Interpretability
    # ============================================================
    "shap": {
        "intuition": "Distribue 'crédit' de la prédiction entre les features. Comme partager "
                       "le mérite d'une victoire entre les joueurs d'une équipe (théorie des jeux).",
        "when_to_use": "Expliquer pourquoi un modèle a prédit X. Audit, fairness, debug.",
        "example": "Prédiction crédit refusé. SHAP : -0.4 dette, -0.2 revenu, +0.1 âge → "
                     "principalement dette.",
        "pitfall": "Valeurs SHAP correlation ≠ causation. Reste explication post-hoc.",
    },
    "attention_viz": {
        "intuition": "Visualise quel input le modèle 'regarde' pour produire chaque sortie. "
                       "BertViz montre lignes entre tokens : épaisses = forte attention.",
        "when_to_use": "Debug NLP. Comprendre quoi attend le modèle.",
        "example": "BERT [CLS] attend pas mal sur tous tokens. Heads spécialisés : "
                     "syntaxe vs sémantique.",
        "pitfall": "Attention ≠ explanation fidèle. Travaux récents critiquent.",
    },

    # ============================================================
    # Phase Space / Dynamoscope
    # ============================================================
    "phase_space_3d": {
        "intuition": "Plot 3D où chaque axe = variable du système. Une trajectoire = "
                       "l'évolution. Lorenz fait un papillon. Pendule oscille en ellipse.",
        "when_to_use": "Visualiser dynamiques continues (Lorenz, pendule, neural activations).",
        "example": "Lorenz : (x, y, z) → 3D trajectory 'attractor étrange' aux ailes papillon.",
        "pitfall": "Si données > 3D, projeter via PCA peut perdre topologie. UMAP préserve mieux.",
    },
    "takens_embedding": {
        "intuition": "Théorème Takens 1981 : reconstruction de l'attracteur à partir d'UNE "
                       "seule variable observable + délais. Magique.",
        "when_to_use": "Quand tu n'as qu'un signal 1D (ECG, audio) mais veux explorer dynamique.",
        "example": "x(t), x(t-10), x(t-20) → trajectoire 3D qui ressemble à l'attracteur original.",
        "pitfall": "Choix de τ et m crucial. τ = premier minimum mutual info. m = false nearest "
                     "neighbors.",
    },

    # ============================================================
    # Adaptive / Metamorph
    # ============================================================
    "critical_grow_net": {
        "intuition": "Réseau qui grandit ou se rétracte pendant l'entraînement pour rester "
                       "près du 'bord du chaos' (criticality). Comme un cerveau qui développe "
                       "ses neurones où c'est utile.",
        "when_to_use": "Architecture search runtime. Recherche métamorphique. "
                          "Self-organizing systems.",
        "example": "M1 ARK avec target_beta=-2 (SOC) atteint 79% MNIST en 2 epochs avec "
                     "mutations actives.",
        "pitfall": "Coût compute mutation. Faut Observer + Trigger bien réglés.",
    },
    "dyadic_mirror": {
        "intuition": "Deux agents qui apprennent à se prédire mutuellement (Mead-Vygotsky). "
                       "Modélise la conscience comme dialogue interne.",
        "when_to_use": "Multi-agent learning. Self-supervised via prédiction mutuelle.",
        "example": "M5 ARK : 2 SelfRefAgent partagent état latent + cross-prediction loss.",
        "pitfall": "Train stabilité difficile. Cross-prediction peut collapse vers identity.",
    },

    # ============================================================
    # Reinforcement Learning
    # ============================================================
    "rl_agent": {
        "intuition": "Entité qui prend des décisions. Comme un joueur d'échecs : observe la "
                       "situation, choisit un coup, observe le résultat, apprend.",
        "when_to_use": "Problèmes séquentiels avec feedback différé : jeux, robotique, "
                          "publicité.",
        "example": "AlphaGo agent : input = position, output = move probability.",
        "pitfall": "Sample inefficient. Demande millions d'épisodes en général.",
    },
    "ppo": {
        "intuition": "Améliore policy mais pas trop vite (trust region). Comme apprendre à "
                       "conduire : adapter mais éviter ré-écrire tout le permis.",
        "when_to_use": "Default RL moderne. RLHF pour LLM fine-tuning. ChatGPT alignment.",
        "example": "PPO clip=0.2 limite ratio nouvelle/ancienne policy entre [0.8, 1.2].",
        "pitfall": "Hyperparams sensibles. Reward shaping critique.",
    },

    # ============================================================
    # Theory
    # ============================================================
    "scaling_laws": {
        "intuition": "Empiriquement : loss ∝ N^α. Plus tu donnes compute/data/params, mieux "
                       "ça marche. Loi de puissance révélée par Kaplan/Chinchilla.",
        "when_to_use": "Planifier expériences. Allocation budget compute optimale.",
        "example": "Chinchilla : N tokens ≈ 20 × N_params pour optimal. 70B model → ~1.4T tokens.",
        "pitfall": "Hold seulement dans certains régimes. Saturation possible. Pas magique.",
    },
    "grokking": {
        "intuition": "Le modèle stagne au plateau d'overfit, puis BOOM généralisation soudaine "
                       "après long entraînement. Comme 'click' soudain de compréhension.",
        "when_to_use": "Hypothèse mécanistique pour LLM emergence. Étudier transitions de phase.",
        "example": "Modular arithmetic : memorize 90% accuracy 1000 steps. Puis grok → 100% "
                     "vers 10000 steps.",
        "pitfall": "Pas toujours grok. Demande weight decay + long entraînement.",
    },
}
