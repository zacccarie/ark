"""Glossaire massif ARK — données structurées.

Source : ai_architecture_builder_master_glossary_and_integration_prompt.md
Phase 29 : 500+ entrées organisées en 23 familles UI.

Chaque entrée :
- name : id snake_case
- title : display
- kind : 'node' (PyTorch implementable) | 'tool' (lib externe) | 'concept' (def only) | 'paradigm' (strategy)
- short : 1-line
- long : paragraphe
- params : list[(name, kind, default, options?)] — pour kind=node
- equation : LaTeX-friendly string (optionnel)
- related : tuple noms liés
"""
from __future__ import annotations

# ============================================================
# Catégories UI hiérarchiques (menus palette)
# ============================================================

CATEGORIES = [
    "Foundations",
    "Neural Networks",
    "Sequential",
    "Transformers",
    "NLP",
    "Computer Vision",
    "Generative AI",
    "Reinforcement Learning",
    "Graph Systems",
    "Unsupervised",
    "Evolutionary Systems",
    "Dynamical Systems",
    "Biological Inspiration",
    "Cognitive Architectures",
    "Interpretability",
    "Optimization",
    "Training",
    "Distributed Systems",
    "Hardware",
    "Simulation",
    "Emergence",
    "Multi-Agent Systems",
    "Adaptive Architectures",
    "Experimental Research",
    "Math",
]


# ============================================================
# Concepts massifs (auto-générés via boucles ci-dessous)
# ============================================================

CONCEPTS: list[dict] = []


def C(name, title, category, kind="concept", short="", long="", params=None,
       equation=None, related=None):
    CONCEPTS.append({
        "name": name, "title": title, "category": category, "kind": kind,
        "short": short, "long": long, "params": params or [],
        "equation": equation, "related": tuple(related or []),
    })


# ============ FOUNDATIONS — Algèbre linéaire ============

C("vector", "Vector", "Foundations", "node",
  "Tenseur 1D.", "Tableau ordonné de scalaires. Base de tout calcul ML.",
  params=[("size", "int", 128)])

C("matrix", "Matrix", "Foundations", "node",
  "Tenseur 2D.", "Représente transformation linéaire R^n → R^m. Multiplication = composition.",
  params=[("rows", "int", 64), ("cols", "int", 64)])

C("tensor_nd", "Tensor (N-D)", "Foundations", "node",
  "Tenseur N-D générique.", "Généralisation matrice à N dimensions. Coeur PyTorch.",
  params=[("shape", "str", "1,3,224,224")])

C("dot_product", "Dot Product", "Foundations", "node",
  "Produit scalaire.", "Σxi·yi. Mesure d'alignement. Base de l'attention QK^T.",
  equation="x·y = Σ xi·yi")

C("outer_product", "Outer Product", "Foundations", "node",
  "Produit tensoriel.", "x⊗y = xy^T. Rang 1. Construit rang supérieur par somme.")

C("svd", "SVD", "Foundations", "node",
  "Décomposition en valeurs singulières.", "A = UΣV^T. Base low-rank approximation, PCA, latent extraction.",
  equation="A = U Σ V^T",
  params=[("full_matrices", "bool", False), ("k", "int", 0)])

C("pca", "PCA", "Foundations", "node",
  "Analyse en composantes principales.", "Projection sur directions max variance. SVD du data centré.",
  params=[("n_components", "int", 2)])

C("eigh", "Eigenvalues / Eigenvectors", "Foundations", "node",
  "Décomposition spectrale symétrique.", "Av = λv. Base de stabilité dynamique + analyse spectrale.",
  equation="A v = λ v")

C("qr_decomp", "QR Decomposition", "Foundations", "node",
  "A = QR.", "Q orthogonal, R triangulaire sup. Pour least-squares + Gram-Schmidt.")

C("lu_decomp", "LU Decomposition", "Foundations", "node",
  "A = LU.", "L triangulaire inf, U sup. Pour résolution Ax=b.")

C("l1_norm", "L1 Norm", "Foundations", "node",
  "Σ|xi|.", "Sparsité induite. Lasso regularization.")

C("l2_norm", "L2 Norm", "Foundations", "node",
  "√Σxi².", "Norme euclidienne. Weight decay = pénalité L2.")

C("cosine_similarity", "Cosine Similarity", "Foundations", "node",
  "cos(x,y) = x·y / (||x||||y||).", "Invariant à la magnitude. Retrieval, embeddings.",
  equation="cos(x,y) = (x·y)/(||x|| ||y||)")

C("projection", "Projection", "Foundations", "node",
  "p = (x·u/||u||²)u.", "Projette x sur direction u. Base de PCA/Gram-Schmidt.")

C("orthogonality", "Orthogonality", "Foundations", "concept",
  "x·y = 0.", "Independence directionnelle. Base de bonnes représentations.")

C("matrix_rank", "Matrix Rank", "Foundations", "concept",
  "Nombre de vecteurs indépendants.", "Capacité expressive. Low-rank = compression.")

C("spectral_decomposition", "Spectral Decomposition", "Foundations", "concept",
  "Diagonalisation matrice symétrique.", "Spectre dynamique. ARK target_beta=-2 = SOC.")

C("low_rank_approx", "Low-Rank Approximation", "Foundations", "node",
  "Garde k singular values top.", "LoRA = rank-r delta. Compression sans perte massive.",
  params=[("rank", "int", 8)])

# ============ FOUNDATIONS — Probabilités ============

C("gaussian", "Gaussian Distribution", "Foundations", "node",
  "N(μ,σ²).", "Distribution normale. CLT. Base VAE prior.",
  params=[("mean", "float", 0.0), ("std", "float", 1.0), ("shape", "str", "128")])

C("bernoulli", "Bernoulli", "Foundations", "node",
  "P(X=1)=p.", "Binaire. Dropout = Bernoulli mask.",
  params=[("p", "float", 0.5), ("shape", "str", "128")])

C("multinomial", "Multinomial", "Foundations", "node",
  "K-classes probabilities.", "Softmax output = multinomial. Categorical sampling.",
  params=[("n_classes", "int", 10), ("temperature", "float", 1.0)])

C("poisson", "Poisson", "Foundations", "node",
  "P(k)=λ^k e^-λ/k!.", "Comptages, spikes neurones, événements rares.",
  params=[("rate", "float", 1.0)])

C("exponential_dist", "Exponential", "Foundations", "node",
  "p(x)=λe^-λx.", "Temps inter-événements. Memoryless.",
  params=[("rate", "float", 1.0)])

C("entropy", "Entropy H(X)", "Foundations", "node",
  "−Σp log p.", "Mesure d'incertitude. Max = uniform.",
  equation="H(X) = -Σ p(x) log p(x)")

C("kl_divergence", "KL Divergence", "Foundations", "node",
  "KL(P||Q) = Σp log(p/q).", "Asymétrique. Base VAE loss, RLHF.",
  equation="D_KL(P||Q) = Σ p log(p/q)")

C("js_divergence", "JS Divergence", "Foundations", "node",
  "Jensen-Shannon = symétrisée KL.", "Borné [0,1]. GAN original loss.")

C("cross_entropy_loss", "Cross-Entropy", "Foundations", "node",
  "−Σ y log(ŷ).", "Loss classification standard. Stable softmax+log fused.",
  equation="H(y,ŷ) = -Σ y_i log ŷ_i")

C("mutual_info", "Mutual Information", "Foundations", "node",
  "I(X;Y) = H(X)+H(Y)−H(X,Y).", "Dépendance non-linéaire. InfoBottleneck.")

C("mle", "Maximum Likelihood Estimation", "Foundations", "concept",
  "argmax θ P(D|θ).", "Principe inférence statistique classique.")

C("map_estimate", "MAP Estimation", "Foundations", "concept",
  "argmax θ P(θ|D).", "Bayes + prior. MLE + régularisation = MAP.")

C("bayes_inference", "Bayesian Inference", "Foundations", "paradigm",
  "P(θ|D) ∝ P(D|θ)P(θ).", "Quantifie incertitude. Posterior over params.")

C("monte_carlo", "Monte Carlo", "Foundations", "node",
  "Estimation par échantillonnage.", "E[f] ≈ (1/N)Σf(xi). Brute mais universel.",
  params=[("n_samples", "int", 1000)])

C("mcmc", "MCMC", "Foundations", "tool",
  "Markov Chain Monte Carlo.", "Hamiltonian, Metropolis. Sample posterior intractable.")

C("variance", "Variance", "Foundations", "node",
  "Var(X) = E[(X−μ)²].", "Dispersion. Bias-variance tradeoff.")

C("covariance", "Covariance", "Foundations", "node",
  "Cov(X,Y).", "Co-variation. Base PCA.")

C("correlation", "Correlation", "Foundations", "node",
  "ρ = Cov/(σxσy).", "Cov normalisée. ρ ∈ [-1,1].")

C("bootstrap", "Bootstrap", "Foundations", "tool",
  "Resample avec remise.", "Estime intervalles confiance non-paramétrique.")

C("naive_bayes", "Naive Bayes", "Foundations", "node",
  "P(y|x) ∝ P(y)Π P(xi|y).", "Indépendance conditionnelle. Baseline texte rapide.")

C("hmm", "Hidden Markov Model", "Foundations", "node",
  "État caché Markov.", "Viterbi, Baum-Welch. Pré-RNN séquentiel.")

C("markov_chain", "Markov Chain", "Foundations", "concept",
  "P(Xt+1|Xt).", "État futur dépend seulement présent. Base MCMC + RL.")

# ============ FOUNDATIONS — Optimisation ============

C("gradient_descent", "Gradient Descent", "Optimization", "node",
  "θ ← θ − η∇L.", "Base. Garanties convex. Plateau non-convex.",
  params=[("lr", "float", 1e-2)])

C("sgd_optim", "SGD", "Optimization", "node",
  "Mini-batch stochastic.", "Noise = régularisation implicite.",
  params=[("lr", "float", 1e-2), ("momentum", "float", 0.9)])

C("adamw", "AdamW", "Optimization", "node",
  "Adam + decoupled weight decay.", "Transformer default. Plus stable que Adam+L2.",
  params=[("lr", "float", 1e-3), ("weight_decay", "float", 0.01)])

C("rmsprop", "RMSProp", "Optimization", "node",
  "Adaptive lr per-param (Hinton).", "Bon pour RNN.",
  params=[("lr", "float", 1e-3), ("alpha", "float", 0.99)])

C("adagrad", "Adagrad", "Optimization", "node",
  "lr/√Σg².", "Adaptive. Lent en pratique long terme.",
  params=[("lr", "float", 1e-2)])

C("adafactor", "Adafactor", "Optimization", "node",
  "Memory-efficient Adam.", "Factorisé. T5 default.",
  params=[("lr", "float", 1e-3)])

C("lion_optim", "Lion", "Optimization", "node",
  "Sign momentum.", "Google 2023. Plus rapide convergence parfois.",
  params=[("lr", "float", 1e-4)])

C("nesterov", "Nesterov Momentum", "Optimization", "concept",
  "Look-ahead momentum.", "Anticipation = convergence plus rapide.")

C("gradient_clipping", "Gradient Clipping", "Training", "node",
  "Borne ||∇|| ≤ c.", "Évite explosion RNN/Transformer.",
  params=[("max_norm", "float", 1.0)])

C("weight_decay", "Weight Decay", "Optimization", "node",
  "L2 sur weights.", "Régularisation. AdamW = découplé du gradient.",
  params=[("decay", "float", 0.01)])

C("early_stopping", "Early Stopping", "Training", "node",
  "Stop si val plateau.", "Anti-overfit.",
  params=[("patience", "int", 5)])

C("cosine_decay", "Cosine LR Decay", "Optimization", "node",
  "lr = lr_max·0.5(1+cos(πt/T)).", "Warmup + cosine = SOTA modern.",
  params=[("T_max", "int", 1000)])

C("warmup", "LR Warmup", "Optimization", "node",
  "Ramp lr from 0.", "Stabilise début training Transformer.",
  params=[("warmup_steps", "int", 500)])

C("hessian", "Hessian", "Optimization", "concept",
  "∂²L/∂θ². Matrice courbure.", "Newton methods. Trop coûteux full.")

C("jacobian", "Jacobian", "Optimization", "concept",
  "∂y/∂x.", "Sensibilité output→input. Backprop = JVP.")

C("lagrangian", "Lagrangian", "Optimization", "concept",
  "L = f + λg.", "Contraintes. KKT conditions.")

C("convex_opt", "Convex Optimization", "Optimization", "concept",
  "f convexe → minimum global garanti.", "Boyd & Vandenberghe.")

C("variational_methods", "Variational Methods", "Optimization", "concept",
  "Min/Max sur fonctions.", "VAE ELBO, variational inference.")

C("hyperparam_opt", "Hyperparameter Optimization", "Optimization", "tool",
  "Grid/Random/Bayes/ES.", "Optuna, Ray Tune.")

C("bayesian_opt", "Bayesian Optimization", "Optimization", "tool",
  "GP surrogate + acquisition.", "Sample-efficient.")

C("evolution_strategies", "Evolution Strategies", "Evolutionary Systems", "node",
  "(μ,λ)-ES.", "Black-box, parallelizable. OpenAI ES.",
  params=[("pop_size", "int", 50), ("sigma", "float", 0.1)])

# ============ PARADIGMES APPRENTISSAGE ============

for p in ["supervised", "unsupervised", "semi_supervised", "self_supervised",
            "reinforcement", "meta_learning", "transfer", "continual",
            "online", "federated", "curriculum", "multi_task", "active_learning",
            "few_shot", "zero_shot", "one_shot"]:
    title = p.replace("_", " ").title() + " Learning"
    C(f"paradigm_{p}", title, "Training", "paradigm",
      f"{title} paradigm.", f"Stratégie d'apprentissage : {title}.")

# ============ CONCEPTS GENERAUX ML ============

ml_concepts = [
    ("generalization", "Generalization", "Performance sur unseen data."),
    ("overfitting", "Overfitting", "Mémorise train, échoue val."),
    ("underfitting", "Underfitting", "Capacité insuffisante."),
    ("embedding", "Embedding", "Représentation dense apprise."),
    ("latent_space", "Latent Space", "Espace interne compressé."),
    ("attention_concept", "Attention", "Pondération dépendante du contenu."),
    ("bottleneck", "Bottleneck", "Goulot informationnel."),
    ("sparse_computation", "Sparse Computation", "Seulement subset actif."),
    ("dynamic_routing", "Dynamic Routing", "Path adapté au input."),
    ("compression", "Compression", "Réduction taille modèle."),
    ("distillation", "Distillation", "Teacher → student."),
    ("quantization", "Quantization", "fp32 → int8/int4."),
    ("pruning", "Pruning", "Suppression weights faibles."),
    ("emergence_concept", "Emergence", "Comportement complexe via règles simples."),
    ("scaling_laws", "Scaling Laws", "L ∝ N^α (Chinchilla/Hoffmann)."),
    ("catastrophic_forgetting", "Catastrophic Forgetting", "Nouvelle tâche écrase ancienne."),
    ("plasticity", "Plasticity", "Capacité modification post-training."),
]
for name, title, desc in ml_concepts:
    C(name, title, "Foundations", "concept", desc, desc)


# ============ ACTIVATIONS ============

for fn in ["sigmoid", "tanh", "relu", "leaky_relu", "elu", "gelu", "silu", "mish",
             "softmax", "softplus"]:
    C(f"act_{fn}", fn.upper().replace("_", " "), "Neural Networks", "node",
      f"Activation {fn}.",
      f"Fonction d'activation {fn} appliquée élément-wise.",
      params=[("inplace", "bool", False)] if fn in ("relu", "leaky_relu") else [])


# ============ LAYERS NN ============

nn_layers = [
    ("conv1d", "Conv1D", "Convolution 1D (audio, séquences).",
     [("in_channels", "int", 1), ("out_channels", "int", 32), ("kernel_size", "int", 3)]),
    ("conv2d", "Conv2D", "Convolution 2D (image).",
     [("in_channels", "int", 3), ("out_channels", "int", 32), ("kernel_size", "int", 3),
      ("stride", "int", 1), ("padding", "int", 0)]),
    ("conv3d", "Conv3D", "Convolution 3D (vidéo, volumes).",
     [("in_channels", "int", 1), ("out_channels", "int", 16), ("kernel_size", "int", 3)]),
    ("conv_transpose2d", "ConvTranspose2D", "Upsampling apprenable.",
     [("in_channels", "int", 32), ("out_channels", "int", 16), ("kernel_size", "int", 4),
      ("stride", "int", 2)]),
    ("maxpool2d", "MaxPool2D", "Max sur fenêtre.", [("kernel_size", "int", 2)]),
    ("avgpool2d", "AvgPool2D", "Moyenne sur fenêtre.", [("kernel_size", "int", 2)]),
    ("batchnorm1d", "BatchNorm1D", "Normalisation batch 1D.", [("num_features", "int", 64)]),
    ("batchnorm2d", "BatchNorm2D", "Normalisation batch 2D.", [("num_features", "int", 32)]),
    ("layernorm", "LayerNorm", "Norm par sample (Transformer).", [("normalized_shape", "int", 768)]),
    ("groupnorm", "GroupNorm", "Norm par groupe channels.", [("num_groups", "int", 8), ("num_channels", "int", 32)]),
    ("dropout_layer", "Dropout", "Mask aléatoire training.", [("p", "float", 0.5)]),
    ("dropout2d_layer", "Dropout2D", "Mask channel.", [("p", "float", 0.5)]),
    ("embedding_layer", "Embedding", "Table lookup id→vec.",
     [("num_embeddings", "int", 10000), ("embedding_dim", "int", 768)]),
    ("flatten_layer", "Flatten", "Reshape vers (B, -1).", []),
    ("identity_layer", "Identity", "Pass-through.", []),
]
for n, t, s, p in nn_layers:
    C(n, t, "Neural Networks", "node", s, s, params=p)


# ============ ARCHITECTURES SEQUENTIELLES ============

seq_layers = [
    ("rnn_layer", "RNN", "Récurrent vanilla.",
     [("input_size", "int", 128), ("hidden_size", "int", 256), ("num_layers", "int", 1)]),
    ("lstm_layer", "LSTM", "Long Short-Term Memory.",
     [("input_size", "int", 128), ("hidden_size", "int", 256), ("num_layers", "int", 1)]),
    ("gru_layer", "GRU", "Gated Recurrent Unit.",
     [("input_size", "int", 128), ("hidden_size", "int", 256), ("num_layers", "int", 1)]),
    ("bidirectional_rnn", "Bidirectional RNN", "Forward + backward.",
     [("input_size", "int", 128), ("hidden_size", "int", 256)]),
    ("temporal_conv", "Temporal Convolution", "Causal 1D conv pour séquences.",
     [("channels", "int", 64), ("kernel_size", "int", 3), ("dilation", "int", 1)]),
    ("state_space_model", "State Space Model", "ẋ=Ax+Bu, y=Cx+Du (Mamba base).",
     [("d_model", "int", 256), ("d_state", "int", 16)]),
    ("reservoir_computing", "Reservoir Computing", "Echo state network.",
     [("reservoir_size", "int", 500), ("spectral_radius", "float", 0.9)]),
]
for n, t, s, p in seq_layers:
    C(n, t, "Sequential", "node", s, s, params=p)

C("seq2seq", "Seq2Seq", "Sequential", "node",
  "Encoder-decoder pré-Transformer.", "Encode séquence source → décode cible.",
  params=[("hidden", "int", 256)])

C("teacher_forcing", "Teacher Forcing", "Sequential", "concept",
  "Feed ground truth au décodeur.", "Accélère convergence training Seq2Seq.")

C("neural_turing_machine", "Neural Turing Machine", "Sequential", "node",
  "RNN + memoire externe différentiable.",
  "Read/write heads sur memory bank. Apprend algorithmes simples.",
  params=[("memory_size", "int", 128), ("memory_dim", "int", 32)])

C("differentiable_neural_computer", "DNC", "Sequential", "concept",
  "NTM++ avec linked list.", "DeepMind 2016.")

C("memory_networks", "Memory Networks", "Sequential", "concept",
  "External memory + attention.", "Sukhbaatar et al.")


# ============ TRANSFORMERS ============

C("self_attention", "Self-Attention", "Transformers", "node",
  "softmax(QK^T/√d)V.", "Coeur Transformer. Q,K,V depuis même input.",
  equation="A = softmax(QK^T/√d_k) V",
  params=[("d_model", "int", 768), ("d_k", "int", 64)])

C("multi_head_attention", "Multi-Head Attention", "Transformers", "node",
  "h heads parallèles.", "Concat([head1...headh])W^O.",
  params=[("d_model", "int", 768), ("n_heads", "int", 12)])

C("cross_attention", "Cross-Attention", "Transformers", "node",
  "Q from decoder, K,V from encoder.", "Encoder-decoder Transformer.",
  params=[("d_model", "int", 768), ("n_heads", "int", 12)])

C("positional_encoding", "Positional Encoding", "Transformers", "node",
  "Sinusoidal absolute.", "PE(pos,2i)=sin(pos/10000^(2i/d)).",
  params=[("d_model", "int", 768), ("max_len", "int", 2048)])

C("rotary_embeddings", "RoPE", "Transformers", "node",
  "Rotation matrix per position.", "Relative encoding. Llama/Mistral default.",
  params=[("d_model", "int", 768), ("base", "float", 10000.0)])

C("kv_cache", "KV Cache", "Transformers", "concept",
  "Cache K,V across decode steps.", "O(N) inference vs O(N²) recompute.")

C("causal_mask", "Causal Masking", "Transformers", "node",
  "Upper-triangular -inf.", "Empêche fuite future en decoder.",
  params=[("seq_len", "int", 512)])

C("tokenizer_bpe", "BPE Tokenizer", "Transformers", "tool",
  "Byte-pair encoding.", "GPT2/Llama. Sub-word.")

C("flash_attention", "FlashAttention", "Transformers", "node",
  "IO-aware exact attention.", "Tri Dao. ~3x faster, ~10x memory.",
  params=[("causal", "bool", True)])

C("grouped_query_attention", "GQA", "Transformers", "node",
  "n_kv_heads < n_heads.", "Llama2 70B+. Memory savings KV cache.",
  params=[("n_heads", "int", 32), ("n_kv_heads", "int", 8)])

C("multi_query_attention", "MQA", "Transformers", "node",
  "1 KV head shared.", "Extrême GQA. Inference rapide.",
  params=[("n_heads", "int", 32)])

C("sparse_attention", "Sparse Attention", "Transformers", "node",
  "Mask O(N√N) ou O(N log N).", "Longformer, BigBird.",
  params=[("window_size", "int", 256)])

C("linear_attention", "Linear Attention", "Transformers", "node",
  "φ(Q)φ(K)^T V.", "O(N). RWKV, Performer.",
  params=[("d_model", "int", 768)])

# Models familles (concept)
models = ["GPT", "BERT", "T5", "UL2", "PaLM", "Gemini", "Claude", "Llama",
          "Mistral", "Mixtral", "DeepSeek", "Qwen", "Falcon", "BLOOM", "Phi",
          "RWKV", "RetNet", "Mamba", "Hyena"]
for m in models:
    C(f"model_{m.lower()}", m, "Transformers", "concept",
      f"Famille modèle {m}.", f"Modèle Transformer (ou successeur) : {m}.")

C("moe", "Mixture of Experts", "Transformers", "node",
  "Gated routing → N experts → weighted sum.",
  "Sparse activation. Mixtral, Switch Transformer.",
  params=[("n_experts", "int", 8), ("top_k", "int", 2)])

# Decoding strategies
C("beam_search", "Beam Search", "Transformers", "node",
  "Top-k partial sequences.", "Trade-off width/quality.",
  params=[("beam_size", "int", 5)])

C("top_k_sampling", "Top-k Sampling", "Transformers", "node",
  "Sample within top-k logits.", "Diversité contrôlée.",
  params=[("k", "int", 50), ("temperature", "float", 1.0)])

C("top_p_sampling", "Top-p (Nucleus) Sampling", "Transformers", "node",
  "Sample within cumsum ≤ p.", "Adaptive vocabulary size.",
  params=[("p", "float", 0.9), ("temperature", "float", 1.0)])

C("temperature_sampling", "Temperature", "Transformers", "node",
  "softmax(logits/T).", "T<1 = sharp, T>1 = diffuse.",
  params=[("temperature", "float", 1.0)])

C("speculative_decoding", "Speculative Decoding", "Transformers", "concept",
  "Draft model + verify.", "2-3x speedup inference.")

C("rag_pipeline", "RAG", "Transformers", "node",
  "Retrieval + generation.", "Embed query → kNN docs → context inject → generate.",
  params=[("top_k", "int", 5)])


# ============ NLP ============

nlp_concepts = [
    ("bow", "Bag of Words", "Frequency vector.", "node"),
    ("tfidf", "TF-IDF", "Term frequency · inverse doc freq.", "node"),
    ("word2vec", "Word2Vec", "Embeddings dense via skipgram/CBOW.", "tool"),
    ("glove", "GloVe", "Global Vectors. Co-occurrence matrix factorization.", "tool"),
    ("fasttext", "FastText", "Subword embeddings.", "tool"),
    ("sentence_embeddings", "Sentence Embeddings", "Phrase → vec dense.", "tool"),
    ("semantic_search", "Semantic Search", "Embed query + kNN.", "node"),
    ("ner", "Named Entity Recognition", "Identifie entités.", "concept"),
    ("translation", "Machine Translation", "Seq2Seq cross-lingue.", "concept"),
    ("summarization", "Summarization", "Long → court.", "concept"),
    ("question_answering", "Question Answering", "Q+context → A.", "concept"),
    ("parsing_syntax", "Syntactic Parsing", "Arbre syntaxique.", "concept"),
    ("information_retrieval", "Information Retrieval", "Query → docs ranked.", "concept"),
]
for n, t, s, k in nlp_concepts:
    C(n, t, "NLP", k, s, s)


# ============ COMPUTER VISION ============

cv_archs = ["LeNet", "AlexNet", "VGG", "ResNet", "DenseNet", "EfficientNet",
            "ConvNeXt", "ViT", "Swin", "U-Net", "YOLO", "Faster-RCNN", "Mask-RCNN", "SAM"]
for a in cv_archs:
    nm = a.lower().replace("-", "_")
    C(f"arch_{nm}", a, "Computer Vision", "concept",
      f"Architecture {a}.", f"Architecture vision : {a}.")

cv_tasks = ["classification", "segmentation", "object_detection", "ocr",
              "pose_estimation", "optical_flow", "depth_estimation",
              "super_resolution", "image_captioning"]
for t in cv_tasks:
    title = t.replace("_", " ").title()
    C(f"task_{t}", title, "Computer Vision", "concept",
      f"Tâche vision : {title}.", f"Tâche {title}.")


# ============ GENERATIVE AI ============

gen_models = [
    ("gan", "GAN", "Generator vs Discriminator.", "node"),
    ("dcgan", "DCGAN", "Deep Conv GAN.", "concept"),
    ("stylegan", "StyleGAN", "Style-based.", "concept"),
    ("cyclegan", "CycleGAN", "Unpaired domain transfer.", "concept"),
    ("ddpm", "DDPM", "Denoising Diffusion Probabilistic Model.", "node"),
    ("stable_diffusion", "Stable Diffusion", "Latent diffusion.", "template"),
    ("flux_model", "Flux", "Rectified flow text-to-image (Black Forest).", "concept"),
    ("midjourney", "Midjourney", "Service propriétaire.", "concept"),
    ("dalle", "DALL·E", "OpenAI text-to-image.", "concept"),
    ("latent_diffusion", "Latent Diffusion", "Diffusion en espace VAE.", "concept"),
    ("flow_matching", "Flow Matching", "Apprend champ de vecteurs.", "concept"),
    ("rectified_flows", "Rectified Flows", "Straighten transport.", "concept"),
    ("autoregressive_gen", "Autoregressive Generation", "P(x) = Π P(xi|x<i).", "concept"),
]
for n, t, s, k in gen_models:
    C(n, t, "Generative AI", k, s, s)

C("guidance_scale", "Guidance Scale", "Generative AI", "node",
  "CFG = (1+w)pos − w·neg.", "Force du conditioning.",
  params=[("scale", "float", 7.5)])

C("noise_schedule", "Noise Schedule", "Generative AI", "node",
  "β_t sequence.", "Linear, cosine, karras.",
  params=[("schedule", "choice", "cosine", ["linear", "cosine", "karras"])])

C("denoising_step", "Denoising Step", "Generative AI", "node",
  "x_{t-1} = f(x_t, t, ε_θ).", "Une étape sampling.")

C("inpainting", "Inpainting", "Generative AI", "concept", "Régénère région masquée.", "")
C("outpainting", "Outpainting", "Generative AI", "concept", "Étend image au-delà bords.", "")
C("text_to_image", "Text-to-Image", "Generative AI", "concept", "Prompt → image.", "")
C("text_to_video", "Text-to-Video", "Generative AI", "concept", "Prompt → vidéo.", "")
C("image_to_image", "Image-to-Image", "Generative AI", "concept", "Img + prompt → img.", "")


# ============ REINFORCEMENT LEARNING ============

rl_concepts = [
    ("rl_agent", "Agent", "Entité qui agit.", "concept"),
    ("rl_environment", "Environment", "Système monde extérieur.", "concept"),
    ("rl_policy", "Policy π(a|s)", "Comportement.", "concept"),
    ("rl_reward", "Reward r(s,a)", "Signal scalaire.", "concept"),
    ("value_function", "Value Function V(s)", "Expected return.", "concept"),
    ("bellman_eq", "Bellman Equation", "V(s)=E[r+γV(s')].", "concept"),
    ("explore_exploit", "Exploration/Exploitation", "ε-greedy, UCB.", "concept"),
]
for n, t, s, k in rl_concepts:
    C(n, t, "Reinforcement Learning", k, s, s)

rl_algos = [
    ("q_learning", "Q-Learning", "Off-policy TD."),
    ("dqn", "DQN", "Deep Q-Network + replay."),
    ("ppo", "PPO", "Proximal Policy Optim. RLHF default."),
    ("trpo", "TRPO", "Trust Region PO."),
    ("sac", "SAC", "Soft Actor-Critic. Maximum entropy."),
    ("actor_critic", "Actor-Critic", "Policy + value baseline."),
    ("policy_gradients", "Policy Gradients", "∇θ E[R]."),
    ("mcts", "MCTS", "Tree search + rollouts. AlphaZero."),
    ("rlhf", "RLHF", "Reward model from human pref."),
    ("rlaif", "RLAIF", "Reward from AI feedback."),
]
for n, t, s in rl_algos:
    C(n, t, "Reinforcement Learning", "concept", s, s)


# ============ UNSUPERVISED ============

unsup = [
    ("kmeans", "K-Means", "Partitionnement par centroides.", "node",
     [("n_clusters", "int", 8), ("max_iter", "int", 100)]),
    ("dbscan", "DBSCAN", "Density-based clustering.", "node",
     [("eps", "float", 0.5), ("min_samples", "int", 5)]),
    ("hierarchical_clustering", "Hierarchical Clustering", "Agglomerative/divisive.", "tool", []),
    ("spectral_clustering", "Spectral Clustering", "Eigenvectors of Laplacian.", "tool", []),
    ("gmm", "Gaussian Mixture Model", "Σ_k w_k N(μ_k,Σ_k).", "node",
     [("n_components", "int", 5)]),
    ("tsne", "t-SNE", "Préserve proximités locales.", "tool",
     [("perplexity", "int", 30)]),
    ("umap_dim", "UMAP", "Topologie locale + globale.", "tool",
     [("n_neighbors", "int", 15)]),
    ("isomap", "Isomap", "Géodésique manifold.", "tool", []),
]
for n, t, s, k, p in unsup:
    C(n, t, "Unsupervised", k, s, s, params=p)

# Autoencoders
C("autoencoder", "Autoencoder", "Neural Networks", "node",
  "Encoder → bottleneck → Decoder.", "Reconstruction loss.",
  params=[("in_dim", "int", 784), ("latent_dim", "int", 32)])

C("sparse_autoencoder", "Sparse Autoencoder", "Neural Networks", "node",
  "AE + L1 sur activations.", "Feature decomposition. Anthropic interpretability.",
  params=[("latent_dim", "int", 1024), ("l1_coef", "float", 1e-3)])

C("vae", "Variational Autoencoder", "Generative AI", "node",
  "Encoder μ,σ → z~N → Decoder.", "ELBO = reconstruction − KL.",
  params=[("in_dim", "int", 784), ("latent_dim", "int", 32)])

C("denoising_ae", "Denoising Autoencoder", "Neural Networks", "node",
  "Reconstruct from noisy input.", "Self-supervised representation.",
  params=[("noise_level", "float", 0.1)])

C("vq_vae", "VQ-VAE", "Generative AI", "node",
  "Latent discret via codebook.", "Base DALL-E, MusicGen.",
  params=[("num_embeddings", "int", 512), ("embedding_dim", "int", 64)])


# ============ GRAPH SYSTEMS ============

graph_items = [
    ("gnn", "GNN", "Graph Neural Network. Message passing.",
     [("in_dim", "int", 64), ("hidden_dim", "int", 64)]),
    ("gat", "GAT", "Graph Attention. Attention weights edges.",
     [("in_dim", "int", 64), ("heads", "int", 4)]),
    ("graphsage", "GraphSAGE", "Inductive sampling neighbors.",
     [("hidden_dim", "int", 64)]),
    ("message_passing", "Message Passing", "h_i ← UPDATE(h_i, AGG{m_ij}).", []),
    ("knowledge_graph", "Knowledge Graph", "Triplets (h,r,t).", []),
    ("node_embedding", "Node Embedding", "DeepWalk, node2vec.",
     [("embedding_dim", "int", 128)]),
    ("dynamic_graph", "Dynamic Graph", "Evolving topology.", []),
    ("hypergraph", "Hypergraph", "Edges over k>2 nodes.", []),
]
for n, t, s, p in graph_items:
    C(n, t, "Graph Systems", "node" if p else "concept", s, s, params=p)


# ============ BIOLOGICAL INSPIRATION ============

bio = [
    ("hebbian", "Hebbian Learning", "Δw ∝ pre·post (fire together wire together).", "node",
     [("eta", "float", 1e-3)]),
    ("stdp", "STDP", "Spike-Timing Dependent Plasticity.", "concept", []),
    ("synaptic_plasticity", "Synaptic Plasticity", "Modification poids learning.", "concept", []),
    ("neurogenesis", "Artificial Neurogenesis", "Création de neurones runtime (ARK M1).", "concept", []),
    ("neural_fields", "Neural Fields", "Continuous neural pop dynamics.", "node",
     [("size", "int", 64)]),
    ("homeostasis", "Homeostasis", "Régulation activité moyenne.", "concept", []),
    ("structural_plasticity", "Structural Plasticity", "Création/suppression synapses.", "concept", []),
    ("morphogenesis", "Morphogenesis", "Formation pattern développemental.", "node",
     [("size", "int", 64)]),
    ("reaction_diffusion", "Reaction-Diffusion", "∂u/∂t = D∇²u + f(u,v) (Turing).", "node",
     [("D_u", "float", 0.16), ("D_v", "float", 0.08), ("f", "float", 0.035), ("k", "float", 0.065)]),
    ("self_organizing_systems", "Self-Organizing Systems", "Order from local rules.", "concept", []),
]
for n, t, s, k, p in bio:
    C(n, t, "Biological Inspiration", k, s, s, params=p)


# ============ EVOLUTIONARY SYSTEMS ============

evo = [
    ("genetic_algorithm", "Genetic Algorithm", "Crossover + mutation + selection.", "node",
     [("pop_size", "int", 100), ("mutation_rate", "float", 0.01)]),
    ("neuroevolution", "Neuroevolution", "Evolve NN weights/topology.", "concept", []),
    ("neat_algo", "NEAT", "Stanley & Miikkulainen 2002. Speciation.", "concept", []),
    ("hyperneat", "HyperNEAT", "Indirect encoding via CPPN.", "concept", []),
    ("cma_es", "CMA-ES", "Covariance Matrix Adaptation ES.", "node",
     [("pop_size", "int", 30), ("sigma", "float", 0.5)]),
    ("swarm_intelligence", "Swarm Intelligence", "Collective behavior simple agents.", "concept", []),
    ("boids", "Boids", "Separation + alignment + cohesion (Reynolds).", "node",
     [("n_boids", "int", 100), ("perception_radius", "float", 5.0)]),
    ("aco", "Ant Colony Optimization", "Pheromone trails.", "concept", []),
    ("pso", "Particle Swarm Optimization", "Velocity update toward best.", "node",
     [("n_particles", "int", 50), ("inertia", "float", 0.7)]),
]
for n, t, s, k, p in evo:
    C(n, t, "Evolutionary Systems", k, s, s, params=p)


# ============ DYNAMICAL SYSTEMS ============

dyn = [
    ("chaos", "Chaos", "Sensibilité conditions initiales.", "concept", []),
    ("attractor", "Attractor", "Set vers lequel système converge.", "node",
     [("kind", "choice", "lorenz", ["lorenz", "rossler", "henon"])]),
    ("lyapunov_exp", "Lyapunov Exponent", "λ = lim (1/t)log||δx(t)||/||δx0||.", "node",
     [("n_steps", "int", 10000)]),
    ("nonlinear_dynamics", "Nonlinear Dynamics", "ẋ = f(x), f nonlinéaire.", "concept", []),
    ("bifurcation", "Bifurcation", "Changement qualitatif comportement.", "concept", []),
    ("synchronization", "Synchronization", "Kuramoto coupled oscillators.", "node",
     [("n_oscillators", "int", 100), ("coupling", "float", 1.0)]),
    ("oscillator", "Oscillator", "Périodique limit cycle.", "node",
     [("frequency", "float", 1.0)]),
    ("dynamic_network", "Dynamic Network", "Edges/weights évoluent.", "concept", []),
    ("dynamic_field", "Dynamic Field", "Field PDE evolution.", "concept", []),
    ("topological_dynamics", "Topological Dynamics", "Structure topo évolue.", "concept", []),
    ("criticality_concept", "Criticality", "Edge of chaos. SOC.", "concept", []),
    ("self_organization", "Self-Organization", "Order emergent.", "concept", []),
]
for n, t, s, k, p in dyn:
    C(n, t, "Dynamical Systems", k, s, s, params=p)


# ============ COGNITIVE ARCHITECTURES ============

cog = [
    ("world_model", "World Model", "P(s'|s,a) appris.", "node",
     [("state_dim", "int", 128), ("action_dim", "int", 4)]),
    ("active_inference", "Active Inference", "Minimize free energy. Friston.", "concept", []),
    ("predictive_coding", "Predictive Coding", "Hierarchical prediction error.", "concept", []),
    ("free_energy", "Free Energy Principle", "F = surprise + complexity.", "concept", []),
    ("cognitive_arch", "Cognitive Architecture", "SOAR, ACT-R.", "concept", []),
    ("symbolic_ai", "Symbolic AI", "Logique formelle.", "concept", []),
    ("neuro_symbolic", "Neuro-Symbolic AI", "NN + reasoning.", "concept", []),
    ("embodied_ai", "Embodied AI", "Corps + environnement.", "concept", []),
    ("agentic_workflow", "Agentic Workflow", "Multi-step tool-use.", "concept", []),
    ("long_term_memory", "Long-Term Memory", "Persistent across episodes.", "node",
     [("capacity", "int", 10000)]),
    ("episodic_memory", "Episodic Memory", "Specific past events.", "node",
     [("capacity", "int", 1000)]),
]
for n, t, s, k, p in cog:
    C(n, t, "Cognitive Architectures", k, s, s, params=p)


# ============ INTERPRETABILITY ============

interp = [
    ("shap", "SHAP", "Shapley values feature attribution.", "tool"),
    ("lime", "LIME", "Local linear approximation.", "tool"),
    ("attention_viz", "Attention Visualization", "Heatmap attention weights.", "node"),
    ("saliency_map", "Saliency Map", "|∂y/∂x|.", "node"),
    ("feature_attribution", "Feature Attribution", "Contribution chaque feature.", "concept"),
    ("activation_steering", "Activation Steering", "Edit activations runtime.", "concept"),
    ("circuit_analysis", "Circuit Analysis", "Sub-graphs computation Anthropic.", "concept"),
    ("mechanistic_interp", "Mechanistic Interpretability", "Reverse-engineer NN.", "concept"),
    ("latent_viz", "Latent Space Viz", "PCA/UMAP/t-SNE des activations.", "node"),
]
for n, t, s, k in interp:
    C(n, t, "Interpretability", k, s, s)


# ============ SAFETY / ALIGNMENT ============

safety = [
    ("ai_alignment", "AI Alignment", "Goals modèle = goals humain."),
    ("constitutional_ai", "Constitutional AI", "Anthropic. Principles."),
    ("reward_hacking", "Reward Hacking", "Exploite proxy reward."),
    ("instrumental_conv", "Instrumental Convergence", "Sub-goals universels."),
    ("ai_safety", "AI Safety", "Préventif risques."),
]
for n, t, s in safety:
    C(n, t, "Interpretability", "concept", s, s)


# ============ TRAINING — LOSSES ============

losses = [
    ("loss_mse", "MSE Loss", "Σ(y−ŷ)². Régression."),
    ("loss_mae", "MAE Loss", "Σ|y−ŷ|. Robuste outliers."),
    ("loss_bce", "BCE Loss", "Binary cross-entropy."),
    ("loss_huber", "Huber Loss", "MSE + MAE quadratique/linéaire."),
    ("loss_focal", "Focal Loss", "(1−p)^γ. Classes déséquilibrées."),
    ("loss_triplet", "Triplet Loss", "max(0, d(a,p)−d(a,n)+α)."),
    ("loss_contrastive", "Contrastive Loss", "Push apart neg, pull together pos."),
    ("loss_dice", "Dice Loss", "2|X∩Y|/(|X|+|Y|). Segmentation."),
    ("loss_perceptual", "Perceptual Loss", "Feature space distance VGG."),
]
for n, t, s in losses:
    p = [("reduction", "choice", "mean", ["mean", "sum", "none"])]
    C(n, t, "Training", "node", s, s, params=p)


# ============ DISTRIBUTED ============

dist = [
    ("data_parallel", "Data Parallelism", "Split batch over devices."),
    ("model_parallel", "Model Parallelism", "Split model across devices."),
    ("pipeline_parallel", "Pipeline Parallelism", "Stages × micro-batches."),
    ("tensor_parallel", "Tensor Parallelism", "Split matmul (Megatron)."),
    ("zero_optimizer", "ZeRO", "Shard optimizer state."),
    ("ddp", "DistributedDataParallel", "PyTorch DDP."),
    ("fsdp", "FSDP", "Fully Sharded Data Parallel."),
    ("nccl_backend", "NCCL", "NVIDIA collective comms."),
]
for n, t, s in dist:
    C(n, t, "Distributed Systems", "concept", s, s)


# ============ HARDWARE ============

hw = [("gpu", "GPU"), ("tpu", "TPU"), ("npu", "NPU"), ("fpga", "FPGA"),
       ("asic", "ASIC"), ("cuda", "CUDA"), ("rocm", "ROCm"),
       ("tensor_cores", "Tensor Cores")]
for n, t in hw:
    C(n, t, "Hardware", "concept", f"Hardware/runtime : {t}.", f"{t} ML accelerator/SDK.")


# ============ ADAPTIVE ARCHITECTURES ============

adapt = [
    ("dynamic_routing_arch", "Dynamic Routing", "Router NN choisit path.", "node",
     [("n_routes", "int", 4), ("temperature", "float", 1.0)]),
    ("sparse_experts", "Sparse Experts", "MoE avec top-k routing.", "node",
     [("n_experts", "int", 8), ("top_k", "int", 2)]),
    ("entropic_router", "Entropic Router", "Maximise entropy distribution routing.", "node",
     [("n_routes", "int", 4), ("entropy_coef", "float", 0.1)]),
    ("reconfigurable_arch", "Reconfigurable Architecture", "Topology runtime mutable.", "concept", []),
    ("self_modifying_graph", "Self-Modifying Graph", "Edges added/removed by self.", "concept", []),
    ("artificial_neurogenesis", "Artificial Neurogenesis", "Création neurones run-time.", "concept", []),
    ("dynamic_pruning", "Dynamic Pruning", "Suppression weights selon métric.", "node",
     [("sparsity_target", "float", 0.5)]),
    ("structural_growth", "Structural Growth", "Net2Net widen/deepen.", "node",
     [("growth_factor", "float", 1.5)]),
    ("topological_adaptation", "Topological Adaptation", "Graph rewires.", "concept", []),
    ("persistent_memory", "Persistent Memory", "External long-term store.", "node",
     [("size", "int", 1024)]),
    ("differentiable_memory", "Differentiable Memory", "Read/write soft.", "concept", []),
]
for n, t, s, k, p in adapt:
    C(n, t, "Adaptive Architectures", k, s, s, params=p)


# ============ MULTI-AGENT ============

multi = [
    ("autonomous_agent", "Autonomous Agent", "Boucle perceive→plan→act.", "node",
     [("memory_size", "int", 100)]),
    ("inter_agent_comm", "Inter-Agent Communication", "Shared message bus.", "node",
     [("channel_dim", "int", 64)]),
    ("shared_memory_block", "Shared Memory", "Tous agents lisent/écrivent.", "node",
     [("size", "int", 1024)]),
    ("coordination", "Coordination", "Consensus/leader/voting.", "concept", []),
    ("distributed_system_arch", "Distributed System", "Agents over network.", "concept", []),
    ("agent_hierarchy", "Agent Hierarchy", "Boss/worker.", "concept", []),
]
for n, t, s, k, p in multi:
    C(n, t, "Multi-Agent Systems", k, s, s, params=p)


# ============ SIMULATION & EMERGENCE ============

sim = [
    ("cellular_automaton", "Cellular Automaton", "Local rules → emergence (Wolfram).", "node",
     [("rule", "int", 110), ("size", "int", 128), ("steps", "int", 256)]),
    ("game_of_life", "Game of Life", "Conway. B3/S23.", "node",
     [("size", "int", 64), ("steps", "int", 100)]),
    ("ddlab_rbn", "Random Boolean Network", "Kauffman NK-model.", "node",
     [("n_nodes", "int", 100), ("k", "int", 2)]),
    ("turing_pattern", "Turing Patterns", "Reaction-diffusion stationnaires.", "node",
     [("steps", "int", 1000)]),
    ("agent_based_model", "Agent-Based Model", "Many micro-agents → macro.", "concept", []),
    ("ising_model", "Ising Model", "Spins ±1 + coupling.", "node",
     [("size", "int", 64), ("temperature", "float", 2.0)]),
    ("kuramoto_sync", "Kuramoto Sync", "Coupled oscillators phase.", "node",
     [("n", "int", 100), ("K", "float", 1.0)]),
    ("lorenz_attractor", "Lorenz Attractor", "ẋ=σ(y−x), ẏ=x(ρ−z)−y, ż=xy−βz.", "node",
     [("sigma", "float", 10.0), ("rho", "float", 28.0), ("beta", "float", 2.667)]),
]
for n, t, s, k, p in sim:
    C(n, t, "Simulation", k, s, s, params=p)


# ============ EXPERIMENTAL / EMERGING ============

emerging = [
    ("foundation_model", "Foundation Model", "Large pretrained generalist."),
    ("multimodal_model", "Multimodal Model", "Text+image+audio joint."),
    ("agentic_ai", "Agentic AI", "Tool-using autonomous."),
    ("tool_using_model", "Tool-Using Model", "LLM invoque APIs."),
    ("long_context_model", "Long-Context Model", "1M+ tokens."),
    ("retrieval_native", "Retrieval-Native", "RAG integrated training."),
    ("diffusion_transformer", "Diffusion Transformer (DiT)", "DiT remplace UNet."),
    ("test_time_compute", "Test-Time Compute", "More inference compute = better answer (o1)."),
    ("self_improving", "Self-Improving Agents", "Recursive self-training."),
    ("memory_augmented", "Memory-Augmented Systems", "External persistent KB."),
    ("adaptive_routing_arch", "Adaptive Routing Architectures", "Path runtime."),
    ("sparse_expert_systems", "Sparse Expert Systems", "MoE families."),
    ("liquid_nn", "Liquid Neural Networks", "Continuous-time NN (MIT)."),
    ("neuromorphic", "Neuromorphic Computing", "Spike-based hardware."),
    ("event_driven_nn", "Event-Driven NN", "Async spikes."),
    ("snn", "Spiking Neural Networks", "Discrete spikes + LIF neurons."),
    ("ca_ml_hybrid", "Cellular Automata ML Hybrid", "CA + NN."),
    ("emergent_intelligence", "Emergent Intelligence", "Capabilities scale."),
    ("phase_transition_learning", "Phase Transitions in Learning", "Grokking, abrupt jumps."),
    ("grokking", "Grokking", "Generalization saute après plateau."),
    ("info_bottleneck", "Information Bottleneck", "Tishby. min I(X,T) − βI(T,Y)."),
    ("kolmogorov_complexity", "Kolmogorov Complexity", "Min program length."),
    ("compression_hypothesis", "Compression Hypothesis", "Generalization = compression."),
    ("lottery_ticket", "Lottery Ticket Hypothesis", "Sparse subnetwork wins."),
    ("in_context_learning", "In-Context Learning", "Few-shot via prompt."),
    ("latent_rollout", "Latent Rollout Prediction", "Future predict in latent."),
    ("predictive_latent_dynamics", "Predictive Latent Dynamics", "Apprend f: z_t → z_{t+1}."),
]
for n, t, s in emerging:
    C(n, t, "Experimental Research", "concept", s, s)


# ============ AUDIO / ROBOTICS / FINANCE / BIO ============

audio = [("stt", "Speech-to-Text"), ("tts", "Text-to-Speech"),
          ("voice_cloning", "Voice Cloning"), ("audio_diffusion", "Audio Diffusion"),
          ("music_gen", "Music Generation")]
for n, t in audio:
    C(n, t, "Experimental Research", "concept", f"Audio domain : {t}.", f"{t} audio model.")

robotics = [("slam", "SLAM"), ("motor_control", "Motor Control"),
              ("planning_robot", "Planning"), ("sim2real", "Sim2Real"),
              ("robotics_rl", "Robotics RL")]
for n, t in robotics:
    C(n, t, "Experimental Research", "concept", f"Robotics : {t}.", f"{t}.")

finance = [("forecasting", "Forecasting"), ("ts_models", "Time-Series Models"),
             ("volatility_model", "Volatility Modeling"),
             ("regime_detection", "Regime Detection"),
             ("quant_models", "Quant Trading")]
for n, t in finance:
    C(n, t, "Experimental Research", "concept", f"Finance : {t}.", f"{t} financial ML.")

bio_apps = [("alphafold", "AlphaFold"), ("protein_folding", "Protein Folding"),
              ("phylogenetics_app", "Phylogenetics"), ("bioinformatics", "Bioinformatics"),
              ("genomic_foundation", "Genomic Foundation Models")]
for n, t in bio_apps:
    C(n, t, "Experimental Research", "concept", f"Bio : {t}.", f"{t} computational biology.")


# ============ FRAMEWORKS / TOOLS ============

frameworks = [("pytorch", "PyTorch"), ("tensorflow", "TensorFlow"),
                ("keras", "Keras"), ("jax", "JAX"), ("onnx", "ONNX"),
                ("hf_transformers", "Hugging Face Transformers"),
                ("langchain", "LangChain"), ("llamaindex", "LlamaIndex"),
                ("vllm", "vLLM"), ("numpy", "NumPy"), ("pandas", "Pandas"),
                ("scikit_learn", "scikit-learn"), ("scipy", "SciPy"),
                ("matplotlib", "Matplotlib"), ("tensorboard", "TensorBoard"),
                ("wandb", "Weights & Biases")]
for n, t in frameworks:
    C(n, t, "Hardware", "tool", f"Framework : {t}.", f"{t} ML framework/lib.")


# ============ STATS TESTS ============

C("null_hypothesis", "Null Hypothesis", "Foundations", "concept",
  "H0 : pas d'effet.", "Stats inférentielle base.")

C("confidence_interval", "Confidence Interval", "Foundations", "node",
  "[θ̂−z·σ, θ̂+z·σ].", "95% = z≈1.96.",
  params=[("alpha", "float", 0.05)])

C("estimator", "Estimator", "Foundations", "concept",
  "θ̂(D).", "Statistique calculée depuis échantillon.")

C("bias_variance", "Bias-Variance Tradeoff", "Foundations", "concept",
  "MSE = bias² + variance + irreducible.",
  "Plus complexe = moins biais + plus variance.")


# ============ EXPORT FINAL ============

def get_all_concepts() -> list[dict]:
    return CONCEPTS


def by_category() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for c in CONCEPTS:
        out.setdefault(c["category"], []).append(c)
    return out


def by_kind(kind: str) -> list[dict]:
    return [c for c in CONCEPTS if c["kind"] == kind]
