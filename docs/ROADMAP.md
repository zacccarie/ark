# ARK roadmap

## Phase 0 — Scaffold ✓ (cette session)

- [x] Repo `ark/` + pyproject + .gitignore
- [x] Design tokens (`ark/ui/design_tokens.py`) : dark + Arial + accent rouge pur
- [x] QSS global (`ark/ui/style.py`)
- [x] MainWindow + menubar + toolbar + statusbar
- [x] 3 QDockWidget panels : Palette, Inspector, Terminal
- [x] Node editor : QGraphicsScene + QGraphicsView (zoom, pan)
- [x] NodeItem + SocketItem + ConnectionItem (bezier)
- [x] Search overlay (double-clic, filtrage instantané)
- [x] Registry initial : 20+ nodes types (Input/Neural/Math/Code/Observe/Viz/Training)
- [x] Entry point `python -m ark`

## Phase 1 — Interactions de base

- [ ] Drag socket → socket pour créer connexion (preview pendant drag)
- [ ] Validation type-matching à la connexion (rejet si mismatch)
- [ ] Multi-input dynamique : node residual_add ajoute un connecteur à chaque connexion
- [ ] Resize handles fonctionnels (coin BR du node)
- [ ] Bypass toggle (switch dans inspector + double-clic header)
- [ ] Error overlay sur node (rouge + message)
- [ ] Undo/redo
- [ ] Context menu (right-click)

## Phase 2 — Compile + train (migration depuis ai-arch-builder)

- [ ] `ark/core/graph_compiler.py` : port de l'ancien graph_compiler v2 (DAG, handlers, vague A-C nodes)
- [ ] `ark/core/trainer.py` : training loop avec MNIST/CIFAR/text
- [ ] Auto-mutate triggers fonctionnels
- [ ] Observer live metrics → terminal panel
- [ ] Compile errors → set_error() sur nodes incorrects

## Phase 3 — Persistence + Export

- [ ] Save/load JSON metamorph-arch-v1
- [ ] Export PyTorch nn.Module (.py)
- [ ] Export ONNX
- [ ] Export TensorFlow (via tf2onnx ou portage)
- [ ] Recent files
- [ ] Project workspaces (multi-graphes)

## Phase 4 — Nodes ComfyUI natifs

Copier les nodes ComfyUI essentiels comme types ARK (pas wrapper, pas serveur séparé) :

- [ ] **KSampler** : diffusion sampling (DDIM, DPM++, Euler, etc.)
- [ ] **DiffusionModel** loader (.safetensors, .ckpt)
- [ ] **VAEEncode** / **VAEDecode**
- [ ] **CLIPTextEncode** (positive / negative prompts)
- [ ] **LoadImage** / **SaveImage**
- [ ] **EmptyLatentImage** (W, H, batch)
- [ ] **ControlNet** apply
- [ ] **LoRA** loader + strength

Importer modèles tous formats : `.safetensors`, `.ckpt`, `.pt`, `.bin`, `.gguf`, `.onnx`.

## Phase 5 — Viz + thumbnails

- [ ] Thumbnails embarqués dans nodes (image, vidéo séquence, audio waveform, graphe)
- [ ] Node **plot_scatter** / **plot_line** avec matplotlib intégré
- [ ] Node **plot_3d** (matplotlib mpl_toolkits ou QtWebEngine + Three.js)
- [ ] Viz live des activations training (similaire à viz_3d_scatter ai-arch-builder)
- [ ] Mesures dynamiques (β, λ, intrinsic_dim) plottés en temps réel

## Phase 6 — Périphériques + sources externes

- [ ] Node **webcam** (capture caméra)
- [ ] Node **microphone** (capture audio)
- [ ] Node **file_watcher** (drop folder)
- [ ] Node **http_fetch** (HuggingFace Hub direct download)

## Phase 7 — Workspace + presets

- [ ] Layout presets : workspace / compact / code-first / training-focus
- [ ] Save layout custom
- [ ] Multi-projets (onglets en haut)
- [ ] Templates (graphes prédéfinis pour démarrer)
- [ ] Connexion HuggingFace (login, search models)
- [ ] Connexion ComfyUI (si user veut hybrid wrapper en plus du native)

## Décisions architecturales

| Question | Choix |
|---|---|
| Stack UI | PySide6 (Qt 6 LGPL) |
| Backend ML | PyTorch (principal) + HF transformers |
| ComfyUI | Nodes copiés natifs (pas wrapper) |
| Format JSON | metamorph-arch-v1 (compat avec ancien ai-arch-builder + metamorph) |
| Node editor | QGraphicsScene custom (pas de lib externe) |
| Style | QSS global depuis design_tokens.py |

## Conventions

- Code : commentaires français, identifiers `snake_case` (PEP8 Python).
- UI text : français.
- Esthétique : tout doit passer un design-critique. Pas de dégradés, pas de fluo, accent unique rouge.
