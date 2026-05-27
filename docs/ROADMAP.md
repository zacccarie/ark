# ARK roadmap

## Phases livrées (19 phases, 9 commits, 65 tests, ~8500 lignes)

| Phase | Statut | Contenu |
|---|---|---|
| **0** Scaffold | ✓ | PySide6, design tokens, MainWindow, 3 dock panels, node editor minimal |
| **1** Interactions | ✓ | Drag connections + validation type + multi-input + resize + context menu |
| **1.5** Undo/Redo | ✓ | QUndoStack + 5 QUndoCommand subclasses |
| **2** Compile + Train | ✓ | graph_compiler 14+ handlers, TrainerWorker QThread, errors → red |
| **3** Save/Load JSON | ✓ | Format ark-arch-v1 (compat metamorph-arch-v1) |
| **4** ComfyUI specs | ✓ | 10 NodeSpecs Diffusion (KSampler, VAE, CLIP, LoRA, ControlNet) |
| **4 stub** | ✓ | Handlers déterministes sans modèles téléchargés |
| **5** Matplotlib viz | ✓ | MetricChart embarqué inspector observer node |
| **6** Devices | ✓ | webcam (cv2) + microphone (sounddevice) handlers, lazy import |
| **7** Layout presets | ✓ | 4 presets (workspace/compact/code/training) + save/load custom |
| **8** Export PyTorch/ONNX | ✓ | Script .py standalone (LazyLinear) + ONNX (opset 17) |
| **9+10** Observer+auto-mutate | ✓ | Port complet vagues B/C ai-arch-builder |
| **11** Thumbnails | ✓ | NodeItem render mini (image damier/webcam REC/audio waveform/sparkline) |
| **12** Multi-projets | ✓ | QTabWidget central, scènes indépendantes, undo par tab |
| **13** Templates | ✓ | 8 templates (MLP/CNN/ResNet/Transformer/VG/Diffusion/M1/M5) |
| **14** Templates browser | ✓ | Dialog preview + favoris ~/.ark/favorites.json |
| **15** Import legacy | ✓ | ai-arch-builder JSON → ARK conversion auto (kebab→snake, camel→snake) |
| **16** Agent CLI | ✓ | `python -m ark.agent_api` 11 commandes headless |
| **17** Metamorph bridge | ✓ | M1 CriticalGrowNet + M2 MetaController + M5 DyadicMirror comme nodes ARK |
| **18** Doc + bench | ✓ | Ce document + bench/compile_benchmark.py |

## Bench compile (offscreen, MPS dispo)

| Template | params | modules | compile (ms) | forward (ms) | shape |
|---|---|---|---|---|---|
| mlp_mnist | 101 770 | 2 | 5.7 ± 11 | 0.19 | [1,28,28]→[10] |
| cnn_cifar | 8 409 418 | 4 | 25.4 ± 1.5 | 4.91 | [3,32,32]→[10] |
| resnet_block | 166 682 | 6 | 1.5 ± 0.4 | 0.97 | [3,32,32]→[10] |
| transformer_mini | 77 194 | 3 | 1.0 ± 0.1 | 1.22 | [16,64]→[10] |
| vg_mutable | 52 154 | 5 | 0.8 ± 0.1 | 0.21 | [1,28,28]→[10] |
| metamorph_m1 | 107 370 | 5 | 1.3 ± 0.7 | 0.17 | [1,28,28]→[10] |
| metamorph_m5 | 66 212 | 2 | 1.1 ± 0.2 | 0.16 | [1,28,28]→[10] |
| diffusion_basic | — | — | ERR | — | multi-output (Phase 4 réelle) |

Compile sub-millisecond pour majorité. CNN cifar 25ms (8M params).

## Reste roadmap (différé)

| Phase | Statut | Pourquoi différé |
|---|---|---|
| **4 réelle** ML backend diffusers | TODO | Demande 5GB téléchargement SD 1.5 + diffusers actif |
| **11.5** Live thumbnails dynamiques | TODO | Demande webcam réelle active en sandbox |
| **19** Multi-output compile (diffusion) | TODO | Refactor GraphModel pour graphe DAG multi-sortie |
| **20** WebSocket sync ARK ↔ metamorph | TODO | Coordination live entre les 2 apps |
| **21** Plugin system | TODO | Permettre user d'ajouter NodeSpec custom |

## Décisions architecturales fixées

| Question | Choix |
|---|---|
| Stack UI | PySide6 (Qt 6 LGPL, natif) |
| Backend ML | PyTorch (principal) + diffusers stubs |
| ComfyUI | Nodes copiés natifs (stubs Phase 4) |
| Format JSON | ark-arch-v1 (compat metamorph-arch-v1) |
| Node editor | QGraphicsScene custom (pas de lib externe) |
| Style | QSS global depuis design_tokens.py |
| Metamorph integration | Bridge lazy import via ark/core/metamorph_bridge.py |

## Conventions

- Code : commentaires français, identifiers `snake_case` (PEP8 Python).
- UI text : français.
- Esthétique : dark, accent rouge pur `#E11D2A`, typo Arial+monospace.
- Pas de dépendances ML lourdes au cold install.
