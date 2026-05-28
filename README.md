# ARK

> Architecture builder node-based. Python natif (PySide6). Refonte complète de `ai-arch-builder`.

## Pourquoi

Application de design d'architectures neuronales par graphe nodal. Réécrite from scratch en **Python pur** (sortie d'Electron) pour intégration native avec l'écosystème ML (PyTorch, HuggingFace, ComfyUI, TensorFlow).

## Philosophie design

- **Sobriété maximale** : dark UI uniquement. Aucun gradient, aucune couleur fluo.
- **Accent unique** : rouge pur (`#E11D2A`). Pas d'autres couleurs vives.
- **Typo** : Arial famille (titres+corps) + monospace (valeurs, IDs, code).
- **Espacement** : système 4-8-12-16-24-32-48.
- **Coins droits ou très légèrement arrondis** (2-4px max).
- **Inspiration** : Blender Geometry Nodes, TouchDesigner, ComfyUI — niveau pro.

## Stack

- **PySide6** (Qt 6, LGPL, natif Mac/Linux/Windows).
- **PyTorch** + écosystème ML.
- **HuggingFace Hub** (modèles, LoRA, transformers).
- **ComfyUI nodes** copiés comme types natifs ARK (Phase 4).

## Layout

QMainWindow avec QDockWidget pour panneaux multi-fenêtres, à la Blender :

```
┌─────────────────────────────────────────────────────────────────┐
│ MenuBar : Fichier  Édition  Affichage  Exécution  Aide          │
├─────────────────────────────────────────────────────────────────┤
│ ToolBar : ▶ Compiler  ⏵ Entraîner  ■ Stop  ⌕ Rechercher node    │
├──────────┬─────────────────────────────────────┬────────────────┤
│          │                                     │                │
│ Palette  │           Node Editor               │   Inspector    │
│ (gauche) │     (centre — QGraphicsScene)       │   (droite)     │
│          │                                     │                │
│          │                                     │                │
├──────────┴─────────────────────────────────────┴────────────────┤
│ Terminal · Python · Metrics (sous-onglets)                      │
├─────────────────────────────────────────────────────────────────┤
│ StatusBar : ARK · ready                                          │
└─────────────────────────────────────────────────────────────────┘
```

Tous les panneaux sont **drag-undock-redimension-fermable** nativement (QDockWidget).

## Quickstart

```bash
cd ~/claude-code-projects/ark
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
ark                       # via entry point pyproject
# ou
./bin/ark                 # via launcher shell standalone
```

## Interactions node editor

- **Souris molette** : zoom (anchor sous curseur).
- **Bouton milieu / espace+drag** : pan.
- **Double-clic vide** : ouvre search overlay (filtrage instantané).
- **Drag socket → socket** : crée connexion.
- **Suppr / Backspace** : delete sélection.
- **Double-clic header node** : toggle bypass.
- **Ctrl+F** : search overlay.

## Roadmap

Voir `docs/ROADMAP.md` pour le plan détaillé par phase.

### Phase 0 (cette session) ✓
- Scaffold repo + PySide6 + design tokens.
- MainWindow + 3 dock panels (Palette, Inspector, Terminal).
- Node editor (scene + view zoom/pan + nodes + sockets + connections bezier).
- Search overlay double-clic.
- Registry initial (Input / Neural / Math / Code / Observe / Viz / Training).

### Phase 1
- Connexions drag-and-drop fonctionnelles (start/end socket).
- Multi-input nodes dynamiques (residual_add, concat).
- Validation pré-compile + nodes rouges si erreur.
- Bypass switch fonctionnel.

### Phase 2
- Compile graph → PyTorch nn.Module (migration de l'ancien graph_compiler).
- Training loop + auto-mutate.
- Observer/Trigger live metrics.

### Phase 3
- Save/load metamorph-arch-v1 JSON.
- Export PyTorch / ONNX / TensorFlow / Python script.

### Phase 4
- Nodes ComfyUI natifs (KSampler, VAE, CLIP, Save Image).
- Connexion HuggingFace Hub.

### Phase 5
- Visualisations 3D (Three.js → QtWebEngine ou matplotlib direct).
- Miniatures nodes (image, vidéo, graphe).

### Phase 6
- Periphériques (webcam, micro) comme nodes input.
- Workflows multi-projets, templates.

### Phase 7
- Layout presets (workspace / compact / code-first).
- Layout save/load.

## License

MIT (à confirmer).
