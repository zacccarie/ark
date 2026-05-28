# ARK — Audit Production Readiness
Phase 40 · 2026-05-28

## 1. Métriques globales

| Domaine | Mesure |
|---|---|
| Tests | **269/269 passing** ✅ |
| Lint (ruff) | **0 erreurs** ✅ |
| LOC code | 15 750 |
| LOC tests | 3 432 |
| Fichiers Python | 64 |
| NodeSpec total | **232** dans 32 catégories |
| Handlers compile | **103** |
| Cold start | 1.2s ✅ |
| Memory baseline | 338 MB |
| Compile 4 nodes | 5.5ms ⚡ |
| Forward bs=8 | 0.49ms ⚡ |
| Throughput | 16 353 samples/s |

## 2. Coverage Spec ↔ Handler

- **117 specs orphelines** (NodeSpec sans handler compile)
- **0 handlers orphelins**

Specs orphelines = intentionnellement concept-only :
- Optimizers (`adam`, `adamw`, `lion`, `rmsprop`) — nodes documentaires
- Distributions (`bernoulli`, `multinomial`, `poisson`) — pour glossaire
- RL algos (`ppo`, `trpo`, `sac`) — concept seulement
- Modèles nommés (`gpt`, `bert`, `claude`, etc.)
- Théories (`grokking`, `scaling_laws`)

### Recommandation R1 : marquer concept-only dans UI
Distinguer visuellement les nodes "concept" (palette + glossaire) vs nodes "exécutable" :
- Icône lecture seule sur palette
- Tooltip "non-compilable"
- Indication Inspector

### Recommandation R2 : implementer optimizers fonctionnels
`adam/adamw/sgd/rmsprop/lion` doivent retourner un torch.optim.* configuré.
Permet pipeline complet sans hardcoded optim dans trainer.

## 3. Architecture review

### Forces ✅
- **Lazy imports** partout (metamorph, dynamoscope, bertviz, plotly, diffusers)
- **Graceful fallback** si lib absente
- **Topologie acyclique** : registry → graph_compiler → handlers
- **Format JSON versionné** : ark-arch-v1 + ark-group-v1
- **41 phases livrées** sans régression majeure
- **3 bridges externes** : metamorph, dynamoscope, AI-Scientist
- **Style QSS unifié** : design_tokens.py source unique
- **Auto-registration** : 159 nodes générés depuis glossary_data

### Dette technique ⚠️
1. **TODO restant** : `ark/core/export.py:189` (`# TODO: {ntype} — not yet exportable`)
   → Export Python standalone incomplet pour custom handlers
2. **Print stderr direct** dans registry/graph_compiler en cas d'échec
   → Devrait utiliser logging avec niveaux
3. **Pas de logger central** : utilise print() partout
4. **gestion erreurs handlers** silencieuse (try/except pass)
5. **Tests manquants** :
   - End-to-end UI workflow (drag depuis palette → compile → train)
   - Multi-tab undo/redo
   - Save/load avec connexions multi-output

## 4. Performance

### Excellents ✅
- Cold start 1.2s (acceptable Mac M2)
- Compile ~5ms
- Forward 0.5ms/batch
- Lint 0 erreurs

### À surveiller ⚠️
- **Memory 338MB** baseline → cause = PyTorch + transformers chargés
- Build .app PyInstaller = **1.0GB** (très gros)
  → R3 : split deps optional[ml] de base
- Glossary parse à chaque `default_registry()` (475 entries)
  → R4 : cache module-level singleton

## 5. Design Production Readiness

### Phase 36 design refresh livré ✅
- Zinc neutrals soft
- Multi-tier borders (subtle/default/strong/focus)
- 4 colors semantic distincts (success/warn/info/error)
- Pill badges status
- Animations module (fade/pulse/slide)
- Status dot per node
- Global state badge bottom bar

### Inspector Phase 38 ✅
- 7 sections didactiques (role/function/equation/intuition/when/example/pitfall/related)
- Couleurs sémantiques par section
- Fallback automatique GlossaryEntry

### Manques UX ⚠️
1. **Pas de keyboard shortcuts cheat-sheet** visible
   → R5 : `Cmd+/` ouvre overlay raccourcis
2. **Pas de tutorial interactif** au premier lancement
   → R6 : onboarding 3-step pour scene minimale + compile
3. **Pas de breadcrumb** dans Inspector pour naviguer concepts
4. **Pas de "recent files"** dans menu Fichier
5. **Connexions invisibles** si nodes éloignés (pas de minimap)
   → R7 : minimap overview en coin scene
6. **Pas de "Run"** boutton toolbar évident pour compile rapide
   → R8 : bouton flottant "▶ Compile" canvas

## 6. Workflow audit

### Workflows fonctionnels ✅
- Create scene → drag nodes → connect → compile → train
- Save/load JSON
- Ctrl+G groups composite
- F1 help page
- Ctrl+Shift+B benchmark dashboard
- Ctrl+Shift+D arch diagram

### Workflows incomplets ⚠️
1. **Live training visualization** : trainer worker existe mais pas de connection live au dashboard benchmark
   → R9 : signal Trainer.metrics_emitted → Dashboard update real-time
2. **Pas de "Stop training"** clear si user lance trainer
3. **Diffusion Phase 4** réelle : code OK mais pas testé bout-en-bout avec vrai SD checkpoint
4. **Multi-output compile** Phase 19 OK mais peu de tests

## 7. Sécurité / robustesse

### Bon ✅
- Lazy imports → pas de crash si extras absents
- Subprocess timeout sur ai-scientist (10min) + autoresearch (1h)
- File picker filtres extension stricts
- Group payload validation version

### À renforcer ⚠️
1. **Pas de sandbox subprocess** : autoresearch loop peut exécuter code arbitraire (par design)
2. **Pas de signature checkpoint** safetensors → load externe risque pickle exploit
   → R10 : warn user avant load_state_dict depuis sources non-trusted
3. **Pas de rate-limit** sur Inspector file dialog ouvertures

## 8. Production-Readiness Checklist

### ✅ Done
- [x] Tests 269/269
- [x] Lint clean
- [x] Format JSON versionné
- [x] Lazy imports
- [x] Custom .app build
- [x] Remote GitHub
- [x] Help page complète
- [x] 232 NodeSpec organisées

### ⚠️ Reste
- [ ] **Logger central** (remplace print)
- [ ] **Onboarding** premier lancement
- [ ] **Minimap** scene
- [ ] **Live training stream** vers dashboard
- [ ] **Optimizers** handlers fonctionnels
- [ ] **Cheat-sheet** raccourcis
- [ ] **Export Python** custom handlers
- [ ] **Singleton registry** (cache parse glossary)
- [ ] **CI/CD** pipeline (currently 0 GitHub Actions)
- [ ] **Crash reporter** pour usage prod
- [ ] **Icon .icns** custom pour .app

## 9. Recommandations prioritaires

### Court terme (livrables 1h)
1. **R8** : bouton flottant "▶ Compile" canvas
2. **R5** : overlay raccourcis Cmd+/
3. **R4** : cache registry singleton

### Moyen terme (1 jour)
4. **R9** : live training → dashboard signal
5. **R7** : minimap scene overview
6. **R2** : optimizers fonctionnels Adam/AdamW/etc

### Long terme
7. **R6** : tutorial onboarding interactif
8. **CI/CD** GitHub Actions tests + build
9. **Logger central** avec niveaux

## 10. Score global

| Dimension | Score | Note |
|---|---|---|
| Code quality | 9/10 | Lint clean, tests OK |
| Architecture | 9/10 | Modulaire, extensible, bridges sains |
| Performance | 9/10 | Cold 1.2s, compile 5ms, forward 0.5ms |
| UX/Design | 7.5/10 | Phase 36+38 mais manques onboarding/minimap |
| Documentation | 9/10 | 475 concepts didactiques + Help page |
| Tests | 8.5/10 | Coverage solide unitaires, manque E2E UI |
| Production prêt | 7/10 | Manque logger, CI, onboarding |

**Score global moyen : 8.4/10** ✅

ARK est **production-ready pour usage perso recherche**.
Pour distribution publique large : compléter R5/R6/R8 + CI + logger central.

## 11. Comparaison ComfyUI (rappel)
ARK couvre niches que ComfyUI ne touche pas :
- Training native (PyTorch)
- Architecture mutable runtime (M1/M2/M5)
- Dynamics analytics (Lyapunov, RQA, DMD, phase space)
- BertViz + 3D plots scientifiques
- Help didactique vulgarisée
- Benchmark/audit dashboard
- Groups composites réutilisables

**Positionnement clair : ARK = ML research toolbench, pas image generation.**
