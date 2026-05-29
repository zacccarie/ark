"""Phase 65 — ML Architect AI agent.

Expert IA en ML/DL/data systems. Propose architectures, critique modèles,
suggère améliorations. Output ARK ark-arch-v1 JSON applicable directement.

Pas de framework agent lourd (langchain etc). Boucle simple :
  user msg → LLM(messages + tools_doc) → extract arch JSON if present
"""
from __future__ import annotations
import json
import re
from typing import Iterable, Optional

from ark.core.llm_providers import LLMProvider, get_provider, DEFAULT_MODELS


SYSTEM_PROMPT = """Tu es ARK Architect, expert IA absolu en machine learning,
deep learning, systèmes de données et architectures neuronales.

DOMAINES MAÎTRISÉS :
• Vision (CNN, ResNet, ViT, ConvNeXt, EfficientNet, U-Net, SwinV2)
• NLP / LLM (Transformer, RoPE, FlashAttn, MoE, GQA, SSM/Mamba, RetNet)
• Diffusion (DDPM, DDIM, SD, ControlNet, AnimateDiff, LDM, EDM)
• Audio (MusicGen, AudioLDM, Whisper, Bark, EnCodec)
• Multimodal (CLIP, BLIP, LLaVA, Flamingo, Whisper-LLM)
• RL (PPO, SAC, DQN, MuZero, IRIS world-models)
• GNN / graphes (GCN, GAT, GIN, message passing)
• Optimisation (Adam, AdamW, Lion, Sophia, schedule cosine/warmup)
• Data systems (Parquet, Arrow, Iceberg, DuckDB, vector DBs, RAG)
• Théorie ML (PAC, double-descent, NTK, lottery ticket, scaling laws)

NODES ARK DISPONIBLES (par catégorie) :
{nodes_catalog}

WORKFLOW :
1. Comprends l'intent utilisateur (tâche, données, contraintes compute).
2. Si problème mal posé → pose 1-2 questions ciblées.
3. Si clair → réponds avec :
   - Réflexion courte (2-4 lignes max)
   - Architecture proposée (texte concis : nodes + connexions)
   - Si applicable : BLOC JSON ark-arch-v1 prêt à charger (entre balises
     ```ark-arch et ```).
4. Justifications scientifiques (cite papers si pertinent).
5. Mentionne trade-offs (compute, mémoire, expressivité, généralisation).

FORMAT JSON ATTENDU (si proposition complète) :
```ark-arch
{{
  "version": "ark-arch-v1",
  "name": "Nom court",
  "nodes": [
    {{"id":"n1","type":"text_input","params":{{"prompt":"..."}},"pos":[0,200],"width":220}},
    ...
  ],
  "connections": [
    {{"from_node":"n1","from_socket":0,"to_node":"n2","to_socket":0}},
    ...
  ],
  "metadata": {{"tags":["..."]}}
}}
```

CONTRAINTES :
- NEVER invent un type de node absent du catalogue.
- Pos : grille 280px horizontal, 200px vertical.
- Réponses en français.
- Concis. Pas de blabla. Caveman-style autorisé.
"""


def _nodes_catalog() -> str:
    """Construit catalogue compact des nodes ARK pour le system prompt."""
    try:
        from ark.nodes.registry import default_registry
        r = default_registry()
        by_cat: dict[str, list[str]] = {}
        for s in r.all():
            by_cat.setdefault(s.category, []).append(s.name)
        lines = []
        for cat, names in sorted(by_cat.items()):
            lines.append(f"  [{cat}] " + ", ".join(sorted(names)))
        return "\n".join(lines)
    except Exception as e:
        return f"  (erreur catalogue: {e})"


def system_prompt() -> str:
    return SYSTEM_PROMPT.format(nodes_catalog=_nodes_catalog())


# ============================================================
# Agent class
# ============================================================

class ArchitectAgent:
    def __init__(self, provider_name: str = "anthropic",
                 model: Optional[str] = None):
        self.provider_name = provider_name
        self.model = model or DEFAULT_MODELS.get(provider_name, "")
        self._provider: Optional[LLMProvider] = None
        # Conversation history (excluding system prompt)
        self.history: list[dict] = []

    def set_provider(self, provider_name: str, model: Optional[str] = None):
        self.provider_name = provider_name
        self.model = model or DEFAULT_MODELS.get(provider_name, "")
        self._provider = None

    def _get_provider(self) -> LLMProvider:
        if self._provider is None:
            self._provider = get_provider(self.provider_name)
        return self._provider

    def reset(self):
        self.history = []

    def chat(self, user_msg: str, scene_summary: str = "",
             stream: bool = False):
        """Envoie message. Si stream=True, retourne generator de chunks.
        Sinon retourne string complète."""
        prov = self._get_provider()
        msgs = [{"role": "system", "content": system_prompt()}]
        if scene_summary:
            msgs.append({
                "role": "system",
                "content": f"ARCHITECTURE COURANTE :\n{scene_summary}",
            })
        msgs += self.history
        msgs.append({"role": "user", "content": user_msg})
        try:
            result = prov.chat(msgs, model=self.model, stream=stream)
        except Exception as e:
            err = f"[erreur provider {self.provider_name}] {e}"
            self.history += [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": err},
            ]
            if stream:
                def _gen():
                    yield err
                return _gen()
            return err
        if stream:
            full = []
            def _wrap():
                for chunk in result:
                    full.append(chunk)
                    yield chunk
                self.history += [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": "".join(full)},
                ]
            return _wrap()
        self.history += [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": result},
        ]
        return result


# ============================================================
# Extraction utilities
# ============================================================

_ARCH_BLOCK_RE = re.compile(r"```ark-arch\s*([\s\S]+?)```",
                                  re.MULTILINE)


def extract_arch_json(text: str) -> Optional[dict]:
    """Extrait premier bloc ```ark-arch ... ``` du texte. Parse JSON."""
    m = _ARCH_BLOCK_RE.search(text)
    if not m:
        return None
    payload = m.group(1).strip()
    try:
        obj = json.loads(payload)
        if obj.get("version") == "ark-arch-v1":
            return obj
    except Exception:
        return None
    return None


def summarize_scene(scene) -> str:
    """Construit résumé textuel scene actuelle pour l'agent."""
    if scene is None:
        return "scene vide"
    try:
        from ark.ui.node_editor.node_item import NodeItem
        nodes = [n for n in scene.items() if isinstance(n, NodeItem)]
        if not nodes:
            return "scene vide"
        lines = [f"Nodes ({len(nodes)}) :"]
        for n in nodes:
            params = ", ".join(f"{k}={v}" for k, v in list(n.params.items())[:4])
            lines.append(f"  {n.spec.name} (id={n.id[:8]}) : {params}")
        # Connections
        cn = 0
        for it in scene.items():
            if hasattr(it, "src_socket") and hasattr(it, "dst_socket"):
                cn += 1
        lines.append(f"Connexions : {cn}")
        return "\n".join(lines)
    except Exception as e:
        return f"(erreur résumé: {e})"
