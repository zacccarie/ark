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

═══════════════════════════════════════════════════════════════
NODES ARK DISPONIBLES (catalogue compact par catégorie)
═══════════════════════════════════════════════════════════════
{nodes_catalog}

═══════════════════════════════════════════════════════════════
NODES PHARES (spec détaillée — params + types)
═══════════════════════════════════════════════════════════════
{nodes_detailed}

═══════════════════════════════════════════════════════════════
TEMPLATES PRÉFABRIQUÉS (chargeables via op spawn_template)
═══════════════════════════════════════════════════════════════
{templates_list}

═══════════════════════════════════════════════════════════════
ACTIONS APP DISPONIBLES (ops dans bloc ark-action)
═══════════════════════════════════════════════════════════════
Tu peux modifier la scène COURANTE atomiquement via un bloc :
```ark-action
[
  {{"op": "node_add", "type": "<type_name>", "params": {{...}},
                       "pos": [x, y], "id": "<optional_id>"}},
  {{"op": "node_edit", "id": "<node_id>", "params": {{...}}}},
  {{"op": "node_remove", "id": "<node_id>"}},
  {{"op": "connect", "from": "<src_id>", "from_socket": 0,
                      "to": "<dst_id>", "to_socket": 0}},
  {{"op": "disconnect", "from": "<src_id>", "from_socket": 0,
                         "to": "<dst_id>", "to_socket": 0}},
  {{"op": "spawn_template", "name": "<template_name>"}},
  {{"op": "compile"}},
  {{"op": "train"}},
  {{"op": "log", "message": "..."}}
]
```

Pour proposition complète (nouveau projet) → bloc ark-arch :
```ark-arch
{{
  "version": "ark-arch-v1",
  "name": "Nom court",
  "nodes": [
    {{"id":"n1","type":"text_input","params":{{"prompt":"..."}},
      "pos":[0,200],"width":220}}
  ],
  "connections": [
    {{"from_node":"n1","from_socket":0,"to_node":"n2","to_socket":0}}
  ],
  "metadata": {{"tags":["..."]}}
}}
```

═══════════════════════════════════════════════════════════════
WORKFLOW
═══════════════════════════════════════════════════════════════
1. Comprends intent (tâche, données, contraintes compute).
2. Si problème mal posé → 1-2 questions ciblées.
3. Si scène vide ou nouvelle tâche → bloc ark-arch (full project).
4. Si scène existe et user demande modification → bloc ark-action
   (préfère ark-action pour éviter perte travail user).
5. Justifications scientifiques courtes (cite papers si pertinent).
6. Mentionne trade-offs (compute, mémoire, expressivité).

CONTRAINTES :
- NEVER invent un type de node absent du catalogue.
- Pos : grille 280px horizontal, 200px vertical.
- Réponses en français, concises. Caveman-style autorisé.
- Si user demande "ajoute X" → ark-action node_add, PAS ark-arch.
- Toujours nodes IDs cohérents quand connect (utiliser ceux que tu
  viens de node_add, ou ceux fournis dans ARCHITECTURE COURANTE).
"""


# Sélection nodes à détailler (params + descriptions complets)
_DETAILED_NODES = {
    # Input
    "text_input", "image_input", "tabular_input",
    "load_image_file", "load_video", "load_sound", "load_csv",
    "torchvision_dataset", "synthetic_dataset",
    # Neural
    "dense", "conv2d", "conv_transpose2d", "max_pool", "avg_pool",
    "batch_norm", "layer_norm", "group_norm", "dropout",
    "embedding", "positional_encoding", "transformer_block",
    "self_attention", "residual_connection",
    # Loss + optim
    "cross_entropy", "mse_loss", "adam", "sgd",
    # Diffusion
    "timestep_embedding", "class_embedding", "cross_attention",
    "diffusion_scheduler", "unet_mini", "ddim_sampler",
    # Multimodal
    "text_to_frame_seq", "video_writer", "text_to_melody",
    "audio_synth", "audio_player", "audio_writer",
    "imagenet_classifier", "caption_formatter",
    "musicgen_audio", "bark_audio", "text2video_diffusion",
    # Tiny LLM
    "tiny_llm_corpus", "tiny_llm_model", "tiny_llm_generate",
    # Output Viewer
    "text_viewer", "number_display", "image_viewer",
    "top_k_predictions", "prob_distribution",
    # Code
    "code_block",
}


def _nodes_catalog() -> str:
    """Catalogue compact : juste names par catégorie."""
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


def _nodes_detailed() -> str:
    """Spec détaillée pour les nodes phares (params + kinds + defaults)."""
    try:
        from ark.nodes.registry import default_registry
        r = default_registry()
        lines = []
        for name in sorted(_DETAILED_NODES):
            s = r.get(name)
            if s is None:
                continue
            param_strs = []
            for p in s.params:
                opts = ""
                if p.kind == "choice" and p.options:
                    opts = f"∈[{'|'.join(map(str, p.options))}]"
                rng = ""
                if p.min is not None or p.max is not None:
                    rng = f" ({p.min}..{p.max})"
                param_strs.append(
                    f"{p.name}:{p.kind}={p.default}{opts}{rng}")
            params_txt = ", ".join(param_strs) if param_strs else "—"
            short_desc = (s.description or "").split(".")[0][:120]
            inputs = ", ".join(f"{x.label}:{x.socket_type}" for x in s.inputs)
            outputs = ", ".join(f"{x.label}:{x.socket_type}" for x in s.outputs)
            lines.append(
                f"• {s.name} [{s.category}] — {short_desc}\n"
                f"    in: ({inputs or 'none'}) → out: ({outputs or 'none'})\n"
                f"    params: {params_txt}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"  (erreur détails: {e})"


def _templates_list() -> str:
    try:
        from ark.core.templates import TEMPLATES_LIST
        return "\n".join(f"  - {n} : {label}" for n, label in TEMPLATES_LIST)
    except Exception:
        return "  (none)"


def system_prompt() -> str:
    return SYSTEM_PROMPT.format(
        nodes_catalog=_nodes_catalog(),
        nodes_detailed=_nodes_detailed(),
        templates_list=_templates_list(),
    )


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
_ACTION_BLOCK_RE = re.compile(r"```ark-action\s*([\s\S]+?)```",
                                     re.MULTILINE)


def extract_actions(text: str) -> Optional[list]:
    """Extrait premier bloc ```ark-action JSON array."""
    m = _ACTION_BLOCK_RE.search(text)
    if not m:
        return None
    try:
        arr = json.loads(m.group(1).strip())
        if isinstance(arr, list):
            return arr
    except Exception:
        return None
    return None


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
