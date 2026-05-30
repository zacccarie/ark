"""Phase 81 — Taxonomie du type de modèle + critère de validité associé.

Le squelette workflow (build→pipeline→train→test→bench→ship) est universel.
Seule la MÉTRIQUE de validité change selon le type de modèle :

  - classifier   → accuracy / F1 / confusion (a une vérité : label)
  - regressor    → MAE / MSE (vérité continue)
  - lm           → perplexity / bits-per-byte (self-supervised, pas de label)
  - generative   → FID / CLIP-score (AUCUNE vérité ponctuelle, distributionnel)
  - world_model  → rollout error horizon-N (futures frames, multi-step)

`detect_kind` infère le type depuis les types de nodes présents dans la scène.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass
class KindSpec:
    kind: str               # classifier|regressor|lm|generative|world_model|unknown
    label: str              # libellé FR
    primary_metric: str     # accuracy|mae|perplexity|fid|rollout_mse
    metric_direction: str   # "higher" (mieux ↑) | "lower" (mieux ↓)
    needs_labels: bool      # vérité ponctuelle dispo ?
    has_ground_truth: bool  # peut-on calculer une vérité exacte ?
    demo_hint: str          # démo "ça marche"
    pass_threshold: Optional[float] = None  # seuil auto-validation (si applicable)
    secondary: tuple = ()   # autres métriques utiles


_KINDS = {
    "classifier": KindSpec(
        "classifier", "Classification", "accuracy", "higher", True, True,
        "1 image → label reconnu", pass_threshold=0.85,
        secondary=("f1", "confusion", "top_k")),
    "regressor": KindSpec(
        "regressor", "Régression", "mae", "lower", True, True,
        "1 input → valeur prédite", pass_threshold=None,
        secondary=("mse", "r2")),
    "lm": KindSpec(
        "lm", "Modèle de langage", "perplexity", "lower", False, True,
        "prompt → complétion cohérente", pass_threshold=None,
        secondary=("bits_per_byte", "cross_entropy")),
    "generative": KindSpec(
        "generative", "Génératif (image/audio)", "fid", "lower", False, False,
        "prompt → échantillon plausible", pass_threshold=None,
        secondary=("clip_score", "inception_score")),
    "world_model": KindSpec(
        "world_model", "World model", "rollout_mse", "lower", True, True,
        "state → N frames futures", pass_threshold=None,
        secondary=("ssim", "lpips", "horizon")),
    "unknown": KindSpec(
        "unknown", "Indéterminé", "loss", "lower", False, False,
        "—", pass_threshold=None),
}

# Marqueurs de types de nodes (substring match, insensible à la casse)
_WORLD = {"rssm", "rollout", "vg_block", "dynamics", "world_model", "regime",
          "phase_space", "latent_state"}
_GEN = {"diffusion", "unet", "ddim", "ddpm", "vae", "scheduler",
        "cross_attention", "timestep_embedding", "gan", "musicgen", "bark",
        "text2video", "text2image", "stable_diffusion", "decoder_image",
        "img2img", "text_to_frame", "text_to_melody"}
_LM = {"tiny_llm", "causal_attention", "causal_self_attention", "gpt",
       "transformer_block", "token_embedding", "char_lm", "llm_generate",
       "positional", "self_attention"}
_CLS = {"mnist_recognizer", "softmax", "cross_entropy", "classif",
        "top_k", "logits", "argmax"}
_REG = {"mse", "regression", "linear_out", "scalar_out"}


def detect_kind(node_types: Iterable[str]) -> KindSpec:
    """Infère le KindSpec depuis les types de nodes (priorité : spécifique
    d'abord)."""
    types = {str(t).lower() for t in node_types}

    def hit(markers: set) -> bool:
        return any(any(mk in t for t in types) for mk in markers)

    if hit(_WORLD):
        return _KINDS["world_model"]
    if hit(_GEN):
        return _KINDS["generative"]
    if hit(_LM):
        return _KINDS["lm"]
    if hit(_CLS):
        return _KINDS["classifier"]
    if hit(_REG):
        return _KINDS["regressor"]
    # Défaut : présence de couches denses/conv → classifieur (cas démo MNIST)
    if any(("dense" in t or "linear" in t or "conv" in t) for t in types):
        return _KINDS["classifier"]
    return _KINDS["unknown"]


def detect_kind_from_scene(scene) -> KindSpec:
    try:
        names = [n.spec.name for n in scene.nodes()]
    except Exception:
        names = []
    return detect_kind(names)


def kind_spec(kind: str) -> KindSpec:
    return _KINDS.get(kind, _KINDS["unknown"])
