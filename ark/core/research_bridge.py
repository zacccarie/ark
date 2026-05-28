"""Bridge ARK ↔ AI-Scientist / autoresearch / scientific-visualization.

Phase 27 : 3 node types pour piloter pipelines recherche externes depuis ARK.

Nodes sont pass-through (Identity) côté compile graphe ML. L'exécution réelle
des outils externes (subprocess) se fait via méthodes `.run()` déclenchables
manuellement (futur : bouton Inspector + menu contextuel).

- IdeaGeneratorWrap : appelle `~/.claude/skills/ai-scientist/scripts/ideas.py`
- AutoResearchLoopWrap : appelle autoresearch loop (script wrapper)
- SciVizWrap : génère matplotlib viz depuis metrics dict ou JSON
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


AI_SCIENTIST_PYTHON = "/opt/anaconda3/envs/ai_scientist/bin/python"
AI_SCIENTIST_IDEAS_SCRIPT = Path.home() / ".claude" / "skills" / "ai-scientist" / "scripts" / "ideas.py"
AUTORESEARCH_SCRIPT = Path.home() / "claude-code-projects" / "autoresearch_loop.py"


@dataclass
class RunResult:
    ok: bool
    stdout: str
    stderr: str
    output_path: str | None = None
    data: Any = None


# ============================================================
# Wrappers nn.Module (pass-through pour compatibilité graphe)
# ============================================================

class IdeaGeneratorWrap(nn.Module):
    """Pass-through. Stocke params pour exécution externe via .run()."""

    def __init__(self, project_dir: str = "", n_ideas: int = 5,
                 model: str = "claude-3-5-sonnet-20241022"):
        super().__init__()
        self.project_dir = project_dir
        self.n_ideas = n_ideas
        self.model = model
        self.last_result: RunResult | None = None

    def forward(self, x):
        return x  # identité

    def run(self) -> RunResult:
        """Lance ai-scientist ideas via subprocess."""
        if not self.project_dir:
            return RunResult(False, "", "project_dir vide")
        pdir = Path(self.project_dir).expanduser()
        if not pdir.exists():
            return RunResult(False, "", f"project_dir introuvable : {pdir}")
        if not AI_SCIENTIST_IDEAS_SCRIPT.exists():
            return RunResult(False, "", f"script absent : {AI_SCIENTIST_IDEAS_SCRIPT}")
        cmd = [
            AI_SCIENTIST_PYTHON,
            str(AI_SCIENTIST_IDEAS_SCRIPT),
            str(pdir),
            "--n", str(self.n_ideas),
            "--model", self.model,
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            ideas_path = pdir / "ideas.json"
            data = None
            if ideas_path.exists():
                with open(ideas_path) as f:
                    data = json.load(f)
            res = RunResult(r.returncode == 0, r.stdout, r.stderr,
                              str(ideas_path) if ideas_path.exists() else None, data)
            self.last_result = res
            return res
        except subprocess.TimeoutExpired:
            return RunResult(False, "", "timeout 10min")
        except Exception as e:
            return RunResult(False, "", repr(e))


class AutoResearchLoopWrap(nn.Module):
    """Lance loop autoresearch (Karpathy pattern) : edit → run → measure → keep/revert."""

    def __init__(self, project_dir: str = "", max_iters: int = 10,
                 metric: str = "val_acc", goal: str = "maximize"):
        super().__init__()
        self.project_dir = project_dir
        self.max_iters = max_iters
        self.metric = metric
        self.goal = goal
        self.last_result: RunResult | None = None

    def forward(self, x):
        return x

    def run(self) -> RunResult:
        if not self.project_dir:
            return RunResult(False, "", "project_dir vide")
        pdir = Path(self.project_dir).expanduser()
        if not pdir.exists():
            return RunResult(False, "", f"project_dir introuvable : {pdir}")
        # Cherche script dans projet ou fallback générique
        script = pdir / "autoresearch_loop.py"
        if not script.exists():
            return RunResult(False, "", f"autoresearch_loop.py absent dans {pdir}")
        env = os.environ.copy()
        env["MAX_ITERS"] = str(self.max_iters)
        env["METRIC"] = self.metric
        env["GOAL"] = self.goal
        try:
            r = subprocess.run([sys.executable, str(script)],
                                 capture_output=True, text=True,
                                 timeout=3600, env=env, cwd=str(pdir))
            res = RunResult(r.returncode == 0, r.stdout, r.stderr, None, None)
            self.last_result = res
            return res
        except subprocess.TimeoutExpired:
            return RunResult(False, "", "timeout 1h")
        except Exception as e:
            return RunResult(False, "", repr(e))


class SciVizWrap(nn.Module):
    """Génère viz scientifique (matplotlib) depuis metrics ou JSON path."""

    def __init__(self, plot_kind: str = "line", title: str = "ARK metrics",
                 output_path: str = "./output/sci_viz.png",
                 metric_keys: str = "loss,val_acc"):
        super().__init__()
        self.plot_kind = plot_kind  # line | bar | scatter | hist | heatmap
        self.title = title
        self.output_path = output_path
        self.metric_keys = [k.strip() for k in metric_keys.split(",") if k.strip()]
        self.last_result: RunResult | None = None

    def forward(self, x):
        return x

    def run(self, data: dict | list | None = None,
              source_json: str | None = None) -> RunResult:
        """data prioritaire ; sinon charge source_json."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return RunResult(False, "", "matplotlib absent")

        if data is None and source_json:
            p = Path(source_json).expanduser()
            if not p.exists():
                return RunResult(False, "", f"json absent : {p}")
            with open(p) as f:
                data = json.load(f)
        if data is None:
            return RunResult(False, "", "aucune donnée fournie")

        # Style sobre cohérent ARK (dark + rouge accent)
        plt.style.use("dark_background")
        accent = "#E11D2A"
        fig, ax = plt.subplots(figsize=(8, 5), dpi=120)

        if isinstance(data, dict):
            for i, key in enumerate(self.metric_keys):
                if key not in data:
                    continue
                ys = data[key]
                xs = list(range(len(ys)))
                color = accent if i == 0 else f"C{i}"
                if self.plot_kind == "line":
                    ax.plot(xs, ys, color=color, label=key, linewidth=1.5)
                elif self.plot_kind == "bar":
                    ax.bar([x + i * 0.3 for x in xs], ys, width=0.3,
                            color=color, label=key)
                elif self.plot_kind == "scatter":
                    ax.scatter(xs, ys, color=color, label=key, s=20)
                elif self.plot_kind == "hist":
                    ax.hist(ys, bins=30, color=color, label=key, alpha=0.7)
        elif isinstance(data, list):
            ys = data
            xs = list(range(len(ys)))
            ax.plot(xs, ys, color=accent, linewidth=1.5)

        ax.set_title(self.title, fontsize=12)
        ax.grid(True, alpha=0.2)
        if self.metric_keys and isinstance(data, dict):
            ax.legend(loc="best", framealpha=0.3)

        out = Path(self.output_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, facecolor="black")
        plt.close(fig)

        res = RunResult(True, f"saved {out}", "", str(out), None)
        self.last_result = res
        return res


# ============================================================
# Handlers ARK
# ============================================================

def h_idea_generator(node, in_shapes):
    m = IdeaGeneratorWrap(
        project_dir=str(node.params.get("project_dir", "")),
        n_ideas=int(node.params.get("n_ideas", 5)),
        model=str(node.params.get("model", "claude-3-5-sonnet-20241022")),
    )
    m._ark_node_id = node.id
    out_shape = in_shapes[0] if in_shapes else (1,)
    return m, out_shape


def h_autoresearch_loop(node, in_shapes):
    m = AutoResearchLoopWrap(
        project_dir=str(node.params.get("project_dir", "")),
        max_iters=int(node.params.get("max_iters", 10)),
        metric=str(node.params.get("metric", "val_acc")),
        goal=str(node.params.get("goal", "maximize")),
    )
    m._ark_node_id = node.id
    out_shape = in_shapes[0] if in_shapes else (1,)
    return m, out_shape


def h_sci_viz(node, in_shapes):
    m = SciVizWrap(
        plot_kind=str(node.params.get("plot_kind", "line")),
        title=str(node.params.get("title", "ARK metrics")),
        output_path=str(node.params.get("output_path", "./output/sci_viz.png")),
        metric_keys=str(node.params.get("metric_keys", "loss,val_acc")),
    )
    m._ark_node_id = node.id
    out_shape = in_shapes[0] if in_shapes else (1,)
    return m, out_shape


def register_research_handlers(handlers_dict: dict) -> bool:
    handlers_dict["idea_generator"] = h_idea_generator
    handlers_dict["autoresearch_loop"] = h_autoresearch_loop
    handlers_dict["sci_viz"] = h_sci_viz
    return True


def register_research_specs(registry) -> bool:
    from ark.nodes.registry import NodeSpec, ParamSpec, SocketSpec

    registry.register(NodeSpec(
        name="idea_generator", title="Idea Generator (AI-Scientist)",
        category="Research",
        inputs=[SocketSpec("trigger", "any")],
        outputs=[SocketSpec("ideas", "any")],
        params=[
            ParamSpec("project_dir", "str", ""),
            ParamSpec("n_ideas", "int", 5, min=1, max=50),
            ParamSpec("model", "choice", "claude-3-5-sonnet-20241022",
                        options=[
                            "claude-3-5-sonnet-20241022",
                            "claude-3-opus-20240229",
                            "claude-3-5-haiku-20241022",
                            "gpt-4o-2024-05-13",
                            "gpt-4o-mini",
                        ]),
        ],
        description="Génère idées recherche via Sakana AI-Scientist. project_dir doit contenir seed_ideas.json + prompt.json + experiment.py.",
        param_help={
            "project_dir": "Dossier projet avec seed_ideas.json/prompt.json/experiment.py.",
            "n_ideas": "Nombre d'idées à générer.",
            "model": "LLM utilisé pour idéation.",
        },
    ))
    registry.register(NodeSpec(
        name="autoresearch_loop", title="AutoResearch Loop",
        category="Research",
        inputs=[SocketSpec("trigger", "any")],
        outputs=[SocketSpec("metrics", "any")],
        params=[
            ParamSpec("project_dir", "str", ""),
            ParamSpec("max_iters", "int", 10, min=1, max=1000),
            ParamSpec("metric", "str", "val_acc"),
            ParamSpec("goal", "choice", "maximize",
                        options=["maximize", "minimize"]),
        ],
        description="Loop autoresearch (Karpathy) : edit → run → measure → keep/revert via git.",
        param_help={
            "project_dir": "Projet contenant autoresearch_loop.py.",
            "max_iters": "Itérations max avant arrêt.",
            "metric": "Nom de la métrique à optimiser.",
            "goal": "Direction d'optimisation.",
        },
    ))
    registry.register(NodeSpec(
        name="sci_viz", title="Scientific Viz",
        category="Research",
        inputs=[SocketSpec("metrics", "any")],
        outputs=[SocketSpec("image", "image")],
        params=[
            ParamSpec("plot_kind", "choice", "line",
                        options=["line", "bar", "scatter", "hist"]),
            ParamSpec("title", "str", "ARK metrics"),
            ParamSpec("output_path", "str", "./output/sci_viz.png"),
            ParamSpec("metric_keys", "str", "loss,val_acc"),
        ],
        description="Visualisation scientifique matplotlib (style dark + accent #E11D2A).",
        has_thumbnail=True,
        param_help={
            "plot_kind": "Type de plot.",
            "title": "Titre du graphe.",
            "output_path": "Chemin PNG sortie.",
            "metric_keys": "Clés metrics (comma-separated) à plotter depuis dict.",
        },
    ))
    return True
