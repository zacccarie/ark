"""Phase 61 — QWebChannel bridge for React WebEngine panels.

Expose Python methods + Qt signals to React via QWebChannel. React side
loads qrc:///qtwebchannel/qwebchannel.js, instantiates QWebChannel, gets
'bridge' object with these slots/signals.
"""
from __future__ import annotations
import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import QObject, Signal, Slot, Qt


class ArkBridge(QObject):
    """Façade Python → JS. All Slots() callable from JS side.

    Signals JS-side subscribed via `bridge.signalName.connect(cb)`.
    """

    # Signals — dicts passed as JSON strings (QWebChannel dict-convert is
    # version-fragile: "_pythonToCppCopy: Cannot copy-convert dict to C++").
    selection_changed = Signal(str)       # JSON node data or ""
    log_appended = Signal(str)            # one log line
    template_loaded = Signal(str)         # template name
    compile_progress = Signal(float, str)  # 0..1, state

    # Architect signals
    architect_chunk = Signal(str)        # streamed token
    architect_done = Signal(str)          # full final message
    architect_arch_proposed = Signal(str)  # JSON arch
    architect_actions_proposed = Signal(str)  # JSON list of ops
    # Modal signals (Phase 68)
    modal_log = Signal(str)
    modal_done = Signal(str)  # JSON result

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._mw = main_window  # weakref-ish, set lazy
        # Phase 65 : lazy architect agent
        self._architect = None

    # ============================================================
    # Palette
    # ============================================================

    @Slot(result=list)
    def list_categories(self):
        from ark.nodes.registry import default_registry
        r = default_registry()
        cats = sorted({s.category for s in r.all()})
        return cats

    @Slot(str, result=list)
    def list_nodes_in_category(self, category: str):
        from ark.nodes.registry import default_registry
        r = default_registry()
        nodes = [s for s in r.all() if s.category == category]
        nodes.sort(key=lambda s: s.title)
        return [{"name": s.name, "title": s.title} for s in nodes]

    # ============================================================
    # Inspector
    # ============================================================

    @Slot(result=str)
    def get_selected_node(self):
        import json as _json
        if self._mw is None or not hasattr(self._mw, "scene"):
            return ""
        sel = [it for it in self._mw.scene.selectedItems()
                if hasattr(it, "spec")]
        if not sel:
            return ""
        return _json.dumps(self._serialize_node(sel[0]))

    def _serialize_node(self, node) -> dict:
        spec = node.spec
        out = {
            "id": node.id,
            "type": spec.name,
            "title": spec.title,
            "category": spec.category,
            "description": spec.description,
            "role": getattr(spec, "role", ""),
            "function": getattr(spec, "function", ""),
            "params": [
                {
                    "name": p.name,
                    "kind": p.kind,
                    "default": p.default,
                    "options": list(p.options) if p.options else [],
                    "min": p.min,
                    "max": p.max,
                    "file_filter": getattr(p, "file_filter", ""),
                }
                for p in spec.params
            ],
            "values": dict(node.params),
        }
        # Live outputs
        mod = None
        if hasattr(self._mw, "_compiled") and self._mw._compiled is not None:
            md = getattr(self._mw._compiled.model, "modules_dict", {})
            try:
                mod = md.get(node.id)
            except Exception:
                mod = None
        if mod is not None:
            for attr in ("last_text", "last_value", "last_generated",
                          "last_audio_path"):
                v = getattr(mod, attr, None)
                if v not in (None, ""):
                    out[attr] = v
        # Preview image base64 if present
        qimg = getattr(node, "_preview_qimage", None)
        if qimg is not None:
            try:
                from PySide6.QtCore import QBuffer, QByteArray, QIODevice
                buf = QBuffer()
                buf.open(QIODevice.OpenModeFlag.WriteOnly)
                qimg.save(buf, "PNG")
                b64 = base64.b64encode(bytes(buf.data())).decode("ascii")
                out["preview_image"] = f"data:image/png;base64,{b64}"
            except Exception:
                pass
        return out

    @Slot(str)
    def spawn_node(self, node_type: str):
        """Phase 62 : palette click → create node at viewport center."""
        if self._mw is None or not hasattr(self._mw, "scene"):
            return
        try:
            view = getattr(self._mw, "view", None)
            if view is not None:
                from PySide6.QtCore import QPointF
                center = view.mapToScene(view.viewport().rect().center())
                x, y = center.x(), center.y()
            else:
                x, y = 200.0, 200.0
            self._mw.scene._raw_create_node(node_type, float(x), float(y))
        except Exception as e:
            self.emit_log(f"[spawn_node error] {e}")

    @Slot(str, str, str, result=str)
    def pick_file(self, node_id: str, param: str, file_filter: str):
        """Phase 76 : ouvre QFileDialog, set le param, retourne le chemin."""
        from PySide6.QtWidgets import QFileDialog
        flt = file_filter or "Tous fichiers (*)"
        path, _ = QFileDialog.getOpenFileName(self._mw, "Choisir un fichier",
                                                  "", flt)
        if not path:
            return ""
        # Applique au node + preview immédiate si image
        if self._mw is not None and hasattr(self._mw, "scene"):
            for it in self._mw.scene.items():
                if getattr(it, "id", None) == node_id and hasattr(it, "set_param"):
                    it.set_param(param, path)
                    self._maybe_preview_image(it, path)
                    break
        return path

    def _maybe_preview_image(self, node, path: str):
        """Phase 77b : si fichier image, affiche miniature dans le node direct."""
        low = path.lower()
        if not low.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif",
                                ".webp", ".tiff")):
            return
        try:
            from PIL import Image
            import numpy as np
            import torch
            img = Image.open(path).convert("RGB")
            img.thumbnail((256, 256))
            arr = np.array(img).astype("float32") / 255.0
            t = torch.from_numpy(arr.transpose(2, 0, 1))
            if hasattr(node, "set_preview_image"):
                node.set_preview_image(t.clamp(0, 1))
        except Exception:
            pass

    @Slot(str, str, "QVariant")
    def set_node_param(self, node_id: str, name: str, value: Any):
        if self._mw is None or not hasattr(self._mw, "scene"):
            return
        for it in self._mw.scene.items():
            if getattr(it, "id", None) == node_id and hasattr(it, "set_param"):
                it.set_param(name, value)
                return

    # ============================================================
    # Toolbar actions
    # ============================================================

    @Slot(result=list)
    def list_templates(self):
        from ark.core.templates import TEMPLATES_LIST
        return [{"name": n, "label": l} for n, l in TEMPLATES_LIST]

    @Slot(str)
    def load_template(self, name: str):
        if self._mw and hasattr(self._mw, "_load_template"):
            self._mw._load_template(name)
            self.template_loaded.emit(name)

    @Slot(int)
    def compile(self, batch: int = 1):
        if self._mw and hasattr(self._mw, "_on_compile"):
            self._mw._on_compile()

    @Slot(int)
    def train(self, batch: int = 1):
        if self._mw and hasattr(self._mw, "_on_train"):
            self._mw._on_train()

    @Slot()
    def stop(self):
        if self._mw and hasattr(self._mw, "_on_stop"):
            try:
                self._mw._on_stop()
            except Exception:
                pass

    @Slot()
    def open_extensions(self):
        # Phase 64 : Extensions button → templates browser (most useful action)
        if self._mw is None:
            return
        for attr in ("_open_extensions", "_on_browse_templates"):
            fn = getattr(self._mw, attr, None)
            if callable(fn):
                fn()
                return

    @Slot()
    def open_settings(self):
        if self._mw is None:
            return
        for attr in ("_open_settings", "_on_open_shortcuts",
                          "_on_open_help_page"):
            fn = getattr(self._mw, attr, None)
            if callable(fn):
                fn()
                return

    @Slot()
    def toggle_inspector(self):
        if self._mw and hasattr(self._mw, "_toggle_inspector"):
            self._mw._toggle_inspector()

    # ============================================================
    # View controls (zoom / fit / map / snap)
    # ============================================================

    @Slot(float)
    def set_zoom(self, factor: float):
        if self._mw is None or not hasattr(self._mw, "view"):
            return
        v = self._mw.view
        try:
            v.resetTransform()
            v.scale(factor, factor)
        except Exception:
            pass

    @Slot()
    def fit_view(self):
        if self._mw is None or not hasattr(self._mw, "view"):
            return
        v = self._mw.view
        try:
            rect = self._mw.scene.itemsBoundingRect()
            if rect.isEmpty():
                return
            v.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        except Exception:
            pass

    @Slot(bool)
    def toggle_minimap(self, on: bool):
        if self._mw and hasattr(self._mw, "_minimap"):
            try:
                self._mw._minimap.setVisible(on)
            except Exception:
                pass

    @Slot(bool)
    def toggle_snap(self, on: bool):
        if self._mw and hasattr(self._mw, "scene"):
            try:
                self._mw.scene.snap_to_grid = on
            except Exception:
                pass

    # ============================================================
    # Audio
    # ============================================================

    @Slot(str)
    def play_audio(self, path: str):
        if not path or not Path(path).exists():
            return
        import platform, subprocess
        sysname = platform.system()
        if sysname == "Darwin":
            subprocess.Popen(["afplay", path],
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
        elif sysname == "Linux":
            subprocess.Popen(["aplay", "-q", path],
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
        else:
            subprocess.Popen(
                ["powershell", "-c",
                 f"(New-Object Media.SoundPlayer '{path}').PlaySync();"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @Slot(str)
    def download_audio(self, src: str):
        if not src or not Path(src).exists():
            return
        from PySide6.QtWidgets import QFileDialog
        dest, _ = QFileDialog.getSaveFileName(
            self._mw, "Save audio as…",
            str(Path.home() / "Downloads" / Path(src).name),
            "Audio (*.wav);;All files (*)")
        if dest:
            import shutil
            shutil.copy2(src, dest)

    # ============================================================
    # Architect agent (Phase 65)
    # ============================================================

    def _architect_instance(self):
        if self._architect is None:
            from ark.core.architect_agent import ArchitectAgent
            self._architect = ArchitectAgent()
        return self._architect

    @Slot(result=str)
    def architect_providers(self):
        import json as _json
        from ark.core.llm_providers import available_providers, DEFAULT_MODELS
        return _json.dumps(
            [{"name": p, "default_model": DEFAULT_MODELS.get(p, "")}
             for p in available_providers()])

    @Slot(result=str)
    def architect_get_config(self):
        import json as _json
        from ark.core.llm_providers import load_secrets
        agent = self._architect_instance()
        return _json.dumps({
            "provider": agent.provider_name,
            "model": agent.model,
            "secrets_present": {k: bool(v.get("api_key") or v.get("base_url"))
                                  for k, v in load_secrets().items()},
        })

    @Slot(str, str, str)
    def architect_set_provider(self, provider: str, model: str, api_key: str):
        from ark.core.llm_providers import load_secrets, save_secrets
        agent = self._architect_instance()
        if api_key:
            secrets = load_secrets()
            secrets.setdefault(provider, {})
            if provider == "ollama":
                secrets[provider]["base_url"] = api_key
            else:
                secrets[provider]["api_key"] = api_key
            save_secrets(secrets)
        agent.set_provider(provider, model or None)

    @Slot()
    def architect_reset(self):
        agent = self._architect_instance()
        agent.reset()

    @Slot(str)
    def architect_chat(self, user_msg: str):
        """Stream response. Emits architect_chunk(str) per token,
        architect_done(full) at end, architect_arch_proposed(dict) if
        ark-arch-v1 block detected."""
        from ark.core.architect_agent import summarize_scene, extract_arch_json
        agent = self._architect_instance()
        scene = getattr(self._mw, "scene", None)
        summary = summarize_scene(scene)
        # Stream in background thread
        from PySide6.QtCore import QThread, Signal as _S, QObject
        class _Worker(QObject):
            chunk_sig = _S(str)
            done_sig = _S(str)

            def __init__(s, agent, msg, summary):
                super().__init__()
                s.agent = agent
                s.msg = msg
                s.summary = summary

            def run(s):
                try:
                    full = []
                    for chunk in s.agent.chat(s.msg, scene_summary=s.summary,
                                                    stream=True):
                        full.append(chunk)
                        s.chunk_sig.emit(chunk)
                    s.done_sig.emit("".join(full))
                except Exception as e:
                    err = f"[architect error] {e}"
                    s.chunk_sig.emit(err)
                    s.done_sig.emit(err)

        self._arch_thread = QThread()
        self._arch_worker = _Worker(agent, user_msg, summary)
        self._arch_worker.moveToThread(self._arch_thread)
        self._arch_worker.chunk_sig.connect(self.architect_chunk)
        self._arch_worker.done_sig.connect(self._on_architect_done)
        self._arch_thread.started.connect(self._arch_worker.run)
        self._arch_thread.start()

    def _on_architect_done(self, full: str):
        import json as _json
        from ark.core.architect_agent import extract_arch_json, extract_actions
        arch = extract_arch_json(full)
        if arch is not None:
            self.architect_arch_proposed.emit(_json.dumps(arch))
        acts = extract_actions(full)
        if acts is not None:
            self.architect_actions_proposed.emit(_json.dumps(acts))
        self.architect_done.emit(full)
        try:
            self._arch_thread.quit()
            self._arch_thread.wait(2000)
        except Exception:
            pass

    @Slot(str)
    def architect_apply_arch(self, arch_json: str):
        """Charge l'arch proposée comme nouveau tab. Input = JSON string."""
        import json as _json
        if self._mw is None:
            return
        try:
            arch = _json.loads(arch_json) if isinstance(arch_json, str) else arch_json
            name = arch.get("name", "Architect")
            if hasattr(self._mw, "tabs"):
                self._mw.tabs.new_project(name, payload=arch)
        except Exception as e:
            self.emit_log(f"[architect apply error] {e}")

    @Slot(str, result=str)
    def architect_apply_actions(self, actions_json: str):
        """Phase 66 : applique liste d'ops sur scène courante. JSON in/out."""
        import json as _json
        try:
            actions = _json.loads(actions_json) if isinstance(actions_json, str) else actions_json
        except Exception as e:
            return _json.dumps({"ok": 0, "fail": 0, "errors": [f"bad json: {e}"]})
        if self._mw is None:
            return _json.dumps({"ok": 0, "fail": 0, "errors": ["no main window"]})
        scene = getattr(self._mw, "scene", None)
        view = getattr(self._mw, "view", None)
        if scene is None:
            return _json.dumps({"ok": 0, "fail": 0, "errors": ["no scene"]})
        ok, fail, errors = 0, 0, []
        # ID alias map (LLM-provided ids → real node ids after creation)
        id_map: dict[str, str] = {}

        def _resolve_id(s: str) -> Optional[str]:
            if s in id_map:
                return id_map[s]
            for it in scene.items():
                if getattr(it, "id", None) == s:
                    return s
            return None

        for i, act in enumerate(actions):
            try:
                op = act.get("op")
                if op == "node_add":
                    pos = act.get("pos") or [200.0, 200.0]
                    node = scene._raw_create_node(
                        act["type"], float(pos[0]), float(pos[1]))
                    if node is None:
                        raise ValueError(f"unknown type {act['type']}")
                    # Apply params via set_param
                    for k, v in (act.get("params") or {}).items():
                        if hasattr(node, "set_param"):
                            node.set_param(k, v)
                    if "id" in act:
                        id_map[act["id"]] = node.id
                    ok += 1
                elif op == "node_edit":
                    rid = _resolve_id(act["id"])
                    if rid is None:
                        raise ValueError(f"node id not found: {act['id']}")
                    for it in scene.items():
                        if getattr(it, "id", None) == rid:
                            for k, v in (act.get("params") or {}).items():
                                it.set_param(k, v)
                            break
                    ok += 1
                elif op == "node_remove":
                    rid = _resolve_id(act["id"])
                    if rid is None:
                        raise ValueError(f"node id not found: {act['id']}")
                    for it in list(scene.items()):
                        if getattr(it, "id", None) == rid:
                            scene.removeItem(it)
                            break
                    ok += 1
                elif op == "connect":
                    src_id = _resolve_id(act["from"])
                    dst_id = _resolve_id(act["to"])
                    if not src_id or not dst_id:
                        raise ValueError("connect ids unresolved")
                    src_node = dst_node = None
                    for it in scene.items():
                        if getattr(it, "id", None) == src_id:
                            src_node = it
                        elif getattr(it, "id", None) == dst_id:
                            dst_node = it
                    if not src_node or not dst_node:
                        raise ValueError("connect nodes missing")
                    fs = int(act.get("from_socket", 0))
                    ts = int(act.get("to_socket", 0))
                    src_sock = src_node.output_sockets[fs]
                    dst_sock = dst_node.input_sockets[ts]
                    scene._raw_create_connection(src_sock, dst_sock)
                    ok += 1
                elif op == "spawn_template":
                    from ark.core.templates import get_template
                    tpl = get_template(act["name"])
                    if tpl is None:
                        raise ValueError(f"unknown template {act['name']}")
                    if hasattr(self._mw, "tabs"):
                        self._mw.tabs.new_project(
                            tpl.get("name", act["name"]), payload=tpl)
                    ok += 1
                elif op == "compile":
                    if hasattr(self._mw, "_on_compile"):
                        self._mw._on_compile()
                    ok += 1
                elif op == "train":
                    if hasattr(self._mw, "_on_train"):
                        self._mw._on_train()
                    ok += 1
                elif op == "log":
                    self.emit_log(f"[architect] {act.get('message', '')}")
                    ok += 1
                else:
                    raise ValueError(f"unknown op: {op}")
            except Exception as e:
                fail += 1
                errors.append(f"op {i} ({act.get('op')}): {e}")
        if view is not None:
            try:
                view.viewport().update()
            except Exception:
                pass
        return _json.dumps({"ok": ok, "fail": fail, "errors": errors})

    @Slot(result=str)
    def architect_get_scene_json(self):
        """Renvoie scene actuelle sérialisée pour debug/inspection LLM."""
        import json as _json
        if self._mw is None or not hasattr(self._mw, "scene"):
            return "{}"
        try:
            from ark.core.format import scene_to_payload
            return _json.dumps(scene_to_payload(self._mw.scene, name="current"))
        except Exception:
            return "{}"

    # ============================================================
    # Modal remote compute (Phase 68)
    # ============================================================

    @Slot(result=str)
    def viz_get_model_config(self):
        """Phase 69 : extrait config GPT depuis tiny_llm_model OU
        transformer_block de la scène courante pour le visualizer 3D."""
        import json as _json
        if self._mw is None or not hasattr(self._mw, "scene"):
            return ""
        try:
            for it in self._mw.scene.items():
                spec = getattr(it, "spec", None)
                if spec is None:
                    continue
                p = it.params
                if spec.name == "tiny_llm_model":
                    d = int(p.get("d_model", 64))
                    return _json.dumps({
                        "nLayers": int(p.get("n_blocks", 2)),
                        "nHeads": int(p.get("n_heads", 4)),
                        "nEmbd": d,
                        "seqLen": int(p.get("block_size", 64)),
                        "vocab": 96,
                        "name": p.get("save_name", "tiny_llm"),
                    })
                if spec.name == "transformer_block":
                    d = int(p.get("d_model", p.get("embed_dim", 128)))
                    return _json.dumps({
                        "nLayers": int(p.get("num_layers", p.get("depth", 2))),
                        "nHeads": int(p.get("num_heads", p.get("n_heads", 4))),
                        "nEmbd": d,
                        "seqLen": int(p.get("max_len", 64)),
                        "vocab": int(p.get("vocab_size", 1000)),
                        "name": "transformer",
                    })
            # Phase 70 : pas de transformer → mode générique (toute archi)
            return _json.dumps(self._build_generic_layers())
        except Exception as e:
            return _json.dumps({"generic": True, "layers": [],
                                  "error": str(e)})

    def _build_generic_layers(self) -> dict:
        """Topo-ordonne la scène → couches pour viz 3D générique.

        Phase 75b : compile headless → vraies shapes + introspection modules
        (params réels, neurones, sous-couches). Fallback heuristique si
        compile échoue.
        """
        from ark.core.graph_compiler import _topo_sort_scene
        CAT_COLOR = {
            "Input": "embed", "Neural": "mlp", "Neural Networks": "mlp",
            "Convolution": "attn", "Normalization": "ln",
            "Activation": "softmax", "Diffusion": "attn",
            "Transformers": "attn", "Tiny LLM": "attn",
            "Output Viewer": "output", "Data": "proj",
            "Loss": "output", "Optimizer": "proj", "Demo": "embed",
            "Multimodal": "attn",
        }
        try:
            ordered, incoming, outgoing, routed = _topo_sort_scene(self._mw.scene)
        except Exception:
            ordered = self._mw.scene.nodes()
            incoming = {}

        # Compile headless pour shapes + modules réels
        shapes_by_id, modules_by_id = {}, {}
        try:
            from ark.core.format import scene_to_payload
            from ark.core.headless_scene import compile_payload
            payload = scene_to_payload(self._mw.scene, name="viz")
            ncls = payload.get("metadata", {}).get("num_classes")
            cg = compile_payload(payload, num_classes=ncls)
            shapes_by_id = getattr(cg.model, "shapes_by_id", {}) or {}
            md = getattr(cg.model, "modules_dict", {})
            modules_by_id = dict(md) if hasattr(md, "keys") else {}
        except Exception:
            pass

        def _introspect(mod):
            """Retourne (params, neurons, subs[list str])."""
            import torch.nn as nn
            if mod is None:
                return 0, 0, []
            try:
                params = sum(p.numel() for p in mod.parameters())
            except Exception:
                params = 0
            subs, neurons = [], 0
            try:
                for sm in mod.modules():
                    if isinstance(sm, nn.Linear):
                        subs.append(f"Linear {sm.in_features}→{sm.out_features}")
                        neurons = sm.out_features
                    elif isinstance(sm, nn.Conv2d):
                        subs.append(
                            f"Conv2d {sm.in_channels}→{sm.out_channels} "
                            f"k{sm.kernel_size[0]}")
                        neurons = sm.out_channels
                    elif isinstance(sm, nn.Conv1d):
                        subs.append(
                            f"Conv1d {sm.in_channels}→{sm.out_channels}")
                        neurons = sm.out_channels
                    elif isinstance(sm, (nn.LayerNorm, nn.BatchNorm1d,
                                            nn.BatchNorm2d, nn.GroupNorm)):
                        subs.append(type(sm).__name__)
                    elif isinstance(sm, nn.MultiheadAttention):
                        subs.append(f"MHA dim{sm.embed_dim} h{sm.num_heads}")
                    elif isinstance(sm, nn.Embedding):
                        subs.append(
                            f"Embedding {sm.num_embeddings}×{sm.embedding_dim}")
            except Exception:
                pass
            return params, neurons, subs[:8]

        def _fmt_shape(s):
            if isinstance(s, tuple):
                if s and isinstance(s[0], tuple):
                    return " | ".join("×".join(map(str, t)) for t in s)
                return "×".join(map(str, s))
            return str(s)

        layers = []
        for n in ordered:
            spec = n.spec
            p = n.params
            mod = modules_by_id.get(n.id)
            params, neurons, subs = _introspect(mod)
            out_shape = shapes_by_id.get(n.id)
            # taille : params réels si dispo, sinon heuristique
            if params > 0:
                size = params
            else:
                size = 1
                for key in ("neurons", "channels", "out_features",
                              "hidden_dim", "out_dim", "features", "d_model"):
                    if key in p:
                        try:
                            size = int(p[key]); break
                        except Exception:
                            pass
            # sub résumé : sous-couches OU params
            if subs:
                sub = " · ".join(subs[:3])
            else:
                sub = " ".join(f"{k}={p[k]}" for k in
                                 ("neurons", "channels", "kernel", "activation")
                                 if k in p)
            layers.append({
                "id": n.id,
                "type": spec.name,
                "label": spec.title,
                "sub": sub.strip(),
                "category": spec.category,
                "kind": CAT_COLOR.get(spec.category, "proj"),
                "size": int(max(1, size)),
                "params": int(params),
                "neurons": int(neurons),
                "subs": subs,
                "outShape": _fmt_shape(out_shape) if out_shape else "",
                "role": getattr(spec, "role", ""),
                "func": getattr(spec, "function", ""),
                "nIn": len(incoming.get(n.id, [])) if isinstance(incoming, dict) else 0,
            })
        name = "ARK scène"
        try:
            name = self._mw.tabs.tabText(self._mw.tabs.currentIndex())
        except Exception:
            pass
        total_params = sum(l["params"] for l in layers)
        return {"generic": True, "name": name, "layers": layers,
                "totalParams": total_params}

    # ============================================================
    # Workflow lifecycle (Phase 73)
    # ============================================================

    @Slot(result=str)
    def workflow_get_state(self):
        """Analyse scène + état compile/train → phases lifecycle + checklist."""
        import json as _json
        from pathlib import Path
        if self._mw is None or not hasattr(self._mw, "scene"):
            return _json.dumps({"modelType": "unknown", "phases": []})
        try:
            nodes = self._mw.scene.nodes()
        except Exception:
            nodes = []
        types = [n.spec.name for n in nodes]
        cats = [n.spec.category for n in nodes]

        # Détection type de modèle
        has = lambda *xs: any(t in types for t in xs)
        if has("tiny_llm_model", "transformer_block", "tiny_llm_generate"):
            mtype = "llm"
        elif has("unet_mini", "diffusion_scheduler", "ddim_sampler",
                    "diffusion_model_loader", "musicgen_audio",
                    "text2video_diffusion", "text_to_frame_seq"):
            mtype = "generative"
        elif has("dyadic_mirror", "self_ref_agent", "world_model",
                    "critical_grow_net"):
            mtype = "world"
        elif has("dense", "conv2d", "imagenet_classifier"):
            mtype = "classifier"
        else:
            mtype = "generic"

        # États
        n_nodes = len(nodes)
        has_source = any(c == "Input" for c in cats) or has(
            "image_input", "tabular_input", "text_input", "tiny_llm_corpus",
            "torchvision_dataset")
        has_transform = any(c in ("Neural", "Neural Networks", "Diffusion",
                                       "Tiny LLM") for c in cats)
        has_output = has("top_k_predictions", "number_display", "text_viewer",
                            "image_viewer", "prob_distribution", "save_image",
                            "audio_writer", "video_writer", "ddim_sampler",
                            "tiny_llm_generate", "caption_formatter")
        compiled = getattr(self._mw, "_compiled", None) is not None
        # Checkpoints (preuve d'entraînement)
        ck_dirs = [Path.home() / ".ark" / "demo_checkpoints",
                     Path.home() / ".ark" / "modal_checkpoints",
                     Path.home() / ".ark" / "tiny_llm"]
        trained = any(d.exists() and any(d.glob("*.pt")) for d in ck_dirs)

        # Templates de checklist par type
        DEFS = {
            "classifier": {
                "label": "Classifieur (vision/tabulaire)",
                "metric": "accuracy", "gen": "Inférence (prédire classe)",
            },
            "llm": {
                "label": "LLM / texte génératif",
                "metric": "perplexité / loss", "gen": "Génération de texte",
            },
            "generative": {
                "label": "Modèle génératif (diffusion/audio/vidéo)",
                "metric": "FID / loss reconstruction",
                "gen": "Génération (échantillonnage)",
            },
            "world": {
                "label": "World model / agent",
                "metric": "reward / prédiction d'état",
                "gen": "Rollout / simulation",
            },
            "generic": {
                "label": "Modèle générique",
                "metric": "loss", "gen": "Inférence",
            },
        }
        d = DEFS.get(mtype, DEFS["generic"])

        phases = [
            {
                "key": "conception", "title": "1 · Conception",
                "icon": "blocks",
                "done": has_source and has_transform,
                "current": not (has_source and has_transform),
                "desc": "Assembler le graphe : source de données → couches de "
                          "transformation → sortie.",
                "checks": [
                    {"label": "Node source / données en entrée",
                     "done": has_source},
                    {"label": "Couches de transformation (dense/conv/...)",
                     "done": has_transform},
                    {"label": "Node de sortie / lecture résultat",
                     "done": has_output},
                ],
                "action": None,
            },
            {
                "key": "compile", "title": "2 · Compilation",
                "icon": "cpu",
                "done": compiled,
                "current": (has_source and has_transform) and not compiled,
                "desc": "Vérifie que les formes (shapes) s'enchaînent sans "
                          "erreur. Construit le graphe PyTorch exécutable.",
                "checks": [
                    {"label": "Graphe compile sans erreur", "done": compiled},
                    {"label": "Formes d'entrée/sortie cohérentes",
                     "done": compiled},
                ],
                "action": "compile",
            },
            {
                "key": "train", "title": "3 · Entraînement",
                "icon": "activity",
                "done": trained,
                "current": compiled and not trained,
                "desc": f"Ajuster les poids sur données réelles. Surveiller "
                          f"que la {d['metric']} s'améliore. Local ou Modal (GPU).",
                "checks": [
                    {"label": "Optimizer + loss présents OU dataset chargé",
                     "done": has("adam", "sgd", "cross_entropy", "mse_loss")
                     or has("torchvision_dataset", "tiny_llm_corpus")},
                    {"label": "Entraînement lancé (checkpoint produit)",
                     "done": trained},
                    {"label": f"{d['metric']} diminue / converge",
                     "done": trained},
                ],
                "action": "train",
            },
            {
                "key": "validation", "title": "4 · Validation / Test",
                "icon": "check-circle",
                "done": False,
                "current": trained,
                "desc": "Évaluer sur données JAMAIS vues (test set). Mesurer "
                          f"la {d['metric']} de généralisation. Détecter "
                          "sur-apprentissage (train ✓ mais test ✗).",
                "checks": [
                    {"label": "Évaluer sur jeu de test séparé", "done": False},
                    {"label": "Métrique test ≈ métrique entraînement",
                     "done": False},
                    {"label": "Pas d'overfitting flagrant", "done": False},
                ],
                "action": "compile",
            },
            {
                "key": "production", "title": f"5 · {d['gen']}",
                "icon": "rocket",
                "done": False,
                "current": False,
                "desc": "Utiliser le modèle entraîné sur de nouvelles entrées. "
                          "Exporter (.py/ONNX) pour déployer.",
                "checks": [
                    {"label": "Sortie cohérente sur nouvelles données",
                     "done": False},
                    {"label": "Export (.py / ONNX / checkpoint)", "done": trained},
                ],
                "action": None,
            },
        ]
        # current = première phase non-done
        cur_set = False
        for ph in phases:
            ph["current"] = False
        for ph in phases:
            if not ph["done"] and not cur_set:
                ph["current"] = True
                cur_set = True

        return _json.dumps({
            "modelType": mtype,
            "modelTypeLabel": d["label"],
            "nNodes": n_nodes,
            "phases": phases,
        })

    @Slot()
    def raise_workflow(self):
        if self._mw and hasattr(self._mw, "dock_workflow"):
            try:
                self._mw.dock_workflow.setVisible(True)
                self._mw.dock_workflow.raise_()
            except Exception:
                pass

    @Slot()
    def raise_viz3d(self):
        if self._mw and hasattr(self._mw, "dock_viz3d"):
            try:
                self._mw.dock_viz3d.setVisible(True)
                self._mw.dock_viz3d.raise_()
            except Exception:
                pass

    @Slot()
    def raise_compute(self):
        if self._mw and hasattr(self._mw, "dock_compute"):
            try:
                self._mw.dock_compute.setVisible(True)
                self._mw.dock_compute.raise_()
            except Exception:
                pass

    @Slot(result=str)
    def modal_get_config(self):
        import json as _json
        try:
            from ark.core.modal_backend import (
                modal_available, token_configured, GPU_CHOICES)
            return _json.dumps({
                "available": modal_available(),
                "token_configured": token_configured(),
                "gpus": GPU_CHOICES,
            })
        except Exception as e:
            return _json.dumps({"available": False, "token_configured": False,
                                  "gpus": [], "error": str(e)})

    @Slot(str, str)
    def modal_set_token(self, token_id: str, token_secret: str):
        from ark.core.modal_backend import save_token
        save_token(token_id, token_secret)

    @Slot(str, int, str, str)
    def modal_train(self, payload_or_empty: str, epochs: int,
                    gpu: str, dataset: str):
        """Lance training Modal en QThread. Stream via modal_log,
        résultat via modal_done. payload_or_empty='' → scène courante."""
        import json as _json
        # Build payload from current scene if none provided
        payload = None
        if payload_or_empty:
            try:
                payload = _json.loads(payload_or_empty)
            except Exception:
                payload = None
        if payload is None:
            try:
                from ark.core.format import scene_to_payload
                payload = scene_to_payload(self._mw.scene, name="Modal Train")
            except Exception as e:
                self.modal_done.emit(_json.dumps(
                    {"ok": False, "error": f"scene serialize: {e}"}))
                return

        from PySide6.QtCore import QThread, Signal as _S, QObject

        class _Worker(QObject):
            log_sig = _S(str)
            done_sig = _S(str)

            def __init__(s, payload, epochs, gpu, dataset):
                super().__init__()
                s.payload = payload
                s.epochs = epochs
                s.gpu = gpu
                s.dataset = dataset

            def run(s):
                import json as __j
                from ark.core.modal_backend import submit_training
                res = submit_training(
                    s.payload, epochs=s.epochs, gpu=s.gpu,
                    dataset=s.dataset,
                    on_log=lambda m: s.log_sig.emit(m))
                s.done_sig.emit(__j.dumps(res))

        self._modal_thread = QThread()
        self._modal_worker = _Worker(payload, int(epochs), gpu, dataset)
        self._modal_worker.moveToThread(self._modal_thread)
        self._modal_worker.log_sig.connect(self.modal_log)
        self._modal_worker.log_sig.connect(self.emit_log)  # also to terminal
        self._modal_worker.done_sig.connect(self._on_modal_done)
        self._modal_thread.started.connect(self._modal_worker.run)
        self._modal_thread.start()

    def _on_modal_done(self, result_json: str):
        self.modal_done.emit(result_json)
        try:
            self._modal_thread.quit()
            self._modal_thread.wait(3000)
        except Exception:
            pass

    # ============================================================
    # Lifecycle hooks (called by MainWindow)
    # ============================================================

    def emit_selection(self, node):
        import json as _json
        try:
            data = self._serialize_node(node) if node else None
            self.selection_changed.emit(_json.dumps(data) if data else "")
        except Exception:
            self.selection_changed.emit("")

    def emit_log(self, line: str):
        self.log_appended.emit(line)
