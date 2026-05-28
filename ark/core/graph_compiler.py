"""ARK graph compiler — port adapté depuis ai-arch-builder.

Compile une scene (NodeScene) → GraphModel (nn.Module DAG topologique).
Erreurs accrochées aux nodes via set_error() pour feedback visuel.

Diffère du port ai-arch-builder par :
- Lit directement NodeItem.spec.name (pas un graph dict JSON).
- Mappage params : nom canonique (e.g. 'kernel_size' au lieu de 'kernelSize').
- Multi-input dynamique : utilise toutes les input_sockets connectées.
"""
from __future__ import annotations
from collections import defaultdict, deque
import torch
import torch.nn as nn


class GraphCompileError(Exception):
    def __init__(self, message: str, node=None):
        super().__init__(message)
        self.node = node  # NodeItem pour set_error visual feedback


ACTIVATIONS = {
    "none": nn.Identity, "relu": nn.ReLU, "gelu": nn.GELU,
    "tanh": nn.Tanh, "sigmoid": nn.Sigmoid,
    "softmax": lambda: nn.Softmax(dim=-1),
}


def _act(name: str) -> nn.Module:
    return ACTIVATIONS.get((name or "none").lower(), nn.Identity)()


def _parse_shape(s):
    if isinstance(s, (list, tuple)):
        return tuple(int(x) for x in s)
    if isinstance(s, str):
        cleaned = s.strip().strip("[]()").replace(" ", "")
        if not cleaned:
            return ()
        return tuple(int(x) for x in cleaned.split(","))
    return ()


class FunctionalOp(nn.Module):
    def __init__(self, fn, name="func"):
        super().__init__()
        self.fn = fn
        self.name = name
    def forward(self, *args):
        return self.fn(*args)


# ============================================================
# Handlers — chacun retourne (module|None, out_shape)
# ============================================================

def h_image_input(node, in_shapes):
    shape = _parse_shape(node.params.get("shape", "[1, 28, 28]"))
    if len(shape) == 3 and shape[-1] in (1, 3) and shape[0] not in (1, 3):
        shape = (shape[2], shape[0], shape[1])
    return None, shape if shape else (1, 28, 28)


def h_tabular_input(node, in_shapes):
    n = int(node.params.get("features", 32))
    return None, (n,)


def h_text_input(node, in_shapes):
    T = int(node.params.get("max_length", 64))
    D = int(node.params.get("embed_dim", 128))
    return None, (T, D)


def h_dense(node, in_shapes):
    if len(in_shapes) != 1:
        raise GraphCompileError("dense requires single input", node)
    in_shape = in_shapes[0]
    flat = 1
    for s in in_shape:
        flat *= int(s)
    out_feat = int(node.params.get("neurons", 128))
    layers = []
    if len(in_shape) > 1:
        layers.append(nn.Flatten())
    layers.append(nn.Linear(flat, out_feat))
    layers.append(_act(node.params.get("activation", "relu")))
    dr = float(node.params.get("dropout", 0.0))
    if dr > 0:
        layers.append(nn.Dropout(dr))
    return nn.Sequential(*layers), (out_feat,)


def h_conv2d(node, in_shapes):
    if len(in_shapes) != 1 or len(in_shapes[0]) != 3:
        raise GraphCompileError("conv2d requires (C, H, W) input", node)
    C, H, W = in_shapes[0]
    F = int(node.params.get("filters", 32))
    K = int(node.params.get("kernel_size", 3))
    pad = K // 2
    m = nn.Sequential(
        nn.Conv2d(C, F, kernel_size=K, padding=pad),
        _act(node.params.get("activation", "relu")),
    )
    return m, (F, H, W)


def h_normalize(node, in_shapes):
    if len(in_shapes) != 1:
        raise GraphCompileError("normalize requires single input", node)
    shape = in_shapes[0]
    kind = node.params.get("kind", "batchnorm")
    if kind == "batchnorm":
        if len(shape) == 3:
            return nn.BatchNorm2d(shape[0]), shape
        if len(shape) == 1:
            return nn.BatchNorm1d(shape[0]), shape
    if kind in ("layernorm", "groupnorm"):
        return nn.LayerNorm(shape[-1]), shape
    raise GraphCompileError(f"normalize: shape/kind unsupported {shape}/{kind}", node)


def h_residual_add(node, in_shapes):
    if len(in_shapes) < 2:
        raise GraphCompileError("residual_add needs >= 2 inputs", node)
    ref = in_shapes[0]
    for s in in_shapes[1:]:
        if s != ref:
            raise GraphCompileError(f"residual_add shape mismatch {s} vs {ref}", node)

    def _sum(*ts):
        out = ts[0]
        for t in ts[1:]:
            out = out + t
        return out
    return FunctionalOp(_sum, "residual_add"), ref


class _SelfAttn(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
    def forward(self, x):
        out, _ = self.mha(x, x, x)
        return out


def _adjust_heads(D, req):
    if D % req == 0:
        return req
    for h in (8, 4, 2, 1):
        if D % h == 0:
            return h
    return 1


def h_attention(node, in_shapes):
    if len(in_shapes) != 1 or len(in_shapes[0]) != 2:
        raise GraphCompileError("attention needs (T, D) input", node)
    T, D = in_shapes[0]
    nh = _adjust_heads(D, int(node.params.get("n_heads", 4)))
    return _SelfAttn(D, nh), (T, D)


class _TransformerBlk(nn.Module):
    def __init__(self, d, h, d_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))
    def forward(self, x):
        n = self.norm1(x); a, _ = self.attn(n, n, n); x = x + a
        x = x + self.ffn(self.norm2(x))
        return x


def h_transformer_block(node, in_shapes):
    if len(in_shapes) != 1 or len(in_shapes[0]) != 2:
        raise GraphCompileError("transformer_block needs (T, D) input", node)
    T, D = in_shapes[0]
    nh = _adjust_heads(D, int(node.params.get("n_heads", 4)))
    dff = int(node.params.get("d_ff", 4 * D))
    return _TransformerBlk(D, nh, dff), (T, D)


class _MoESwitch(nn.Module):
    def __init__(self, d_in, d_out, n_experts):
        super().__init__()
        self.router = nn.Linear(d_in, n_experts)
        self.experts = nn.ModuleList([nn.Linear(d_in, d_out) for _ in range(n_experts)])
        self.n_experts = n_experts
        self.d_out = d_out
    def forward(self, x):
        sh = x.shape
        xf = x.reshape(-1, sh[-1])
        idx = self.router(xf).argmax(dim=-1)
        out = torch.zeros(xf.shape[0], self.d_out, device=x.device, dtype=x.dtype)
        for i, e in enumerate(self.experts):
            m = idx == i
            if m.any():
                out[m] = e(xf[m])
        return out.reshape(*sh[:-1], self.d_out)


def h_moe_switch(node, in_shapes):
    if len(in_shapes) != 1:
        raise GraphCompileError("moe_switch single input", node)
    sh = in_shapes[0]
    n = int(node.params.get("n_experts", 4))
    d_out = int(node.params.get("d_out", sh[-1]))
    return _MoESwitch(sh[-1], d_out, n), tuple(list(sh[:-1]) + [d_out])


class _VGBlock(nn.Module):
    """MLP mutable (vague B) avec widen/prune function-preserving."""
    def __init__(self, in_d, out_d, hidden, act):
        super().__init__()
        self.in_d, self.out_d, self.hidden_dim = in_d, out_d, hidden
        self.fc1 = nn.Linear(in_d, hidden)
        self.act = _act(act)
        self.fc2 = nn.Linear(hidden, out_d)
        self.activation_name = act

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

    @torch.no_grad()
    def widen(self, delta: int) -> "_VGBlock":
        if delta <= 0:
            return self
        new_h = self.hidden_dim + delta
        new_fc1 = nn.Linear(self.in_d, new_h).to(self.fc1.weight.device)
        new_fc1.weight[: self.hidden_dim].copy_(self.fc1.weight)
        new_fc1.bias[: self.hidden_dim].copy_(self.fc1.bias)
        nn.init.kaiming_uniform_(new_fc1.weight[self.hidden_dim:], a=5 ** 0.5)
        new_fc1.bias[self.hidden_dim:].zero_()
        new_fc2 = nn.Linear(new_h, self.out_d).to(self.fc2.weight.device)
        new_fc2.weight[:, : self.hidden_dim].copy_(self.fc2.weight)
        new_fc2.bias.copy_(self.fc2.bias)
        new_fc2.weight[:, self.hidden_dim:].zero_()
        self.fc1, self.fc2, self.hidden_dim = new_fc1, new_fc2, new_h
        return self

    @torch.no_grad()
    def prune(self, delta: int) -> "_VGBlock":
        if delta <= 0:
            return self
        if delta >= self.hidden_dim:
            raise ValueError(f"Cannot prune {delta} from hidden={self.hidden_dim}")
        new_h = self.hidden_dim - delta
        col_norms = self.fc2.weight.abs().sum(dim=0)
        top = col_norms.argsort(descending=True)[:new_h]
        new_fc1 = nn.Linear(self.in_d, new_h).to(self.fc1.weight.device)
        new_fc1.weight.copy_(self.fc1.weight[top])
        new_fc1.bias.copy_(self.fc1.bias[top])
        new_fc2 = nn.Linear(new_h, self.out_d).to(self.fc2.weight.device)
        new_fc2.weight.copy_(self.fc2.weight[:, top])
        new_fc2.bias.copy_(self.fc2.bias)
        self.fc1, self.fc2, self.hidden_dim = new_fc1, new_fc2, new_h
        return self


def h_vg_block(node, in_shapes):
    if len(in_shapes) != 1 or len(in_shapes[0]) != 1:
        raise GraphCompileError("vg_block needs 1D input", node)
    in_d = in_shapes[0][0]
    out_d = int(node.params.get("out_dim", 64))
    h = int(node.params.get("start_hidden", 32))
    m = _VGBlock(in_d, out_d, h, node.params.get("activation", "gelu"))
    # Tag avec node id pour mutation runtime
    m._ark_node_id = node.id
    return m, (out_d,)


class _ObserverModule(nn.Module):
    """Pass-through qui calcule eff_rank / intrinsic_dim / power_slope / mean_abs."""
    def __init__(self, metrics: list[str], buffer_size: int = 256):
        super().__init__()
        self.metrics = metrics
        self.buffer_size = buffer_size
        self.activation_buffer: list[torch.Tensor] = []
        self.last_metrics: dict = {}

    def _eff_rank(self, X: torch.Tensor) -> float:
        Xc = X - X.mean(dim=0, keepdim=True)
        try:
            s = torch.linalg.svdvals(Xc.float())
        except RuntimeError:
            return 0.0
        s = s[s > 1e-9]
        if s.numel() == 0:
            return 0.0
        p = s / s.sum()
        ent = -(p * (p + 1e-12).log()).sum()
        return float(ent.exp().item())

    def _intrinsic_dim(self, X: torch.Tensor, k: int = 10) -> float:
        n = X.shape[0]
        if n <= k:
            return float("nan")
        d = torch.cdist(X.float(), X.float())
        d.fill_diagonal_(float("inf"))
        vals, _ = d.topk(k, largest=False)
        log_r = (vals[:, -1:].log() - vals[:, :-1].log()).clamp(min=1e-9)
        m = (k - 1) / log_r.sum(dim=1)
        return float(((k - 2) / (k - 1) * m.mean()).item())

    def _power_slope(self, traj: torch.Tensor) -> float:
        if traj.ndim == 2:
            sig = traj.float().mean(dim=1)
        else:
            sig = traj.float().flatten()
        T_ = sig.shape[0]
        if T_ < 8:
            return float("nan")
        fft = torch.fft.rfft(sig - sig.mean()).abs()
        p = (fft ** 2)[1:]
        freqs = torch.arange(1, p.shape[0] + 1, dtype=sig.dtype)
        x = freqs.log(); y = (p + 1e-12).log()
        var = ((x - x.mean()) ** 2).sum()
        if var.item() < 1e-9:
            return float("nan")
        cov = ((x - x.mean()) * (y - y.mean())).sum()
        return float((cov / var).item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            X = x.reshape(x.shape[0], -1) if x.ndim > 2 else x
            metrics: dict = {}
            if "mean_abs" in self.metrics:
                metrics["mean_abs"] = float(X.abs().mean().item())
            if "eff_rank" in self.metrics and X.shape[0] > 1:
                metrics["eff_rank"] = self._eff_rank(X)
            if "intrinsic_dim" in self.metrics and X.shape[0] > 12:
                metrics["intrinsic_dim"] = self._intrinsic_dim(X, k=min(10, X.shape[0] - 1))
            self.activation_buffer.append(X.detach().mean(dim=0).cpu())
            if len(self.activation_buffer) > self.buffer_size:
                self.activation_buffer.pop(0)
            if "power_slope" in self.metrics and len(self.activation_buffer) >= 8:
                try:
                    traj = torch.stack(self.activation_buffer, dim=0)
                    metrics["power_slope"] = self._power_slope(traj)
                except RuntimeError:
                    self.activation_buffer.clear()
            self.last_metrics = metrics
        return x


def h_observer(node, in_shapes):
    if len(in_shapes) != 1:
        raise GraphCompileError("observer needs single input", node)
    raw = node.params.get("metrics", "eff_rank,intrinsic_dim,power_slope,mean_abs")
    metrics = [m.strip() for m in raw.split(",") if m.strip()]
    buf = int(node.params.get("buffer_size", 256))
    m = _ObserverModule(metrics, buf)
    m._ark_node_id = node.id
    return m, in_shapes[0]


class _TriggerRule(nn.Module):
    """Évalue condition `metric op threshold` → action mutate target_node."""
    _OPS = {
        "<": lambda a, b: a < b, ">": lambda a, b: a > b,
        "<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b,
        "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
    }

    def __init__(self, metric: str, op: str, threshold: float,
                 action_op: str, target_node: str, delta: int = 8):
        super().__init__()
        self.metric_name = metric
        self.op = op
        self.threshold = float(threshold)
        self.action_op = action_op
        self.target_node = target_node
        self.delta = int(delta)

    def forward(self, x):
        return x

    def evaluate(self, metrics: dict) -> dict | None:
        if self.metric_name not in metrics:
            return None
        v = metrics[self.metric_name]
        if v != v:  # NaN
            return None
        if self._OPS[self.op](v, self.threshold):
            return {
                "action": "mutate",
                "node_id": self.target_node,
                "op": self.action_op,
                "delta": self.delta,
                "triggered_by": {"metric": self.metric_name, "value": v,
                                   "op": self.op, "threshold": self.threshold},
            }
        return None


def h_trigger_rule(node, in_shapes):
    if len(in_shapes) != 1:
        raise GraphCompileError("trigger_rule needs single input", node)
    target = node.params.get("target_node", "")
    if not target:
        raise GraphCompileError("trigger_rule: target_node param required", node)
    m = _TriggerRule(
        metric=node.params.get("metric", "eff_rank"),
        op=node.params.get("op", ">"),
        threshold=float(node.params.get("threshold", 0.0)),
        action_op=node.params.get("action_op", "widen"),
        target_node=target,
        delta=int(node.params.get("delta", 8)),
    )
    m._ark_node_id = node.id
    return m, in_shapes[0]


def h_math_op(node, in_shapes):
    op = node.params.get("op", "add")
    if op in ("add",):  fn = lambda a, b: a + b
    elif op == "sub":  fn = lambda a, b: a - b
    elif op == "mul":  fn = lambda a, b: a * b
    elif op == "div":  fn = lambda a, b: a / b
    elif op == "pow":  fn = lambda a, b: a ** b
    elif op == "min":  fn = lambda a, b: torch.minimum(a, b)
    elif op == "max":  fn = lambda a, b: torch.maximum(a, b)
    elif op == "mod":  fn = lambda a, b: a % b
    else:
        raise GraphCompileError(f"math_op unknown op {op}", node)
    if len(in_shapes) < 2:
        raise GraphCompileError("math_op needs 2 inputs", node)
    return FunctionalOp(fn, f"math_{op}"), in_shapes[0]


def h_constant(node, in_shapes):
    v = float(node.params.get("value", 1.0))
    class _Const(nn.Module):
        def __init__(self): super().__init__(); self.v = v
        def forward(self, x=None): return torch.tensor(self.v)
    return _Const(), ()


def h_code_line(node, in_shapes):
    """Exécute la ligne / bloc Python sur l'input. UNSAFE — pour dev only."""
    code = node.params.get("code", "result = x")

    class _CodeOp(nn.Module):
        def __init__(self, src):
            super().__init__()
            self.src = src
        def forward(self, x):
            ns = {"x": x, "torch": torch, "result": None}
            try:
                exec(self.src, ns)
            except Exception as e:
                raise RuntimeError(f"code_line: {e}")
            r = ns.get("result", x)
            return r if isinstance(r, torch.Tensor) else torch.as_tensor(r)
    return _CodeOp(code), (in_shapes[0] if in_shapes else ())


HANDLERS = {
    "image_input": h_image_input,
    "tabular_input": h_tabular_input,
    "text_input": h_text_input,
    "dense": h_dense,
    "conv2d": h_conv2d,
    "normalize": h_normalize,
    "residual_add": h_residual_add,
    "attention": h_attention,
    "transformer_block": h_transformer_block,
    "moe_switch": h_moe_switch,
    "vg_block": h_vg_block,
    "observer": h_observer,
    "trigger_rule": h_trigger_rule,
    "math_op": h_math_op,
    "constant": h_constant,
    "code_line": h_code_line,
}

# Devices (Phase 6) — lazy imports
def _lazy_register_devices():
    try:
        from ark.core.devices import h_webcam, h_microphone
        HANDLERS["webcam"] = h_webcam
        HANDLERS["microphone"] = h_microphone
    except Exception:
        pass

_lazy_register_devices()


# Diffusion stubs (Phase 4 stub) — pas de download modèles, juste pipeline
def _register_diffusion_stubs():
    from ark.core.diffusion_stubs import (
        h_diffusion_model_loader, h_clip_text_encode, h_empty_latent_image,
        h_k_sampler, h_vae_decode, h_vae_encode, h_load_image, h_save_image,
        h_lora_loader, h_controlnet_apply,
    )
    HANDLERS["diffusion_model_loader"] = h_diffusion_model_loader
    HANDLERS["clip_text_encode"] = h_clip_text_encode
    HANDLERS["empty_latent_image"] = h_empty_latent_image
    HANDLERS["k_sampler"] = h_k_sampler
    HANDLERS["vae_decode"] = h_vae_decode
    HANDLERS["vae_encode"] = h_vae_encode
    HANDLERS["load_image"] = h_load_image
    HANDLERS["save_image"] = h_save_image
    HANDLERS["lora_loader"] = h_lora_loader
    HANDLERS["controlnet_apply"] = h_controlnet_apply

_register_diffusion_stubs()


# Metamorph integration (Phase 17) — lazy import si dispo
def _register_metamorph():
    try:
        from ark.core.metamorph_bridge import register_metamorph_handlers
        register_metamorph_handlers(HANDLERS)
    except Exception:
        pass

_register_metamorph()


# Research integration (Phase 27)
def _register_research():
    try:
        from ark.core.research_bridge import register_research_handlers
        register_research_handlers(HANDLERS)
    except Exception:
        pass

_register_research()


# Glossary auto-handlers (Phase 29) — 50+ handlers PyTorch génériques
def _register_glossary_handlers():
    try:
        from ark.core.auto_register import register_glossary_handlers
        register_glossary_handlers(HANDLERS)
    except Exception as e:
        import sys
        print(f"[ARK] auto_register handlers failed: {e}", file=sys.stderr)

_register_glossary_handlers()


INPUT_TYPES = {"image_input", "tabular_input", "text_input"}
NON_FORWARD_TYPES = {"adam", "cross_entropy", "plot_scatter", "plot_line"}
SKIP_TYPES = INPUT_TYPES | NON_FORWARD_TYPES


# ============================================================
# GraphModel + compile
# ============================================================

class GraphModel(nn.Module):
    """DAG nn.Module avec support multi-output socket routing (Phase 19).

    `incoming_routed[dst_id]` = list[(src_id, from_socket, to_socket)] préservant
    ordre des connexions ET indices de socket. Permet aux nodes source qui
    retournent tuple (diffusion_model_loader → model/clip/vae) de router le bon
    élément vers le bon downstream.

    `source_node_ids` = nodes sans incoming (input/loader/empty_latent/load_image
    etc.) — leur forward() est appelé sans argument et leur output peut être
    tensor OU tuple. cache[nid] stocke toujours le tensor/tuple complet.
    """

    def __init__(self, ordered_nodes, modules_by_id, input_id, output_id,
                 incoming, incoming_routed=None, source_node_ids=None):
        super().__init__()
        self.ordered_nodes = ordered_nodes
        self.modules_dict = nn.ModuleDict(
            {nid: m for nid, m in modules_by_id.items() if m is not None}
        )
        self.input_id = input_id
        self.output_id = output_id
        self.incoming = dict(incoming)
        self.incoming_routed: dict = incoming_routed or {}
        self.source_node_ids: set = set(source_node_ids or [input_id])

    def _select_socket(self, value, idx: int):
        """Extrait l'élément `idx` d'un output. Si value est tuple/list, sélectionne.
        Sinon retourne value direct (idx ignoré pour single-output nodes)."""
        if isinstance(value, (tuple, list)):
            if idx < len(value):
                return value[idx]
            return value[0]
        return value

    def forward(self, x):
        cache: dict = {self.input_id: x}
        # Lance les autres source nodes (sans incoming) qui ne sont pas l'input_id
        for nid in self.source_node_ids:
            if nid in cache:
                continue
            mod = self.modules_dict[nid] if nid in self.modules_dict else None
            if mod is None:
                continue
            try:
                cache[nid] = mod()  # source : pas d'argument
            except TypeError:
                # Fallback : module attend un x (legacy)
                cache[nid] = mod(None)

        for node in self.ordered_nodes:
            nid = node.id
            if nid in cache:
                continue
            ups = self.incoming.get(nid, [])
            routed = self.incoming_routed.get(nid)
            if not ups:
                continue
            if routed:
                # Multi-input avec socket routing
                in_t = [self._select_socket(cache[src], src_sock)
                        for (src, src_sock, _) in routed]
            else:
                in_t = [cache[u] for u in ups]
            mod = self.modules_dict[nid] if nid in self.modules_dict else None
            if mod is None:
                cache[nid] = in_t[0]
                continue
            if isinstance(mod, FunctionalOp):
                cache[nid] = mod(*in_t)
            else:
                # Single-input modules : prend 1er. Sinon expand args.
                if len(in_t) == 1:
                    cache[nid] = mod(in_t[0])
                else:
                    cache[nid] = mod(*in_t)
        if self.output_id not in cache:
            raise RuntimeError(f"Output {self.output_id} not produced")
        out = cache[self.output_id]
        # Final : si tuple, retourne 1er élément (sortie principale)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


class CompiledGraph:
    def __init__(self, model, input_shape, output_shape, num_classes=None):
        self.model = model
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_classes = num_classes


def _topo_sort_scene(scene):
    """Construit incoming/outgoing edges depuis scene.connections + topo sort.

    Phase 19 : retourne aussi `incoming_routed` qui préserve les indices socket
    (src_socket_idx, dst_socket_idx) pour chaque arête, utile au forward.

    Trie les inputs par dst_socket_idx pour respecter l'ordre attendu côté
    handler (KSampler : model, positive, negative, latent en sockets 0,1,2,3).
    """
    nodes = scene.nodes()
    by_id = {n.id: n for n in nodes}
    incoming = defaultdict(list)
    outgoing = defaultdict(list)
    incoming_routed: dict = defaultdict(list)
    for conn in scene.connections():
        if conn.start_socket is None or conn.end_socket is None:
            continue
        src = conn.start_socket.parent_node.id
        dst = conn.end_socket.parent_node.id
        src_idx = getattr(conn.start_socket, "index", 0)
        dst_idx = getattr(conn.end_socket, "index", 0)
        incoming[dst].append(src)
        outgoing[src].append(dst)
        incoming_routed[dst].append((src, src_idx, dst_idx))
    # Sort incoming par dst_socket_idx pour ordre stable
    for dst in incoming_routed:
        incoming_routed[dst].sort(key=lambda t: t[2])
        # Sync incoming dans même ordre
        incoming[dst] = [t[0] for t in incoming_routed[dst]]
    indeg = {nid: len(incoming[nid]) for nid in by_id}
    queue = deque([nid for nid, d in indeg.items() if d == 0])
    order_ids = []
    while queue:
        nid = queue.popleft()
        order_ids.append(nid)
        for nxt in outgoing[nid]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    if len(order_ids) != len(by_id):
        raise GraphCompileError("Graph has a cycle or disconnected component")
    return [by_id[nid] for nid in order_ids], incoming, outgoing, incoming_routed


def compile_scene(scene, num_classes: int | None = None,
                   clear_errors: bool = True) -> CompiledGraph:
    """Compile scene → GraphModel.

    Si erreur, accroche message au node concerné via set_error() puis raise.
    Si clear_errors True, reset les erreurs des nodes avant compilation.
    """
    if clear_errors:
        for n in scene.nodes():
            n.set_error(None)

    nodes = scene.nodes()
    if not nodes:
        raise GraphCompileError("Empty graph")

    ordered, incoming, outgoing, incoming_routed = _topo_sort_scene(scene)

    # Phase 19 : SOURCE_TYPES = nodes sans incoming attendu (loader, latent, etc.)
    SOURCE_TYPES = INPUT_TYPES | {
        "diffusion_model_loader", "empty_latent_image", "load_image",
        "webcam", "microphone", "constant", "hf_model_loader",
        "hf_tokenizer",
    }

    # Input node : 1er INPUT_TYPES, sinon 1er SOURCE_TYPES
    input_node = next((n for n in ordered if n.spec.name in INPUT_TYPES), None)
    if input_node is None:
        input_node = next((n for n in ordered if n.spec.name in SOURCE_TYPES), None)
    if input_node is None:
        raise GraphCompileError("Missing source node (input/loader/etc.)")
    input_id = input_node.id

    # Tous les source nodes (sans incoming OU type source)
    source_node_ids = {
        n.id for n in ordered
        if n.spec.name in SOURCE_TYPES or not incoming.get(n.id)
    }

    # Shapes propagation
    shapes_by_id: dict = {}
    modules_by_id: dict = {}

    try:
        for node in ordered:
            ntype = node.spec.name
            handler = HANDLERS.get(ntype)
            if handler is None and ntype not in NON_FORWARD_TYPES:
                raise GraphCompileError(f"Unknown handler for type '{ntype}'", node)
            if ntype in NON_FORWARD_TYPES:
                continue
            in_ids = incoming.get(node.id, [])
            # Phase 19 : utilise routed pour résoudre shapes par socket idx
            routed = incoming_routed.get(node.id, [])
            if routed:
                in_shapes = []
                for src_id, src_sock, _dst_sock in routed:
                    if src_id not in shapes_by_id:
                        continue
                    src_shape = shapes_by_id[src_id]
                    # Si tuple de shapes (multi-output), prend src_sock
                    if isinstance(src_shape, tuple) and len(src_shape) > 0 and \
                       isinstance(src_shape[0], tuple):
                        in_shapes.append(src_shape[src_sock] if src_sock < len(src_shape) else src_shape[0])
                    else:
                        in_shapes.append(src_shape)
            else:
                in_shapes = [shapes_by_id[u] for u in in_ids if u in shapes_by_id]
            try:
                mod, out_shape = handler(node, in_shapes)
            except GraphCompileError:
                raise
            except Exception as e:
                raise GraphCompileError(f"{ntype}: {e}", node)
            modules_by_id[node.id] = mod
            shapes_by_id[node.id] = out_shape
    except GraphCompileError as e:
        if e.node is not None:
            e.node.set_error(str(e))
        raise

    # Output : dernier node non-skip
    output_id = None
    for n in reversed(ordered):
        if n.spec.name not in SKIP_TYPES and n.id in shapes_by_id:
            output_id = n.id
            break
    if output_id is None:
        raise GraphCompileError("No output computable")

    # Classifier head auto si num_classes mismatch (skip si tuple/multi-output)
    final_shape = shapes_by_id[output_id]
    if num_classes is not None and isinstance(final_shape, tuple) and \
       len(final_shape) > 0 and not isinstance(final_shape[0], tuple):
        if len(final_shape) != 1 or final_shape[0] != num_classes:
            flat = 1
            for s in final_shape:
                if isinstance(s, int) and s > 0:
                    flat *= int(s)
                else:
                    flat = None
                    break
            if flat is not None:
                head_id = f"_auto_head_{output_id}"
                head = nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(flat, num_classes))
                modules_by_id[head_id] = head
                shapes_by_id[head_id] = (num_classes,)
                incoming[head_id] = [output_id]
                incoming_routed[head_id] = [(output_id, 0, 0)]
                class _Pseudo:
                    def __init__(self, id): self.id = id
                ordered.append(_Pseudo(head_id))
                output_id = head_id
                final_shape = (num_classes,)

    model = GraphModel(ordered, modules_by_id, input_id, output_id,
                         incoming, incoming_routed=dict(incoming_routed),
                         source_node_ids=source_node_ids)
    # input_shape : si tuple-of-tuples (multi-output source), prend 1ère
    in_sh = shapes_by_id[input_id]
    if isinstance(in_sh, tuple) and len(in_sh) > 0 and isinstance(in_sh[0], tuple):
        in_sh = in_sh[0]
    return CompiledGraph(model, in_sh, final_shape, num_classes)
