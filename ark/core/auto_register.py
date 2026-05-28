"""Phase 29 — Auto-registration depuis glossary_data.

Génère :
1. NodeSpec dans le registry pour kind='node' (et 'template' pour future)
2. Handlers PyTorch dans HANDLERS pour items implémentables
3. GlossaryEntry additionnelles depuis tous les concepts

Strategy : 95% NodeSpec only (palette + glossary), 50+ avec vrai handler.
"""
from __future__ import annotations
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Custom modules (non-natifs PyTorch)
# ============================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal absolute position encoding."""
    def __init__(self, d_model: int = 768, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                         * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class RotaryEmbedding(nn.Module):
    """RoPE — rotation matrix per position."""
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim

    def forward(self, x):
        seq_len = x.shape[-2]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos(), emb.sin()
        # x : (..., seq, dim)
        x1, x2 = x[..., : self.dim // 2], x[..., self.dim // 2:]
        rot = torch.cat([-x2, x1], dim=-1)
        return x * cos + rot * sin


class SelfAttention(nn.Module):
    """Self-attention QKV via 1 Linear."""
    def __init__(self, d_model: int = 768, d_k: int = 64):
        super().__init__()
        self.d_k = d_k
        self.qkv = nn.Linear(d_model, 3 * d_k)
        self.out = nn.Linear(d_k, d_model)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        attn = torch.softmax(q @ k.transpose(-2, -1) / (self.d_k ** 0.5), dim=-1)
        return self.out(attn @ v)


class CausalMaskModule(nn.Module):
    def __init__(self, seq_len: int = 512):
        super().__init__()
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, x):
        seq = x.shape[-2]
        m = self.mask[:seq, :seq]
        return x.masked_fill(m, float("-inf"))


class ReactionDiffusion(nn.Module):
    """Gray-Scott Turing pattern. forward = N steps."""
    def __init__(self, D_u: float = 0.16, D_v: float = 0.08,
                 f: float = 0.035, k: float = 0.065, steps: int = 100):
        super().__init__()
        self.D_u = D_u; self.D_v = D_v; self.f = f; self.k = k
        self.steps = steps
        # Laplacian kernel
        lap = torch.tensor([[0.05, 0.2, 0.05],
                             [0.2, -1.0, 0.2],
                             [0.05, 0.2, 0.05]]).view(1, 1, 3, 3)
        self.register_buffer("lap", lap)

    def forward(self, x):
        # x : (B, 2, H, W) — channel 0 = u, 1 = v
        u, v = x[:, :1], x[:, 1:2]
        for _ in range(self.steps):
            Lu = F.conv2d(u, self.lap, padding=1)
            Lv = F.conv2d(v, self.lap, padding=1)
            uvv = u * v * v
            du = self.D_u * Lu - uvv + self.f * (1 - u)
            dv = self.D_v * Lv + uvv - (self.f + self.k) * v
            u = u + du; v = v + dv
        return torch.cat([u, v], dim=1)


class LorenzAttractor(nn.Module):
    """Lorenz system via Euler. Input (B, 3) state → output (B, 3)."""
    def __init__(self, sigma: float = 10.0, rho: float = 28.0,
                 beta: float = 2.667, dt: float = 0.01, steps: int = 100):
        super().__init__()
        self.sigma = sigma; self.rho = rho; self.beta = beta
        self.dt = dt; self.steps = steps

    def forward(self, x):
        for _ in range(self.steps):
            xs, ys, zs = x[..., 0], x[..., 1], x[..., 2]
            dx = self.sigma * (ys - xs)
            dy = xs * (self.rho - zs) - ys
            dz = xs * ys - self.beta * zs
            x = x + self.dt * torch.stack([dx, dy, dz], dim=-1)
        return x


class CellularAutomaton(nn.Module):
    """Elementary CA Wolfram. forward propage N steps."""
    def __init__(self, rule: int = 110, size: int = 128, steps: int = 256):
        super().__init__()
        self.rule = rule; self.size = size; self.steps = steps

    def forward(self, x):
        # x : (B, size). Si zeros, seed centre = 1
        B = x.shape[0]
        if x.abs().sum() < 1e-6:
            x = torch.zeros(B, self.size, device=x.device)
            x[:, self.size // 2] = 1.0
        for _ in range(self.steps):
            left = torch.roll(x, 1, dims=-1)
            right = torch.roll(x, -1, dims=-1)
            idx = (left * 4 + x * 2 + right).long()
            new = torch.tensor([(self.rule >> i) & 1 for i in range(8)],
                                  device=x.device, dtype=x.dtype)
            x = new[idx]
        return x


class GameOfLife(nn.Module):
    def __init__(self, size: int = 64, steps: int = 100):
        super().__init__()
        self.size = size; self.steps = steps
        kernel = torch.tensor([[1.0, 1.0, 1.0],
                                [1.0, 0.0, 1.0],
                                [1.0, 1.0, 1.0]]).view(1, 1, 3, 3)
        self.register_buffer("kernel", kernel)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        for _ in range(self.steps):
            n = F.conv2d(x, self.kernel, padding=1)
            x = ((n == 3) | ((x == 1) & (n == 2))).float()
        return x


class KuramotoSync(nn.Module):
    def __init__(self, n: int = 100, K: float = 1.0, dt: float = 0.05,
                 steps: int = 200):
        super().__init__()
        self.n = n; self.K = K; self.dt = dt; self.steps = steps
        self.register_buffer("omega", torch.randn(n) * 0.5)

    def forward(self, x):
        # x : (B, n) phases
        for _ in range(self.steps):
            diff = x.unsqueeze(-1) - x.unsqueeze(-2)
            coupling = (self.K / self.n) * torch.sin(diff).sum(-1)
            x = x + self.dt * (self.omega + coupling)
        return x


class IsingModel(nn.Module):
    def __init__(self, size: int = 64, temperature: float = 2.0, steps: int = 100):
        super().__init__()
        self.size = size; self.T = temperature; self.steps = steps

    def forward(self, x):
        # x : (B, size, size) spins ±1
        if x.dim() == 1 or (x.abs().sum() < 1e-6):
            x = torch.sign(torch.randn(1, self.size, self.size, device=x.device))
        for _ in range(self.steps):
            # Single Metropolis sweep simplifié
            i = torch.randint(0, self.size, (1,)).item()
            j = torch.randint(0, self.size, (1,)).item()
            s = x[..., i, j]
            up = x[..., (i + 1) % self.size, j]
            dn = x[..., (i - 1) % self.size, j]
            lt = x[..., i, (j - 1) % self.size]
            rt = x[..., i, (j + 1) % self.size]
            dE = 2 * s * (up + dn + lt + rt)
            flip = (torch.rand_like(dE) < torch.exp(-dE / self.T)).float()
            x[..., i, j] = s * (1 - 2 * flip)
        return x


class BoidsModule(nn.Module):
    def __init__(self, n_boids: int = 100, perception: float = 5.0, steps: int = 50):
        super().__init__()
        self.n = n_boids; self.r = perception; self.steps = steps

    def forward(self, x):
        # x : (B, n, 4) = [px, py, vx, vy]
        for _ in range(self.steps):
            pos = x[..., :2]
            vel = x[..., 2:]
            # Pairwise distances
            d = torch.cdist(pos, pos)
            mask = (d > 0) & (d < self.r)
            # Cohesion : aller vers centre voisins
            neigh = mask.unsqueeze(-1) * pos.unsqueeze(-2)
            count = mask.sum(-1, keepdim=True).clamp(min=1)
            center = neigh.sum(-2) / count
            coh = (center - pos) * 0.01
            # Alignement : aligner vel
            v_n = mask.unsqueeze(-1) * vel.unsqueeze(-2)
            align = (v_n.sum(-2) / count - vel) * 0.05
            # Separation
            sep_mask = (d > 0) & (d < self.r * 0.3)
            diff = pos.unsqueeze(-2) - pos.unsqueeze(-3)
            sep = (sep_mask.unsqueeze(-1) * diff).sum(-2) * -0.05
            new_vel = vel + coh + align + sep
            # Clip vel
            speed = new_vel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            new_vel = new_vel / speed * speed.clamp(max=2.0)
            new_pos = pos + new_vel * 0.1
            x = torch.cat([new_pos, new_vel], dim=-1)
        return x


class HebbianLayer(nn.Module):
    """Linear + Δw = η·post·pre."""
    def __init__(self, in_dim: int, out_dim: int, eta: float = 1e-3):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.eta = eta

    def forward(self, x):
        out = self.linear(x)
        if self.training:
            with torch.no_grad():
                # Outer product mean batch
                delta = self.eta * (out.t() @ x) / x.shape[0]
                self.linear.weight.add_(delta)
        return out


class EntropyNode(nn.Module):
    def forward(self, x):
        p = torch.softmax(x, dim=-1)
        return -(p * torch.log(p.clamp_min(1e-12))).sum(-1)


class KLDivergenceNode(nn.Module):
    def forward(self, p, q):
        p_safe = p.clamp_min(1e-12)
        q_safe = q.clamp_min(1e-12)
        return (p_safe * (p_safe.log() - q_safe.log())).sum(-1)


class CosineSimilarityNode(nn.Module):
    def forward(self, x, y):
        return F.cosine_similarity(x, y, dim=-1)


class DotProductNode(nn.Module):
    def forward(self, x, y):
        return (x * y).sum(-1)


class L1NormNode(nn.Module):
    def forward(self, x):
        return x.abs().sum(-1)


class L2NormNode(nn.Module):
    def forward(self, x):
        return x.norm(dim=-1)


class SVDNode(nn.Module):
    def __init__(self, k: int = 0):
        super().__init__()
        self.k = k

    def forward(self, x):
        U, S, V = torch.linalg.svd(x, full_matrices=False)
        if self.k > 0:
            U = U[..., :self.k]; S = S[..., :self.k]; V = V[..., :self.k, :]
        return U @ torch.diag_embed(S) @ V


# ============================================================
# Handler factories
# ============================================================

def _shape_out(in_shapes, fallback=(1,)):
    return in_shapes[0] if in_shapes else fallback


def _make_simple_handler(module_factory, shape_fn=None):
    def handler(node, in_shapes):
        m = module_factory(node)
        s = shape_fn(node, in_shapes) if shape_fn else _shape_out(in_shapes)
        m._ark_node_id = getattr(node, "id", "auto")
        return m, s
    return handler


# Mapping concept name → factory (callable taking node returning nn.Module)
HANDLER_FACTORIES: dict = {
    # Activations
    "act_sigmoid": lambda n: nn.Sigmoid(),
    "act_tanh": lambda n: nn.Tanh(),
    "act_relu": lambda n: nn.ReLU(inplace=bool(n.params.get("inplace", False))),
    "act_leaky_relu": lambda n: nn.LeakyReLU(inplace=bool(n.params.get("inplace", False))),
    "act_elu": lambda n: nn.ELU(),
    "act_gelu": lambda n: nn.GELU(),
    "act_silu": lambda n: nn.SiLU(),
    "act_mish": lambda n: nn.Mish(),
    "act_softmax": lambda n: nn.Softmax(dim=-1),
    "act_softplus": lambda n: nn.Softplus(),
    # Layers
    "conv1d": lambda n: nn.Conv1d(int(n.params.get("in_channels", 1)),
                                       int(n.params.get("out_channels", 32)),
                                       int(n.params.get("kernel_size", 3))),
    "conv2d": lambda n: nn.Conv2d(int(n.params.get("in_channels", 3)),
                                       int(n.params.get("out_channels", 32)),
                                       int(n.params.get("kernel_size", 3)),
                                       stride=int(n.params.get("stride", 1)),
                                       padding=int(n.params.get("padding", 0))),
    "conv3d": lambda n: nn.Conv3d(int(n.params.get("in_channels", 1)),
                                       int(n.params.get("out_channels", 16)),
                                       int(n.params.get("kernel_size", 3))),
    "conv_transpose2d": lambda n: nn.ConvTranspose2d(int(n.params.get("in_channels", 32)),
                                                          int(n.params.get("out_channels", 16)),
                                                          int(n.params.get("kernel_size", 4)),
                                                          stride=int(n.params.get("stride", 2))),
    "maxpool2d": lambda n: nn.MaxPool2d(int(n.params.get("kernel_size", 2))),
    "avgpool2d": lambda n: nn.AvgPool2d(int(n.params.get("kernel_size", 2))),
    "batchnorm1d": lambda n: nn.BatchNorm1d(int(n.params.get("num_features", 64))),
    "batchnorm2d": lambda n: nn.BatchNorm2d(int(n.params.get("num_features", 32))),
    "layernorm": lambda n: nn.LayerNorm(int(n.params.get("normalized_shape", 768))),
    "groupnorm": lambda n: nn.GroupNorm(int(n.params.get("num_groups", 8)),
                                              int(n.params.get("num_channels", 32))),
    "dropout_layer": lambda n: nn.Dropout(float(n.params.get("p", 0.5))),
    "dropout2d_layer": lambda n: nn.Dropout2d(float(n.params.get("p", 0.5))),
    "embedding_layer": lambda n: nn.Embedding(int(n.params.get("num_embeddings", 10000)),
                                                    int(n.params.get("embedding_dim", 768))),
    "flatten_layer": lambda n: nn.Flatten(),
    "identity_layer": lambda n: nn.Identity(),
    # Sequential
    "rnn_layer": lambda n: nn.RNN(int(n.params.get("input_size", 128)),
                                       int(n.params.get("hidden_size", 256)),
                                       num_layers=int(n.params.get("num_layers", 1)),
                                       batch_first=True),
    "lstm_layer": lambda n: nn.LSTM(int(n.params.get("input_size", 128)),
                                         int(n.params.get("hidden_size", 256)),
                                         num_layers=int(n.params.get("num_layers", 1)),
                                         batch_first=True),
    "gru_layer": lambda n: nn.GRU(int(n.params.get("input_size", 128)),
                                       int(n.params.get("hidden_size", 256)),
                                       num_layers=int(n.params.get("num_layers", 1)),
                                       batch_first=True),
    # Transformer pieces
    "self_attention": lambda n: SelfAttention(int(n.params.get("d_model", 768)),
                                                    int(n.params.get("d_k", 64))),
    "multi_head_attention": lambda n: nn.MultiheadAttention(
        int(n.params.get("d_model", 768)),
        int(n.params.get("n_heads", 12)), batch_first=True),
    "positional_encoding": lambda n: PositionalEncoding(
        int(n.params.get("d_model", 768)), int(n.params.get("max_len", 2048))),
    "rotary_embeddings": lambda n: RotaryEmbedding(
        int(n.params.get("d_model", 768)), float(n.params.get("base", 10000.0))),
    "causal_mask": lambda n: CausalMaskModule(int(n.params.get("seq_len", 512))),
    # Math
    "entropy": lambda n: EntropyNode(),
    "kl_divergence": lambda n: KLDivergenceNode(),
    "cosine_similarity": lambda n: CosineSimilarityNode(),
    "dot_product": lambda n: DotProductNode(),
    "l1_norm": lambda n: L1NormNode(),
    "l2_norm": lambda n: L2NormNode(),
    "svd": lambda n: SVDNode(int(n.params.get("k", 0))),
    # Bio / dynamical
    "reaction_diffusion": lambda n: ReactionDiffusion(
        float(n.params.get("D_u", 0.16)), float(n.params.get("D_v", 0.08)),
        float(n.params.get("f", 0.035)), float(n.params.get("k", 0.065))),
    "lorenz_attractor": lambda n: LorenzAttractor(
        float(n.params.get("sigma", 10.0)), float(n.params.get("rho", 28.0)),
        float(n.params.get("beta", 2.667))),
    "cellular_automaton": lambda n: CellularAutomaton(
        int(n.params.get("rule", 110)), int(n.params.get("size", 128)),
        int(n.params.get("steps", 256))),
    "game_of_life": lambda n: GameOfLife(int(n.params.get("size", 64)),
                                                int(n.params.get("steps", 100))),
    "kuramoto_sync": lambda n: KuramotoSync(int(n.params.get("n", 100)),
                                                    float(n.params.get("K", 1.0))),
    "ising_model": lambda n: IsingModel(int(n.params.get("size", 64)),
                                                float(n.params.get("temperature", 2.0))),
    "boids": lambda n: BoidsModule(int(n.params.get("n_boids", 100)),
                                          float(n.params.get("perception_radius", 5.0))),
    "hebbian": lambda n: HebbianLayer(int(n.params.get("in_dim", 64)),
                                              int(n.params.get("out_dim", 64)),
                                              float(n.params.get("eta", 1e-3))),
    # Generative — autoencoders
    "autoencoder": lambda n: _build_autoencoder(int(n.params.get("in_dim", 784)),
                                                       int(n.params.get("latent_dim", 32))),
    "vae": lambda n: _build_vae(int(n.params.get("in_dim", 784)),
                                       int(n.params.get("latent_dim", 32))),
}


def _build_autoencoder(in_dim: int, latent_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, 128), nn.ReLU(),
        nn.Linear(128, latent_dim), nn.ReLU(),
        nn.Linear(latent_dim, 128), nn.ReLU(),
        nn.Linear(128, in_dim), nn.Sigmoid(),
    )


def _build_vae(in_dim: int, latent_dim: int) -> nn.Module:
    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc_mu = nn.Linear(in_dim, latent_dim)
            self.fc_lv = nn.Linear(in_dim, latent_dim)
            self.dec = nn.Sequential(nn.Linear(latent_dim, in_dim), nn.Sigmoid())

        def forward(self, x):
            mu = self.fc_mu(x); lv = self.fc_lv(x)
            std = torch.exp(0.5 * lv)
            z = mu + std * torch.randn_like(std)
            return self.dec(z)
    return VAE()


# ============================================================
# Param translation : data format → ParamSpec
# ============================================================

def _to_param_spec(params_list):
    from ark.nodes.registry import ParamSpec
    out = []
    for p in params_list:
        # p = (name, kind, default) ou (name, kind, default, options)
        if len(p) == 3:
            name, kind, default = p
            options = []
        else:
            name, kind, default, options = p[0], p[1], p[2], p[3]
        out.append(ParamSpec(name=name, kind=kind, default=default, options=options or []))
    return out


# ============================================================
# Auto-registration
# ============================================================

def register_glossary_specs(registry) -> int:
    """Enregistre tous les NodeSpec depuis glossary_data.CONCEPTS."""
    from ark.core.glossary_data import CONCEPTS
    from ark.nodes.registry import NodeSpec, SocketSpec

    count = 0
    existing_names = {s.name for s in registry.all()}

    for c in CONCEPTS:
        # Skip si déjà enregistré (eg. metamorph/research)
        if c["name"] in existing_names:
            continue
        # Skip concept/paradigm/tool : seulement node + template
        if c["kind"] not in ("node", "template"):
            continue

        # Inputs/outputs génériques (1 tensor → 1 tensor) sauf cas spéciaux
        inputs = [SocketSpec("in", "tensor")]
        outputs = [SocketSpec("out", "tensor")]

        spec = NodeSpec(
            name=c["name"],
            title=c["title"],
            category=c["category"],
            inputs=inputs,
            outputs=outputs,
            params=_to_param_spec(c.get("params", [])),
            description=c.get("long", c.get("short", "")),
        )
        try:
            registry.register(spec)
            count += 1
        except Exception:
            pass
    return count


def register_glossary_handlers(handlers_dict: dict) -> int:
    """Enregistre les handlers PyTorch pour concepts implémentables."""
    count = 0
    for name, factory in HANDLER_FACTORIES.items():
        if name in handlers_dict:
            continue
        handlers_dict[name] = _make_simple_handler(factory)
        count += 1
    return count


def auto_register_all(registry, handlers_dict: dict) -> tuple[int, int]:
    """Pipeline : specs + handlers."""
    n_specs = register_glossary_specs(registry)
    n_handlers = register_glossary_handlers(handlers_dict)
    return n_specs, n_handlers
