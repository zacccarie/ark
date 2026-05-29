"""Phase 60 — Diffusion primitives.

Permettent construire LLM ET text2image/class2image from scratch dans
l'éditeur ARK.

Nodes :
- timestep_embedding   : (B,) int → (B, dim) sinusoidal
- diffusion_scheduler  : forward q_sample / reverse sample step DDIM
- class_embedding      : (B,) int → (B, dim) learned embed
- cross_attention      : (Q, ctx) → attn output
- unet_mini            : mini U-Net class/text conditional, 2 down + 2 up
- ddim_sampler         : génère depuis modèle entraîné

Compat : tensors PyTorch, signature (in, out, params).
"""
from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Sinusoidal timestep embedding
# ============================================================

class TimestepEmbedding(nn.Module):
    """(B,) integer t → (B, dim) sinusoidal positional, projeté MLP."""

    def __init__(self, dim: int = 128, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def _sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(0, half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, t):
        if isinstance(t, int):
            t = torch.tensor([t], dtype=torch.long)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        emb = self._sinusoidal(t)
        return self.mlp(emb)


# ============================================================
# Class embedding (label → vector for conditioning)
# ============================================================

class ClassEmbedding(nn.Module):
    """(B,) int label → (B, dim) learned embedding. Pour cond class2image."""

    def __init__(self, num_classes: int = 10, dim: int = 128):
        super().__init__()
        self.embed = nn.Embedding(num_classes + 1, dim)  # +1 unconditional
        self.num_classes = num_classes
        self.dim = dim

    def forward(self, labels):
        if isinstance(labels, int):
            labels = torch.tensor([labels], dtype=torch.long)
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        return self.embed(labels.long())


# ============================================================
# Cross attention (Q from feature map, K/V from context)
# ============================================================

class CrossAttention(nn.Module):
    """Multi-head cross attention.

    Q from x (B, N, D_q), K/V from ctx (B, M, D_ctx) → (B, N, D_q).
    Supporte image features flattened (B, C, H, W) → (B, H*W, C).
    """

    def __init__(self, dim: int = 64, n_heads: int = 4,
                 ctx_dim: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        ctx_dim = ctx_dim or dim
        head_dim = dim // n_heads
        assert head_dim * n_heads == dim
        self.head_dim = head_dim
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(ctx_dim, dim, bias=False)
        self.v = nn.Linear(ctx_dim, dim, bias=False)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, ctx):
        # Flatten spatial dim if image
        was_image = (x.dim() == 4)
        if was_image:
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(1)  # (B, D_ctx) → (B, 1, D_ctx)
        B, N, _ = x.shape
        _, M, _ = ctx.shape
        q = self.q(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(ctx).reshape(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(ctx).reshape(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.softmax((q @ k.transpose(-1, -2)) * scale, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, self.dim)
        out = self.out(out)
        if was_image:
            out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out


# ============================================================
# Diffusion scheduler (DDPM linear beta)
# ============================================================

class DiffusionScheduler(nn.Module):
    """Linear beta DDPM. 2 modes :

    - "forward" : input x_0 → return (x_t, noise) en samplant t aléatoire
    - "noise_only" : input (x_0, t, noise) → return x_t
    - "sample" : input (x_t, t, predicted_noise) → return x_{t-1} (DDIM)
    """

    def __init__(self, num_timesteps: int = 200,
                 beta_start: float = 1e-4, beta_end: float = 0.02,
                 mode: str = "forward"):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.mode = mode
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_acp", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_1macp", torch.sqrt(1.0 - alphas_cumprod))
        # State for last forward
        self.last_t: Optional[torch.Tensor] = None
        self.last_noise: Optional[torch.Tensor] = None
        self.last_x_t: Optional[torch.Tensor] = None

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_acp_t = self.sqrt_acp[t].view(-1, *[1] * (x_0.dim() - 1))
        sqrt_1macp_t = self.sqrt_1macp[t].view(-1, *[1] * (x_0.dim() - 1))
        return sqrt_acp_t * x_0 + sqrt_1macp_t * noise, noise

    def ddim_step(self, x_t, t, predicted_noise, eta: float = 0.0):
        """Single DDIM reverse step."""
        if isinstance(t, int):
            t = torch.tensor([t], device=x_t.device)
        acp_t = self.alphas_cumprod[t].view(-1, *[1] * (x_t.dim() - 1))
        if (t > 0).all():
            acp_prev = self.alphas_cumprod[t - 1].view(-1, *[1] * (x_t.dim() - 1))
        else:
            acp_prev = torch.ones_like(acp_t)
        # Predict x_0
        pred_x0 = (x_t - torch.sqrt(1 - acp_t) * predicted_noise) / torch.sqrt(acp_t)
        pred_x0 = pred_x0.clamp(-1, 1)
        # Direction term
        dir_xt = torch.sqrt(1 - acp_prev) * predicted_noise
        x_prev = torch.sqrt(acp_prev) * pred_x0 + dir_xt
        return x_prev

    def forward(self, x):
        if self.mode == "forward":
            # x = x_0 (B, ...) → sample t random, return x_t (replaces input)
            if not isinstance(x, torch.Tensor):
                return x
            B = x.shape[0]
            t = torch.randint(0, self.num_timesteps, (B,), device=x.device)
            x_t, noise = self.q_sample(x, t)
            self.last_t = t
            self.last_noise = noise
            self.last_x_t = x_t
            return x_t
        return x


# ============================================================
# Mini U-Net (class/text conditional, 2 down + middle + 2 up)
# ============================================================

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_c) if out_c >= 8 else nn.GroupNorm(1, out_c)
        self.norm2 = nn.GroupNorm(8, out_c) if out_c >= 8 else nn.GroupNorm(1, out_c)
        self.cond_mlp = nn.Linear(cond_dim, out_c)
        self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, cond):
        h = self.conv1(x)
        h = self.norm1(h)
        # Add conditioning (broadcast over spatial)
        c = self.cond_mlp(cond).unsqueeze(-1).unsqueeze(-1)
        h = h + c
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return F.silu(h + self.skip(x))


class UNetMini(nn.Module):
    """Mini U-Net pour DDPM class/text conditional MNIST/CIFAR.

    Input  : x_t (B, C, H, W) + cond (B, D_cond)
    Output : predicted noise (B, C, H, W)
    """

    def __init__(self, in_ch: int = 1, base_ch: int = 32, cond_dim: int = 128,
                 use_attn: bool = True):
        super().__init__()
        self.in_ch = in_ch
        self.base_ch = base_ch
        self.cond_dim = cond_dim
        ch1 = base_ch
        ch2 = base_ch * 2
        ch3 = base_ch * 4
        # Down
        self.in_conv = nn.Conv2d(in_ch, ch1, 3, padding=1)
        self.d1 = ResBlock(ch1, ch1, cond_dim)
        self.d2 = ResBlock(ch1, ch2, cond_dim)
        self.down = nn.Conv2d(ch2, ch2, 4, stride=2, padding=1)  # /2
        self.d3 = ResBlock(ch2, ch3, cond_dim)
        # Middle with cross attn
        self.use_attn = use_attn
        if use_attn:
            self.attn = CrossAttention(dim=ch3, n_heads=4, ctx_dim=cond_dim)
        self.mid = ResBlock(ch3, ch3, cond_dim)
        # Up
        self.up = nn.ConvTranspose2d(ch3, ch2, 4, stride=2, padding=1)
        self.u1 = ResBlock(ch2 + ch2, ch2, cond_dim)  # skip d2 cat
        self.u2 = ResBlock(ch2, ch1, cond_dim)
        self.out_conv = nn.Conv2d(ch1, in_ch, 3, padding=1)

    def forward(self, x, cond):
        # x : (B, C, H, W), cond : (B, D_cond)
        h0 = self.in_conv(x)
        h1 = self.d1(h0, cond)
        h2 = self.d2(h1, cond)
        h3 = self.down(h2)
        h4 = self.d3(h3, cond)
        if self.use_attn:
            h4 = h4 + self.attn(h4, cond)
        h5 = self.mid(h4, cond)
        u = self.up(h5)
        # Crop or pad if mismatch
        if u.shape[-2:] != h2.shape[-2:]:
            u = F.interpolate(u, size=h2.shape[-2:])
        h6 = self.u1(torch.cat([u, h2], dim=1), cond)
        h7 = self.u2(h6, cond)
        return self.out_conv(h7)


# ============================================================
# DDIM Sampler (post-training generation)
# ============================================================

class DDIMSampler(nn.Module):
    """Génère images depuis modèle DDPM entraîné.

    Input : cond (B, D_cond) → noise → sample → image (B, C, H, W).
    Référence au scheduler + unet via attr `_scheduler` `_unet` set runtime.
    """

    def __init__(self, n_steps: int = 50, channels: int = 1,
                 height: int = 28, width: int = 28, batch_size: int = 4):
        super().__init__()
        self.n_steps = n_steps
        self.channels = channels
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self._scheduler: Optional[DiffusionScheduler] = None
        self._unet: Optional[UNetMini] = None
        self.last_samples: Optional[torch.Tensor] = None

    def forward(self, cond):
        if self._scheduler is None or self._unet is None:
            return torch.zeros(self.batch_size, self.channels,
                                  self.height, self.width)
        if isinstance(cond, int):
            cond = torch.full((self.batch_size,), cond, dtype=torch.long)
        device = next(self._unet.parameters()).device
        B = cond.shape[0] if cond.dim() >= 1 else self.batch_size
        # Embed cond if int labels (else assume already embedded)
        if cond.dim() == 1:
            # Need class_embedding upstream → caller passes embedded
            pass
        x_t = torch.randn(B, self.channels, self.height, self.width,
                              device=device)
        T = self._scheduler.num_timesteps
        step = T // self.n_steps
        timesteps = list(range(0, T, step))[::-1]
        for t_int in timesteps:
            t = torch.full((B,), t_int, dtype=torch.long, device=device)
            with torch.no_grad():
                eps = self._unet(x_t, cond.to(device))
            x_t = self._scheduler.ddim_step(x_t, t, eps)
        # Map [-1, 1] → [0, 1]
        out = ((x_t + 1) / 2).clamp(0, 1)
        self.last_samples = out
        return out


# ============================================================
# Handlers
# ============================================================

def h_timestep_embedding(node, in_shapes):
    m = TimestepEmbedding(
        dim=int(node.params.get("dim", 128)),
        max_period=int(node.params.get("max_period", 10000)),
    )
    return m, (m.dim,)


def h_class_embedding(node, in_shapes):
    m = ClassEmbedding(
        num_classes=int(node.params.get("num_classes", 10)),
        dim=int(node.params.get("dim", 128)),
    )
    return m, (m.dim,)


def h_cross_attention(node, in_shapes):
    m = CrossAttention(
        dim=int(node.params.get("dim", 64)),
        n_heads=int(node.params.get("n_heads", 4)),
        ctx_dim=int(node.params.get("ctx_dim", 128)),
    )
    return m, in_shapes[0] if in_shapes else (1, 64)


def h_diffusion_scheduler(node, in_shapes):
    m = DiffusionScheduler(
        num_timesteps=int(node.params.get("num_timesteps", 200)),
        beta_start=float(node.params.get("beta_start", 1e-4)),
        beta_end=float(node.params.get("beta_end", 0.02)),
        mode=node.params.get("mode", "forward"),
    )
    return m, in_shapes[0] if in_shapes else (1,)


def h_unet_mini(node, in_shapes):
    m = UNetMini(
        in_ch=int(node.params.get("in_ch", 1)),
        base_ch=int(node.params.get("base_ch", 32)),
        cond_dim=int(node.params.get("cond_dim", 128)),
        use_attn=bool(node.params.get("use_attn", True)),
    )
    return m, in_shapes[0] if in_shapes else (1, 1, 28, 28)


def h_ddim_sampler(node, in_shapes):
    m = DDIMSampler(
        n_steps=int(node.params.get("n_steps", 50)),
        channels=int(node.params.get("channels", 1)),
        height=int(node.params.get("height", 28)),
        width=int(node.params.get("width", 28)),
        batch_size=int(node.params.get("batch_size", 4)),
    )
    return m, (m.batch_size, m.channels, m.height, m.width)


def register_diffusion_handlers(handlers_dict: dict) -> bool:
    handlers_dict["timestep_embedding"] = h_timestep_embedding
    handlers_dict["class_embedding"] = h_class_embedding
    handlers_dict["cross_attention"] = h_cross_attention
    handlers_dict["diffusion_scheduler"] = h_diffusion_scheduler
    handlers_dict["unet_mini"] = h_unet_mini
    handlers_dict["ddim_sampler"] = h_ddim_sampler
    return True


def register_diffusion_specs(registry) -> bool:
    from ark.nodes.registry import NodeSpec, ParamSpec, SocketSpec

    CAT = "Diffusion"

    registry.register(NodeSpec(
        name="timestep_embedding", title="Timestep Embedding",
        category=CAT,
        inputs=[SocketSpec("t", "tensor")],
        outputs=[SocketSpec("embed", "tensor")],
        params=[
            ParamSpec("dim", "int", 128, min=16, max=1024),
            ParamSpec("max_period", "int", 10000, min=100, max=100000),
        ],
        description="Sinusoidal encoding du timestep (DDPM). (B,) int → (B, dim) "
                       "via cos/sin + projection MLP. Standard diffusion.",
        role="Encode diffusion timestep t.",
        function="cos/sin freq exponential + Linear→SiLU→Linear.",
    ))

    registry.register(NodeSpec(
        name="class_embedding", title="Class Embedding (cond)",
        category=CAT,
        inputs=[SocketSpec("labels", "tensor")],
        outputs=[SocketSpec("embed", "tensor")],
        params=[
            ParamSpec("num_classes", "int", 10, min=2, max=10000),
            ParamSpec("dim", "int", 128, min=16, max=1024),
        ],
        description="Conditioning class label → vecteur learned. (B,) int → (B, dim). "
                       "num_classes+1 (slot 'unconditional').",
        role="Class label conditioning embedding.",
        function="nn.Embedding(num_classes+1, dim).",
    ))

    registry.register(NodeSpec(
        name="cross_attention", title="Cross Attention",
        category=CAT,
        inputs=[SocketSpec("x", "tensor"), SocketSpec("ctx", "tensor")],
        outputs=[SocketSpec("out", "tensor")],
        params=[
            ParamSpec("dim", "int", 64, min=8, max=1024),
            ParamSpec("n_heads", "int", 4, min=1, max=32),
            ParamSpec("ctx_dim", "int", 128, min=8, max=1024),
        ],
        description="Multi-head cross attention. Q de x, K/V de ctx. "
                       "Supporte image (B,C,H,W) auto-flatten ↔ (B,N,C). "
                       "Bloc essentiel SD/UNet pour cond texte ou classe.",
        role="Cross attention image ← text/class.",
        function="Q·K^T / sqrt(d) → softmax → ·V.",
    ))

    registry.register(NodeSpec(
        name="diffusion_scheduler", title="Diffusion Scheduler",
        category=CAT,
        inputs=[SocketSpec("x_0", "tensor")],
        outputs=[SocketSpec("x_t", "tensor")],
        params=[
            ParamSpec("num_timesteps", "int", 200, min=10, max=2000),
            ParamSpec("beta_start", "float", 1e-4, min=1e-6, max=0.1),
            ParamSpec("beta_end", "float", 0.02, min=1e-4, max=0.5),
            ParamSpec("mode", "choice", "forward",
                          options=["forward", "noise_only"]),
        ],
        description="DDPM linear beta schedule. Forward q_sample(x_0, t, ε) → x_t. "
                       "Random t sampling pendant training. Output x_t, "
                       "stocke t + noise dans attrs pour MSE loss downstream.",
        role="DDPM forward noising.",
        function="x_t = sqrt(ᾱ_t)·x_0 + sqrt(1-ᾱ_t)·ε.",
    ))

    registry.register(NodeSpec(
        name="unet_mini", title="UNet Mini (cond)",
        category=CAT,
        inputs=[SocketSpec("x_t", "tensor"), SocketSpec("cond", "tensor")],
        outputs=[SocketSpec("noise_pred", "tensor")],
        params=[
            ParamSpec("in_ch", "int", 1, min=1, max=16),
            ParamSpec("base_ch", "int", 32, min=8, max=256),
            ParamSpec("cond_dim", "int", 128, min=16, max=1024),
            ParamSpec("use_attn", "bool", True),
        ],
        description="U-Net mini 2-down/2-up + middle cross-attn. Prédit "
                       "noise ε pour DDPM. MNIST 28x28 OK avec base_ch=32 "
                       "(~150k params). CIFAR base_ch=64.",
        role="Conditional noise prediction U-Net.",
        function="ResBlock×4 + Conv down/up + CrossAttn middle.",
    ))

    registry.register(NodeSpec(
        name="ddim_sampler", title="DDIM Sampler",
        category=CAT,
        inputs=[SocketSpec("cond", "tensor")],
        outputs=[SocketSpec("samples", "tensor")],
        params=[
            ParamSpec("n_steps", "int", 50, min=5, max=1000),
            ParamSpec("channels", "int", 1, min=1, max=4),
            ParamSpec("height", "int", 28, min=8, max=512),
            ParamSpec("width", "int", 28, min=8, max=512),
            ParamSpec("batch_size", "int", 4, min=1, max=64),
        ],
        description="Génère images depuis modèle entraîné DDPM. "
                       "Reverse DDIM N steps depuis noise pur conditionné cond. "
                       "Output (B, C, H, W) ∈ [0, 1].",
        role="DDIM reverse sampling generation.",
        function="x_T ~ N(0,1) → loop reverse → x_0.",
        has_thumbnail=True,
    ))
    return True
