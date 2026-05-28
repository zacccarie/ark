"""Phase 4 réelle : backend diffusers HuggingFace.

Activé via param `real_backend=True` sur diffusion_model_loader.
Lazy import : si diffusers absent, raise RuntimeError explicite.

Modèle par défaut : runwayml/stable-diffusion-v1-5 (~4GB, cache HF local).
Override : checkpoint = repo_id HF ou chemin local .safetensors.

API minimale compatible diffusion_stubs._DiffusionModelSentinel :
- model.unet, model.scheduler, model.tokenizer accessibles
- vae.encode(image), vae.decode(latent)
- clip.encode(text) → embeddings
"""
from __future__ import annotations
import torch
import torch.nn as nn
from pathlib import Path


def _ensure_diffusers():
    try:
        import diffusers  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError as e:
        raise RuntimeError(
            "diffusers/transformers absents. Install : "
            "uv pip install -e '.[ml]' (ou pip install diffusers transformers)"
        ) from e


def _device_auto() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ============================================================
# Pipeline holder
# ============================================================

class _RealPipelineHolder:
    """Garde le pipeline diffusers complet (unet+vae+clip+scheduler+tokenizer)."""

    def __init__(self, checkpoint: str, device: str = "auto"):
        _ensure_diffusers()
        from diffusers import StableDiffusionPipeline
        self.checkpoint = checkpoint
        self.device = _device_auto() if device == "auto" else device
        dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        # Si checkpoint = path local .safetensors → from_single_file
        ck = Path(checkpoint)
        if ck.exists() and ck.suffix in (".safetensors", ".ckpt"):
            self.pipe = StableDiffusionPipeline.from_single_file(
                str(ck), torch_dtype=dtype, safety_checker=None,
            )
        else:
            # Repo HF (ex. runwayml/stable-diffusion-v1-5)
            self.pipe = StableDiffusionPipeline.from_pretrained(
                checkpoint, torch_dtype=dtype, safety_checker=None,
            )
        self.pipe = self.pipe.to(self.device)
        # Désactive progress bar (sortie sur stderr pollue UI)
        self.pipe.set_progress_bar_config(disable=True)

    def __repr__(self):
        return f"<RealDiffusionPipe:{self.checkpoint}@{self.device}>"


class _RealCLIPAdapter:
    """Mimic _CLIPSentinel API mais via tokenizer+text_encoder vrais."""

    def __init__(self, pipe_holder: _RealPipelineHolder):
        self.holder = pipe_holder

    def encode(self, text: str) -> torch.Tensor:
        p = self.holder.pipe
        tokens = p.tokenizer(text, padding="max_length",
                              max_length=p.tokenizer.model_max_length,
                              truncation=True, return_tensors="pt")
        tokens = tokens.input_ids.to(self.holder.device)
        with torch.no_grad():
            emb = p.text_encoder(tokens)[0]
        return emb  # (1, 77, 768)


class _RealVAEAdapter:
    """Mimic _VAESentinel API via pipe.vae."""

    def __init__(self, pipe_holder: _RealPipelineHolder):
        self.holder = pipe_holder

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        # image (B, 3, H, W) ∈ [0,1] → latent (B, 4, H/8, W/8)
        img = image.to(self.holder.device, dtype=self.holder.pipe.vae.dtype)
        img = img * 2.0 - 1.0  # normalize [-1,1]
        with torch.no_grad():
            latent = self.holder.pipe.vae.encode(img).latent_dist.sample()
        return latent * self.holder.pipe.vae.config.scaling_factor

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        lat = latent.to(self.holder.device, dtype=self.holder.pipe.vae.dtype)
        lat = lat / self.holder.pipe.vae.config.scaling_factor
        with torch.no_grad():
            img = self.holder.pipe.vae.decode(lat).sample
        img = (img / 2 + 0.5).clamp(0, 1)
        return img


# ============================================================
# Modules (compat avec graph_compiler — forward returns expected shape)
# ============================================================

class DiffusionModelLoaderReal(nn.Module):
    """Charge pipeline réel + retourne tuple (model, clip, vae) wrappés."""

    def __init__(self, checkpoint: str, device: str = "auto"):
        super().__init__()
        self.checkpoint = checkpoint
        self.device = device
        self._holder: _RealPipelineHolder | None = None

    def _load(self):
        if self._holder is None:
            self._holder = _RealPipelineHolder(self.checkpoint, self.device)

    def forward(self, x=None):
        self._load()
        return (
            self._holder,  # model (pipe complet)
            _RealCLIPAdapter(self._holder),
            _RealVAEAdapter(self._holder),
        )


class CLIPTextEncodeReal(nn.Module):
    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def forward(self, clip):
        if hasattr(clip, "encode"):
            return clip.encode(self.text)
        raise RuntimeError("clip input n'a pas de .encode()")


class KSamplerReal(nn.Module):
    """Sampling diffusers réel via scheduler + unet step."""

    def __init__(self, seed: int = 42, steps: int = 20, cfg: float = 7.0,
                 sampler: str = "euler", scheduler: str = "normal",
                 denoise: float = 1.0):
        super().__init__()
        self.seed = seed
        self.steps = steps
        self.cfg = cfg
        self.sampler = sampler
        self.scheduler = scheduler
        self.denoise = denoise

    def _make_scheduler(self, base_scheduler):
        from diffusers import (
            EulerDiscreteScheduler, EulerAncestralDiscreteScheduler,
            HeunDiscreteScheduler, DPMSolverMultistepScheduler, DDIMScheduler,
        )
        mp = {
            "euler": EulerDiscreteScheduler,
            "euler_ancestral": EulerAncestralDiscreteScheduler,
            "heun": HeunDiscreteScheduler,
            "dpmpp_2m": DPMSolverMultistepScheduler,
            "ddim": DDIMScheduler,
        }
        cls = mp.get(self.sampler, EulerDiscreteScheduler)
        return cls.from_config(base_scheduler.config)

    def forward(self, model, positive, negative, latent):
        if not isinstance(model, _RealPipelineHolder):
            raise RuntimeError("KSamplerReal demande _RealPipelineHolder en input model")
        pipe = model.pipe
        device = model.device
        dtype = pipe.unet.dtype

        sched = self._make_scheduler(pipe.scheduler)
        sched.set_timesteps(self.steps, device=device)

        g = torch.Generator(device="cpu").manual_seed(self.seed)
        # Init latent : noise pur si latent vide, sinon partial denoise
        noise = torch.randn(latent.shape, generator=g, device="cpu").to(device, dtype)
        if latent.abs().sum() < 1e-6:
            lat = noise * sched.init_noise_sigma
        else:
            lat = latent.to(device, dtype) + noise * self.denoise

        pos = positive.to(device, dtype)
        neg = negative.to(device, dtype) if negative is not None else torch.zeros_like(pos)
        # Concat pour classifier-free guidance
        embeds = torch.cat([neg, pos], dim=0)

        with torch.no_grad():
            for t in sched.timesteps:
                lat_in = torch.cat([lat, lat], dim=0)
                lat_in = sched.scale_model_input(lat_in, t)
                noise_pred = pipe.unet(lat_in, t, encoder_hidden_states=embeds).sample
                neg_pred, pos_pred = noise_pred.chunk(2)
                noise_pred = neg_pred + self.cfg * (pos_pred - neg_pred)
                lat = sched.step(noise_pred, t, lat).prev_sample

        return lat


class VAEDecodeReal(nn.Module):
    def forward(self, vae, latent):
        if not hasattr(vae, "decode"):
            raise RuntimeError("vae input n'a pas de .decode()")
        return vae.decode(latent)


class VAEEncodeReal(nn.Module):
    def forward(self, vae, image):
        if not hasattr(vae, "encode"):
            raise RuntimeError("vae input n'a pas de .encode()")
        return vae.encode(image)


# ============================================================
# Dispatch : choisit Real vs Stub selon param `real_backend`
# ============================================================

def make_diffusion_loader(checkpoint: str, device: str = "auto",
                            real_backend: bool = False):
    if real_backend:
        return DiffusionModelLoaderReal(checkpoint, device)
    from ark.core.diffusion_stubs import DiffusionModelLoaderStub
    return DiffusionModelLoaderStub(checkpoint, device)


def make_clip_encode(text: str, real_backend: bool = False):
    if real_backend:
        return CLIPTextEncodeReal(text)
    from ark.core.diffusion_stubs import CLIPTextEncodeStub
    return CLIPTextEncodeStub(text)


def make_ksampler(seed: int, steps: int, cfg: float, sampler: str,
                    scheduler: str, denoise: float, real_backend: bool = False):
    kw = dict(seed=seed, steps=steps, cfg=cfg, sampler=sampler,
                 scheduler=scheduler, denoise=denoise)
    if real_backend:
        return KSamplerReal(**kw)
    from ark.core.diffusion_stubs import KSamplerStub
    return KSamplerStub(**kw)


def make_vae_decode(real_backend: bool = False):
    if real_backend:
        return VAEDecodeReal()
    from ark.core.diffusion_stubs import VAEDecodeStub
    return VAEDecodeStub()


def make_vae_encode(real_backend: bool = False):
    if real_backend:
        return VAEEncodeReal()
    from ark.core.diffusion_stubs import VAEEncodeStub
    return VAEEncodeStub()
