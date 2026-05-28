"""Stubs Phase 4 ComfyUI nodes — pas de modèles téléchargés, signaux fake.

Permettent de tester le pipeline visuel + interactions sans dépendance
gigas diffusers/transformers. Phase 4 réelle remplacera par diffusers
vrais.

Convention :
- DiffusionModelLoaderStub : retourne sentinel objects (model/clip/vae).
- CLIPTextEncodeStub : retourne tensor (1, 77, 768) random fixe par texte.
- EmptyLatentImageStub : tensor zeros (B, 4, H/8, W/8).
- KSamplerStub : ajoute noise progressif (fake diffusion sampling).
- VAEDecodeStub : convertit latent (B, 4, H, W) → image (B, 3, H*8, W*8).
- VAEEncodeStub : inverse.
- LoadImageStub : retourne tensor 512x512 noise déterministe par path hash.
- SaveImageStub : sauvegarde via PIL si dispo, sinon torch.save .pt.
- LoRALoaderStub : pass-through (model, clip) avec tag.
- ControlNetApplyStub : pass-through conditioning.
"""
from __future__ import annotations
import hashlib
import torch
import torch.nn as nn


# ============================================================
# Sentinels
# ============================================================

class _DiffusionModelSentinel:
    def __init__(self, checkpoint: str, device: str = "auto"):
        self.checkpoint = checkpoint
        self.device = device
    def __repr__(self):
        return f"<DiffusionModel:{self.checkpoint}>"


class _CLIPSentinel:
    def __init__(self, checkpoint: str):
        self.checkpoint = checkpoint
    def encode(self, text: str) -> torch.Tensor:
        # Deterministic per text
        seed = int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)
        g = torch.Generator().manual_seed(seed)
        return torch.randn(1, 77, 768, generator=g)


class _VAESentinel:
    def __init__(self, checkpoint: str):
        self.checkpoint = checkpoint
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        # Latent (B, 4, h, w) → image (B, 3, h*8, w*8) (fake : upsample + clip)
        B, C, h, w = latent.shape
        if C != 4:
            latent = latent.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
        img = torch.nn.functional.interpolate(latent, scale_factor=8, mode="bilinear",
                                                 align_corners=False)
        img = img[:, :3]  # take 3 channels
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        return img
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        # image (B, 3, H, W) → latent (B, 4, H/8, W/8)
        B, C, H, W = image.shape
        l = torch.nn.functional.interpolate(image, scale_factor=1 / 8, mode="bilinear",
                                              align_corners=False)
        # Pad to 4 channels
        pad = torch.zeros(B, 4 - C, l.shape[2], l.shape[3])
        return torch.cat([l, pad], dim=1) if C < 4 else l[:, :4]


# ============================================================
# Modules
# ============================================================

class DiffusionModelLoaderStub(nn.Module):
    def __init__(self, checkpoint: str, device: str = "auto"):
        super().__init__()
        self.checkpoint = checkpoint
        self.device = device
    def forward(self, x=None):
        # Retourne tuple (model, clip, vae) sentinels
        return (
            _DiffusionModelSentinel(self.checkpoint, self.device),
            _CLIPSentinel(self.checkpoint),
            _VAESentinel(self.checkpoint),
        )


class CLIPTextEncodeStub(nn.Module):
    def __init__(self, text: str):
        super().__init__()
        self.text = text
    def forward(self, clip):
        if hasattr(clip, "encode"):
            return clip.encode(self.text)
        # Fallback random
        return torch.randn(1, 77, 768)


class EmptyLatentImageStub(nn.Module):
    def __init__(self, width: int, height: int, batch_size: int = 1):
        super().__init__()
        self.width = width
        self.height = height
        self.batch_size = batch_size
    def forward(self, x=None):
        return torch.zeros(self.batch_size, 4, self.height // 8, self.width // 8)


class KSamplerStub(nn.Module):
    """Fake sampler : ajoute noise progressif puis "convergence" déterministe."""
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
    def forward(self, model, positive, negative, latent):
        g = torch.Generator().manual_seed(self.seed)
        # Initial noise injection
        noise = torch.randn(latent.shape, generator=g) * self.denoise
        # "Sample" : interpole linéairement latent → noise puis applique sigmoid pour
        # mimer convergence
        for _ in range(self.steps):
            noise = noise * 0.95
        # Influence positive/negative en mélangeant leur norme dans le latent
        # (totalement fake mais déterministe)
        p_w = positive.float().mean().item()
        n_w = negative.float().mean().item() if negative is not None else 0.0
        cond_scale = (p_w - n_w) * self.cfg / 14.0
        out = latent + noise + cond_scale * torch.ones_like(latent) * 0.05
        return out


class VAEDecodeStub(nn.Module):
    def forward(self, vae, latent):
        if hasattr(vae, "decode"):
            return vae.decode(latent)
        # Fallback : upsample x8
        return torch.nn.functional.interpolate(latent[:, :3], scale_factor=8,
                                                  mode="bilinear", align_corners=False)


class VAEEncodeStub(nn.Module):
    def forward(self, vae, image):
        if hasattr(vae, "encode"):
            return vae.encode(image)
        return torch.nn.functional.interpolate(image[:, :4], scale_factor=1 / 8,
                                                  mode="bilinear", align_corners=False)


class LoadImageStub(nn.Module):
    def __init__(self, path: str):
        super().__init__()
        self.path = path
    def forward(self, x=None):
        # Si le fichier existe, charge via PIL
        try:
            from PIL import Image
            import numpy as np
            from pathlib import Path
            p = Path(self.path)
            if p.exists():
                img = Image.open(p).convert("RGB")
                arr = np.array(img).astype("float32") / 255.0
                t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
                mask = torch.ones(1, 1, t.shape[2], t.shape[3])
                return (t, mask)
        except Exception:
            pass
        # Fallback deterministic random
        seed = int(hashlib.sha1(self.path.encode("utf-8")).hexdigest()[:8], 16)
        g = torch.Generator().manual_seed(seed)
        return (torch.randn(1, 3, 512, 512, generator=g),
                torch.ones(1, 1, 512, 512))


class SaveImageStub(nn.Module):
    def __init__(self, prefix: str = "ark_", out_dir: str = "./output"):
        super().__init__()
        self.prefix = prefix
        self.out_dir = out_dir
    def forward(self, image):
        from pathlib import Path
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        import time
        ts = int(time.time() * 1000)
        path = Path(self.out_dir) / f"{self.prefix}{ts}.png"
        try:
            from PIL import Image
            import numpy as np
            img = image[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            arr = (img * 255).astype("uint8")
            Image.fromarray(arr).save(path)
        except Exception:
            torch.save(image, str(path.with_suffix(".pt")))
        return image  # pass-through


class LoRALoaderStub(nn.Module):
    def __init__(self, lora_name: str, strength_model: float = 1.0,
                 strength_clip: float = 1.0):
        super().__init__()
        self.lora_name = lora_name
        self.strength_model = strength_model
        self.strength_clip = strength_clip
    def forward(self, model, clip):
        # Pass-through (in real Phase 4 : merge LoRA delta dans model.weights)
        return (model, clip)


class ControlNetApplyStub(nn.Module):
    def __init__(self, strength: float = 1.0):
        super().__init__()
        self.strength = strength
    def forward(self, conditioning, control_net, image):
        # Pass-through avec scale
        return conditioning * (1.0 + self.strength * 0.1)


# ============================================================
# Handlers (pour graph_compiler HANDLERS)
# ============================================================

def h_diffusion_model_loader(node, in_shapes):
    from ark.core.diffusion_real import make_diffusion_loader
    m = make_diffusion_loader(
        node.params.get("checkpoint", "model.safetensors"),
        node.params.get("device", "auto"),
        bool(node.params.get("real_backend", False)),
    )
    return m, (("model_opaque",), ("clip_opaque",), ("vae_opaque",))


def h_clip_text_encode(node, in_shapes):
    from ark.core.diffusion_real import make_clip_encode
    return make_clip_encode(
        node.params.get("text", ""),
        bool(node.params.get("real_backend", False)),
    ), (1, 77, 768)


def h_empty_latent_image(node, in_shapes):
    return EmptyLatentImageStub(
        int(node.params.get("width", 512)),
        int(node.params.get("height", 512)),
        int(node.params.get("batch_size", 1)),
    ), (4, -1, -1)


def h_k_sampler(node, in_shapes):
    from ark.core.diffusion_real import make_ksampler
    return make_ksampler(
        seed=int(node.params.get("seed", 42)),
        steps=int(node.params.get("steps", 20)),
        cfg=float(node.params.get("cfg", 7.0)),
        sampler=node.params.get("sampler", "euler"),
        scheduler=node.params.get("scheduler", "normal"),
        denoise=float(node.params.get("denoise", 1.0)),
        real_backend=bool(node.params.get("real_backend", False)),
    ), (4, -1, -1)


def h_vae_decode(node, in_shapes):
    from ark.core.diffusion_real import make_vae_decode
    return make_vae_decode(bool(node.params.get("real_backend", False))), (3, -1, -1)


def h_vae_encode(node, in_shapes):
    from ark.core.diffusion_real import make_vae_encode
    return make_vae_encode(bool(node.params.get("real_backend", False))), (4, -1, -1)


def h_load_image(node, in_shapes):
    # 2 outputs : image (3, 512, 512) + mask (1, 512, 512)
    return LoadImageStub(node.params.get("path", "input.png")), ((3, 512, 512), (1, 512, 512))


def h_save_image(node, in_shapes):
    return SaveImageStub(node.params.get("prefix", "ark_"),
                            node.params.get("dir", "./output")), (3, -1, -1)


def h_lora_loader(node, in_shapes):
    return LoRALoaderStub(
        node.params.get("lora_name", "lora.safetensors"),
        float(node.params.get("strength_model", 1.0)),
        float(node.params.get("strength_clip", 1.0)),
    ), (("model_opaque",), ("clip_opaque",))


def h_controlnet_apply(node, in_shapes):
    return ControlNetApplyStub(float(node.params.get("strength", 1.0))), (1, 77, 768)
