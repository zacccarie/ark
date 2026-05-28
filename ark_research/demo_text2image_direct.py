"""Demo text2image REAL via diffusion_real modules directement.

Skip graph_compiler bug multi-output. Instancie modules ARK avec real_backend=True
et les chaîne manuellement = équivalent pipeline ComfyUI mais sans Qt scene.

Modèle léger : OFA-Sys/small-stable-diffusion-v0 (~400MB) ou sd-tiny.
"""
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARK_ROOT = HERE.parent
sys.path.insert(0, str(ARK_ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import torch
from ark.app import create_app
from ark.core.diffusion_real import (
    DiffusionModelLoaderReal, CLIPTextEncodeReal, KSamplerReal,
    VAEDecodeReal,
)


def section(t):
    print(f"\n{'=' * 60}\n  {t}\n{'=' * 60}")


# Config
MODEL = "OFA-Sys/small-stable-diffusion-v0"
PROMPT = "a cute cyberpunk cat sitting on a neon sign at night, vivid colors"
NEGATIVE = "blurry, low quality, distorted, bad anatomy"
STEPS = 20
SEED = 42
W, H = 512, 512


section("1. Setup ARK + diffusers")
app = create_app()


section("2. Instancie modules ARK (real_backend=True)")
print(f"Model    : {MODEL}")
print(f"Prompt   : '{PROMPT}'")
print(f"Negative : '{NEGATIVE}'")
print(f"Steps    : {STEPS}, Seed : {SEED}, Size : {W}x{H}")

loader = DiffusionModelLoaderReal(checkpoint=MODEL, device="auto")
pos_encoder = CLIPTextEncodeReal(text=PROMPT)
neg_encoder = CLIPTextEncodeReal(text=NEGATIVE)
sampler = KSamplerReal(seed=SEED, steps=STEPS, cfg=7.5,
                              sampler="euler", scheduler="normal", denoise=1.0)
vae_decoder = VAEDecodeReal()


section("3. Forward chain (premier run télécharge ~400MB)")
t0 = time.perf_counter()

print("→ Loader...")
model, clip, vae = loader()
t1 = time.perf_counter()
print(f"  Loaded in {t1 - t0:.1f}s (device={model.device})")

print("→ CLIP encode positive...")
positive = pos_encoder(clip)
print(f"  shape={tuple(positive.shape)}")

print("→ CLIP encode negative...")
negative = neg_encoder(clip)
print(f"  shape={tuple(negative.shape)}")

print("→ Empty latent...")
latent = torch.zeros(1, 4, H // 8, W // 8)
print(f"  shape={tuple(latent.shape)}")

print(f"→ KSampler ({STEPS} steps)...")
ts = time.perf_counter()
sampled_latent = sampler(model, positive, negative, latent)
te = time.perf_counter()
print(f"  done in {te - ts:.1f}s, shape={tuple(sampled_latent.shape)}")

print("→ VAE decode...")
image = vae_decoder(vae, sampled_latent)
print(f"  shape={tuple(image.shape)}, range=[{image.min():.3f}, {image.max():.3f}]")


section("4. Save PNG")
from PIL import Image
import numpy as np

out_dir = HERE / "demo_output"
out_dir.mkdir(exist_ok=True)
img = image[0]  # (3, H, W)
arr = img.permute(1, 2, 0).cpu().numpy()
arr = arr.clip(0, 1)
arr = (arr * 255).astype("uint8")
out_path = out_dir / "t2i_real_sd.png"
Image.fromarray(arr).save(out_path)
total = time.perf_counter() - t0
print(f"✓ Image sauvée : {out_path}")
print(f"  shape : {arr.shape}")
print(f"  Total pipeline : {total:.1f}s")
