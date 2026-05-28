"""Demo text2image ARK : construit pipeline diffusion + génère image.

Stub backend (instant, pipeline déterministe).
Pour real SD : real_backend=True nécessite diffusers + transformers + 4GB SD1.5.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARK_ROOT = HERE.parent
sys.path.insert(0, str(ARK_ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import torch
from ark.app import create_app
from ark.nodes.registry import default_registry
from ark.ui.node_editor.scene import NodeScene
from ark.ui.node_editor.connection_item import ConnectionItem
from ark.core.graph_compiler import compile_scene


def section(t):
    print(f"\n{'=' * 60}\n  {t}\n{'=' * 60}")


section("1. Setup ARK")
app = create_app()
registry = default_registry()
scene = NodeScene(registry=registry)
print(f"Registry : {len(registry.all())} NodeSpec")


# Helper
def add(t, x, y, **p):
    n = scene._raw_create_node(t, x=x, y=y)
    for k, v in p.items():
        if k in n.params:
            n.params[k] = v
    return n


def conn(a, b, src=0, dst=0):
    c = ConnectionItem(start_socket=a.output_sockets[src],
                            end_socket=b.input_sockets[dst])
    a.output_sockets[src].add_connection(c)
    b.input_sockets[dst].add_connection(c)
    scene._raw_add_connection(c)
    return c


section("2. Build text2image scene")

# Pipeline diffusion ComfyUI-style :
# loader (model+clip+vae) → [model→ksampler, clip→positive/negative encode → ksampler,
#                              empty_latent → ksampler → vae_decode → save_image]
loader = add("diffusion_model_loader", 50, 100,
                checkpoint="runwayml/stable-diffusion-v1-5",
                device="auto", real_backend=False)
pos = add("clip_text_encode", 300, 50,
            text="a cyberpunk cat in neon city at night, masterpiece, 4k",
            real_backend=False)
neg = add("clip_text_encode", 300, 200,
            text="blurry, low quality", real_backend=False)
latent = add("empty_latent_image", 300, 350,
                width=512, height=512, batch_size=1)
sampler = add("k_sampler", 600, 200,
                 seed=42, steps=20, cfg=7.5,
                 sampler="euler", scheduler="normal", denoise=1.0,
                 real_backend=False)
vae_dec = add("vae_decode", 900, 200, real_backend=False)
save = add("save_image", 1150, 200,
             prefix="t2i_demo_", dir=str(HERE / "demo_output"))

# Connexions :
# loader output 0 = model → sampler input 0 (model)
# loader output 1 = clip → pos input 0 + neg input 0
# loader output 2 = vae → vae_dec input 0
# pos output → sampler input 1 (positive)
# neg output → sampler input 2 (negative)
# latent output → sampler input 3 (latent)
# sampler output → vae_dec input 1 (latent)
# vae_dec output → save_image input 0
conn(loader, sampler, src=0, dst=0)  # model
conn(loader, pos, src=1, dst=0)       # clip → pos.clip
conn(loader, neg, src=1, dst=0)       # clip → neg.clip
conn(loader, vae_dec, src=2, dst=0)   # vae → vae_dec.vae
conn(pos, sampler, src=0, dst=1)      # positive cond
conn(neg, sampler, src=0, dst=2)      # negative cond
conn(latent, sampler, src=0, dst=3)   # latent
conn(sampler, vae_dec, src=0, dst=1)  # sampled latent → vae_dec.latent
conn(vae_dec, save, src=0, dst=0)     # image → save

print(f"Scene : {len(list(scene.nodes()))} nodes, "
        f"{len(list(scene.connections()))} connexions")
for n in scene.nodes():
    print(f"  {n.spec.title}")


section("3. Compile pipeline → nn.Module")
cg = compile_scene(scene)
model = cg.model
print(f"Compile OK")
print(f"Input shape  : {cg.input_shape}")
print(f"Output shape : {cg.output_shape}")
n_params = sum(p.numel() for p in model.parameters())
print(f"Params (stubs léger) : {n_params:,}")


section("4. Forward pipeline (stub backend → noise déterministe)")
import time

model.eval()
out_dir = HERE / "demo_output"
out_dir.mkdir(exist_ok=True)

with torch.no_grad():
    t0 = time.perf_counter()
    # Pas d'input réel : tout vient des sources internes (empty_latent_image)
    dummy = torch.zeros(1, 4, 64, 64)
    out = model(dummy)
    if isinstance(out, tuple):
        out = out[0]
    t1 = time.perf_counter()

print(f"Forward : {(t1 - t0) * 1000:.1f}ms")
print(f"Output  : type={type(out).__name__}, shape={tuple(out.shape) if hasattr(out, 'shape') else 'N/A'}")


section("5. Sauvegarde image")

# Convert tensor → PNG via PIL
try:
    from PIL import Image
    import numpy as np
    if hasattr(out, "dim") and out.dim() >= 3:
        # Take batch 0 si dim 4
        img = out[0] if out.dim() == 4 else out
        # Si shape (C, H, W), normalize + permute
        if img.dim() == 3:
            img = img.permute(1, 2, 0)
        arr = img.cpu().numpy()
        # Normalize [0, 1]
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        arr = (arr * 255).clip(0, 255).astype("uint8")
        # If single channel → grayscale
        if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
            arr = arr.squeeze()
            Image.fromarray(arr).save(out_dir / "t2i_demo.png")
        else:
            # Take first 3 channels si plus
            if arr.shape[2] > 3:
                arr = arr[:, :, :3]
            Image.fromarray(arr).save(out_dir / "t2i_demo.png")
        print(f"✓ Image sauvée : {out_dir / 't2i_demo.png'}")
        print(f"  shape : {arr.shape}, dtype : {arr.dtype}")
        print(f"  range : [{mn:.3f}, {mx:.3f}]")
except Exception as e:
    print(f"Erreur save : {e}")


section("6. Récap pipeline architecturale validée")
print(f"""
Pipeline text2image complet en {len(list(scene.nodes()))} nodes :
  1. Diffusion Model Loader   → (model, clip, vae)
  2. CLIP Text Encode (pos)   → conditioning positive
  3. CLIP Text Encode (neg)   → conditioning negative
  4. Empty Latent Image       → latent (1, 4, 64, 64)
  5. KSampler                 → latent débruité
  6. VAE Decode               → image (1, 3, 512, 512)
  7. Save Image               → PNG disque

Stub backend = pipeline déterministe pour tests.
Pour vraie génération SD1.5 :
    pip install diffusers transformers
    + dans Inspector : real_backend = True sur loader/sampler/clip/vae
    Premier run télécharge ~4GB depuis HuggingFace cache.
""")

print(f"Output dir : {out_dir.absolute()}")
