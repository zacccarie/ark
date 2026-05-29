"""Demo text2image REAL : real_backend=True avec modèle SD léger.

Modèle : OFA-Sys/small-stable-diffusion-v0 (~400MB) ou stabilityai/sd-turbo (1 step).
Premier run télécharge depuis HuggingFace cache (~/.cache/huggingface/).
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
from ark.nodes.registry import default_registry
from ark.ui.node_editor.scene import NodeScene
from ark.ui.node_editor.connection_item import ConnectionItem
from ark.core.graph_compiler import compile_scene


def section(t):
    print(f"\n{'=' * 60}\n  {t}\n{'=' * 60}")


section("1. Setup ARK + diffusers REAL backend")
app = create_app()
registry = default_registry()
scene = NodeScene(registry=registry)

# Modèle SD compact ~400MB
MODEL = "OFA-Sys/small-stable-diffusion-v0"
STEPS = 25
PROMPT = "a cute cyberpunk cat sitting on a neon sign, vivid colors, 4k"
NEGATIVE = "blurry, low quality, distorted"


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


section("2. Build scene t2i avec real_backend=True")

loader = add("diffusion_model_loader", 50, 100,
                checkpoint=MODEL, device="auto", real_backend=True)
pos = add("clip_text_encode", 300, 50, text=PROMPT, real_backend=True)
neg = add("clip_text_encode", 300, 200, text=NEGATIVE, real_backend=True)
latent = add("empty_latent_image", 300, 350,
                width=512, height=512, batch_size=1)
sampler = add("k_sampler", 600, 200,
                 seed=42, steps=STEPS, cfg=7.5,
                 sampler="euler", scheduler="normal", denoise=1.0,
                 real_backend=True)
vae_dec = add("vae_decode", 900, 200, real_backend=True)

conn(loader, sampler, src=0, dst=0)
conn(loader, pos, src=1, dst=0)
conn(loader, neg, src=1, dst=0)
conn(loader, vae_dec, src=2, dst=0)
conn(pos, sampler, src=0, dst=1)
conn(neg, sampler, src=0, dst=2)
conn(latent, sampler, src=0, dst=3)
conn(sampler, vae_dec, src=0, dst=1)

print(f"Scene : {len(list(scene.nodes()))} nodes")
print(f"Model : {MODEL}")
print(f"Prompt: '{PROMPT}'")
print(f"Steps : {STEPS}")


section("3. Compile + forward (premier run télécharge ~400MB)")
print("Download progress visible si premier run...")
t0 = time.perf_counter()
cg = compile_scene(scene)
model = cg.model
t1 = time.perf_counter()
print(f"Compile : {(t1 - t0):.1f}s")

print("\nForward...")
model.eval()
t2 = time.perf_counter()
try:
    with torch.no_grad():
        dummy = torch.zeros(1, 4, 64, 64)
        out = model(dummy)
        if isinstance(out, tuple):
            out = out[0]
    t3 = time.perf_counter()
    print(f"Forward : {(t3 - t2):.1f}s")
    print(f"Output  : shape={tuple(out.shape)}, range=[{out.min():.3f}, {out.max():.3f}]")
except Exception as e:
    print(f"Erreur forward : {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


section("4. Sauvegarde image")
from PIL import Image
import numpy as np

out_dir = HERE / "demo_output"
out_dir.mkdir(exist_ok=True)
img = out[0]  # batch 0
if img.dim() == 3:
    img = img.permute(1, 2, 0)
arr = img.cpu().numpy()
arr = arr.clip(0, 1)  # SD output déjà [0, 1]
arr = (arr * 255).astype("uint8")
out_path = out_dir / "t2i_real_sd.png"
Image.fromarray(arr).save(out_path)
print(f"✓ Image sauvée : {out_path}")
print(f"  shape : {arr.shape}, prompt : '{PROMPT}'")
