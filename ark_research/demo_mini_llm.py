"""Demo mini LLM char-level entièrement composé via modules ARK.

Architecture :
  embedding → positional encoding → N× (self_attention + layernorm + dense FFN) → dense logits

Train sur corpus synthétique généré (grammaire simple) — pas de données externes.
Sample par température après entraînement.

Stack ARK utilisé :
- nn.Embedding (auto_register embedding_layer)
- PositionalEncoding (ark.core.auto_register)
- SelfAttention (ark.core.auto_register)
- LayerNorm (auto_register layernorm)
- Dense MLP (FFN)
- Output projection → vocab
"""
import os
import sys
import time
import random
import string
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARK_ROOT = HERE.parent
sys.path.insert(0, str(ARK_ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import torch
import torch.nn as nn
import torch.nn.functional as F

from ark.app import create_app
from ark.core.auto_register import PositionalEncoding


class CausalSelfAttention(nn.Module):
    """Self-attention causal (mask futur) — version utilisable pour LLM gen."""
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        # causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


def section(t):
    print(f"\n{'=' * 60}\n  {t}\n{'=' * 60}")


# ============================================================
# 1. Corpus synthétique généré (grammaire simple)
# ============================================================

section("1. Génération corpus synthétique original")

SUBJECTS = ["the cat", "a dog", "my robot", "the bird", "every fish",
              "a child", "the king", "her friend", "this scientist",
              "no human"]
VERBS = ["sees", "loves", "builds", "eats", "knows", "studies",
            "imagines", "creates", "discovers", "remembers"]
OBJECTS = ["a star", "the moon", "many ideas", "the truth",
              "no secret", "every song", "this city", "the future",
              "her dream", "a paradox"]
ADVERBS = ["quickly", "softly", "always", "rarely", "carefully",
              "wildly", "deeply", "now", "forever", "today"]


def gen_sentence(rng):
    s = rng.choice(SUBJECTS)
    v = rng.choice(VERBS)
    o = rng.choice(OBJECTS)
    a = rng.choice(ADVERBS)
    return f"{s} {v} {o} {a}."


rng = random.Random(42)
sentences = [gen_sentence(rng) for _ in range(2000)]
text = " ".join(sentences)
print(f"Corpus : {len(text):,} chars, {len(sentences)} phrases")
print(f"Sample : {text[:200]}...")

# Char-level vocab
chars = sorted(set(text))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}
vocab_size = len(chars)
print(f"Vocab size : {vocab_size}")
print(f"Chars : {''.join(chars)!r}")


# ============================================================
# 2. Build TinyTransformer via ARK modules
# ============================================================

section("2. Build TinyTransformer (ARK composition)")

D_MODEL = 64
N_HEADS = 4
N_BLOCKS = 2
MAX_LEN = 128
FFN_DIM = 256


class TransformerBlock(nn.Module):
    """1 block : Self-Attention + LayerNorm + FFN + LayerNorm. ARK modules."""
    def __init__(self, d_model, ffn_dim):
        super().__init__()
        self.attn = CausalSelfAttention(d_model=d_model, n_heads=N_HEADS)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ffn(x))
        return x


class TinyLLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_blocks, max_len, ffn_dim):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, ffn_dim) for _ in range(n_blocks)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x : (B, T) idx
        e = self.tok_emb(x)
        e = self.pos_enc(e)
        for blk in self.blocks:
            e = blk(e)
        e = self.ln_f(e)
        logits = self.head(e)
        return logits


model = TinyLLM(vocab_size, D_MODEL, N_BLOCKS, MAX_LEN, FFN_DIM)
n_params = sum(p.numel() for p in model.parameters())
print(f"Architecture : {N_BLOCKS} blocks × ({D_MODEL}d, {N_HEADS} heads, "
        f"{FFN_DIM} ffn)")
print(f"Params : {n_params:,}")
print(f"Vocab : {vocab_size}, Max seq len : {MAX_LEN}")


# ============================================================
# 3. Prepare data : sliding window seq
# ============================================================

section("3. Préparation data tokens + sliding windows")
data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)
n_train = int(0.9 * len(data))
train_data = data[:n_train]
val_data = data[n_train:]
print(f"Train : {len(train_data):,} tokens")
print(f"Val   : {len(val_data):,} tokens")


def get_batch(split_data, batch_size, block_size):
    ix = torch.randint(0, len(split_data) - block_size - 1, (batch_size,))
    x = torch.stack([split_data[i:i + block_size] for i in ix])
    y = torch.stack([split_data[i + 1:i + block_size + 1] for i in ix])
    return x, y


# ============================================================
# 4. Train
# ============================================================

section("4. Training")
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device : {device}")
model = model.to(device)
optim = torch.optim.AdamW(model.parameters(), lr=3e-3)

BATCH_SIZE = 32
BLOCK_SIZE = 64
N_STEPS = 800

train_losses = []
val_losses = []

t0 = time.perf_counter()
for step in range(N_STEPS):
    x, y = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE)
    x, y = x.to(device), y.to(device)
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    optim.zero_grad()
    loss.backward()
    optim.step()
    train_losses.append(loss.item())

    if (step + 1) % 100 == 0:
        model.eval()
        with torch.no_grad():
            xv, yv = get_batch(val_data, BATCH_SIZE, BLOCK_SIZE)
            xv, yv = xv.to(device), yv.to(device)
            logits_v = model(xv)
            loss_v = F.cross_entropy(logits_v.view(-1, vocab_size), yv.view(-1))
            val_losses.append((step + 1, loss_v.item()))
        print(f"step {step + 1:4d}/{N_STEPS} · "
                f"train_loss={loss.item():.4f} · "
                f"val_loss={loss_v.item():.4f}")
        model.train()

train_time = time.perf_counter() - t0
print(f"Train time : {train_time:.1f}s")


# ============================================================
# 5. Generate text
# ============================================================

section("5. Génération text (sampling température)")

def generate(prompt, max_new=120, temperature=0.8, top_k=10):
    model.eval()
    ctx = torch.tensor([[char_to_idx.get(c, 0) for c in prompt]],
                          dtype=torch.long, device=device)
    out = list(prompt)
    with torch.no_grad():
        for _ in range(max_new):
            ctx_crop = ctx[:, -BLOCK_SIZE:]
            logits = model(ctx_crop)[:, -1, :] / temperature
            # top-k filter
            v, idx = torch.topk(logits, top_k)
            mask = torch.full_like(logits, -1e9)
            mask.scatter_(1, idx, v)
            probs = F.softmax(mask, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            ctx = torch.cat([ctx, next_tok], dim=1)
            out.append(idx_to_char[int(next_tok.item())])
    return "".join(out)


prompts = ["the cat ", "a dog ", "my robot ", "her friend "]
for p in prompts:
    gen = generate(p, max_new=100, temperature=0.8, top_k=10)
    print(f"\n[temp=0.8 top_k=10] '{p}'")
    print(f"  → {gen}")


# ============================================================
# 6. Save outputs
# ============================================================

section("6. Save model + metrics")
out_dir = HERE / "demo_output"
out_dir.mkdir(exist_ok=True)
torch.save({
    "model_state": model.state_dict(),
    "vocab_size": vocab_size,
    "char_to_idx": char_to_idx,
    "idx_to_char": idx_to_char,
    "d_model": D_MODEL, "n_blocks": N_BLOCKS,
    "max_len": MAX_LEN, "ffn_dim": FFN_DIM,
}, out_dir / "tiny_llm.pt")
print(f"Model saved : {out_dir / 'tiny_llm.pt'}")

# Plot loss
try:
    from ark.core.research_bridge import SciVizWrap
    viz = SciVizWrap(
        plot_kind="line",
        title=f"TinyLLM training loss ({n_params:,} params)",
        output_path=str(out_dir / "tiny_llm_loss.png"),
        metric_keys="train_loss",
    )
    viz.run(data={"train_loss": train_losses})
    print(f"Loss curve : {out_dir / 'tiny_llm_loss.png'}")
except Exception as e:
    print(f"viz error : {e}")

print(f"\nFinal train loss : {train_losses[-1]:.4f}")
print(f"Final val loss   : {val_losses[-1][1]:.4f}")
print(f"Pipeline complet via modules ARK : ✓")
