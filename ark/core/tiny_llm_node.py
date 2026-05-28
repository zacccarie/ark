"""Phase 54 — TinyLLM nodes ARK : LLM char-level réutilisable.

3 nodes :
- tiny_llm_corpus    : génère corpus synthétique (subject/verb/object/adverb)
- tiny_llm_model     : instancie TinyLLM module (+ train opt sur corpus)
- tiny_llm_generate  : sampling text depuis prompt + checkpoint

Architecture : embedding → PositionalEncoding → N× (CausalSelfAttention +
LayerNorm + FFN) → LayerNorm → Linear head.

State_dict + vocab + config sauvés dans ~/.ark/tiny_llm/<name>.pt
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


MODELS_DIR = Path.home() / ".ark" / "tiny_llm"


# ============================================================
# Corpus synthétique (grammaire originale ARK)
# ============================================================

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


def generate_synthetic_corpus(n_sentences: int = 2000, seed: int = 42) -> str:
    rng = random.Random(seed)
    out = []
    for _ in range(n_sentences):
        s = rng.choice(SUBJECTS)
        v = rng.choice(VERBS)
        o = rng.choice(OBJECTS)
        a = rng.choice(ADVERBS)
        out.append(f"{s} {v} {o} {a}.")
    return " ".join(out)


# ============================================================
# Modules
# ============================================================

class CausalSelfAttention(nn.Module):
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
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class PosEnc(nn.Module):
    def __init__(self, d_model, max_len):
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


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
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
    def __init__(self, vocab_size: int, d_model: int = 64, n_blocks: int = 2,
                 n_heads: int = 4, ffn_dim: int = 256, max_len: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PosEnc(d_model, max_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_dim) for _ in range(n_blocks)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        e = self.tok_emb(x)
        e = self.pos_enc(e)
        for blk in self.blocks:
            e = blk(e)
        e = self.ln_f(e)
        return self.head(e)


# ============================================================
# Wrappers ARK (compat handlers)
# ============================================================

class TinyLLMCorpusNode(nn.Module):
    """Génère corpus synthétique. Pass-through tensor (B, 1) avec chars indices."""

    def __init__(self, n_sentences: int = 2000, seed: int = 42):
        super().__init__()
        self.n_sentences = n_sentences
        self.seed = seed
        self.text = ""
        self.vocab = []
        self.char_to_idx = {}
        self.idx_to_char = {}

    def build(self):
        self.text = generate_synthetic_corpus(self.n_sentences, self.seed)
        self.vocab = sorted(set(self.text))
        self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx_to_char = {i: c for i, c in enumerate(self.vocab)}
        return self.text

    def forward(self, x=None):
        if not self.text:
            self.build()
        idxs = torch.tensor([self.char_to_idx[c] for c in self.text],
                                dtype=torch.long)
        return idxs.unsqueeze(0)  # (1, N)


class TinyLLMModelNode(nn.Module):
    """Build + optionnellement train TinyLLM sur corpus reçu."""

    def __init__(self, d_model: int = 64, n_blocks: int = 2, n_heads: int = 4,
                 ffn_dim: int = 256, max_len: int = 128,
                 train_steps: int = 0, lr: float = 3e-3,
                 batch_size: int = 32, block_size: int = 64,
                 vocab_size: int = 26, save_name: str = "default"):
        super().__init__()
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.ffn_dim = ffn_dim
        self.max_len = max_len
        self.train_steps = train_steps
        self.lr = lr
        self.batch_size = batch_size
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.save_name = save_name
        self.model: TinyLLM | None = None
        self.last_loss: float | None = None
        self.last_corpus_tokens: torch.Tensor | None = None

    def _build_model(self, vocab_size):
        self.model = TinyLLM(
            vocab_size=vocab_size, d_model=self.d_model,
            n_blocks=self.n_blocks, n_heads=self.n_heads,
            ffn_dim=self.ffn_dim, max_len=self.max_len)
        return self.model

    def _train(self, data: torch.Tensor):
        device = ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = self.model.to(device)
        optim = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        for step in range(self.train_steps):
            ix = torch.randint(0, len(data) - self.block_size - 1,
                                  (self.batch_size,))
            x = torch.stack([data[i:i + self.block_size] for i in ix]).to(device)
            y = torch.stack([data[i + 1:i + self.block_size + 1] for i in ix]).to(device)
            logits = self.model(x)
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), y.view(-1))
            optim.zero_grad()
            loss.backward()
            optim.step()
            self.last_loss = float(loss.item())

    def forward(self, x):
        # x : (1, N) char indices ou (N,)
        if x.dim() == 2:
            tokens = x[0]
        else:
            tokens = x
        self.last_corpus_tokens = tokens
        vocab_size = int(tokens.max().item() + 1)
        self.vocab_size = max(vocab_size, self.vocab_size)
        if self.model is None:
            self._build_model(self.vocab_size)
        if self.train_steps > 0:
            self._train(tokens)
        # Save checkpoint
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "config": dict(
                d_model=self.d_model, n_blocks=self.n_blocks,
                n_heads=self.n_heads, ffn_dim=self.ffn_dim,
                max_len=self.max_len, vocab_size=self.vocab_size),
        }, MODELS_DIR / f"{self.save_name}.pt")
        return tokens  # pass-through pour chaîner


class TinyLLMGenerateNode(nn.Module):
    """Charge checkpoint + génère text depuis prompt."""

    def __init__(self, load_name: str = "default",
                 prompt: str = "the cat ",
                 max_new: int = 100, temperature: float = 0.8,
                 top_k: int = 10, corpus_seed: int = 42,
                 n_sentences: int = 2000):
        super().__init__()
        self.load_name = load_name
        self.prompt = prompt
        self.max_new = max_new
        self.temperature = temperature
        self.top_k = top_k
        self.corpus_seed = corpus_seed
        self.n_sentences = n_sentences
        self.last_generated: str = ""

    def _rebuild_vocab(self):
        text = generate_synthetic_corpus(self.n_sentences, self.corpus_seed)
        vocab = sorted(set(text))
        return {c: i for i, c in enumerate(vocab)}, \
                  {i: c for i, c in enumerate(vocab)}

    def forward(self, x=None):
        ckpt_path = MODELS_DIR / f"{self.load_name}.pt"
        if not ckpt_path.exists():
            self.last_generated = f"[checkpoint absent : {ckpt_path}]"
            return torch.zeros(1)
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt["config"]
        model = TinyLLM(**cfg)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        char_to_idx, idx_to_char = self._rebuild_vocab()
        ctx = torch.tensor([[char_to_idx.get(c, 0) for c in self.prompt]],
                              dtype=torch.long)
        out_chars = list(self.prompt)
        with torch.no_grad():
            block_size = min(64, cfg["max_len"])
            for _ in range(self.max_new):
                ctx_crop = ctx[:, -block_size:]
                logits = model(ctx_crop)[:, -1, :] / self.temperature
                v, idx = torch.topk(logits, self.top_k)
                mask = torch.full_like(logits, -1e9)
                mask.scatter_(1, idx, v)
                probs = torch.softmax(mask, dim=-1)
                next_tok = torch.multinomial(probs, 1)
                ctx = torch.cat([ctx, next_tok], dim=1)
                out_chars.append(idx_to_char.get(int(next_tok.item()), "?"))
        self.last_generated = "".join(out_chars)
        return torch.tensor([len(self.last_generated)], dtype=torch.float32)


# ============================================================
# Handlers + Specs
# ============================================================

def h_tiny_llm_corpus(node, in_shapes):
    m = TinyLLMCorpusNode(
        n_sentences=int(node.params.get("n_sentences", 2000)),
        seed=int(node.params.get("seed", 42)),
    )
    m._ark_node_id = getattr(node, "id", "corpus")
    return m, (1,)


def h_tiny_llm_model(node, in_shapes):
    m = TinyLLMModelNode(
        d_model=int(node.params.get("d_model", 64)),
        n_blocks=int(node.params.get("n_blocks", 2)),
        n_heads=int(node.params.get("n_heads", 4)),
        ffn_dim=int(node.params.get("ffn_dim", 256)),
        max_len=int(node.params.get("max_len", 128)),
        train_steps=int(node.params.get("train_steps", 0)),
        lr=float(node.params.get("lr", 3e-3)),
        batch_size=int(node.params.get("batch_size", 32)),
        block_size=int(node.params.get("block_size", 64)),
        save_name=node.params.get("save_name", "default"),
    )
    m._ark_node_id = getattr(node, "id", "tinyllm")
    return m, (1,)


def h_tiny_llm_generate(node, in_shapes):
    m = TinyLLMGenerateNode(
        load_name=node.params.get("load_name", "default"),
        prompt=node.params.get("prompt", "the cat "),
        max_new=int(node.params.get("max_new", 100)),
        temperature=float(node.params.get("temperature", 0.8)),
        top_k=int(node.params.get("top_k", 10)),
        corpus_seed=int(node.params.get("corpus_seed", 42)),
        n_sentences=int(node.params.get("n_sentences", 2000)),
    )
    m._ark_node_id = getattr(node, "id", "tinyllm_gen")
    return m, (1,)


def register_tiny_llm_handlers(handlers_dict: dict) -> bool:
    handlers_dict["tiny_llm_corpus"] = h_tiny_llm_corpus
    handlers_dict["tiny_llm_model"] = h_tiny_llm_model
    handlers_dict["tiny_llm_generate"] = h_tiny_llm_generate
    return True


def register_tiny_llm_specs(registry) -> bool:
    from ark.nodes.registry import NodeSpec, ParamSpec, SocketSpec

    CAT = "Tiny LLM"

    registry.register(NodeSpec(
        name="tiny_llm_corpus", title="TinyLLM Corpus (synthetic)",
        category=CAT,
        outputs=[SocketSpec("tokens", "tensor")],
        params=[
            ParamSpec("n_sentences", "int", 2000, min=10, max=100000),
            ParamSpec("seed", "int", 42),
        ],
        description="Génère corpus synthétique char-level via grammaire originale "
                       "(10 subjects × 10 verbs × 10 objects × 10 adverbs). 26 chars vocab.",
        role="Synthetic corpus generator for LLM training.",
        function="random.choice(SUBJECTS) + VERBS + OBJECTS + ADVERBS + '.'. "
                    "Tokenize char-level → tensor (1, N) indices.",
        has_thumbnail=True,
    ))

    registry.register(NodeSpec(
        name="tiny_llm_model", title="TinyLLM Model (build + train)",
        category=CAT,
        inputs=[SocketSpec("corpus", "tensor")],
        outputs=[SocketSpec("trained_tokens", "tensor")],
        params=[
            ParamSpec("d_model", "int", 64, min=16, max=512),
            ParamSpec("n_blocks", "int", 2, min=1, max=12),
            ParamSpec("n_heads", "int", 4, min=1, max=16),
            ParamSpec("ffn_dim", "int", 256, min=32, max=2048),
            ParamSpec("max_len", "int", 128, min=16, max=2048),
            ParamSpec("train_steps", "int", 800, min=0, max=100000),
            ParamSpec("lr", "float", 3e-3, min=1e-6, max=1.0),
            ParamSpec("batch_size", "int", 32, min=1, max=1024),
            ParamSpec("block_size", "int", 64, min=8, max=1024),
            ParamSpec("save_name", "str", "default"),
        ],
        description="TinyLLM Transformer char-level. Build architecture + train "
                       "optionnel sur corpus reçu. Save ~/.ark/tiny_llm/<save_name>.pt.",
        role="Mini Transformer char-level builder + trainer.",
        function="N× (CausalSelfAttention + LayerNorm + FFN). AdamW lr=3e-3. "
                    "train_steps=0 = skip training, juste build architecture.",
        has_thumbnail=True,
    ))

    registry.register(NodeSpec(
        name="tiny_llm_generate", title="TinyLLM Generate (sample)",
        category=CAT,
        outputs=[SocketSpec("output_len", "tensor")],
        params=[
            ParamSpec("load_name", "str", "default"),
            ParamSpec("prompt", "multiline", "the cat "),
            ParamSpec("max_new", "int", 100, min=1, max=2000),
            ParamSpec("temperature", "float", 0.8, min=0.05, max=2.0),
            ParamSpec("top_k", "int", 10, min=1, max=100),
            ParamSpec("corpus_seed", "int", 42),
            ParamSpec("n_sentences", "int", 2000),
        ],
        description="Charge checkpoint TinyLLM + génère text par sampling "
                       "température + top_k. Output dans node.last_generated.",
        role="LLM generation sampling.",
        function="Multinomial sampling après top_k filter. Génère caractère par "
                    "caractère, append à context, crop block_size, répète.",
        has_thumbnail=True,
        param_help={
            "prompt": "Texte de départ. Char-level.",
            "temperature": "0.1 = quasi-deterministe / 1+ = créatif. 0.8 sweet spot.",
            "top_k": "Sampling sur K meilleurs candidats. Plus haut = plus diverse.",
        },
    ))
    return True
