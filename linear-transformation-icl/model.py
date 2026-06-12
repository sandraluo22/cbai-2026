"""Models behind one interface: a tiny train-from-scratch GPT, and a Llama loader.

The tiny model is the workhorse for Balls & Urns: it is the only setting where the
ground-truth belief is known, the separation knob is controllable, and we have
intermediate checkpoints — and it is cheap (trains on CPU in a minute or two).

Both paths expose the same capture surface: ``blocks()`` returns the list of
residual-stream modules to hook, and the model is a plain ``nn.Module`` whose
forward returns next-token logits. Activation capture (``activations.py``) hooks
``blocks()`` either way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TinyGPTConfig:
    vocab: int = 2
    d_model: int = 64
    n_layer: int = 2
    n_head: int = 2
    mlp_ratio: int = 4
    block_size: int = 256


class Block(nn.Module):
    """Pre-norm transformer block; forward returns the residual stream after the block."""

    def __init__(self, cfg: TinyGPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.mlp_ratio * cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.mlp_ratio * cfg.d_model, cfg.d_model),
        )

    def forward(self, x, attn_mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: TinyGPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos = nn.Embedding(cfg.block_size, cfg.d_model)
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.lnf = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)

    def blocks(self):
        """Residual-stream modules to hook (one per layer)."""
        return list(self.h)

    def forward(self, idx):
        B, C = idx.shape
        pos = torch.arange(C, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        mask = torch.triu(torch.full((C, C), float("-inf"), device=idx.device), diagonal=1)
        for blk in self.h:
            x = blk(x, mask)
        return self.head(self.lnf(x))            # (B, C, vocab)


# --------------------------------------------------------------------------- #
# Training (online: a fresh batch every step)                                  #
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    steps: int = 2000
    batch: int = 128
    lr: float = 3e-3
    weight_decay: float = 0.01
    log_spaced_ckpts: bool = True
    device: str = "cpu"
    seed: int = 0
    # If set {"K":int,"diversity":float}, train on a CONTINUUM of sources drawn
    # fresh from Dirichlet each sequence (so the model learns general source
    # inference, not a fixed finite set). Otherwise train on the given SourceSet.
    online_dirichlet: dict | None = None


def _online_batch(sources, C, batch, rng: np.random.Generator):
    """Sample `batch` sequences, each from a uniformly chosen fixed source."""
    src_ids = rng.integers(sources.S, size=batch)
    seqs = np.empty((batch, C), dtype=np.int64)
    for b in range(batch):
        seqs[b] = rng.choice(sources.K, size=C, p=sources.W[src_ids[b]])
    return torch.from_numpy(seqs)


def _online_dirichlet_batch(K, diversity, C, batch, rng: np.random.Generator):
    """Each sequence: draw a fresh source w~Dirichlet(D), then sample C tokens."""
    seqs = np.empty((batch, C), dtype=np.int64)
    for b in range(batch):
        w = rng.dirichlet(diversity * np.ones(K))
        seqs[b] = rng.choice(K, size=C, p=w)
    return torch.from_numpy(seqs)


def _ckpt_steps(total: int) -> set[int]:
    s, t = {0, total - 1}, 1
    while t < total:
        s.add(t)
        t *= 2
    return s


def train_tiny(sources, model_cfg: TinyGPTConfig, train_cfg: TrainConfig):
    """Train a TinyGPT online on Balls & Urns. Returns (model, checkpoints).

    checkpoints: dict {step -> cpu state_dict} at log-spaced steps (for the
    training/developmental axis in Section 6).
    """
    torch.manual_seed(train_cfg.seed)
    rng = np.random.default_rng(train_cfg.seed + 777)   # separate stream: sequence sampling
    model = TinyGPT(model_cfg).to(train_cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    ckpt_at = _ckpt_steps(train_cfg.steps) if train_cfg.log_spaced_ckpts else set()
    checkpoints: dict[int, dict] = {}
    losses = []

    od = train_cfg.online_dirichlet
    model.train()
    for step in range(train_cfg.steps):
        if od is not None:
            idx = _online_dirichlet_batch(od["K"], od["diversity"], model_cfg.block_size,
                                          train_cfg.batch, rng).to(train_cfg.device)
        else:
            idx = _online_batch(sources, model_cfg.block_size, train_cfg.batch, rng).to(train_cfg.device)
        logits = model(idx[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, model_cfg.vocab), idx[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if step in ckpt_at:
            checkpoints[step] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.eval()
    return model, checkpoints, losses


def optimal_loss(sources) -> float:
    """Entropy-rate lower bound: expected NLL of the Bayes-optimal predictor.

    For a uniform mixture of sources, a long-context optimum approaches the mean
    per-source entropy H(w_s). Used as a training sanity reference.
    """
    H = -np.sum(sources.W * np.log(np.clip(sources.W, 1e-12, None)), axis=1)
    return float(np.mean(H))


# --------------------------------------------------------------------------- #
# Llama loader (Section 8 transfer; same capture surface) — lazy import        #
# --------------------------------------------------------------------------- #
class LlamaBackend:
    """Wraps Llama-3.1-8B for the in-context (no-training) transfer check.

    Exposes ``blocks()`` (decoder layers) so activation capture is identical.
    """

    def __init__(self, name="meta-llama/Llama-3.1-8B", dtype="bfloat16", device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        td = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype, torch.float32)
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=td)
        if device != "cuda":
            self.model.to(device)
        self.model.eval()
        self.device = device

    def blocks(self):
        return list(self.model.model.layers)
