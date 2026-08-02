"""One shared model instance plays every role (A, B, observer) -- same weights
is what later makes the same-space test a plain dot product. This module owns:

  - loading (bf16, single 24GB card is enough for an 8-9B model)
  - chat formatting (tokenizer chat template, with a plain-text fallback for
    the SMOKE stub which has none)
  - sampled generation, optionally under a steering context
  - last-token full-depth residual capture (hidden_states, all layers)
  - the Steering context manager (adds coef * v to decoder-block outputs)
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Dict, List, Optional

import numpy as np
import torch

from config import Config


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_model(cfg: Config):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if cfg.device == "cpu":
        # Small models thrash catastrophically with one intra-op thread per
        # core on many-core hosts (96-core box: minutes per generation vs
        # 0.7s at 4 threads). Cap threads; GPU runs are left untouched.
        torch.set_num_threads(min(8, os.cpu_count() or 8))
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, dtype=getattr(torch, cfg.dtype)
    )
    model.to(cfg.device)
    model.eval()
    return model, tok


def decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers              # Llama, Gemma2
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h             # GPT-2 stub (SMOKE)
    raise AttributeError("could not locate decoder blocks on this model")


def n_hidden_states(model) -> int:
    """Number of entries in output_hidden_states (= n_blocks + 1; index 0 is
    the embedding output, index k is the output of block k-1)."""
    return len(decoder_blocks(model)) + 1


# ---------------------------------------------------------------------------
# Chat formatting
# ---------------------------------------------------------------------------
def render_chat(tok, messages: List[dict], add_generation_prompt: bool) -> str:
    """messages: [{"role": "system"|"user"|"assistant", "content": str}, ...].
    Uses the tokenizer's chat template when present; otherwise a plain-text
    fallback (SMOKE stub)."""
    if getattr(tok, "chat_template", None):
        try:
            return tok.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            # e.g. Gemma-2-it rejects the "system" role: fold the system
            # prompt into the first user turn and retry.
            if messages and messages[0]["role"] == "system":
                merged = [dict(m) for m in messages[1:]]
                assert merged and merged[0]["role"] == "user"
                merged[0]["content"] = (messages[0]["content"] + "\n\n"
                                        + merged[0]["content"])
                return tok.apply_chat_template(
                    merged, tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            raise
    parts = [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
    text = "\n".join(parts)
    if add_generation_prompt:
        text += "\nAssistant:"
    return text


# ---------------------------------------------------------------------------
# Steering
# ---------------------------------------------------------------------------
class Steering:
    """Context manager: while active, adds  coef * vecs[l+1]  to the output
    hidden state of decoder block l for each l in `layers`.

    `vecs` is indexed like output_hidden_states ([n_blocks+1, d]; row l+1
    corresponds to the output of block l), so steering.py's extracted
    directions and probes.py's probe directions can be passed straight in.
    """

    def __init__(self, model, vecs: np.ndarray, layers, coef: float):
        self.model = model
        self.layers = list(layers)
        self.coef = float(coef)
        p = next(model.parameters())
        self.vecs = torch.as_tensor(np.asarray(vecs)).to(p.device, p.dtype)
        self.handles = []

    def __enter__(self):
        if self.coef == 0.0:
            return self
        blocks = decoder_blocks(self.model)

        def make_hook(block_idx: int):
            v = self.vecs[block_idx + 1] * self.coef

            def hook(_module, _inp, out):
                if isinstance(out, tuple):
                    return (out[0] + v,) + tuple(out[1:])
                return out + v

            return hook

        self.handles = [
            blocks[l].register_forward_hook(make_hook(l)) for l in self.layers
        ]
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        return False


# ---------------------------------------------------------------------------
# Generation and capture
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate(model, tok, messages: List[dict], cfg: Config,
             steer: Optional[Steering] = None,
             max_new_tokens: Optional[int] = None) -> str:
    """Sample one reply to `messages`. Steering (if given) is active for the
    whole generation -- prompt pass and decode steps alike."""
    text = render_chat(tok, messages, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(model.device)
    ctx = steer if steer is not None else nullcontext()
    with ctx:
        out = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_new_tokens=max_new_tokens or cfg.max_new_tokens,
            pad_token_id=tok.pad_token_id,
        )
    return tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def capture_last_token(model, tok, messages: List[dict]) -> np.ndarray:
    """One forward pass over the full rendered context (no generation prompt:
    the context already ends with the agent's own last reply). Returns the
    residual stream at the LAST token for ALL layers: fp16 [n_blocks+1, d]."""
    text = render_chat(tok, messages, add_generation_prompt=False)
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(model.device)
    out = model(input_ids=input_ids, output_hidden_states=True)
    hs = torch.stack([h[0, -1] for h in out.hidden_states])   # [L+1, d]
    return hs.to(torch.float16).cpu().numpy()


@torch.no_grad()
def capture_last_token_text(model, tok, text: str) -> np.ndarray:
    """Same as capture_last_token but for a raw (already rendered) string --
    used by steering.py's contrastive extraction."""
    enc = tok(text, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(model.device)
    out = model(input_ids=input_ids, output_hidden_states=True)
    hs = torch.stack([h[0, -1] for h in out.hidden_states])
    return hs.to(torch.float16).cpu().numpy()
