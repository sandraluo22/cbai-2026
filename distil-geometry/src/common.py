"""Shared plumbing. Adapted from ../lora-geometry/src/common.py."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("MOCK_OUT") or os.path.join(os.path.dirname(HERE), "out")
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# arXiv:2606.00995 recipe: r=8, alpha=32, all linear modules, AdamW, lr 1e-4, 2 epochs.
LORA = dict(r=int(os.environ.get("RANK", 8)),
            lora_alpha=int(os.environ.get("LALPHA", 32)),
            lora_dropout=0.0,          # dropout injects seed noise into dW, which IS the measurement
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])


def out_path(*p):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, *p)


def adapter_root():
    d = out_path("adapters")
    os.makedirs(d, exist_ok=True)
    return d


# Adapter identity "<concept>__b<init>_d<data>". The init seed is split from the
# data seed because dW is anchored to the random lora_A draw: in ../lora-geometry
# the same concept scored 0.788 across a shared init and 0.141 across different
# inits. Cross-concept comparison is only defined within a shared init block.
def item_id(name, iseed, dseed):
    return f"{name}__b{iseed}_d{dseed}"


def parse_item(it):
    name, _, rest = it.partition("__b")
    b, _, d = rest.partition("_d")
    return name, int(b), int(d)


def adapter_dir(name, iseed, dseed):
    d = os.path.join(adapter_root(), item_id(name, iseed, dseed))
    os.makedirs(d, exist_ok=True)
    return d


def load_base():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16,
                                             device_map="auto").eval()
    return m, tok


def chat(tok, system, user, prefill=""):
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": user}]
    try:
        t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
    except TypeError:
        t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return t + prefill


def n_layers(model):
    return model.config.num_hidden_layers


def layer_grid(model):
    env = os.environ.get("LAYERS", "")
    n = n_layers(model)
    return [int(x) for x in env.split(",")] if env else \
        sorted({int(round(f * n)) for f in (0.4, 0.5, 0.6, 0.7)})


@torch.no_grad()
def resid_last(model, tok, texts, layers, bs=8):
    """{layer: (N,d)} residual at the FINAL token. Left padding, so index -1 is
    the real last token for every row. One forward per batch, all layers."""
    acc = {l: [] for l in layers}
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states
        for l in layers:
            acc[l].append(hs[l][:, -1].float().cpu().numpy())
    return {l: np.concatenate(v) for l, v in acc.items()}


class Inject:
    """Add `vec` at every position, at the output of block layer-1, so it lands in
    the stream that resid_last(..., [layer]) reads."""

    def __init__(self, model, layer, vec):
        self.blk = model.model.layers[max(0, layer - 1)]
        self.vec = torch.as_tensor(vec)

    def __enter__(self):
        def f(mod, inp, out):
            tup = isinstance(out, tuple)
            h = (out[0] if tup else out)
            h = h + self.vec.to(h.dtype).to(h.device)
            return ((h,) + tuple(out[1:])) if tup else h
        self.hk = self.blk.register_forward_hook(f)
        return self

    def __exit__(self, *a):
        self.hk.remove()
        return False


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
