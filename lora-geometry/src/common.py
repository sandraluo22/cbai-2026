"""Shared plumbing: model loading, chat templating, residual reads, LoRA config.

Self-contained (vendored from ../trust-vector/src/common.py and
../reciprocal-signaling/recursive-ft/engine.py) so this directory does not depend
on either parent.
"""
from __future__ import annotations

import os

import numpy as np
import torch

BASE = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# One LoRA config for every adapter in the project. Two things matter here and
# both are load-bearing for the weight-space comparison:
#   * target_modules is FIXED across concepts. Comparing dW between adapters
#     that touch different modules is comparing different coordinate systems.
#   * r is FIXED. Rank changes the norm and the effective redundancy of dW, so a
#     rank sweep is a separate arm (RANK env), never mixed into one matrix.
LORA = dict(r=int(os.environ.get("RANK", 16)),
            lora_alpha=int(os.environ.get("RANK", 16)) * 2,
            lora_dropout=0.0,  # dropout adds seed noise to dW for no benefit here
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])

HERE = os.path.dirname(os.path.abspath(__file__))
# MOCK_OUT lets mock_test.py point the real stage-4/6 code at a scratch tree of
# synthetic adapters, so the analysis is exercised as written rather than as a
# reimplementation of itself.
OUT = os.environ.get("MOCK_OUT") or os.path.join(os.path.dirname(HERE), "out")


def out_path(*parts):
    p = os.path.join(OUT, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def adapter_root():
    d = os.path.join(OUT, "adapters")
    os.makedirs(d, exist_ok=True)
    return d


# Adapter identity is "<concept>__b<init-seed>_d<data-seed>". The init seed is
# split out from the data seed because the pilot showed dW is anchored to the
# random lora_A basis: cross-init cosines are pinned near zero by geometry, so
# the init block is the unit within which weight-space comparisons are defined.
def item_id(name, iseed, dseed):
    return f"{name}__b{iseed}_d{dseed}"


def parse_item(it):
    """'french__b0_d1' -> ('french', 0, 1)"""
    name, _, rest = it.partition("__b")
    b, _, d = rest.partition("_d")
    return name, int(b), int(d)


def adapter_dir(name, iseed, dseed):
    d = os.path.join(adapter_root(), item_id(name, iseed, dseed))
    os.makedirs(d, exist_ok=True)
    return d


def load_base(dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=dtype, device_map="auto")
    return model, tok


def n_layers(model):
    return len(model.model.layers)


def read_layer(model):
    """Default read/inject depth: 60% of the way up, the usual sweet spot for
    style directions. Swept in build_vecs.py -- never trust one depth (see the
    trust-vector layer-sign result: an effect that flips sign across depth and
    cancels when summed is not a mechanism)."""
    return int(os.environ.get("LAYER", int(0.6 * n_layers(model))))


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------
def chat(tok, system, user, prefill=""):
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": user}]
    try:
        t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
    except TypeError:
        t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return t + prefill


# ---------------------------------------------------------------------------
# residual reads
# ---------------------------------------------------------------------------
@torch.no_grad()
def resid(model, tok, text, layers, span=None):
    """{layer: d-vector} residual stream, mean-pooled over `span` token indices.

    Takes a LIST of layers and does ONE forward. The earlier single-layer version
    made the caller loop, which cost 5 forward passes per text per read position
    -- 10x the necessary compute in the two read stages.

    hidden_states[l] is the INPUT to block l (l=0 is the embedding output), so
    layer l here is the stream *before* block l -- the same integer the Inject
    hook writes into when it hooks block l-1's output. Read and write at the
    same number.

    span=None -> the last token.
    """
    single = isinstance(layers, int)
    ls = [layers] if single else list(layers)
    enc = tok(text, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items() if k != "offset_mapping"}
    hs = model(**enc, output_hidden_states=True).hidden_states
    out = {}
    for l in ls:
        h = hs[l][0]
        out[l] = (h[-1] if span is None
                  else h[torch.tensor(span, device=h.device)].mean(0)).float().cpu().numpy()
    return out[ls[0]] if single else out


def response_span(tok, prompt_text, full_text):
    """Token indices of the RESPONSE portion of `full_text`.

    The two read positions this project treats as a factor from day one:
      RESPONSE  mean over the response tokens  -- "the behaviour as it happens"
      LAST      the final prompt token          -- "the intention to behave"
    trust-vector spent a week discovering that six tokens of read position
    selects a near-orthogonal, equally-reliable direction. So both are always
    built, and any claim that holds at only one is reported as position-bound.
    """
    n_prompt = len(tok(prompt_text)["input_ids"])
    n_full = len(tok(full_text)["input_ids"])
    return list(range(n_prompt, n_full)) if n_full > n_prompt else [n_full - 1]


class Inject:
    """Add `vec` to the residual stream at block `layer`'s output.

    pos=None -> every position. Prefill-only by construction: the hook indexes
    `pos` against the current sequence, so with a KV cache during generation the
    positions are only valid on the prefill pass. Generation therefore injects at
    all positions or not at all (this exact bug threw a CUDA assert in
    trust-vector).
    """

    def __init__(self, model, layer, vec, pos=None):
        self.model, self.blk = model, max(0, layer - 1)
        self.vec = vec if torch.is_tensor(vec) else torch.tensor(np.asarray(vec))
        self.pos = pos

    def __enter__(self):
        def f(mod, inp, out):
            tup = isinstance(out, tuple)
            h = (out[0] if tup else out).clone()
            v = self.vec.to(h.dtype).to(h.device)
            if self.pos is None:
                h = h + v
            elif len(self.pos):
                h[0, self.pos] = h[0, self.pos] + v
            return ((h,) + tuple(out[1:])) if tup else h
        self.hk = self.model.model.layers[self.blk].register_forward_hook(f)
        return self

    def __exit__(self, *a):
        self.hk.remove()
        return False


# ---------------------------------------------------------------------------
# vector helpers
# ---------------------------------------------------------------------------
def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cos(a, b):
    return float(np.dot(unit(np.asarray(a, dtype=np.float64)),
                        unit(np.asarray(b, dtype=np.float64))))


def rand_like(v, seed=0):
    """Matched-norm random direction -- the floor every steering number is read
    against."""
    r = np.random.default_rng(seed).normal(size=np.shape(v))
    return unit(r) * np.linalg.norm(v)
