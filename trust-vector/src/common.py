"""Shared plumbing: model loading, chat templating, residual reads, injection hooks.

Kept deliberately thin so `mock_test.py` can swap in a fake model/tokenizer with the
same interface (HF-style `model.model.layers`, `output_hidden_states`, offset mapping).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


def load(tag=None):
    """Model loader, vendored in model_spec.py so this directory is self-contained."""
    from model_spec import load as _load
    return _load(tag or os.environ.get("MODEL", "Qwen32"))


# ---------------------------------------------------------------------------
# prompt assembly
# ---------------------------------------------------------------------------
def chat(tok, system, user, prefill=""):
    """Chat-templated prompt + optional prefill (the assistant's forced opening)."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    try:
        t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
    except TypeError:
        t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return t + prefill


def raw(tok, system, user, prefill=""):
    """No chat template — plain continuation. FMT=raw uses this."""
    return (user + prefill)


def fmt_fn():
    return raw if os.environ.get("FMT", "chat") == "raw" else chat


# ---------------------------------------------------------------------------
# character-span -> token-index resolution
# ---------------------------------------------------------------------------
def spans_of(text, needle, which="all"):
    """Character spans of every literal occurrence of `needle`.

    which: "all" | "first" | "last" | int n (0-based occurrence index).
    """
    out, j = [], 0
    while True:
        j = text.find(needle, j)
        if j < 0:
            break
        out.append((j, j + len(needle)))
        j += 1
    if not out:
        return []
    if which == "all":
        return out
    if which == "first":
        return out[:1]
    if which == "last":
        return out[-1:]
    return [out[int(which)]] if int(which) < len(out) else []


def tok_idx(tok, text, spans):
    """Token indices whose character extent overlaps any of `spans`."""
    if not spans:
        return []
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
    offs = enc["offset_mapping"][0].tolist()
    return [i for i, (a, b) in enumerate(offs)
            if b > a and any(a < c1 and b > c0 for (c0, c1) in spans)]


def n_tokens(tok, text):
    return len(tok(text, add_special_tokens=False)["input_ids"])


def first_id(tok, word):
    """First token id of `word` — the unit every bounded logit read is taken over."""
    return tok(word, add_special_tokens=False)["input_ids"][0]


# ---------------------------------------------------------------------------
# reads
# ---------------------------------------------------------------------------
@torch.no_grad()
def resid(model, tok, text, layers, positions=None):
    """{layer: d-vector} from hidden_states.

    positions=None -> last token. Otherwise a list of token indices, mean-pooled.
    hidden_states[l] is the INPUT to block l (l=0 is the embedding output), so layer
    index l here is the residual stream *before* block l — the same convention the
    injection hook writes into when it hooks block l-1's output. `build_vectors.py`
    reads and `steer_qsg.py` writes at the same integer; see README "layer indexing".
    """
    enc = tok(text, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items() if k != "offset_mapping"}
    o = model(**enc, output_hidden_states=True)
    out = {}
    for l in layers:
        h = o.hidden_states[l][0]
        out[l] = (h[-1] if positions is None
                  else h[torch.tensor(positions, device=h.device)].mean(0)).float().cpu().numpy()
    return out


@torch.no_grad()
def margin(model, tok, text, pos_word, neg_word):
    """logit(pos) - logit(neg) at the final position, over first tokens."""
    enc = tok(text, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items() if k != "offset_mapping"}
    lg = model(**enc).logits[0, -1]
    return float(lg[first_id(tok, pos_word)] - lg[first_id(tok, neg_word)])


@torch.no_grad()
def p_first(model, tok, text, words):
    """softmax over the first tokens of `words` — bounded read, returns p(words[0])."""
    enc = tok(text, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items() if k != "offset_mapping"}
    lg = model(**enc).logits[0, -1]
    ids = torch.tensor([first_id(tok, w) for w in words], device=lg.device)
    return float(torch.softmax(lg[ids].float(), 0)[0])


# ---------------------------------------------------------------------------
# injection
# ---------------------------------------------------------------------------
class Inject:
    """Add `vec` to the residual stream at block `layer`'s output, at `pos`.

    pos=None -> every position. Hooking block (layer-1)'s output means the addition
    lands in the same stream that resid(..., layers=[layer]) reads.

    Verification note (audit 2026-08-12): in transformers 5.x, output_hidden_states
    records hidden_states[layer] BEFORE forward-hook modifications propagate, so an
    injection is invisible at hidden_states[layer] but present at [layer+1] onward and
    in the logits. Reads in this repo use unhooked forwards, so derivation is
    unaffected; just do not use hidden_states[layer] to verify an injection.
    """

    def __init__(self, model, layer, vec, pos):
        self.model = model
        self.blk = max(0, layer - 1)
        self.vec = vec if torch.is_tensor(vec) else torch.tensor(vec)
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


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def rand_like(v, seed=0):
    """Matched-norm random direction — the floor every steering arm is read against."""
    r = np.random.default_rng(seed).normal(size=np.shape(v))
    return unit(r) * np.linalg.norm(v)
