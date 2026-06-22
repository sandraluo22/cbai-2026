"""Model loading and hooked residual-stream activation capture.

Both models are run on the EXACT SAME word sequences (the walks). Because the
two tokenizers segment those words differently, we resolve each concept-word
occurrence to a single token position using the tokenizer's offset mapping and
the configured subword rule (CONFIRMED: "last" subword token). Activations are
then paired across models by (walk_id, step), never by token position.

Residual stream = post-block hidden state. We grab it with a forward hook on
each captured decoder block (output[0]); this equals output_hidden_states[i+1]
but is captured explicitly as requested.

Memory: models are loaded and run ONE AT A TIME (see run.py), each walk is a
single-sequence forward pass, and per-occurrence vectors are moved to CPU as
fp16 immediately so the full [seq, d] tensors never accumulate. On an H200 both
models could co-reside, but sequential + disk cache keeps alignment re-runnable
offline and peak memory bounded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import torch

from config import Config
from graph import Walk, occurrence_table


# ---------------------------------------------------------------------------
# Tokenizer alignment: occurrence -> single token index
# ---------------------------------------------------------------------------
def resolve_token_indices(
    tokenizer, walk: Walk, subword_rule: str
) -> List[int]:
    """For each emitted word in the walk, return the token index it aligns to,
    per the subword rule. Uses character offset mapping from the fast tokenizer
    so it is robust to leading-space merging, BOS tokens, and multi-subword
    words. (For subword_rule == 'mean', the *list* of token indices per word is
    needed; see resolve_token_spans.)"""
    spans = resolve_token_spans(tokenizer, walk)
    out = []
    for toks in spans:
        if subword_rule == "last":
            out.append(toks[-1])
        elif subword_rule == "first":
            out.append(toks[0])
        elif subword_rule == "mean":
            raise ValueError("use resolve_token_spans for mean pooling")
        else:
            raise ValueError(f"unknown subword_rule {subword_rule!r}")
    return out


def resolve_token_spans(tokenizer, walk: Walk) -> List[List[int]]:
    """Per emitted word, the list of token indices whose character span overlaps
    the word. Asserts every word maps to >=1 token."""
    enc = tokenizer(
        walk.text,
        return_offsets_mapping=True,
        add_special_tokens=True,
        return_tensors=None,
    )
    offsets = enc["offset_mapping"]
    word_spans = walk.char_spans()

    per_word: List[List[int]] = []
    for (ws, we) in word_spans:
        toks = [
            ti
            for ti, (ts, te) in enumerate(offsets)
            # overlap test; skip special tokens which map to (0, 0)
            if not (ts == 0 and te == 0) and ts < we and te > ws
        ]
        assert toks, f"word span {(ws, we)} matched no tokens in {walk.text!r}"
        per_word.append(toks)
    assert len(per_word) == len(walk.nodes)
    return per_word


# ---------------------------------------------------------------------------
# Hooked capture
# ---------------------------------------------------------------------------
@dataclass
class CaptureResult:
    # acts[layer] : float16 array [N_occurrences, hidden_size]
    acts: Dict[int, np.ndarray]
    meta: Dict[str, np.ndarray]   # walk_id, step, node, context_length (occ order)
    hidden_size: int
    model_name: str


def _decoder_blocks(model):
    """Return the list of decoder blocks for Llama/Gemma-style HF models, with a
    fallback for the tiny GPT-2 stub used by the SMOKE preset."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers              # Llama, Gemma2
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h             # GPT-2 stub
    raise AttributeError("could not locate decoder blocks on this model")


def load_model(name: str, cfg: Config):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    dtype = getattr(torch, cfg.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=dtype,                       # transformers >= 4.56 (was torch_dtype)
        output_hidden_states=False,        # we use explicit hooks instead
    )
    model.to(cfg.device)
    model.eval()
    if not tok.is_fast:
        raise RuntimeError(
            f"{name} did not load a fast tokenizer; offset mapping is required "
            "for occurrence alignment."
        )
    return model, tok


def _register_hooks(model, capture_layers):
    """Register forward hooks that stash the post-block residual stream. Returns
    (grabbed dict, handles list); caller must remove the handles."""
    blocks = _decoder_blocks(model)
    grabbed: Dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(_module, _inp, out):
            # decoder block returns a tuple; first element is the hidden state
            grabbed[layer_idx] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    handles = [blocks[l].register_forward_hook(make_hook(l)) for l in capture_layers]
    return grabbed, handles


def _spans_for_walk(tokenizer, wk: Walk, cfg: Config) -> List[List[int]]:
    if cfg.subword_rule == "mean":
        return resolve_token_spans(tokenizer, wk)
    return [[i] for i in resolve_token_indices(tokenizer, wk, cfg.subword_rule)]


def _run_walk(model, tokenizer, wk: Walk, capture_layers, cfg, grabbed,
              dst_device) -> Dict[int, np.ndarray]:
    """One forward pass over one walk; return per-layer occurrence vectors
    [n_occ_walk, d] as fp16 numpy. `dst_device` is where this model lives."""
    enc = tokenizer(wk.text, add_special_tokens=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(dst_device)
    spans = _spans_for_walk(tokenizer, wk, cfg)

    grabbed.clear()
    model(input_ids=input_ids)

    # fast path: when every occurrence is a single token (the "last"/"first"
    # subword rule), gather all occurrence rows for a layer in ONE indexing op
    # and a single GPU->CPU transfer instead of one transfer per occurrence.
    single_idx = [t[0] for t in spans] if all(len(t) == 1 for t in spans) else None

    out: Dict[int, np.ndarray] = {}
    for l in capture_layers:
        hs = grabbed[l][0]                                         # [seq, d]
        if single_idx is not None:
            out[l] = hs[single_idx].to(torch.float16).cpu().numpy()
        else:                                                      # mean-pool path
            rows = [(hs[toks].mean(0) if len(toks) > 1 else hs[toks[0]])
                    .to(torch.float16).cpu().numpy() for toks in spans]
            out[l] = np.stack(rows)                                # [n_occ_walk, d]
    return out


def _finalize(per_model_rows: Dict[int, List[np.ndarray]], walks, name) -> CaptureResult:
    acts = {l: np.concatenate(rows, axis=0) for l, rows in per_model_rows.items()}
    meta = occurrence_table(walks)
    n = meta["walk_id"].shape[0]
    hidden_size = next(iter(acts.values())).shape[1]
    for l, a in acts.items():
        assert a.shape[0] == n, f"layer {l}: {a.shape[0]} acts vs {n} occurrences"
    return CaptureResult(acts=acts, meta=meta, hidden_size=int(hidden_size),
                         model_name=name)


@torch.no_grad()
def capture(model, tokenizer, walks: List[Walk], capture_layers: Tuple[int, ...],
            cfg: Config, device: str | None = None) -> CaptureResult:
    """Run every walk through ONE model, capturing the post-block residual stream
    at `capture_layers` and extracting one vector per concept-word occurrence."""
    device = device or cfg.device
    grabbed, handles = _register_hooks(model, capture_layers)
    per_layer_rows: Dict[int, List[np.ndarray]] = {l: [] for l in capture_layers}
    try:
        for wk in walks:
            for l, arr in _run_walk(model, tokenizer, wk, capture_layers, cfg,
                                    grabbed, device).items():
                per_layer_rows[l].append(arr)
    finally:
        for h in handles:
            h.remove()
    return _finalize(per_layer_rows, walks, model.name_or_path)


@torch.no_grad()
def capture_many(entries: List[dict], walks: List[Walk], cfg: Config
                 ) -> List[CaptureResult]:
    """Co-resident capture: all models stay loaded and each walk is run through
    every model in ONE shared pass over the corpus. Use when GPU memory holds
    both models at once (e.g. H200). Pairing is unaffected -- both models see the
    same walks in the same order.

    `entries` : list of {"model", "tokenizer", "layers", "name", "device"}.
    NOTE: on a single GPU the per-walk forward passes serialize at the kernel
    level; the gains here are skipping the load/free cycle and iterating the
    corpus once. True overlap would require placing models on separate devices.
    """
    hooks = []
    rows = []
    for e in entries:
        grabbed, handles = _register_hooks(e["model"], e["layers"])
        hooks.append((grabbed, handles))
        rows.append({l: [] for l in e["layers"]})

    try:
        for wk in walks:
            for ei, e in enumerate(entries):
                grabbed = hooks[ei][0]
                dev = e.get("device") or cfg.device
                got = _run_walk(e["model"], e["tokenizer"], wk, e["layers"], cfg,
                                grabbed, dev)
                for l, arr in got.items():
                    rows[ei][l].append(arr)
    finally:
        for _, handles in hooks:
            for h in handles:
                h.remove()

    return [_finalize(rows[ei], walks, entries[ei]["name"]) for ei in range(len(entries))]


def enough_memory_for_parallel(cfg: Config, min_gb: float) -> bool:
    """True if the CUDA device has at least `min_gb` total memory (rough guard
    for co-hosting two 8-9B bf16 models + activations). Non-CUDA -> False."""
    try:
        if not (cfg.device.startswith("cuda") and torch.cuda.is_available()):
            return False
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return total >= min_gb
    except Exception:
        return False


def save_capture(result: CaptureResult, path: str) -> None:
    """Persist activations + metadata to a single .npz (fp16 acts)."""
    payload = {f"layer_{l}": a for l, a in result.acts.items()}
    payload.update({f"meta_{k}": v for k, v in result.meta.items()})
    payload["_hidden_size"] = np.array([result.hidden_size])
    payload["_layers"] = np.array(sorted(result.acts.keys()))
    np.savez_compressed(path, **payload)


def load_capture(path: str) -> CaptureResult:
    z = np.load(path, allow_pickle=False)
    layers = list(z["_layers"])
    acts = {int(l): z[f"layer_{l}"] for l in layers}
    meta = {k[len("meta_"):]: z[k] for k in z.files if k.startswith("meta_")}
    return CaptureResult(acts=acts, meta=meta,
                         hidden_size=int(z["_hidden_size"][0]), model_name="")
