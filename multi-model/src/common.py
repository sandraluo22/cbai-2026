"""Shared plumbing for the multi-model (Llama -> Qwen) transfer experiments.

Both experiments share one spine:

  1. A square-grid graph over 16 semantically-unrelated concept words (reusing
     cross-model's `graph.py` / `config.py`), the exact same object the
     cross-model paper-reproduction and coord-probe work is built on.
  2. `Llama-3.1-8B` (BASE) free-generates a random walk over that grid, with its
     next-token distribution CONSTRAINED to the 16 node words (like
     cross-model's gen_head_ablation). The result is a valid word sequence.
  3. That word sequence is fed to `Qwen3-8B-Base` and we read Qwen's internal
     state -- either the grid geometry (leave-one-node-out coord probe, Exp1) or
     an emotion direction (Exp2). Because both models consume the identical word
     sequence, pairing is by (node, step) exactly as in cross-model.

This module deliberately reuses the cross-model core (graph / models / config)
and the emotion GoEmotions helpers rather than re-implementing them, so the grid
object, tokenizer-alignment rule (last subword token), and per-node-mean
convention are bit-identical to the established runs. It only ADDS the
cross-model generation feed, the additive steering hook, and the emotion-vector
projection read-out.
"""
from __future__ import annotations

import os
import sys
import gc
from dataclasses import replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --- bootstrap: put the sibling cross-model/src and emotion/ on the path ------
# Works locally and on the pod as long as the three dirs keep their relative
# layout (cbai-2026/{cross-model,emotion,multi-model}). Override roots with
# CROSS_MODEL_SRC / EMOTION_DIR if the deploy layout differs.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))                      # cbai-2026
_CM_SRC = os.environ.get("CROSS_MODEL_SRC", os.path.join(_ROOT, "cross-model", "src"))
_EMO_DIR = os.environ.get("EMOTION_DIR", os.path.join(_ROOT, "emotion"))
for _p in (_CM_SRC, _EMO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import torch
except Exception:                                                   # pragma: no cover
    torch = None

from config import get_config                                       # noqa: E402  (cross-model)
import graph as G                                                   # noqa: E402
from graph import Walk                                              # noqa: E402
import models as M                                                  # noqa: E402
from models import resolve_token_spans                              # noqa: E402


# ---------------------------------------------------------------------------
# Model registry. BASE models throughout (consistent with cross-model). Each
# entry: (tag, gated HF id, ungated mirror for bare pods). PRESET=smoke swaps in
# distilgpt2 for both so the whole pipeline runs on CPU in seconds.
# ---------------------------------------------------------------------------
PRESET = os.environ.get("PRESET", "gemma_qwen")
SPEC: Dict[str, Tuple[str, Optional[str]]] = {
    "Llama": ("meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"),
    "Qwen":  ("Qwen/Qwen3-8B-Base", None),
}
if PRESET == "smoke":
    SPEC = {"Llama": ("distilgpt2", None), "Qwen": ("distilgpt2", None)}

# Square grid over 16 concept words -- the canonical cross-model object.
GKW = {"square_grid": dict(graph_type="grid", grid_rows=4, grid_cols=4),
       "smoke_grid":  dict(graph_type="grid", grid_rows=3, grid_cols=3)}


def default_device() -> str:
    return os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")


def make_cfg(graph="square_grid", n_walks=24, walk_length=300, device=None):
    """A gemma_qwen-derived config specialized to a grid + walk size. Keeps the
    frozen-dataclass discipline of cross-model (everything result-affecting here)."""
    gkw = GKW["smoke_grid"] if PRESET == "smoke" else GKW[graph]
    return replace(get_config("gemma_qwen"), **gkw,
                   n_walks=n_walks, walk_length=walk_length,
                   device=device or default_device())


def build_grid(cfg):
    graph = G.build_graph(cfg)
    n = graph.n_nodes
    R = cfg.grid_rows
    coords = np.array([(i // R, i % R) for i in range(n)], dtype=float)  # (row,col)
    return graph, n, coords


def load_model(tag: str, cfg):
    """Load `tag` in bf16 with its fast tokenizer, falling back to the ungated
    mirror on gated-access failure (bare pods)."""
    hf, mirror = SPEC[tag]
    try:
        return M.load_model(hf, cfg)
    except Exception as e:                                          # pragma: no cover
        if mirror is None:
            raise
        print(f"[{tag}] gated load failed ({type(e).__name__}); mirror {mirror}", flush=True)
        return M.load_model(mirror, cfg)


def decoder_blocks(model):
    return M._decoder_blocks(model)


def n_layers(model) -> int:
    return int(model.config.num_hidden_layers)


def free(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def mkwalk(nodes: Sequence[int], graph) -> Walk:
    return Walk(walk_id=0, nodes=list(nodes), words=[graph.words[j] for j in nodes])


def candidate_token_ids(tok, graph, dev):
    """First-subword token id of each node word (with a leading space), matching
    cross-model's constrained-generation candidate set."""
    ids = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in graph.words]
    return torch.tensor(ids, device=dev)


# ---------------------------------------------------------------------------
# Additive steering hook (Exp2). Adds a fixed vector to the POST-BLOCK residual
# stream at the chosen layers, at ALL positions -- the same tensor location the
# emotion vectors were read from, so the dose is interpretable in that space.
# ---------------------------------------------------------------------------
def steer_hooks(blocks, dev, layer_to_vec: Dict[int, np.ndarray]):
    handles = []
    for L, v in layer_to_vec.items():
        vt = torch.tensor(np.asarray(v), device=dev, dtype=torch.float32)

        def hook(_m, _i, out, vt=vt):
            h = out[0] if isinstance(out, tuple) else out
            h = h + vt.to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        handles.append(blocks[L].register_forward_hook(hook))
    return handles


# ---------------------------------------------------------------------------
# Constrained walk generation. Seed with `seed_nodes`, then sample `n_steps`
# more nodes, each time restricting the LM head to the 16 node-word tokens and
# sampling one. `steer` (layer->vector) optionally steers generation. Returns the
# full node list (seed + generated) plus per-step behaviour diagnostics.
# ---------------------------------------------------------------------------
@torch.no_grad() if torch is not None else (lambda f: f)
def generate_walk(model, tok, graph, cand_t, dev, seed_nodes, n_steps,
                  temp=1.0, rng=None, steer: Optional[Dict[int, np.ndarray]] = None,
                  prefix: str = ""):
    """Constrained walk generation. `steer` (layer->vec) optionally perturbs the
    residual stream; `prefix` optionally prepends natural-language context (e.g. a
    sad story) that the model READS while generating -- the emitted walk still
    contains only node words, and neighbour/validity are measured on nodes only.
    The two are mutually-exclusive ways to bias generation (activation vs prompt)."""
    rng = rng or np.random.default_rng(0)
    blocks = decoder_blocks(model)
    nodes = list(seed_nodes)
    nbr_mass, valid = [], []
    pre = (prefix.rstrip() + "\n") if prefix else ""
    handles = steer_hooks(blocks, dev, steer) if steer else []
    try:
        for _ in range(n_steps):
            wk = mkwalk(nodes, graph)
            ids = tok(pre + wk.text, add_special_tokens=True, return_tensors="pt")["input_ids"].to(dev)
            last = model(input_ids=ids).logits[0, -1]
            p = torch.softmax(last[cand_t].float() / temp, 0).cpu().numpy()
            p = p / p.sum()
            prev = nodes[-1]
            nb = graph.neighbors(prev)
            j = int(rng.choice(len(p), p=p))
            nbr_mass.append(float(p[nb].sum()))
            valid.append(int(j in nb))
            nodes.append(j)
    finally:
        for h in handles:
            h.remove()
    return nodes, {"nbr_mass": float(np.mean(nbr_mass)) if nbr_mass else float("nan"),
                   "validity": float(np.mean(valid)) if valid else float("nan")}


# ---------------------------------------------------------------------------
# All-layer per-node-mean residuals over a set of walks (teacher-forced feed).
# `ctxlo` restricts to occurrences with context_length >= ctxlo (the in-context
# regime), matching coord_decode / gen_head_ablation. Returns {L: (n_nodes, H)}.
# ---------------------------------------------------------------------------
@torch.no_grad() if torch is not None else (lambda f: f)
def node_means_all_layers(model, tok, graph, walks, dev, n_nodes, ctxlo=100):
    blocks = decoder_blocks(model)
    nL, H = n_layers(model), model.config.hidden_size
    grabbed: Dict[int, "torch.Tensor"] = {}

    def mk(L):
        def hh(_m, _i, out):
            grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh

    hs = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
    nsum = {L: np.zeros((n_nodes, H)) for L in range(nL)}
    ncnt = np.zeros(n_nodes)
    try:
        for wk in walks:
            ids = tok(wk.text, add_special_tokens=True, return_tensors="pt")["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk)
            single = [t[-1] for t in spans]                      # last-subword rule
            nodes = wk.nodes
            cl = np.arange(1, len(nodes) + 1)
            grabbed.clear()
            model(input_ids=ids)
            for L in range(nL):
                rows = grabbed[L][0][single].float().cpu().numpy()
                for s in range(len(nodes)):
                    if cl[s] >= ctxlo:
                        nsum[L][nodes[s]] += rows[s]
                        if L == 0:
                            ncnt[nodes[s]] += 1
    finally:
        for h in hs:
            h.remove()
    cn = np.maximum(ncnt, 1)
    return {L: nsum[L] / cn[:, None] for L in range(nL)}, ncnt


@torch.no_grad() if torch is not None else (lambda f: f)
def walk_reps_all_layers(model, tok, walks, dev, ctxlo=1):
    """One pooled activation vector PER WALK PER LAYER: the mean residual over the
    walk's node-token positions (last subword of each node, context >= ctxlo).
    Returns array (n_walks, nL, H) -- the feature tensor for a walk-level
    classifier (Exp4)."""
    blocks = decoder_blocks(model)
    nL, H = n_layers(model), model.config.hidden_size
    grabbed: Dict[int, "torch.Tensor"] = {}

    def mk(L):
        def hh(_m, _i, out):
            grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh

    hs = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
    reps = np.zeros((len(walks), nL, H), dtype=np.float32)
    try:
        for wi, wk in enumerate(walks):
            ids = tok(wk.text, add_special_tokens=True, return_tensors="pt")["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk)
            cl = np.arange(1, len(wk.nodes) + 1)
            keep = [t[-1] for s, t in enumerate(spans) if cl[s] >= ctxlo]
            grabbed.clear()
            model(input_ids=ids)
            for L in range(nL):
                reps[wi, L] = grabbed[L][0][keep].float().mean(0).cpu().numpy()
    finally:
        for h in hs:
            h.remove()
    return reps


# ---------------------------------------------------------------------------
# Leave-one-node-out ridge coord probe (identical procedure to cross-model's
# coord_decode / gen_head_ablation: per-axis LOO R^2, alpha chosen on LOO, plus
# an optional label-permutation null for significance).
# ---------------------------------------------------------------------------
ALPHAS = [1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6]


def _r2(y, yh):
    tot = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ((y - yh) ** 2).sum() / tot) if tot > 0 else float("nan")


def _prep_folds(Mn):
    n = Mn.shape[0]
    folds = []
    for k in range(n):
        idx = [i for i in range(n) if i != k]
        Xtr = Mn[idx]
        mu = Xtr.mean(0)
        sd = Xtr.std(0) + 1e-6
        Xs = (Xtr - mu) / sd
        xks = (Mn[k:k + 1] - mu) / sd
        U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
        folds.append((np.array(idx), (xks @ Vt.T).ravel(), U.T.copy(), S))
    return folds


def _loo_bestalpha(folds, y):
    """Per-axis LOO R^2 from PRECOMPUTED folds (SVDs), alpha chosen on LOO. The
    folds are label-independent, so a permutation null reuses them -- computing
    them once instead of per-permutation is a ~nperm-fold speedup (this is the
    coord_decode.py structure)."""
    n = len(folds)
    best = (-9.0, -9.0)
    for a in ALPHAS:
        pred = np.zeros((n, 2))
        for k, (idx, proj, UT, S) in enumerate(folds):
            ytr = y[idx]
            ymu = ytr.mean(0)
            pred[k] = proj @ ((S / (S ** 2 + a))[:, None] * (UT @ (ytr - ymu))) + ymu
        rr, rc = _r2(y[:, 0], pred[:, 0]), _r2(y[:, 1], pred[:, 1])
        if rr + rc > best[0] + best[1]:
            best = (rr, rc)
    return best


def coord_loo_r2(Mn, coords):
    """Per-axis leave-one-node-out R^2, alpha chosen to maximize mean LOO R^2.
    Returns (r2_axis0, r2_axis1). NaN-node rows are dropped."""
    ok = np.isfinite(Mn).all(1)
    Mn = Mn[ok]
    coords = coords[ok]
    if Mn.shape[0] < 6:
        return float("nan"), float("nan")
    return _loo_bestalpha(_prep_folds(Mn), coords)


def coord_loo_r2_with_null(Mn, coords, nperm=200, seed=0):
    ok = np.isfinite(Mn).all(1)
    Mn = Mn[ok]
    coords = coords[ok]
    n = Mn.shape[0]
    if n < 6:
        return (float("nan"),) * 6
    folds = _prep_folds(Mn)                      # computed ONCE; reused for every perm
    rr, rc = _loo_bestalpha(folds, coords)
    if nperm <= 0:
        return rr, rc, float("nan"), float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    nulls = np.array([_loo_bestalpha(folds, coords[rng.permutation(n)]) for _ in range(nperm)])
    p_row = float((np.sum(nulls[:, 0] >= rr) + 1) / (nperm + 1))
    p_col = float((np.sum(nulls[:, 1] >= rc) + 1) / (nperm + 1))
    return rr, rc, float(np.nanmean(nulls[:, 0])), float(np.nanmean(nulls[:, 1])), p_row, p_col


# ---------------------------------------------------------------------------
# Emotion-vector I/O + projection read-out (Exp2).
# ---------------------------------------------------------------------------
def emotion_vec_path(model_tag: str, run_dir: str) -> str:
    return os.path.join(run_dir, f"emotion_vectors_{model_tag}.npz")


def load_emotion_vectors(model_tag: str, run_dir: str):
    """Returns (clean (E,L,H) float32, meta dict). `meta['emotion_names']` gives
    the axis order; `meta['sadness_idx']` the sadness row."""
    z = np.load(emotion_vec_path(model_tag, run_dir), allow_pickle=True)
    clean = z["clean"].astype(np.float32)
    meta = {k: z[k].tolist() for k in ("emotion_names",) if k in z.files}
    names = list(meta.get("emotion_names", []))
    meta["sadness_idx"] = names.index("sadness") if "sadness" in names else None
    meta["L"] = int(clean.shape[1])
    meta["H"] = int(clean.shape[2])
    return clean, meta


def unit(v, eps=1e-9):
    v = np.asarray(v, dtype=np.float64)
    return v / max(np.linalg.norm(v), eps)


@torch.no_grad() if torch is not None else (lambda f: f)
def project_residuals_on_dir(model, tok, text, dev, dir_by_layer: Dict[int, np.ndarray],
                             tok_slice: Optional[slice] = None):
    """Feed `text`, and at each layer in `dir_by_layer` return the mean projection
    of the residual (over tokens in `tok_slice`, default all) onto the given unit
    direction. Positive => residual points toward the direction (e.g. sadness)."""
    blocks = decoder_blocks(model)
    grabbed: Dict[int, "torch.Tensor"] = {}

    def mk(L):
        def hh(_m, _i, out):
            grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh

    hs = [blocks[L].register_forward_hook(mk(L)) for L in dir_by_layer]
    ids = tok(text, add_special_tokens=True, return_tensors="pt")["input_ids"].to(dev)
    grabbed.clear()
    model(input_ids=ids)
    for h in hs:
        h.remove()
    out = {}
    for L, d in dir_by_layer.items():
        hs_L = grabbed[L][0].float().cpu().numpy()               # (seq, H)
        sl = tok_slice if tok_slice is not None else slice(0, hs_L.shape[0])
        seg = hs_L[sl]
        out[L] = float((seg @ unit(d)).mean()) if len(seg) else float("nan")
    return out


# small sad/negative lexicon for a cheap, model-independent story-sadness proxy
SAD_WORDS = {
    "sad", "sadness", "sorrow", "grief", "grieving", "cry", "cried", "crying", "tears",
    "tear", "weep", "wept", "lonely", "alone", "loss", "lost", "mourn", "mourning",
    "despair", "hopeless", "empty", "ache", "aching", "pain", "painful", "hurt",
    "broken", "heartbreak", "heartbroken", "miserable", "misery", "gloom", "gloomy",
    "dark", "cold", "silence", "silent", "gone", "never", "goodbye", "farewell",
    "regret", "remorse", "wound", "wounded", "fade", "faded", "dying", "death", "dead",
}


# Natural-language priming prefixes (Exp3): a genuinely sad passage and a
# length-matched neutral one, used to bias Llama's generation via CONTEXT instead
# of residual steering. Kept deliberately generic (no grid words) so any transfer
# rides the emotional tone, not shared vocabulary.
SAD_STORY = (
    "The house was quiet now. After the funeral everyone left, and I sat alone in "
    "the empty kitchen where she used to hum while making tea. Her coat still hung "
    "by the door. I could not bring myself to move it. Grief settles like dust, in "
    "the corners, on everything she touched. I kept waiting to hear her footsteps, "
    "but there was only the slow ticking of the clock and the ache of a silence that "
    "would never end. I have never felt so utterly, hopelessly alone."
)
NEUTRAL_STORY = (
    "The report was finished now. After the meeting everyone left, and I sat at the "
    "long table where we usually review the quarterly figures. The folder still lay "
    "by the monitor. I decided to file it later. Paperwork accumulates steadily, in "
    "the trays, on every surface of the office. I kept expecting the printer to "
    "start, but there was only the steady hum of the ventilation and the routine "
    "rhythm of an ordinary afternoon that would continue as usual. It was a perfectly "
    "average, unremarkable working day."
)
HAPPY_STORY = (
    "The house was full of light now. After the wedding everyone stayed, and I stood "
    "in the bright kitchen where she used to hum while making tea, laughing at some "
    "joke. Her coat hung by the door beside mine. Joy bubbles up like sunlight, in "
    "the corners, on everything we touch. I kept hearing her footsteps and her "
    "delighted laughter, and there was the cheerful ticking of the clock and the warm "
    "glow of a happiness that felt like it would never end. I have never felt so "
    "wonderfully, radiantly alive."
)


def sad_word_fraction(text: str) -> float:
    toks = [t.strip(".,!?;:\"'()[]").lower() for t in text.split()]
    toks = [t for t in toks if t]
    if not toks:
        return float("nan")
    return sum(t in SAD_WORDS for t in toks) / len(toks)
