"""Per-layer attention atlas: EVERY head's QK-softmax attention heatmap, one slide per
layer (a grid of all heads at that layer). One PDF per model (Qwen, then Llama).

Stimulus = the induction probe stimulus (GLEN random tokens, excl. most/least frequent,
repeated NREP times, SOS prepended), so induction stripes / previous-token / BOS-sink
patterns are all visible across the head grid.

Runs on the pod (eager attention). Env: PRESET MODELS_FILTER(Qwen,Llama) GLEN(25) NREP(4)
OUTDIR DEVICE STIM(induction|walk)
Out: <OUTDIR>/attention_atlas_<model>.pdf
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from dataclasses import replace

try:
    import torch
except Exception:
    torch = None

from config import get_config
import graph as G
from models import resolve_token_spans

PRESET = os.environ.get("PRESET", "gemma_qwen")
if PRESET == "smoke":
    MODELS = [("distilgpt2", "distilgpt2", None)]
else:
    MODELS = [("Qwen", "Qwen/Qwen3-8B-Base", None),
              ("Llama", "meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"),
              ("Gemma", "google/gemma-2-9b", "unsloth/gemma-2-9b")]
_mf = os.environ.get("MODELS_FILTER")
if _mf:
    MODELS = [m for m in MODELS if m[0] in set(_mf.split(","))]
GLEN = int(os.environ.get("GLEN", "25"))
NREP = int(os.environ.get("NREP", "4"))
STIM = os.environ.get("STIM", "induction")        # "induction" (generic) | "walk" (graph task)
GRAPH = os.environ.get("GRAPH", "days")           # which graph for STIM=walk
GKW = {"days": dict(graph_type="ring", ring_size=7, word_set="days"),
       "square_grid": dict(graph_type="grid", grid_rows=4, grid_cols=4),
       "ring": dict(graph_type="ring", ring_size=16),
       "hex": dict(graph_type="hex", hex_rows=4, hex_cols=4)}
WLEN = int(os.environ.get("WLEN", "60"))
OUTDIR = os.environ.get("OUTDIR", "/workspace/cross-model/runs/induction-head/atlas" if PRESET != "smoke" else "runs/smoke_atlas")


def load_eager(tag, hf, mirror, cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = getattr(torch, cfg.dtype)
    def _load(name):
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation="eager")
        model.to(cfg.device).eval()
        return model, tok
    try:
        return _load(hf)
    except Exception:
        if mirror:
            return _load(mirror)
        raise


def stimulus_ids(tok, cfg, dev):
    if STIM == "walk":
        g = replace(cfg, **GKW[GRAPH], walk_length=WLEN, n_walks=1)
        wk = G.generate_walks(G.build_graph(g), g)[0]
        ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
        return ids, f"{GRAPH} walk ({WLEN} steps)"
    V = int(getattr(tok, "vocab_size", None) or len(tok))
    special = set(tok.all_special_ids or [])
    ids_pool = [i for i in range(V) if i not in special]
    lo, hi = int(0.02 * len(ids_pool)), int(0.90 * len(ids_pool))
    rng = np.random.default_rng(0)
    r = rng.choice(ids_pool[lo:hi], size=GLEN, replace=False).tolist()
    sos = tok.bos_token_id if tok.bos_token_id is not None else (tok.eos_token_id or 0)
    return torch.tensor([[sos] + r * NREP], device=dev), f"{GLEN} random toks x{NREP} + SOS"


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    cfg = replace(get_config("gemma_qwen"), device=dev)
    os.makedirs(OUTDIR, exist_ok=True)
    for tag, hf, mirror in MODELS:
        print(f"[{tag}] loading (eager)", flush=True)
        model, tok = load_eager(tag, hf, mirror, cfg)
        ids, stim_desc = stimulus_ids(tok, cfg, dev)
        atts = [a[0].float().cpu().numpy() for a in model(input_ids=ids, output_attentions=True).attentions]
        nL = len(atts); nH = atts[0].shape[0]
        ncol = 8; nrow = int(np.ceil(nH / ncol))
        L = GLEN
        out = f"{OUTDIR}/attention_atlas_{tag}_{STIM}.pdf"
        with PdfPages(out) as pdf:
            for l in range(nL):
                A = atts[l]                                   # [H, S, S]
                fig, ax = plt.subplots(nrow, ncol, figsize=(ncol * 1.7, nrow * 1.7))
                ax = np.array(ax).reshape(-1)
                for h in range(nH):
                    a = ax[h]
                    a.imshow(A[h], cmap="viridis", aspect="auto", vmin=0,
                             vmax=max(0.05, float(np.percentile(A[h], 99))))
                    a.set_title(f"H{h}", fontsize=5, pad=1)
                    a.set_xticks([]); a.set_yticks([])
                for h in range(nH, len(ax)):
                    ax[h].axis("off")
                fig.suptitle(f"{tag}  layer {l}  — every head's attention  [{stim_desc}]  "
                             f"(induction stripe = diagonal at key=query−{L})", fontsize=10)
                fig.tight_layout(rect=[0, 0, 1, 0.97]); pdf.savefig(fig); plt.close(fig)
        print(f"[{tag}] wrote {out}  ({nL} layer-slides x {nH} heads)", flush=True)
        del model, tok
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
