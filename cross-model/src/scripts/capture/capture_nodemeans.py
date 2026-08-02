"""Capture per-(node, layer) MEAN residual-stream activations at high context, for the
divider-basis / graph-spectrum decomposition. Tiny output (n_nodes x d per layer), so we can
pull it back and decompose offline for every model x topology without moving the big
per-occurrence caches.

For a given model TAG, loops over GRAPHS: run the model over random walks, hook every decoder
block's residual output, average each node's activations over all occurrences with context
length >= CTXLO. Saves the node-means plus the graph ADJACENCY + coords so the decomposition can
build the topology-correct Laplacian (grid / ring / hex).

Env: PRESET(gemma_qwen) TAG(Llama|Gemma|Qwen) GRAPHS(square_grid,ring,hex) NWALKS(60) WLEN(300)
     CTXLO(100) OUTDIR DEVICE
Out: <OUTDIR>/nodemeans_<TAG>_<graph>.npz  (layer_0..layer_{nL-1}, adjacency, coords, rows, cols, tag, graph)
"""
from __future__ import annotations
import os, gc
from dataclasses import replace
import numpy as np

try:
    import torch
except Exception:
    torch = None

from config import get_config
import graph as G
import models as M
from models import resolve_token_spans

PRESET = os.environ.get("PRESET", "gemma_qwen")
ALLSPEC = {"Llama": ("meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"),
           "Gemma": ("google/gemma-2-9b", "unsloth/gemma-2-9b"),
           "Qwen":  ("Qwen/Qwen3-8B-Base", None),
           "Qwen32": ("Qwen/Qwen3-32B", None),
           "distilgpt2": ("distilgpt2", None)}
TAG = os.environ.get("TAG", "Llama" if PRESET != "smoke" else "distilgpt2")
GKW = {"square_grid": dict(graph_type="grid", grid_rows=4, grid_cols=4),
       "ring": dict(graph_type="ring", ring_size=16),
       "hex": dict(graph_type="hex", hex_rows=4, hex_cols=4),
       "days": dict(graph_type="ring", ring_size=7, word_set="days")}
GRAPHS = os.environ.get("GRAPHS", "square_grid,ring,hex").split(",")
NWALKS = int(os.environ.get("NWALKS", "60"))
WLEN = int(os.environ.get("WLEN", "300"))
CTXLO = int(os.environ.get("CTXLO", "100"))
OUTDIR = os.environ.get("OUTDIR", "runs/axes/1_decomposition/divider_basis")


def load_with_fallback(hf, mirror, cfg):
    try:
        return M.load_model(hf, cfg)
    except Exception:
        if mirror is None:
            raise
        return M.load_model(mirror, cfg)


@torch.no_grad()
def node_means(model, tok, blocks, cm, graph, walks, dev, n, ctxlo):
    grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    nL = cm.num_hidden_layers
    hs = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
    nsum = {L: np.zeros((n, cm.hidden_size)) for L in range(nL)}; ncnt = np.zeros(n)
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; grabbed.clear()
            model(input_ids=ids); single = [t[-1] for t in spans]; cl = np.arange(1, len(nodes) + 1)
            for L in range(nL):
                rows = grabbed[L][0][single].float().cpu().numpy()
                for s in range(len(nodes)):
                    if cl[s] >= ctxlo:
                        nsum[L][nodes[s]] += rows[s]
                        if L == 0: ncnt[nodes[s]] += 1
    finally:
        for h in hs: h.remove()
    cn = np.maximum(ncnt, 1)
    return {L: (nsum[L] / cn[:, None]).astype(np.float16) for L in range(nL)}, ncnt


def adjacency_matrix(graph):
    n = graph.n_nodes; A = np.zeros((n, n), np.int8)
    for i in range(n):
        for j in graph.adjacency[i]:
            A[i, j] = 1
    return A


def main():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    os.makedirs(OUTDIR, exist_ok=True)
    hf, mirror = ALLSPEC[TAG]
    print(f"[{TAG}] loading", flush=True)
    # load once against the first graph's config, reuse across graphs
    cfg0 = replace(get_config("gemma_qwen"), **GKW[GRAPHS[0]], n_walks=NWALKS, walk_length=WLEN, device=dev)
    model, tok = load_with_fallback(hf, mirror, cfg0)
    cm = model.config; blocks = M._decoder_blocks(model); nL = cm.num_hidden_layers
    for gname in GRAPHS:
        cfg = replace(get_config("gemma_qwen"), **GKW[gname], n_walks=NWALKS, walk_length=WLEN, device=dev)
        graph = G.build_graph(cfg); n = graph.n_nodes
        walks = G.generate_walks(graph, cfg)
        means, ncnt = node_means(model, tok, blocks, cm, graph, walks, dev, n, CTXLO)
        rows = GKW[gname].get("grid_rows") or GKW[gname].get("hex_rows") or 0
        cols = GKW[gname].get("grid_cols") or GKW[gname].get("hex_cols") or 0
        save = {f"layer_{L}": means[L] for L in range(nL)}
        save.update({"adjacency": adjacency_matrix(graph), "coords": np.array(graph.coords, float),
                     "rows": np.array([rows]), "cols": np.array([cols]), "ncnt": ncnt,
                     "tag": np.array([TAG]), "graph": np.array([gname])})
        path = f"{OUTDIR}/nodemeans_{TAG}_{gname}.npz"
        np.savez_compressed(path, **save)
        print(f"[{TAG}/{gname}] n={n} nL={nL} min_occ={int(ncnt.min())} -> {path}", flush=True)
    del model, tok; gc.collect()
    if torch and torch.cuda.is_available(): torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
