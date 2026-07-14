"""Capture node-means for a RANDOM Markov graph (no imposed geometry) — the control for the
grid/ring/hex decompositions. Build a random CONNECTED graph over N nodes (random spanning tree +
extra random edges to ~avg degree DEG), uniform random walks over it, and capture per-(node,layer)
mean residuals at high context. Save adjacency + a spectral 2-D layout so divider_basis can build
the graph-Laplacian eigenbasis and ask: does the model still concentrate on the low-frequency
(coarse) structure, or does a structureless graph give a diffuse, hard-to-compress representation?

Env: PRESET TAG(Llama) N(16) DEG(4) SEED(0) NWALKS(60) WLEN(300) CTXLO(100) OUTDIR DEVICE
Out: <OUTDIR>/nodemeans_<TAG>_random.npz  (layer_* + adjacency + coords + rows/cols=0)
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
from graph import Graph
import models as M
from models import resolve_token_spans

PRESET = os.environ.get("PRESET", "gemma_qwen")
ALLSPEC = {"Llama": ("meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"),
           "Gemma": ("google/gemma-2-9b", "unsloth/gemma-2-9b"),
           "Qwen": ("Qwen/Qwen3-8B-Base", None), "distilgpt2": ("distilgpt2", None)}
TAG = os.environ.get("TAG", "Llama")
N = int(os.environ.get("N", "16")); DEG = int(os.environ.get("DEG", "4")); SEED = int(os.environ.get("SEED", "0"))
NWALKS = int(os.environ.get("NWALKS", "60")); WLEN = int(os.environ.get("WLEN", "300"))
CTXLO = int(os.environ.get("CTXLO", "100"))
OUTDIR = os.environ.get("OUTDIR", "runs/axes/1_decomposition/divider_basis")


def load_with_fallback(hf, mirror, cfg):
    try: return M.load_model(hf, cfg)
    except Exception:
        if mirror is None: raise
        return M.load_model(mirror, cfg)


def random_graph(n, deg, seed):
    rng = np.random.default_rng(seed); edges = set()
    perm = rng.permutation(n)                       # random spanning tree -> connected
    for i in range(1, n):
        j = perm[rng.integers(0, i)]; a, b = int(perm[i]), int(j); edges.add((min(a, b), max(a, b)))
    target = n * deg // 2
    while len(edges) < target:
        a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
        if a != b: edges.add((min(a, b), max(a, b)))
    adj = [[] for _ in range(n)]
    for a, b in edges: adj[a].append(b); adj[b].append(a)
    A = np.zeros((n, n))
    for a, b in edges: A[a, b] = A[b, a] = 1
    L = np.diag(A.sum(1)) - A; w, V = np.linalg.eigh(L)
    coords = V[:, 1:3]                              # spectral 2-D layout
    return [sorted(x) for x in adj], A, coords


@torch.no_grad()
def node_means(model, tok, blocks, cm, walks, dev, n):
    grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    nL = cm.num_hidden_layers; hs = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
    nsum = {L: np.zeros((n, cm.hidden_size)) for L in range(nL)}; ncnt = np.zeros(n)
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; grabbed.clear(); model(input_ids=ids)
            single = [t[-1] for t in spans]; cl = np.arange(1, len(nodes) + 1)
            for L in range(nL):
                rows = grabbed[L][0][single].float().cpu().numpy()
                for s in range(len(nodes)):
                    if cl[s] >= CTXLO:
                        nsum[L][nodes[s]] += rows[s]
                        if L == 0: ncnt[nodes[s]] += 1
    finally:
        for h in hs: h.remove()
    cn = np.maximum(ncnt, 1)
    return {L: (nsum[L] / cn[:, None]).astype(np.float16) for L in range(nL)}, ncnt


def main():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    os.makedirs(OUTDIR, exist_ok=True)
    hf, mirror = ALLSPEC[TAG]
    cfg = replace(get_config("gemma_qwen"), graph_type="ring", ring_size=N, n_walks=NWALKS, walk_length=WLEN, device=dev)
    words = cfg.words()[:N]
    adj, A, coords = random_graph(N, DEG, SEED)
    graph = Graph(n_nodes=N, words=words, adjacency=adj, coords=[tuple(c) for c in coords])
    degs = [len(a) for a in adj]
    print(f"[{TAG}] random graph N={N} edges={int(A.sum()//2)} deg(min/mean/max)={min(degs)}/{np.mean(degs):.1f}/{max(degs)}", flush=True)
    print(f"[{TAG}] loading", flush=True)
    model, tok = load_with_fallback(hf, mirror, cfg)
    cm = model.config; blocks = M._decoder_blocks(model); nL = cm.num_hidden_layers
    walks = G.generate_walks(graph, cfg)
    means, ncnt = node_means(model, tok, blocks, cm, walks, dev, N)
    save = {f"layer_{L}": means[L] for L in range(nL)}
    save.update({"adjacency": A.astype(np.int8), "coords": np.array(coords, float),
                 "rows": np.array([0]), "cols": np.array([0]), "ncnt": ncnt,
                 "tag": np.array([TAG]), "graph": np.array(["random"]), "seed": np.array([SEED])})
    path = f"{OUTDIR}/nodemeans_{TAG}_random.npz"
    np.savez_compressed(path, **save)
    print(f"[{TAG}/random] nL={nL} min_occ={int(ncnt.min())} -> {path}", flush=True)
    del model, tok; gc.collect()
    if torch and torch.cuda.is_available(): torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
