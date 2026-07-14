"""Markov-chain families: for each graph family, capture node-means (-> Laplacian eigenmode power
spectrum, computed offline) AND behavioural accuracy (does the model predict a valid next node?).
Tests which eigenvectors the model relies on across structured vs random chains, and whether accuracy
tracks how compressible/structured the chain is.

Families (N nodes, uniform random walk over undirected neighbours):
  grid, ring, er_random (Erdos-Renyi), sbm2 / sbm4 (planted communities), tree, smallworld.

Per family, one pass records: neighbour mass (softmax mass on the current node's true neighbours),
validity (argmax is a neighbour), and per-(node,layer) mean residuals. Saves nodemeans_<TAG>_<fam>.npz
(layers + adjacency + spectral coords) + an accuracy line.

Env: PRESET TAG(Llama) N(16) SEED(0) NWALKS(60) WLEN(300) CTXLO(100) TEMP(1.0) FAMS OUTDIR DEVICE
Out: <OUTDIR>/nodemeans_<TAG>_<fam>.npz  and  <OUTDIR>/markov_families_<TAG>.json
"""
from __future__ import annotations
import os, json, gc
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
N = int(os.environ.get("N", "16")); SEED = int(os.environ.get("SEED", "0"))
NWALKS = int(os.environ.get("NWALKS", "60")); WLEN = int(os.environ.get("WLEN", "300"))
CTXLO = int(os.environ.get("CTXLO", "100")); TEMP = float(os.environ.get("TEMP", "1.0"))
FAMS = os.environ.get("FAMS", "grid,ring,er_random,sbm2,sbm4,tree,smallworld").split(",")
OUTDIR = os.environ.get("OUTDIR", "runs/axes/1_decomposition/markov_families")


def load_with_fallback(hf, mirror, cfg):
    try: return M.load_model(hf, cfg)
    except Exception:
        if mirror is None: raise
        return M.load_model(mirror, cfg)


def _finish(edges, n):
    """symmetric adjacency + spectral 2-D coords from an edge set."""
    A = np.zeros((n, n))
    for a, b in edges: A[a, b] = A[b, a] = 1
    L = np.diag(A.sum(1)) - A; w, V = np.linalg.eigh(L)
    adj = [sorted(np.where(A[i] > 0)[0].tolist()) for i in range(n)]
    return adj, A, V[:, 1:3]


def _spanning_tree(rng, n, order=None):
    perm = rng.permutation(n) if order is None else np.array(order)
    e = set()
    for i in range(1, n):
        j = perm[rng.integers(0, i)]; a, b = int(perm[i]), int(j); e.add((min(a, b), max(a, b)))
    return e


def build_family(name, n, seed):
    rng = np.random.default_rng(seed)
    if name == "grid":
        r = c = int(round(n ** 0.5)); e = set()
        for i in range(r):
            for j in range(c):
                u = i * c + j
                if j + 1 < c: e.add((u, u + 1))
                if i + 1 < r: e.add((u, u + c))
        return _finish(e, n)
    if name == "ring":
        return _finish({(i, (i + 1) % n) if i + 1 < n else (0, n - 1) for i in range(n)}, n)
    if name == "tree":
        return _finish(_spanning_tree(rng, n), n)
    if name == "er_random":
        e = _spanning_tree(rng, n); tgt = n * 4 // 2
        while len(e) < tgt:
            a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
            if a != b: e.add((min(a, b), max(a, b)))
        return _finish(e, n)
    if name in ("sbm2", "sbm4"):
        k = 2 if name == "sbm2" else 4; block = np.repeat(np.arange(k), n // k)
        e = _spanning_tree(rng, n)                       # backbone -> connected
        pin, pout = 0.75, 0.04
        for a in range(n):
            for b in range(a + 1, n):
                p = pin if block[a] == block[b] else pout
                if rng.random() < p: e.add((a, b))
        return _finish(e, n)
    if name == "smallworld":
        e = {(i, (i + 1) % n) for i in range(n)} | {(i, (i + 2) % n) for i in range(n)}  # ring lattice
        e = {(min(a, b), max(a, b)) for a, b in e}; e = set(e)
        for edge in list(e):                              # rewire ~20%
            if rng.random() < 0.2:
                a = edge[0]; b = int(rng.integers(0, n))
                if b != a: e.discard(edge); e.add((min(a, b), max(a, b)))
        return _finish(e, n)
    raise ValueError(name)


@torch.no_grad()
def capture(model, tok, blocks, cm, graph, cand_t, dev, walks, n):
    grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    nL = cm.num_hidden_layers; hs = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
    nsum = {L: np.zeros((n, cm.hidden_size)) for L in range(nL)}; ncnt = np.zeros(n)
    nbr = 0.0; val = 0; cnt = 0
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; grabbed.clear()
            logits = model(input_ids=ids).logits[0]
            single = [t[-1] for t in spans]; cl = np.arange(1, len(nodes) + 1)
            for L in range(nL):
                rows = grabbed[L][0][single].float().cpu().numpy()
                for s in range(len(nodes)):
                    if cl[s] >= CTXLO:
                        nsum[L][nodes[s]] += rows[s]
                        if L == 0: ncnt[nodes[s]] += 1
            for s in range(len(nodes) - 1):
                if s + 1 < CTXLO: continue
                p = torch.softmax(logits[spans[s][-1]][cand_t].float() / TEMP, 0).cpu().numpy(); p = p / p.sum()
                nb = graph.neighbors(nodes[s]); nbr += float(p[nb].sum()); val += int(int(p.argmax()) in nb); cnt += 1
    finally:
        for h in hs: h.remove()
    cn = np.maximum(ncnt, 1); c = max(cnt, 1)
    means = {L: (nsum[L] / cn[:, None]).astype(np.float16) for L in range(nL)}
    return means, ncnt, {"nbr_mass": nbr / c, "validity": val / c, "n_pred": cnt}


def main():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    os.makedirs(OUTDIR, exist_ok=True)
    hf, mirror = ALLSPEC[TAG]
    cfg = replace(get_config("gemma_qwen"), graph_type="ring", ring_size=N, n_walks=NWALKS, walk_length=WLEN, device=dev)
    words = cfg.words()[:N]
    print(f"[{TAG}] loading", flush=True)
    model, tok = load_with_fallback(hf, mirror, cfg)
    cm = model.config; blocks = M._decoder_blocks(model)
    cand_t = torch.tensor([tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in words], device=dev)
    out = {"model": TAG, "N": N, "seed": SEED, "families": {}}
    for fam in FAMS:
        adj, A, coords = build_family(fam, N, SEED)
        graph = Graph(n_nodes=N, words=words, adjacency=adj, coords=[tuple(c) for c in coords])
        walks = G.generate_walks(graph, cfg)
        means, ncnt, acc = capture(model, tok, blocks, cm, graph, cand_t, dev, walks, N)
        degs = A.sum(1)
        np.savez_compressed(f"{OUTDIR}/nodemeans_{TAG}_{fam}.npz",
                            **{f"layer_{L}": means[L] for L in range(cm.num_hidden_layers)},
                            adjacency=A.astype(np.int8), coords=np.array(coords, float),
                            rows=np.array([0]), cols=np.array([0]), ncnt=ncnt)
        out["families"][fam] = {**acc, "edges": int(A.sum() // 2), "mean_deg": float(degs.mean()),
                                "min_occ": int(ncnt.min())}
        print(f"[{TAG}/{fam:11}] edges={int(A.sum()//2):3d} deg={degs.mean():.1f}  "
              f"nbr_mass={acc['nbr_mass']:.2f}  validity={acc['validity']:.2f}", flush=True)
    json.dump(out, open(f"{OUTDIR}/markov_families_{TAG}.json", "w"), indent=2)
    del model, tok; gc.collect()
    if torch and torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"DONE -> {OUTDIR}/markov_families_{TAG}.json", flush=True)


if __name__ == "__main__":
    main()
