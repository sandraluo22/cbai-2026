"""Experiment 1 -- eigenmode logit lens. Each graph-Laplacian eigenmode k has a residual direction
c_k = Hc_L^T u_k (the d-dim vector the mode writes into the stream at layer L, from clean node-means).
Apply the LN-folded unembedding (logit lens) to c_k at the SECOND-TO-LAST layer to see which node-words
that mode promotes / demotes. The payoff: the parity mode's logit-lens should reproduce the 2-coloring
(so it predicts OPPOSITE-colour = neighbour nodes); coordinate modes should promote one end of an axis.

We build an (n_modes x n_nodes) logit-lens matrix M[k] = dla(c_k) and correlate each row with the mode's
own node pattern u_k, the parity 2-colouring, and the coordinates -- turning "what does mode k mean" into
"what does mode k make the model say".

Env: GEN_MODEL(Llama) GRAPH(square_grid) NWALKS(16) WLEN(300) CTXLO(100) LAP(norm|unnorm) LAYER(auto=nL-2) OUTDIR DEVICE
Out: <OUTDIR>/logit_lens_eigmode_<model>_<graph>.json (+ figure via viz/logit_lens_eigmode_plot.py)
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
import models as M
from models import resolve_token_spans

ALLSPEC = {"Llama": ("meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"),
           "Gemma": ("google/gemma-2-9b", "unsloth/gemma-2-9b"), "Qwen": ("Qwen/Qwen3-8B-Base", None)}
GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
GKW = {"square_grid": dict(graph_type="grid", grid_rows=4, grid_cols=4),
       "ring": dict(graph_type="ring", ring_size=16), "hex": dict(graph_type="hex", hex_rows=4, hex_cols=4),
       "prism": dict(graph_type="prism", prism_k=7)}
GRAPH = os.environ.get("GRAPH", "square_grid"); GS = {"square_grid": "grid"}.get(GRAPH, GRAPH)
NWALKS = int(os.environ.get("NWALKS", "16")); WLEN = int(os.environ.get("WLEN", "300"))
CTXLO = int(os.environ.get("CTXLO", "100")); LAP = os.environ.get("LAP", "norm")
OUTDIR = os.environ.get("OUTDIR", "runs/axes/1_decomposition/logit_lens_eigmode")


def load_with_fallback(hf, mirror, cfg):
    try: return M.load_model(hf, cfg)
    except Exception:
        if mirror is None: raise
        return M.load_model(mirror, cfg)


def two_colour(graph):
    n = graph.n_nodes; col = np.zeros(n)
    for s in range(n):
        if col[s] != 0: continue
        col[s] = 1; st = [s]
        while st:
            u = st.pop()
            for v in graph.adjacency[u]:
                if col[v] == 0: col[v] = -col[u]; st.append(v)
    return col.astype(float)


def laplacian_modes(A, kind):
    d = A.sum(1)
    if kind == "unnorm":
        L = np.diag(d) - A
    else:
        di = 1.0 / np.sqrt(np.maximum(d, 1e-12)); L = np.eye(len(A)) - (di[:, None] * A * di[None, :])
    return np.linalg.eigh(L)


def final_norm(model):
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    return model.transformer.ln_f


@torch.no_grad()
def node_means(model, tok, blocks, cm, graph, dev, walks):
    nL = cm.num_hidden_layers; grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    hooks = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
    n = graph.n_nodes; nsum = {L: np.zeros((n, cm.hidden_size)) for L in range(nL)}; ncnt = np.zeros(n)
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; grabbed.clear()
            model(input_ids=ids); cl = np.arange(1, len(nodes) + 1); single = [t[-1] for t in spans]
            for L in range(nL):
                rows = grabbed[L][0][single].float().cpu().numpy()
                for s in range(len(nodes)):
                    if cl[s] >= CTXLO:
                        nsum[L][nodes[s]] += rows[s]
                        if L == 0: ncnt[nodes[s]] += 1
    finally:
        for h in hooks: h.remove()
    cn = np.maximum(ncnt, 1)
    return {L: nsum[L] / cn[:, None] for L in range(nL)}


def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH], n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg); n = graph.n_nodes; col = two_colour(graph); coords = np.array(graph.coords, float)
    A = np.zeros((n, n))
    for a in range(n):
        for b in graph.adjacency[a]: A[a, b] = 1.0
    w, U = laplacian_modes(A, LAP); U = U / (np.linalg.norm(U, axis=0, keepdims=True) + 1e-12)

    def corr(a, b):
        a = a - a.mean(); b = b - b.mean(); return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    print(f"[{tag}] loading ({GS} n={n}, LAP={LAP})", flush=True)
    model, tok = load_with_fallback(hf, mirror, cfg)
    cm = model.config; blocks = M._decoder_blocks(model); nL = cm.num_hidden_layers
    LAYER = int(os.environ.get("LAYER", str(nL - 2)))                    # second-to-last by default
    walks = G.generate_walks(graph, cfg)

    WU = model.get_output_embeddings().weight.detach().float()
    fn = final_norm(model); is_rms = "rms" in type(fn).__name__.lower(); gamma = fn.weight.detach().float()
    if "gemma" in (getattr(cm, "model_type", "") or "").lower(): gamma = 1.0 + gamma
    cand = [tok(" " + graph.words[i], add_special_tokens=False)["input_ids"][0] for i in range(n)]
    WUn = WU[cand]                                                        # [n, d]

    def dla(vec_np):
        c = torch.tensor(vec_np, dtype=torch.float32, device=dev)
        if not is_rms: c = c - c.mean()
        c = c * gamma; g = WUn @ c; return (g - g.mean()).cpu().numpy()

    means = node_means(model, tok, blocks, cm, graph, dev, walks)
    Hc = means[LAYER] - means[LAYER].mean(0)                              # [n, d]

    modes = []
    Mmat = np.zeros((n, n))                                               # rows = modes, cols = node-word logits
    for k in range(1, n):
        c_k = Hc.T @ U[:, k]                                              # [d] residual direction mode k writes
        ll = dla(c_k); Mmat[k] = ll
        order = np.argsort(-ll)
        modes.append({"mode": k, "eigenvalue": float(w[k]),
                      "corr_selfpattern": corr(ll, U[:, k]),              # does logit-lens reproduce u_k?
                      "corr_parity": corr(ll, col),
                      "corr_coordX": corr(ll, coords[:, 0]), "corr_coordY": corr(ll, coords[:, 1]),
                      "top_promote": [graph.words[i] for i in order[:3]],
                      "top_demote": [graph.words[i] for i in order[-3:]]})
        print(f"  m{k:<2} λ={w[k]:.2f}: LL·u_k={modes[-1]['corr_selfpattern']:+.2f} "
              f"LL·parity={modes[-1]['corr_parity']:+.2f} LL·coordX={modes[-1]['corr_coordX']:+.2f} "
              f"promote={modes[-1]['top_promote']}", flush=True)

    out = {"model": tag, "graph": GRAPH, "lap": LAP, "layer": LAYER, "n_layers": nL, "words": graph.words,
           "logit_lens_matrix": Mmat.tolist(), "eigenvectors": U.tolist(), "two_colour": col.tolist(),
           "coords": coords.tolist(), "modes": modes}
    del model, tok; gc.collect()
    if torch and torch.cuda.is_available(): torch.cuda.empty_cache()
    p = f"{OUTDIR}/logit_lens_eigmode_{tag}_{GS}.json"
    json.dump(out, open(p, "w"), indent=2); print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
