"""DIRECT (additive) attribution: how much does each circuit head WRITE to each eigenmode, measured
by projecting the head's own o_proj residual-write (per node) onto the Laplacian eigenvectors — no
ablation. head-write_node = attn_out[:, head_slice] @ o_proj.weight[:, head_slice]^T ; average per
node (ctx>=CTXLO); centre over nodes; power on mode k = ||Hc_head^T u_k||^2.

Env: GEN_MODEL(Llama) GRAPH(square_grid) NWALKS(16) CTXLO(100) GHSJSON OUTDIR DEVICE
Out: <OUTDIR>/head_direct_mode_write_<model>_<G>.json
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
GKW = {"square_grid": dict(graph_type="grid", grid_rows=4, grid_cols=4)}
GRAPH = os.environ.get("GRAPH", "square_grid"); GS = {"square_grid": "grid"}.get(GRAPH, GRAPH)
NWALKS = int(os.environ.get("NWALKS", "16")); CTXLO = int(os.environ.get("CTXLO", "100"))
GHSJSON = os.environ.get("GHSJSON", "runs/axes/4_circuits/greedy_head_set/ghs_Llama_grid.json")
OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/head_eig_sweep")


def load_with_fallback(hf, mirror, cfg):
    try: return M.load_model(hf, cfg)
    except Exception:
        if mirror is None: raise
        return M.load_model(mirror, cfg)


def attn_proj(block, cm):
    if hasattr(block, "self_attn") and hasattr(block.self_attn, "o_proj"):
        return block.self_attn.o_proj, (getattr(cm, "head_dim", None) or cm.hidden_size // cm.num_attention_heads)
    return block.attn.c_proj, cm.hidden_size // cm.n_head


@torch.no_grad()
def capture_oproj_nodemeans(model, tok, blocks, cm, layers, graph, dev, walks, n):
    """per-node mean of the o_proj INPUT (concatenated head outputs) at each given layer."""
    grabbed = {}
    def mk(L):
        def pre(_m, args): grabbed[L] = args[0].detach()
        return pre
    hooks = [attn_proj(blocks[L], cm)[0].register_forward_pre_hook(mk(L)) for L in layers]
    nsum = {L: np.zeros((n, cm.hidden_size)) for L in layers}; ncnt = np.zeros(n)
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; grabbed.clear(); model(input_ids=ids)
            single = [t[-1] for t in spans]; cl = np.arange(1, len(nodes) + 1); first = layers[0]
            for L in layers:
                rows = grabbed[L][0][single].float().cpu().numpy()
                for s in range(len(nodes)):
                    if cl[s] >= CTXLO:
                        nsum[L][nodes[s]] += rows[s]
                        if L == first: ncnt[nodes[s]] += 1
    finally:
        for h in hooks: h.remove()
    cn = np.maximum(ncnt, 1)[:, None]
    return {L: nsum[L] / cn for L in layers}


def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH], n_walks=NWALKS, walk_length=300, device=dev)
    graph = G.build_graph(cfg); n = graph.n_nodes
    A = np.zeros((n, n))
    for a in range(n):
        for b in graph.adjacency[a]: A[a, b] = 1.0
    w, V = np.linalg.eigh(np.diag(A.sum(1)) - A)                  # unnormalized (matches head_eig_sweep)
    GS_ = json.load(open(GHSJSON))
    circ = {"parity": [tuple(c["head"]) for c in GS_["objectives"]["parity"]["greedy"]],
            "coord":  [tuple(c["head"]) for c in GS_["objectives"]["coord"]["greedy"]]}
    ALLHEADS = os.environ.get("ALLHEADS", "0") == "1"
    model, tok = load_with_fallback(hf, mirror, cfg)
    cm = model.config; blocks = M._decoder_blocks(model); nL, nH = cm.num_hidden_layers, cm.num_attention_heads
    _, hd = attn_proj(blocks[0], cm)
    layers = list(range(nL)) if ALLHEADS else sorted({l for v in circ.values() for l, _ in v})
    print(f"[{tag}] ALLHEADS={ALLHEADS}  capturing {len(layers)} layers", flush=True)
    walks = G.generate_walks(graph, cfg)
    amean = capture_oproj_nodemeans(model, tok, blocks, cm, layers, graph, dev, walks, n)

    def head_power(L, h, WoL):
        slc = slice(h * hd, (h + 1) * hd)
        writ = amean[L][:, slc] @ WoL[:, slc].T                  # (n, hidden) head's per-node residual write
        Hc = writ - writ.mean(0); c = V.T @ Hc
        return (c ** 2).sum(1)[1:]                               # power per mode (raw), skip trivial

    out = {"model": tag, "graph": GRAPH, "eigenvalues": [float(x) for x in w[1:]],
           "circuits": {k: [f"L{l}H{hh}" for l, hh in v] for k, v in circ.items()}}
    if ALLHEADS:
        Dall = np.zeros((nL, nH, len(w) - 1))
        for L in layers:
            WoL = attn_proj(blocks[L], cm)[0].weight.detach().float().cpu().numpy()
            for h in range(nH):
                Dall[L, h] = head_power(L, h, WoL)
            del WoL
            if L % 8 == 0: print(f"  layer {L}/{nL}", flush=True)
        out["nL"] = nL; out["nH"] = nH; out["direct_power_all"] = Dall.tolist()
    else:
        Wo = {L: attn_proj(blocks[L], cm)[0].weight.detach().float().cpu().numpy() for L in layers}
        out["direct_power"] = {f"L{L}H{h}": [float(x) for x in head_power(L, h, Wo[L])]
                               for v in circ.values() for (L, h) in v}
    del model, tok; gc.collect()
    if torch and torch.cuda.is_available(): torch.cuda.empty_cache()
    p = f"{OUTDIR}/head_direct_mode_write_{'all_' if ALLHEADS else ''}{tag}_{GS}.json"
    json.dump(out, open(p, "w"), indent=2); print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
