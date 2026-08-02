"""What does a given head WRITE in each structure? Capture head h's additive residual-write per node
(o_proj-input head-slice @ W_O head-slice), centre over nodes, and project onto the structure's graph-
Laplacian eigenmodes -> a power spectrum over modes. Answers whether L14H26 writes the GRID's parity
(top-eigenvalue checkerboard) but the RING's circular/position modes (low/high-freq), i.e. one head doing
structure-conditioned outputs.

Env: GEN_MODEL(Llama) HEAD("14:26") GRAPHS("grid,ring16,ring15,ring8") NWALKS(10) CTXLO(100) OUTDIR DEVICE
Out: <OUTDIR>/head_write_spectrum_<model>.json
"""
from __future__ import annotations
import os, json, gc
from dataclasses import replace
import numpy as np
import torch

from config import get_config
import graph as G
import models as M
from models import resolve_token_spans
from cross_eigenmode_ablation import ALLSPEC, lw

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
HEAD = os.environ.get("HEAD", "14:26"); HL, HH = (int(x) for x in HEAD.split(":"))
GRAPHS = os.environ.get("GRAPHS", "grid,ring16,ring15,ring8").split(",")
NWALKS = int(os.environ.get("NWALKS", "10")); CTXLO = int(os.environ.get("CTXLO", "100"))
OUTDIR = os.environ.get("OUTDIR", "runs/axes/5_cyclic")

GKW = {"grid": dict(graph_type="grid", grid_rows=4, grid_cols=4)}
for n in (5, 6, 7, 8, 9, 10, 11, 12, 15, 16):
    GKW[f"ring{n}"] = dict(graph_type="ring", ring_size=n)


def attn_proj(block, cm):
    if hasattr(block, "self_attn") and hasattr(block.self_attn, "o_proj"):
        return block.self_attn.o_proj, (getattr(cm, "head_dim", None) or cm.hidden_size // cm.num_attention_heads)
    return block.attn.c_proj, cm.hidden_size // cm.n_head


def lap_modes(graph):
    n = graph.n_nodes; A = np.zeros((n, n))
    for i in range(n):
        for j in graph.neighbors(i): A[i, j] = 1
    L = np.diag(A.sum(1)) - A
    w, U = np.linalg.eigh(L)  # ascending eigenvalue = ascending frequency
    return w, U


@torch.no_grad()
def head_write_nodemeans(model, tok, blocks, cm, graph, walks, dev):
    proj, hd = attn_proj(blocks[HL], cm)
    WO = proj.weight.detach().float()                      # [d, d]; head cols = HH*hd:(HH+1)*hd
    sl = slice(HH * hd, (HH + 1) * hd)
    grab = {}
    def pre(_m, args): grab["x"] = args[0].detach()
    h = proj.register_forward_pre_hook(pre)
    n = graph.n_nodes; nsum = np.zeros((n, cm.hidden_size)); ncnt = np.zeros(n)
    for wk in walks:
        ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
        spans = resolve_token_spans(tok, wk); nodes = wk.nodes; grab.clear(); model(input_ids=ids)
        x = grab["x"][0]                                    # [T, d] o_proj input
        write = (x[:, sl].float() @ WO[:, sl].T)            # [T, d] this head's residual write
        cl = np.arange(1, len(nodes) + 1)
        for s in range(len(nodes)):
            if cl[s] < CTXLO: continue
            nsum[nodes[s]] += write[spans[s][-1]].cpu().numpy(); ncnt[nodes[s]] += 1
    h.remove()
    means = nsum / np.maximum(ncnt[:, None], 1)
    return means


def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model)
    out = {"model": tag, "head": HEAD, "graphs": {}}
    for gname in GRAPHS:
        cfg = replace(get_config("gemma_qwen"), **GKW[gname], n_walks=NWALKS, walk_length=250, device=dev)
        graph = G.build_graph(cfg); walks = G.generate_walks(graph, cfg)
        means = head_write_nodemeans(model, tok, blocks, cm, graph, walks, dev)
        Hc = means - means.mean(0)
        w, U = lap_modes(graph)
        pw = np.array([np.linalg.norm(Hc.T @ U[:, k]) ** 2 for k in range(graph.n_nodes)])
        pw = pw / (pw.sum() + 1e-12)
        n = graph.n_nodes
        # frequency bands: low = bottom third of nonzero modes, high = top third, parity = top eigenvalue mode
        nz = np.arange(1, n)  # skip constant mode 0
        lo = nz[: max(1, len(nz) // 3)]; hi = nz[-max(1, len(nz) // 3):]
        parity_mode = int(np.argmax(w))  # top-eigenvalue = checkerboard/parity (only meaningful if bipartite)
        is_bip = abs(w[parity_mode] - (2.0 if gname == "grid" else 2.0)) < 0.35 if gname.startswith("ring") else True
        out["graphs"][gname] = {
            "n": n, "lambda_max": round(float(w[parity_mode]), 3), "bipartite_parity": bool(gname == "grid" or (n % 2 == 0)),
            "power_per_mode": [round(float(x), 4) for x in pw],
            "top3_modes": [[int(k), round(float(pw[k]), 3), round(float(w[k]), 3)] for k in np.argsort(pw)[::-1][:3]],
            "power_low": round(float(pw[lo].sum()), 3), "power_high": round(float(pw[hi].sum()), 3),
            "power_parity_topeig": round(float(pw[parity_mode]), 3)}
        t = out["graphs"][gname]
        print(f"[{tag}] {gname:7} n={n:2d} λmax={t['lambda_max']:.2f}  top modes={t['top3_modes']}  "
              f"low={t['power_low']} high={t['power_high']} parity(topeig)={t['power_parity_topeig']}", flush=True)
    del model, tok; gc.collect(); torch.cuda.empty_cache()
    p = f"{OUTDIR}/head_write_spectrum_{tag}.json"
    json.dump(out, open(p, "w"), indent=2); print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
