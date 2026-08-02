"""Is L14H26 the most-damaging head for EVERY structure, or only rings? Run the same batched per-head
ablation (drop in next-node neighbour prob) across grid / hex / ring / prism / path, and report each
structure's top heads + where L14H26 ranks. If L14H26 tops geometric graphs but not the path, it's a
geometry head; if it tops the path too, it's generic in-context machinery.

Env: GEN_MODEL(Llama) NWALKS(6) WLEN(250) CTXLO(120) TOPK(12) OUTDIR DEVICE
Out: <OUTDIR>/struct_head_circuit_<model>.json
"""
from __future__ import annotations
import os, json, gc
from dataclasses import replace
import numpy as np
import torch

from config import get_config
import graph as G
import models as M
from cross_eigenmode_ablation import ALLSPEC, lw
from cycle_head_circuit import prep_batch, validity, head_hook, attn_proj

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
NWALKS = int(os.environ.get("NWALKS", "6")); WLEN = int(os.environ.get("WLEN", "250"))
CTXLO = int(os.environ.get("CTXLO", "120")); TOPK = int(os.environ.get("TOPK", "12"))
OUTDIR = os.environ.get("OUTDIR", "runs/axes/5_cyclic")
FOCUS = os.environ.get("FOCUS", "14:26"); FL, FH = (int(x) for x in FOCUS.split(":"))

STRUCTS = [("grid4x4", dict(graph_type="grid", grid_rows=4, grid_cols=4)),
           ("hex4x4", dict(graph_type="hex", hex_rows=4, hex_cols=4)),
           ("ring16", dict(graph_type="ring", ring_size=16)),
           ("prism7", dict(graph_type="prism", prism_k=7)),
           ("path1x16", dict(graph_type="grid", grid_rows=1, grid_cols=16))]


def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    # patch cycle_head_circuit's CTXLO to ours
    import cycle_head_circuit as chc; chc.CTXLO = CTXLO
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model)
    nL = cm.num_hidden_layers; nH = getattr(cm, "num_attention_heads", None) or cm.n_head
    out = {"model": tag, "focus": FOCUS, "structs": {}}
    for name, kw in STRUCTS:
        cfg = replace(get_config("gemma_qwen"), **kw, n_walks=NWALKS, walk_length=WLEN, device=dev)
        graph = G.build_graph(cfg); walks = G.generate_walks(graph, cfg)
        batch, mask, readouts, cand, nbrs = prep_batch(tok, graph, walks, dev)
        base = validity(model, batch, mask, readouts, cand, nbrs)
        hm = np.zeros((nL, nH), np.float32)
        for L in range(nL):
            proj, hd = attn_proj(blocks[L], cm)
            for h in range(nH):
                hh = head_hook(proj, hd, h, dev)
                hm[L, h] = base - validity(model, batch, mask, readouts, cand, nbrs); hh.remove()
        flat = sorted([(float(hm[L, h]), L, h) for L in range(nL) for h in range(nH)], reverse=True)
        rank = [i for i, (_, L, h) in enumerate(flat) if (L, h) == (FL, FH)][0]
        out["structs"][name] = {"n": graph.n_nodes, "base": round(base, 3),
                                "top": [[L, h, round(v, 4)] for v, L, h in flat[:TOPK]],
                                "focus_drop": round(float(hm[FL, FH]), 4), "focus_rank": rank}
        print(f"[{tag}] {name:9} n={graph.n_nodes:2d} base={base:.3f}  top3={[(L,h) for _,L,h in flat[:3]]}  "
              f"{FOCUS} rank={rank} drop={hm[FL,FH]:.3f}", flush=True)
    del model, tok; gc.collect(); torch.cuda.empty_cache()
    p = f"{OUTDIR}/struct_head_circuit_{tag}.json"
    json.dump(out, open(p, "w"), indent=2); print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
