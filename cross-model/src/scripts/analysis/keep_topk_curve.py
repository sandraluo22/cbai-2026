"""How many heads does the in-context circuit need? Rank heads by importance (from the ring-16 ablation map),
then keep the top-K live and MEAN-ablate the rest, sweeping K. Report fraction of clean next-node validity
recovered above the all-ablated floor, per structure. Answers whether the circuit is a few heads or many,
and whether a ring-derived head ranking generalizes to grid/path.

Env: GEN_MODEL(Llama) RANKJSON(cycle_head_circuit ring map) RANKSIZE(16) KS("2,4,8,16,24,32,48,64,96,128,192")
     NWALKS(6) WLEN(250) CTXLO(120) OUTDIR DEVICE
Out: <OUTDIR>/keep_topk_curve_<model>.json
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
from cycle_head_circuit import prep_batch, validity, attn_proj
from keep_only_circuit import cache_means, keep_hooks

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
RANKJSON = os.environ.get("RANKJSON", "runs/axes/5_cyclic/cycle_head_circuit_Llama.json")
RANKSIZE = os.environ.get("RANKSIZE", "16")
KS = [int(x) for x in os.environ.get("KS", "2,4,8,16,24,32,48,64,96,128,192").split(",")]
NWALKS = int(os.environ.get("NWALKS", "6")); WLEN = int(os.environ.get("WLEN", "250")); CTXLO = int(os.environ.get("CTXLO", "120"))
OUTDIR = os.environ.get("OUTDIR", "runs/axes/5_cyclic")

STRUCTS = [("grid4x4", dict(graph_type="grid", grid_rows=4, grid_cols=4)),
           ("ring16", dict(graph_type="ring", ring_size=16)),
           ("path1x16", dict(graph_type="grid", grid_rows=1, grid_cols=16))]


def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    import cycle_head_circuit as chc; chc.CTXLO = CTXLO
    rk = json.load(open(RANKJSON)); hm = np.array(rk["head_map"][RANKSIZE])
    order = [tuple(map(int, np.unravel_index(i, hm.shape))) for i in np.argsort(hm, axis=None)[::-1]]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model)
    nL = cm.num_hidden_layers; nH = getattr(cm, "num_attention_heads", None) or cm.n_head
    out = {"model": tag, "rank_from": f"ring{RANKSIZE}", "KS": KS, "ranked_top": [f"L{L}H{h}" for L, h in order[:12]], "structs": {}}
    for name, kw in STRUCTS:
        cfg = replace(get_config("gemma_qwen"), **kw, n_walks=NWALKS, walk_length=WLEN, device=dev)
        graph = G.build_graph(cfg); walks = G.generate_walks(graph, cfg)
        batch, mask, readouts, cand, nbrs = prep_batch(tok, graph, walks, dev)
        base = validity(model, batch, mask, readouts, cand, nbrs)
        means = cache_means(model, blocks, cm, nL, batch, mask)

        def run(keepset):
            hs = keep_hooks(blocks, cm, nL, nH, means, keepset, dev)
            v = validity(model, batch, mask, readouts, cand, nbrs)
            for h in hs: h.remove()
            return v
        floor = run(set()); denom = base - floor
        row = {"base": round(base, 3), "floor_none": round(floor, 3), "keep": {}}
        for K in KS:
            v = run(set(order[:K]))
            row["keep"][K] = {"validity": round(v, 3),
                              "recovered": round((v - floor) / denom, 3) if denom > 1e-6 else None}
        out["structs"][name] = row
        rec = "  ".join(f"K{K}={row['keep'][K]['recovered']}" for K in KS)
        print(f"[{tag}] {name:9} base={row['base']} floor={row['floor_none']}  recov: {rec}", flush=True)
    del model, tok; gc.collect(); torch.cuda.empty_cache()
    p = f"{OUTDIR}/keep_topk_curve_{tag}.json"
    json.dump(out, open(p, "w"), indent=2); print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
