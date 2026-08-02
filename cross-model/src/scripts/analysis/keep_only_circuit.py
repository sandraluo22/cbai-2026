"""Minimal-circuit test: keep only a few attention heads live and MEAN-ablate every other head (replace its
o_proj-input slice with its mean activation over valid positions), MLPs intact. If {L14H26, L2H22} alone
recover most of next-node neighbour prediction across structures, they are the minimal in-context circuit.
Controls: each head alone, a random-2-head floor (mean of seeds), and all-heads-ablated.

Two-pass: pass 1 caches per-layer o_proj-input means; pass 2 overwrites ablated head slices with them.

Env: GEN_MODEL(Llama) KEEP("14:26,2:22") NWALKS(6) WLEN(250) CTXLO(120) NRAND(3) OUTDIR DEVICE
Out: <OUTDIR>/keep_only_circuit_<model>.json
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

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
KEEP = os.environ.get("KEEP", "14:26,2:22")
KEEP_SET = {tuple(int(x) for x in p.split(":")) for p in KEEP.split(",")}
NWALKS = int(os.environ.get("NWALKS", "6")); WLEN = int(os.environ.get("WLEN", "250"))
CTXLO = int(os.environ.get("CTXLO", "120")); NRAND = int(os.environ.get("NRAND", "3"))
OUTDIR = os.environ.get("OUTDIR", "runs/axes/5_cyclic")

STRUCTS = [("grid4x4", dict(graph_type="grid", grid_rows=4, grid_cols=4)),
           ("hex4x4", dict(graph_type="hex", hex_rows=4, hex_cols=4)),
           ("ring16", dict(graph_type="ring", ring_size=16)),
           ("prism7", dict(graph_type="prism", prism_k=7)),
           ("path1x16", dict(graph_type="grid", grid_rows=1, grid_cols=16))]


def cache_means(model, blocks, cm, nL, batch, mask):
    means = {}
    def mk(L):
        def pre(_m, args):
            x = args[0]; mf = mask.unsqueeze(-1).to(x.dtype)
            means[L] = ((x * mf).sum(dim=(0, 1)) / mf.sum()).detach()
        return pre
    hs = [attn_proj(blocks[L], cm)[0].register_forward_pre_hook(mk(L)) for L in range(nL)]
    model(input_ids=batch, attention_mask=mask)
    for h in hs: h.remove()
    return means


def keep_hooks(blocks, cm, nL, nH, means, keepset, dev):
    hs = []
    for L in range(nL):
        proj, hd = attn_proj(blocks[L], cm)
        ablate = [h for h in range(nH) if (L, h) not in keepset]
        if not ablate:
            continue
        cols = torch.tensor(np.concatenate([np.arange(h * hd, (h + 1) * hd) for h in ablate]), device=dev, dtype=torch.long)
        m = means[L]
        def pre(_m, args, cols=cols, m=m):
            x = args[0].clone(); x[..., cols] = m[cols].to(x.dtype)
            return (x,) + tuple(args[1:])
        hs.append(proj.register_forward_pre_hook(pre))
    return hs


def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    import cycle_head_circuit as chc; chc.CTXLO = CTXLO
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model)
    nL = cm.num_hidden_layers; nH = getattr(cm, "num_attention_heads", None) or cm.n_head
    rng = np.random.default_rng(0)
    keep_list = sorted(KEEP_SET)
    out = {"model": tag, "keep": KEEP, "structs": {}}
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

        res = {"base": round(base, 3), "keep_both": round(run(KEEP_SET), 3)}
        for hd_ in keep_list:
            res[f"keep_{hd_[0]}:{hd_[1]}"] = round(run({hd_}), 3)
        res["keep_none"] = round(run(set()), 3)
        rnd = []
        for r in range(NRAND):
            rset = {(int(rng.integers(nL)), int(rng.integers(nH))) for _ in range(len(KEEP_SET))}
            while len(rset) < len(KEEP_SET): rset.add((int(rng.integers(nL)), int(rng.integers(nH))))
            rnd.append(run(rset))
        res["keep_rand2_mean"] = round(float(np.mean(rnd)), 3); res["keep_rand2_std"] = round(float(np.std(rnd)), 3)
        # fraction of clean recovered above the all-ablated floor
        denom = base - res["keep_none"]
        res["recovered_frac"] = round((res["keep_both"] - res["keep_none"]) / denom, 3) if denom > 1e-6 else None
        out["structs"][name] = res
        solo = "  ".join(f"{a}:{b}={res[f'keep_{a}:{b}']}" for a, b in keep_list)
        print(f"[{tag}] {name:9} base={res['base']} keep_both={res['keep_both']}  [{solo}]  "
              f"none={res['keep_none']} rand2={res['keep_rand2_mean']} recov={res['recovered_frac']}", flush=True)
    del model, tok; gc.collect(); torch.cuda.empty_cache()
    p = f"{OUTDIR}/keep_only_circuit_{tag}.json"
    json.dump(out, open(p, "w"), indent=2); print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
