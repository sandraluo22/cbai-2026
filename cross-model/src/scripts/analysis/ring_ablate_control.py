"""Specificity control for the ring sharing matrix: for each ring n=3..16, compare the drop in next-node
validity from ablating (a) its OWN fitted fundamental direction vs (b) the mean over several RANDOM rank-2
per-layer directions of matched rank. If own >> random floor, the circle ablation is specific even where the
diagonal is not the row-max (the band-not-diagonal shape then reflects shared directions, not a weak effect).

Env: GEN_MODEL(Llama) RINGS(3..16) NWALKS(16) WLEN(220) CTXLO(100) NRAND(5) OUTDIR DEVICE
Out: <OUTDIR>/ring_ablate_control_<model>.json
"""
from __future__ import annotations
import os, json, gc
from dataclasses import replace
import numpy as np
import torch

from config import get_config
import graph as G
import models as M
from cross_eigenmode_ablation import ALLSPEC, lw, node_means, readout_Q, abl_hooks, cycle_lap_modes, torus_neighbour_validity

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
RINGS = [int(x) for x in os.environ.get("RINGS", "3,4,5,6,7,8,9,10,11,12,13,14,15,16").split(",")]
NWALKS = int(os.environ.get("NWALKS", "16")); WLEN = int(os.environ.get("WLEN", "220"))
NRAND = int(os.environ.get("NRAND", "5")); OUTDIR = os.environ.get("OUTDIR", "runs/axes/5_cyclic")


def rand_Q(nL, d, dev, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    Q = {}
    for L in range(nL):
        A = torch.randn(d, 2, generator=g)
        q, _ = torch.linalg.qr(A)
        Q[L] = q.to(dev)
    return Q


def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model); nL = cm.num_hidden_layers; d = cm.hidden_size

    out = {"model": tag, "rings": RINGS, "nrand": NRAND, "rows": {}}
    for n in RINGS:
        cfg = replace(get_config("gemma_qwen"), graph_type="ring", ring_size=n, n_walks=NWALKS, walk_length=WLEN, device=dev)
        ring = G.build_graph(cfg); wk = G.generate_walks(ring, cfg)
        base = torus_neighbour_validity(model, tok, blocks, cm, ring, wk, dev)
        means = node_means(model, tok, blocks, cm, ring, wk, dev)
        cw, cU = cycle_lap_modes(n); Qown = readout_Q(means, cU[:, [1, 2]], nL, dev)
        hs = abl_hooks(blocks, Qown, nL); own = torus_neighbour_validity(model, tok, blocks, cm, ring, wk, dev)
        for h in hs: h.remove()
        rnd = []
        for r in range(NRAND):
            Qr = rand_Q(nL, d, dev, seed=1000 * n + r)
            hs = abl_hooks(blocks, Qr, nL); rnd.append(torus_neighbour_validity(model, tok, blocks, cm, ring, wk, dev))
            for h in hs: h.remove()
        row = {"base": round(base, 3), "own": round(own, 3), "rand_mean": round(float(np.mean(rnd)), 3),
               "rand_std": round(float(np.std(rnd)), 3),
               "own_drop": round(base - own, 3), "rand_drop": round(base - float(np.mean(rnd)), 3)}
        out["rows"][n] = row
        print(f"[{tag}] ring-{n:2d}: base={row['base']} own_drop={row['own_drop']} rand_drop={row['rand_drop']}(±{row['rand_std']})", flush=True)

    del model, tok; gc.collect(); torch.cuda.empty_cache()
    p = f"{OUTDIR}/ring_ablate_control_{tag}.json"
    json.dump(out, open(p, "w"), indent=2); print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
