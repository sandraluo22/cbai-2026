"""Do in-context rings of different sizes share one circular circuit? For each ring size n=3..16, derive
its fundamental circular-position direction (per-layer readout of the n-cycle Laplacian's cos/sin pair from
the ring-n node-means). Then (a) CAUSAL: ablate ring-n's direction and measure ring-m's next-node neighbour
validity, for all pairs -> a sharing matrix; (b) REPRESENTATIONAL: subspace alignment between ring-n and
ring-m direction pairs at a late layer. A shared circuit shows off-diagonal drops and high alignment.

Env: GEN_MODEL(Llama) RINGS(3..16) NWALKS(12) WLEN(200) CTXLO(100) SIMLAYER(30) OUTDIR DEVICE
Out: <OUTDIR>/ring_circuit_sharing_<model>.json
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
NWALKS = int(os.environ.get("NWALKS", "12")); WLEN = int(os.environ.get("WLEN", "200"))
SIMLAYER = int(os.environ.get("SIMLAYER", "30")); OUTDIR = os.environ.get("OUTDIR", "runs/axes/5_cyclic")


def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model); nL = cm.num_hidden_layers

    Q = {}; walks = {}; graphs = {}; dir_rep = {}
    for n in RINGS:
        cfg = replace(get_config("gemma_qwen"), graph_type="ring", ring_size=n, n_walks=NWALKS, walk_length=WLEN, device=dev)
        ring = G.build_graph(cfg); wk = G.generate_walks(ring, cfg); graphs[n] = ring; walks[n] = wk
        means = node_means(model, tok, blocks, cm, ring, wk, dev)
        cw, cU = cycle_lap_modes(n); Q[n] = readout_Q(means, cU[:, [1, 2]], nL, dev)
        q = Q[n].get(SIMLAYER)
        dir_rep[n] = q.cpu().numpy() if q is not None else None
        print(f"[{tag}] ring-{n} direction ready", flush=True)

    # ---- causal sharing matrix: ablate ring-n dir, measure ring-m neighbour validity ----
    base = {}; Mtx = {}
    for m in RINGS:
        base[m] = round(torus_neighbour_validity(model, tok, blocks, cm, graphs[m], walks[m], dev), 3)
        Mtx[m] = {}
        for n in RINGS:
            hs = abl_hooks(blocks, Q[n], nL)
            Mtx[m][n] = round(torus_neighbour_validity(model, tok, blocks, cm, graphs[m], walks[m], dev), 3)
            for h in hs: h.remove()
        print(f"[{tag}] ring-{m} task: base={base[m]} self-ablate={Mtx[m][m]}", flush=True)

    # ---- representational alignment: principal-angle similarity of the 2D direction subspaces ----
    sim = {}
    for a in RINGS:
        sim[a] = {}
        for b in RINGS:
            if dir_rep[a] is None or dir_rep[b] is None: sim[a][b] = None; continue
            s = np.linalg.svd(dir_rep[a].T @ dir_rep[b], compute_uv=False)   # cos of principal angles
            sim[a][b] = round(float((s ** 2).mean()), 3)                      # mean squared cosine in [0,1]

    out = {"model": tag, "rings": RINGS, "baseline": base, "ablate_matrix": Mtx, "subspace_sim": sim, "sim_layer": SIMLAYER}
    del model, tok; gc.collect(); torch.cuda.empty_cache()
    p = f"{OUTDIR}/ring_circuit_sharing_{tag}.json"
    json.dump(out, open(p, "w"), indent=2); print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
