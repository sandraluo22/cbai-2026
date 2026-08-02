"""Characterize HOW the shared ring direction shifts with size. For each ring n=3..16 capture the fundamental
circular-position 2D readout basis at EVERY layer, save the raw stack. Downstream we measure: per-step
principal angle (rotation rate), cumulative arc from the smallest ring, gap-decay of alignment, and the
depth at which the shared circle is tightest.

Env: GEN_MODEL(Llama) RINGS(3..16) NWALKS(16) WLEN(220) CTXLO(100) OUTDIR DEVICE
Out: <OUTDIR>/ring_shift_dirs_<model>.npz  (dirs[n_rings, n_layers, d, 2], rings, layers)
"""
from __future__ import annotations
import os, gc
from dataclasses import replace
import numpy as np
import torch

from config import get_config
import graph as G
import models as M
from cross_eigenmode_ablation import ALLSPEC, lw, node_means, readout_Q, cycle_lap_modes

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
RINGS = [int(x) for x in os.environ.get("RINGS", "3,4,5,6,7,8,9,10,11,12,13,14,15,16").split(",")]
NWALKS = int(os.environ.get("NWALKS", "16")); WLEN = int(os.environ.get("WLEN", "220"))
OUTDIR = os.environ.get("OUTDIR", "runs/axes/5_cyclic")


def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    model, tok = lw(hf, mirror, replace(get_config("gemma_qwen"), device=dev))
    cm = model.config; blocks = M._decoder_blocks(model); nL = cm.num_hidden_layers; d = cm.hidden_size
    layers = list(range(nL))
    dirs = np.zeros((len(RINGS), nL, d, 2), np.float32)
    for i, n in enumerate(RINGS):
        cfg = replace(get_config("gemma_qwen"), graph_type="ring", ring_size=n, n_walks=NWALKS, walk_length=WLEN, device=dev)
        ring = G.build_graph(cfg); wk = G.generate_walks(ring, cfg)
        means = node_means(model, tok, blocks, cm, ring, wk, dev)
        cw, cU = cycle_lap_modes(n); Q = readout_Q(means, cU[:, [1, 2]], nL, dev)
        for L in layers:
            q = Q.get(L)
            if q is not None: dirs[i, L] = q.cpu().numpy()
        print(f"[{tag}] ring-{n} captured", flush=True)
    del model, tok; gc.collect(); torch.cuda.empty_cache()
    p = f"{OUTDIR}/ring_shift_dirs_{tag}.npz"
    np.savez_compressed(p, dirs=dirs, rings=np.array(RINGS), layers=np.array(layers))
    print(f"DONE -> {p}  shape={dirs.shape}", flush=True)


if __name__ == "__main__":
    main()
