"""Capture Llama-3.1-8B on the SAME walks as Gemma/Qwen and run the paper's
representational analysis, so all three are directly comparable.

Outputs (to /root/cmrun/llama/):
  llama_grid_rsa.json   -- grid RSA at every layer (high-context per-node means)
  llama_emergence.json  -- paper-faithful Nw=50 emergence at layer 26
  acts_sub_llama.npz     -- 15k all-layer subsample for offline re-use
Same graph/seed/n_walks (200) as the gemma_qwen run, so emergence is
directly comparable to Gemma L32 / Qwen L28 (paper_faithful.json).
"""
from __future__ import annotations
import os, json, gc
from dataclasses import replace
import numpy as np
import torch

from config import get_config
import graph as G
import models as M

LLAMA = "NousResearch/Meta-Llama-3.1-8B"     # un-gated base mirror (== meta-llama base)
LAYER = 26                                    # paper's featured layer
HICTX = 300
NW = 50
CTX = [5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500, 750, 1000]


def spearman(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def main():
    cfg = replace(get_config("gemma_qwen"), n_walks=200, out_dir="/root/cmrun", name="llama")
    graph = G.build_grid_graph(cfg)
    walks = G.generate_walks(graph, cfg)
    IU = np.triu_indices(16, 1)
    GRIDD = graph.grid_distance_matrix()[IU]
    A_adj = np.zeros((16, 16))
    for n in range(16):
        for m in graph.neighbors(n):
            A_adj[n, m] = 1.0
    layers = tuple(range(32))
    run_dir = f"{cfg.out_dir}/{cfg.name}"
    os.makedirs(run_dir, exist_ok=True)

    print(f"[llama] capturing {LLAMA}, {cfg.n_walks} walks x {cfg.walk_length}, all 32 layers",
          flush=True)
    model, tok = M.load_model(LLAMA, cfg)
    cap = M.capture(model, tok, walks, layers, cfg)
    del model, tok; gc.collect(); torch.cuda.empty_cache()
    node, step, ctx = cap.meta["node"], cap.meta["step"], cap.meta["context_length"]

    def rdm(H):
        return np.linalg.norm(H[:, None, :] - H[None, :, :], axis=2)[IU]

    def means(L, mask):
        X = cap.acts[L].astype(np.float32)
        H = np.full((16, X.shape[1]), np.nan, np.float32)
        for k in range(16):
            mm = mask & (node == k)
            if mm.any():
                H[k] = X[mm].mean(0)
        return H

    def energy_ratio(H):
        D2 = ((H[:, None, :] - H[None, :, :]) ** 2).sum(-1)
        return float(D2[A_adj > 0].mean() / D2[IU].mean())

    # --- grid RSA at every layer (high-context) ---
    hi = ctx >= HICTX
    grid_rsa = {int(L): spearman(rdm(means(L, hi)), GRIDD) for L in layers}
    json.dump(grid_rsa, open(f"{run_dir}/llama_grid_rsa.json", "w"), indent=2)
    best = max(grid_rsa, key=grid_rsa.get)
    print(f"[llama] grid RSA peak L{best}={grid_rsa[best]:.3f}  |  L26={grid_rsa[26]:.3f}", flush=True)

    # --- paper-faithful Nw=50 emergence at layer 26 ---
    emrg = []
    for t in CTX:
        lo = max(0, t - NW)
        win = (step >= lo) & (step < t)
        H = means(LAYER, win)
        emrg.append({"ctx": t, "energy_ratio": energy_ratio(H), "rsa": spearman(rdm(H), GRIDD)})
    json.dump({"layer": LAYER, "emergence": emrg}, open(f"{run_dir}/llama_emergence.json", "w"), indent=2)
    print(f"[llama] emergence L26 rsa: {emrg[0]['rsa']:.2f} -> {emrg[-1]['rsa']:.2f}  "
          f"energy: {emrg[0]['energy_ratio']:.2f} -> {emrg[-1]['energy_ratio']:.2f}", flush=True)

    # --- save 15k all-layer subsample ---
    rng = np.random.default_rng(cfg.seed)
    sidx = np.sort(rng.choice(node.shape[0], 15000, replace=False))
    np.savez(f"{run_dir}/acts_sub_llama.npz",
             **{f"layer_{L}": cap.acts[L][sidx] for L in layers},
             **{f"meta_{k}": v[sidx] for k, v in cap.meta.items()},
             _hidden_size=np.array([cap.hidden_size]), _layers=np.array(layers))
    print("[llama] saved subsample. DONE", flush=True)


if __name__ == "__main__":
    main()
