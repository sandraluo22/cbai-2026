"""Variance-explained vs causal-USAGE, per eigenmode. For each grid, for every Laplacian eigenmode k measure
(a) VARIANCE = sum_L ||Hc_L^T u_k||^2 (how much the node-mean reps vary along mode k across layers), and
(b) USAGE = drop in next-node neighbour validity when that mode's per-layer readout is ablated (project the
residual onto its orthogonal complement at every layer). Then correlate the two: does high variance imply
high causal use, or do they dissociate ("structure in low-variance directions")? Also flags the single
most-used mode vs the single highest-variance mode.

Env: GEN_MODEL(Llama) GRIDS(4x4,6x6,8x8) NWALKS(12) SAMPLES_PER_NODE(120) WLEN_CAP(700) CTXLO(100) OUTDIR DEVICE
Out: <OUTDIR>/mode_var_vs_use_<model>.json
"""
from __future__ import annotations
import os, json, gc
from dataclasses import replace
import numpy as np
import torch

import config as _config
from config import get_config
import graph as G
import models as M
from models import resolve_token_spans
from per_mode_ablate import ALLSPEC, load_with_fallback, forward_collect, build_Q_from_dirs, laplacian_modes, two_colour
from grid_parity_compare import build_word_pool

GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
GRIDS = [int(g.split("x")[0]) for g in os.environ.get("GRIDS", "4x4,6x6,8x8").split(",")]
NWALKS = int(os.environ.get("NWALKS", "12")); SPN = int(os.environ.get("SAMPLES_PER_NODE", "120"))
WLEN_CAP = int(os.environ.get("WLEN_CAP", "700")); CTXLO = int(os.environ.get("CTXLO", "100"))
OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/parity")


def run_grid(model, tok, blocks, cm, dev, k):
    n = k * k
    wl = min(WLEN_CAP, CTXLO + int(np.ceil(n * SPN / NWALKS)))
    cfg = replace(get_config("gemma_qwen"), graph_type="grid", grid_rows=k, grid_cols=k, n_walks=NWALKS, walk_length=wl, device=dev)
    graph = G.build_graph(cfg); col = two_colour(graph); coords = np.array(graph.coords, float)
    A = np.zeros((n, n))
    for a in range(n):
        for b in graph.adjacency[a]: A[a, b] = 1.0
    w, U = laplacian_modes(A, "norm")
    nL = cm.num_hidden_layers
    cand_t = torch.tensor([tok(" " + graph.words[i], add_special_tokens=False)["input_ids"][0] for i in range(n)], device=dev)
    walks = G.generate_walks(graph, cfg)
    base, means = forward_collect(model, tok, blocks, cm, graph, cand_t, col, dev, walks, grab=True)

    def corr(a, b): a = a - a.mean(); b = b - b.mean(); return float(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    def label(kk):
        cx = max(corr(U[:, kk], coords[:, 0]), corr(U[:, kk], coords[:, 1])); cp = corr(U[:, kk], col)
        return "parity" if cp > 0.9 else ("coord" if cx > 0.7 else "other")

    Rdir = {kk: np.stack([(means[L] - means[L].mean(0)).T @ U[:, kk] for L in range(nL)]) for kk in range(1, n)}
    var = {kk: float((Rdir[kk] ** 2).sum()) for kk in range(1, n)}
    vtot = sum(var.values())
    modes = []
    for kk in range(1, n):
        Q = build_Q_from_dirs([[Rdir[kk][L]] for L in range(nL)], nL, dev)
        m, _ = forward_collect(model, tok, blocks, cm, graph, cand_t, col, dev, walks, proj_Q=Q)
        modes.append({"mode": kk, "eigenvalue": float(w[kk]), "label": label(kk),
                      "variance_frac": round(var[kk] / vtot, 4),
                      "usage_d_nbr": round(base["neighbour_validity"] - m["neighbour_validity"], 4)})
    v = np.array([md["variance_frac"] for md in modes]); u = np.array([md["usage_d_nbr"] for md in modes])
    def spear(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return corr(ra.astype(float), rb.astype(float))
    pear = corr(v, u); sp = spear(v, u)
    top_var = max(modes, key=lambda md: md["variance_frac"]); top_use = max(modes, key=lambda md: md["usage_d_nbr"])
    print(f"[{k}x{k}] n={n} base_nbr={base['neighbour_validity']:.3f}  pearson(var,use)={pear:.2f} spearman={sp:.2f}  "
          f"| MOST-VARIANCE m{top_var['mode']}({top_var['label']}) v={top_var['variance_frac']:.3f} u={top_var['usage_d_nbr']:.3f}  "
          f"| MOST-USED m{top_use['mode']}({top_use['label']}) u={top_use['usage_d_nbr']:.3f} v={top_use['variance_frac']:.3f}", flush=True)
    return {"k": k, "n": n, "base_nbr": base["neighbour_validity"], "pearson": round(pear, 3), "spearman": round(sp, 3),
            "top_variance_mode": top_var, "top_used_mode": top_use, "modes": modes}


def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    cfg0 = replace(get_config("gemma_qwen"), device=dev)
    model, tok = load_with_fallback(hf, mirror, cfg0)
    cm = model.config; blocks = M._decoder_blocks(model)
    need = max(k * k for k in GRIDS)
    if need > len(_config.WORDS): _config.WORDS[:] = build_word_pool(tok, need)
    out = {"model": tag, "grids": {}}
    for k in GRIDS:
        out["grids"][f"{k}x{k}"] = run_grid(model, tok, blocks, cm, dev, k)
    del model, tok; gc.collect(); torch.cuda.empty_cache()
    p = f"{OUTDIR}/mode_var_vs_use{os.environ.get('OUTTAG', '')}_{tag}.json"; json.dump(out, open(p, "w"), indent=2); print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
