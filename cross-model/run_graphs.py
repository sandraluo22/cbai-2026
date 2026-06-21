"""Capture ONE model across the ring / hex / days-of-week graphs (model loaded
once, so it downloads once). Per graph: grid RSA at every layer, paper-faithful
Nw=50 emergence at a deep layer, and a 15k all-layer subsample. For the days
condition, also a semantic-distance RSA (context ring vs natural weekday cycle).

Usage:  python run_graphs.py <hf_model_name> <tag>
Env:    NWALKS (100) WLEN (1000) OUTDIR (/root/cmrun) DEVICE DTYPE
Outputs: <OUTDIR>/<graph>/<tag>_analysis.json , <tag>_acts_sub.npz
"""
from __future__ import annotations
import sys, os, json, gc
from dataclasses import replace
import numpy as np
try:
    import torch
except Exception:
    torch = None

from config import get_config, DAYS
import graph as G
import models as M

NW = 50
CTX = [5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
HICTX = 300


def spearman(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def GRAPHS():
    allg = [
        ("ring", dict(graph_type="ring", ring_size=16, word_set="concepts")),
        ("hex",  dict(graph_type="hex", hex_rows=4, hex_cols=4, word_set="concepts")),
        ("days", dict(graph_type="ring", ring_size=7, word_set="days")),
    ]
    only = os.environ.get("GRAPHS_FILTER")           # e.g. "days" or "hex,days"
    if only:
        keep = set(only.split(","))
        return [g for g in allg if g[0] in keep]
    return allg


def analyze(cap, graph, run_dir, tag, deep):
    n = graph.n_nodes
    iu = np.triu_indices(n, 1)
    GD = graph.distance_matrix()[iu]
    node, step, ctx = cap.meta["node"], cap.meta["step"], cap.meta["context_length"]
    layers = sorted(cap.acts)

    def means(L, mask):
        X = cap.acts[L].astype(np.float32)
        H = np.full((n, X.shape[1]), np.nan, np.float32)
        for k in range(n):
            m = mask & (node == k)
            if m.any():
                H[k] = X[m].mean(0)
        return H

    def rdm(H):
        return np.linalg.norm(H[:, None, :] - H[None, :, :], axis=2)[iu]

    hi = ctx >= HICTX
    grid_rsa = {int(L): spearman(rdm(means(L, hi)), GD) for L in layers}
    emrg = []
    for t in CTX:
        lo = max(0, t - NW)
        win = (step >= lo) & (step < t)
        emrg.append({"ctx": t, "rsa": spearman(rdm(means(deep, win)), GD)})
    out = {"n_nodes": n, "grid_rsa": grid_rsa,
           "emergence": {"layer": deep, "rows": emrg}}

    # days-of-week: also correlate against the NATURAL weekday cyclic distance
    if graph.words and graph.words[0] in DAYS:
        nat = [DAYS.index(w) for w in graph.words]
        SD = np.array([[min(abs(nat[i] - nat[j]), 7 - abs(nat[i] - nat[j]))
                        for j in range(n)] for i in range(n)])[iu]
        out["semantic_rsa"] = {int(L): spearman(rdm(means(L, hi)), SD) for L in layers}

    json.dump(out, open(f"{run_dir}/{tag}_analysis.json", "w"), indent=2)
    rng = np.random.default_rng(0)
    sidx = np.sort(rng.choice(node.shape[0], min(15000, node.shape[0]), replace=False))
    np.savez(f"{run_dir}/{tag}_acts_sub.npz",
             **{f"layer_{L}": cap.acts[L][sidx] for L in layers},
             **{f"meta_{k}": v[sidx] for k, v in cap.meta.items()},
             _layers=np.array(layers), _hidden_size=np.array([cap.hidden_size]))
    return grid_rsa


def main():
    model_name, tag = sys.argv[1], sys.argv[2]
    base = replace(get_config("gemma_qwen"),
                   n_walks=int(os.environ.get("NWALKS", "100")),
                   walk_length=int(os.environ.get("WLEN", "1000")),
                   out_dir=os.environ.get("OUTDIR", "/root/cmrun"),
                   device=os.environ.get("DEVICE", "cuda"),
                   dtype=os.environ.get("DTYPE", "bfloat16"))
    print(f"[{tag}] loading {model_name}", flush=True)
    model, tok = M.load_model(model_name, base)
    n_layers = model.config.num_hidden_layers
    deep = int(round(0.8 * (n_layers - 1)))
    for gname, gkw in GRAPHS():
        cfg = replace(base, name=gname, **gkw)
        graph = G.build_graph(cfg)
        walks = G.generate_walks(graph, cfg)
        run_dir = f"{cfg.out_dir}/{gname}"
        os.makedirs(run_dir, exist_ok=True)
        print(f"[{tag}/{gname}] {cfg.n_nodes} nodes, {cfg.n_walks} walks, "
              f"{n_layers} layers", flush=True)
        cap = M.capture(model, tok, walks, tuple(range(n_layers)), cfg)
        gr = analyze(cap, graph, run_dir, tag, deep)
        best = max(gr, key=gr.get)
        extra = ""
        if gname == "days":
            sj = json.load(open(f"{run_dir}/{tag}_analysis.json")).get("semantic_rsa", {})
            if sj:
                extra = f" | semantic peak {max(sj.values()):+.2f}"
        print(f"[{tag}/{gname}] grid RSA peak L{best}={gr[best]:+.3f} "
              f"(deep L{deep}={gr[deep]:+.3f}){extra}", flush=True)
        del cap; gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
    del model, tok; gc.collect()
    if torch and torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[{tag}] DONE all graphs", flush=True)


if __name__ == "__main__":
    main()
