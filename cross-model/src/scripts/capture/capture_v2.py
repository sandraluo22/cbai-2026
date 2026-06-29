"""v2 capture: longer context (walk_length=2000), means over context 1000-2000.

Motivation: in v1 (walk_length=1000, means over ctx>=300) Gemma/Qwen grids look
less clean than Llama's; the hypothesis is the context is too short for the
in-context geometry to fully converge. v2 doubles the walk length and analyses
the second half (ctx in [1000,2000]).

Runs on a CUDA GPU (gated 8-9B models). For each model, captures grid/ring/hex
at EVERY layer, subsamples occurrences across the full context range, and writes
the standardized v2 layout that paths.py expects:

    runs/v2/<graph>/<Tag>_acts_sub.npz      # layer_<L> + meta_{node,step,context_length}
    runs/v2/<graph>/<Tag>_analysis.json     # grid_rsa per layer over ctx in [CTX_LO,CTX_HI]

Per-graph peak RAM ~= n_occ * d * n_layers * 2 bytes (fp16 acts on CPU); at
NWALKS=100, WLEN=2000 that is ~60GB for Gemma. One graph held at a time.

Usage (on the GPU box, from the cross-model/ project root):
    huggingface-cli login                      # all three are gated
    PYTHONPATH=src NWALKS=100 WLEN=2000 OUTDIR=runs/v2 \
        python src/scripts/capture/capture_v2.py
Env: NWALKS(100) WLEN(2000) OUTDIR(runs/v2) SUBSAMPLE(24000) DEVICE(cuda)
     DTYPE(bfloat16) CTX_LO(1000) CTX_HI(2000) MODELS_FILTER(e.g. "Gemma,Qwen")
"""
from __future__ import annotations
import sys, os, json, gc, shutil
from dataclasses import replace
import numpy as np
try:
    import torch
except Exception:
    torch = None

from config import get_config
import graph as G
import models as M

MODELS = [("Llama", "meta-llama/Llama-3.1-8B"),
          ("Gemma", "google/gemma-2-9b"),
          ("Qwen",  "Qwen/Qwen3-8B-Base")]
# ungated mirrors (identical weights) used if the gated original is inaccessible
MIRRORS = {"Llama": "NousResearch/Meta-Llama-3.1-8B",
           "Gemma": "unsloth/gemma-2-9b"}


def free_weights(names):
    """Delete cached HF weights (original + mirror) to stay under disk quota."""
    hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    for name in names:
        if name:
            shutil.rmtree(os.path.join(hub, "models--" + name.replace("/", "--")), ignore_errors=True)


def load_with_fallback(tag, hf, base, M):
    try:
        return M.load_model(hf, base)
    except Exception as e:
        if tag in MIRRORS:
            print(f"[{tag}] {hf} unavailable ({type(e).__name__}); "
                  f"falling back to ungated mirror {MIRRORS[tag]}", flush=True)
            return M.load_model(MIRRORS[tag], base)
        raise
GRAPHS = [("square_grid", dict(graph_type="grid", grid_rows=4, grid_cols=4)),
          ("ring",        dict(graph_type="ring", ring_size=16)),
          ("hex",         dict(graph_type="hex", hex_rows=4, hex_cols=4))]


def spearman(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def npz_ok(path):
    """True iff a complete, loadable subsample npz already exists (for resume)."""
    if not os.path.exists(path):
        return False
    try:
        z = np.load(path)
        ok = ("meta_node" in z.files) and ("layer_0" in z.files)
        z.close()
        return ok
    except Exception:
        return False


def main():
    outdir = os.environ.get("OUTDIR", "runs/v2")
    nsub = int(os.environ.get("SUBSAMPLE", "24000"))
    ctx_lo = int(os.environ.get("CTX_LO", "1000"))
    ctx_hi = int(os.environ.get("CTX_HI", "2000"))
    base = replace(get_config("gemma_qwen"),
                   n_walks=int(os.environ.get("NWALKS", "100")),
                   walk_length=int(os.environ.get("WLEN", "2000")),
                   device=os.environ.get("DEVICE", "cuda"),
                   dtype=os.environ.get("DTYPE", "bfloat16"))
    mfilter = os.environ.get("MODELS_FILTER")
    models = [m for m in MODELS if not mfilter or m[0] in set(mfilter.split(","))]

    for tag, hf in models:
        todo = [(g, kw) for g, kw in GRAPHS
                if not npz_ok(f"{outdir}/{g}/{tag}_acts_sub.npz")]
        if not todo:
            print(f"[{tag}] all graphs already captured, skipping", flush=True)
            continue
        print(f"[{tag}] loading {hf}  (todo: {[g for g, _ in todo]})", flush=True)
        model, tok = load_with_fallback(tag, hf, base, M)
        n_layers = model.config.num_hidden_layers
        for gname, gkw in todo:
            cfg = replace(base, name=gname, **gkw)
            graph = G.build_graph(cfg)
            walks = G.generate_walks(graph, cfg)
            run_dir = f"{outdir}/{gname}"
            os.makedirs(run_dir, exist_ok=True)
            n_occ = sum(len(w.nodes) for w in walks)
            print(f"[{tag}/{gname}] {cfg.n_nodes} nodes, {cfg.n_walks} walks x "
                  f"{cfg.walk_length} = {n_occ:,} occ, {n_layers} layers", flush=True)

            cap = M.capture(model, tok, walks, tuple(range(n_layers)), cfg)
            node = cap.meta["node"]; cl = cap.meta["context_length"]
            n = graph.n_nodes; iu = np.triu_indices(n, 1)
            GD = graph.distance_matrix()[iu]

            # grid RSA per layer over the v2 analysis window [ctx_lo, ctx_hi]
            win = (cl >= ctx_lo) & (cl <= ctx_hi)
            def rsa_at(L):
                X = cap.acts[L].astype(np.float32)
                H = np.stack([X[win & (node == k)].mean(0) for k in range(n)])
                R = np.linalg.norm(H[:, None] - H[None], axis=2)[iu]
                return spearman(R, GD)
            grid_rsa = {int(L): rsa_at(L) for L in sorted(cap.acts)}
            best = max(grid_rsa, key=grid_rsa.get)
            json.dump({"n_nodes": n, "walk_length": cfg.walk_length,
                       "ctx_window": [ctx_lo, ctx_hi], "grid_rsa": grid_rsa},
                      open(f"{run_dir}/{tag}_analysis.json", "w"), indent=2)

            # subsample occurrences across the FULL context range (so per-occ
            # plots and context-RSA have density at every context length)
            rng = np.random.default_rng(0)
            sidx = np.sort(rng.choice(node.shape[0], min(nsub, node.shape[0]), replace=False))
            layers = sorted(cap.acts)
            out_npz = f"{run_dir}/{tag}_acts_sub.npz"
            tmp = f"{run_dir}/{tag}_acts_sub.tmp.npz"   # atomic: write tmp -> rename
            np.savez(tmp,
                     **{f"layer_{L}": cap.acts[L][sidx] for L in layers},
                     **{f"meta_{k}": v[sidx] for k, v in cap.meta.items()},
                     _layers=np.array(layers), _hidden_size=np.array([cap.hidden_size]))
            os.replace(tmp, out_npz)
            print(f"[{tag}/{gname}] grid RSA peak L{best}={grid_rsa[best]:+.3f} "
                  f"-> {run_dir}/{tag}_acts_sub.npz", flush=True)
            del cap; gc.collect()
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
        del model, tok; gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if os.environ.get("FREE_WEIGHTS"):
            free_weights([hf, MIRRORS.get(tag)])
            print(f"[{tag}] freed cached weights", flush=True)
        print(f"[{tag}] DONE all graphs", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
