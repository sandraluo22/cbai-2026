"""Capture a SUBSAMPLE of per-occurrence residual activations (all layers) at high context, for the
3-D viewer's per-occurrence cloud. Tiny: NPTS points per layer, so we can pull it and PCA offline.

Same npz layout as the v2 acts_sub cache (layer_* + meta_node + meta_context_length), so
axis_pca_export.py reads it unchanged. Keeps the first NPTS occurrences with context >= CTXLO.

Env: PRESET TAG(Gemma|Qwen|Llama) GRAPH(square_grid) NWALKS(30) WLEN(300) CTXLO(100) NPTS(400) OUTDIR DEVICE
Out: <OUTDIR>/perocc_<TAG>_<graph>.npz
"""
from __future__ import annotations
import os, gc
from dataclasses import replace
import numpy as np

try:
    import torch
except Exception:
    torch = None

from config import get_config
import graph as G
import models as M
from models import resolve_token_spans

PRESET = os.environ.get("PRESET", "gemma_qwen")
ALLSPEC = {"Llama": ("meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"),
           "Gemma": ("google/gemma-2-9b", "unsloth/gemma-2-9b"),
           "Qwen": ("Qwen/Qwen3-8B-Base", None), "Qwen32": ("Qwen/Qwen3-32B", None), "distilgpt2": ("distilgpt2", None)}
TAG = os.environ.get("TAG", "Gemma")
GKW = {"square_grid": dict(graph_type="grid", grid_rows=4, grid_cols=4),
       "ring": dict(graph_type="ring", ring_size=16), "hex": dict(graph_type="hex", hex_rows=4, hex_cols=4)}
GRAPH = os.environ.get("GRAPH", "square_grid")
NWALKS = int(os.environ.get("NWALKS", "30")); WLEN = int(os.environ.get("WLEN", "300"))
CTXLO = int(os.environ.get("CTXLO", "100")); NPTS = int(os.environ.get("NPTS", "400"))
OUTDIR = os.environ.get("OUTDIR", "runs/axes/2_geometry/axis_geometry")


def load_with_fallback(hf, mirror, cfg):
    try: return M.load_model(hf, cfg)
    except Exception:
        if mirror is None: raise
        return M.load_model(mirror, cfg)


@torch.no_grad()
def main():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    os.makedirs(OUTDIR, exist_ok=True)
    hf, mirror = ALLSPEC[TAG]
    cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH], n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg); n = graph.n_nodes
    print(f"[{TAG}] loading", flush=True)
    model, tok = load_with_fallback(hf, mirror, cfg)
    cm = model.config; blocks = M._decoder_blocks(model); nL = cm.num_hidden_layers
    grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    hs = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
    rows = {L: [] for L in range(nL)}; mnode = []; mctx = []
    walks = G.generate_walks(graph, cfg)
    try:
        for wk in walks:
            if len(mnode) >= NPTS: break
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; grabbed.clear(); model(input_ids=ids)
            single = [t[-1] for t in spans]; cl = np.arange(1, len(nodes) + 1)
            keep = [s for s in range(len(nodes)) if cl[s] >= CTXLO]
            for L in range(nL):
                arr = grabbed[L][0][single].float().cpu().numpy()
                for s in keep:
                    if len(rows[L]) < NPTS: rows[L].append(arr[s])
            for s in keep:
                if len(mnode) < NPTS: mnode.append(nodes[s]); mctx.append(int(cl[s]))
    finally:
        for h in hs: h.remove()
    save = {f"layer_{L}": np.stack(rows[L]).astype(np.float16) for L in range(nL)}
    save["meta_node"] = np.array(mnode); save["meta_context_length"] = np.array(mctx)
    path = f"{OUTDIR}/perocc_{TAG}_{GRAPH}.npz"
    np.savez_compressed(path, **save)
    print(f"[{TAG}/{GRAPH}] kept {len(mnode)} occ x {nL} layers -> {path}", flush=True)
    del model, tok; gc.collect()
    if torch and torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
