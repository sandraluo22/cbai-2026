"""Greedy set-selection: find the set of heads whose JOINT ablation does the most damage to the
coordinate axes (x+y) and to parity -- accounting for redundancy (single-head ranking double-counts
overlapping heads). At each step, add the candidate head (from a top-POOL shortlist ranked by
single-head damage) that maximises the additional damage when ablated together with the current set.

Env: GEN_MODEL(Llama) GRAPH(square_grid) NWALKS(16) K(8) POOL(25) SWEEPJSON OUTDIR DEVICE
Out: <OUTDIR>/ghs_<model>_<graph>.json
"""
from __future__ import annotations
import os, json, gc
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
           "Gemma": ("google/gemma-2-9b", "unsloth/gemma-2-9b"), "Qwen": ("Qwen/Qwen3-8B-Base", None)}
GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
GKW = {"square_grid": dict(graph_type="grid", grid_rows=4, grid_cols=4),
       "ring": dict(graph_type="ring", ring_size=16), "hex": dict(graph_type="hex", hex_rows=4, hex_cols=4)}
GRAPH = os.environ.get("GRAPH", "square_grid")
GS = {"square_grid": "grid"}.get(GRAPH, GRAPH)  # short graph token for filenames
NWALKS = int(os.environ.get("NWALKS", "16")); CTXLO = int(os.environ.get("CTXLO", "100"))
K = int(os.environ.get("K", "8")); POOL = int(os.environ.get("POOL", "25"))
SWEEPJSON = os.environ.get("SWEEPJSON", "runs/axes/4_circuits/head_axis_sweep/head_axis_sweep_Llama_square_grid.json")
OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/greedy_head_set")


def load_with_fallback(hf, mirror, cfg):
    try: return M.load_model(hf, cfg)
    except Exception:
        if mirror is None: raise
        return M.load_model(mirror, cfg)


def two_colour(graph):
    n = graph.n_nodes; col = np.zeros(n)
    for s in range(n):
        if col[s] != 0: continue
        col[s] = 1; st = [s]
        while st:
            u = st.pop()
            for v in graph.adjacency[u]:
                if col[v] == 0: col[v] = -col[u]; st.append(v)
    return col.astype(float)


def unit(v): v = v - v.mean(); return v / (np.linalg.norm(v) + 1e-9)


def attn_proj(block, cm):
    if hasattr(block, "self_attn") and hasattr(block.self_attn, "o_proj"):
        return block.self_attn.o_proj, (getattr(cm, "head_dim", None) or cm.hidden_size // cm.num_attention_heads)
    return block.attn.c_proj, cm.hidden_size // cm.n_head


def by_layer(heads):
    d = {}
    for l, h in heads: d.setdefault(int(l), []).append(int(h))
    return d


def ablation_hooks(blocks, cm, dev, by_layer_map):
    handles = []
    for L, heads in by_layer_map.items():
        proj, hd = attn_proj(blocks[L], cm)
        ct = torch.tensor(np.concatenate([np.arange(h * hd, (h + 1) * hd) for h in heads]), device=dev, dtype=torch.long)
        def pre(_m, args, ct=ct):
            x = args[0].clone(); x[..., ct] = 0
            return (x,) + tuple(args[1:])
        handles.append(proj.register_forward_pre_hook(pre))
    return handles


@torch.no_grad()
def powers(model, tok, blocks, cm, walks, dev, n, layers, cuts, abl):
    grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    caps = [blocks[L].register_forward_hook(mk(L)) for L in layers]
    handles = ablation_hooks(blocks, cm, dev, abl) if abl else []
    nsum = {L: np.zeros((n, cm.hidden_size)) for L in layers}; ncnt = np.zeros(n)
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; grabbed.clear(); model(input_ids=ids)
            single = [t[-1] for t in spans]; cl = np.arange(1, len(nodes) + 1); first = layers[0]
            for L in layers:
                rows = grabbed[L][0][single].float().cpu().numpy()
                for s in range(len(nodes)):
                    if cl[s] >= CTXLO:
                        nsum[L][nodes[s]] += rows[s]
                        if L == first: ncnt[nodes[s]] += 1
    finally:
        for h in caps: h.remove()
        for h in handles: h.remove()
    cn = np.maximum(ncnt, 1)
    out = {}
    for name, (L, u) in cuts.items():
        H = nsum[L] / cn[:, None]; Hc = H - H.mean(0)
        out[name] = float(((Hc.T @ u) ** 2).sum() / ((Hc ** 2).sum() + 1e-12))
    return out


def topk_from(dmap, k):
    idx = np.dstack(np.unravel_index(np.argsort(dmap, axis=None)[::-1], dmap.shape))[0][:k]
    return [(int(l), int(h)) for l, h in idx]


def main():
    dev = os.environ.get("DEVICE", "cuda")
    os.makedirs(OUTDIR, exist_ok=True)
    sw = json.load(open(SWEEPJSON)); Lpk = sw["peak_layer"]
    Dx = np.array(sw["damage"]["x"]); Dy = np.array(sw["damage"]["y"]); Dp = np.array(sw["damage"]["parity"])
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH], n_walks=NWALKS, walk_length=300, device=dev)
    graph = G.build_graph(cfg); n = graph.n_nodes; coords = np.array(graph.coords, float)
    cuts = {"x": (Lpk["x"], unit(coords[:, 0])), "y": (Lpk["y"], unit(coords[:, 1])), "parity": (Lpk["parity"], unit(two_colour(graph)))}
    read_layers = sorted(set(Lpk.values()))
    print(f"[{tag}] loading", flush=True)
    model, tok = load_with_fallback(hf, mirror, cfg)
    cm = model.config; blocks = M._decoder_blocks(model)
    walks = G.generate_walks(graph, cfg)
    clean = powers(model, tok, blocks, cm, walks, dev, n, read_layers, cuts, None)
    print(f"[{tag}] clean power x={clean['x']:.3f} y={clean['y']:.3f} parity={clean['parity']:.3f}", flush=True)

    objectives = {"coord": ((Dx + Dy) / 2, lambda p: (clean["x"] - p["x"]) + (clean["y"] - p["y"]), clean["x"] + clean["y"]),
                  "parity": (Dp, lambda p: clean["parity"] - p["parity"], clean["parity"])}
    out = {"model": tag, "graph": GRAPH, "clean": clean, "K": K, "pool": POOL, "objectives": {}}
    for oname, (dmap, dmg_fn, base) in objectives.items():
        pool = topk_from(dmap, POOL)
        S = []; curve = []
        for step in range(K):
            best, bestval, bestp = None, -1e9, None
            for h in pool:
                if h in S: continue
                p = powers(model, tok, blocks, cm, walks, dev, n, read_layers, cuts, by_layer(S + [h]))
                val = dmg_fn(p)
                if val > bestval: bestval, best, bestp = val, h, p
            S.append(best); curve.append({"head": list(best), "cum_damage": float(bestval),
                                          "cum_frac": float(bestval / base), "powers": bestp})
            print(f"[{tag}/{oname}] step {step+1}: +L{best[0]}H{best[1]}  cum_damage {bestval:.3f} ({bestval/base*100:.0f}% of {base:.2f})", flush=True)
        out["objectives"][oname] = {"base": base, "greedy": curve,
                                    "naive_top": [list(t) for t in topk_from(dmap, K)]}
    del model, tok; gc.collect()
    if torch and torch.cuda.is_available(): torch.cuda.empty_cache()
    json.dump(out, open(f"{OUTDIR}/ghs_{tag}_{GS}.json", "w"), indent=2)
    print(f"DONE -> {OUTDIR}/ghs_{tag}_{GS}.json", flush=True)


if __name__ == "__main__":
    main()
