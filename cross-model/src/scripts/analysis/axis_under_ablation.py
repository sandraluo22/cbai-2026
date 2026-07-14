"""What do the QK/induction and DLA head-group ablations do to the DIVIDER AXES?

divider_basis found the node representation = x + y + parity (near-orthogonal). Here we recompute
the power in each of those cuts while ablating a head group, to see whether a circuit selectively
builds a particular axis:
  - does ablating induction/QK heads collapse the coordinate axes (x, y)?
  - does ablating DLA writer heads collapse parity (the biggest, most causal cut)?

For each condition (clean | ablate_induction | ablate_dla | ablate_random) we zero the head group's
slice into o_proj at ALL positions, recompute per-(node,layer) mean residuals over walks, and report
the fraction of node-mean variance captured by x, y, parity at each layer (+ the peak-over-layers).
Also saves the ablated node-means so the full spectrum / 3-D geometry can be rebuilt offline.

Env: PRESET GEN_MODEL(Llama) GRAPH(square_grid) NWALKS(60) WLEN(300) CTXLO(100) KGROUP(15)
     INDJSON DLAJSON OUTDIR DEVICE
Out: <OUTDIR>/axis_under_ablation_<graph>.json  + nodemeans_ablated_<cond>_<graph>.npz
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
           "Gemma": ("google/gemma-2-9b", "unsloth/gemma-2-9b"),
           "Qwen": ("Qwen/Qwen3-8B-Base", None), "distilgpt2": ("distilgpt2", None)}
GEN_MODEL = os.environ.get("GEN_MODEL", "Llama" if PRESET != "smoke" else "distilgpt2")
GKW = {"square_grid": dict(graph_type="grid", grid_rows=4, grid_cols=4),
       "ring": dict(graph_type="ring", ring_size=16), "hex": dict(graph_type="hex", hex_rows=4, hex_cols=4)}
GRAPH = os.environ.get("GRAPH", "square_grid")
NWALKS = int(os.environ.get("NWALKS", "60")); WLEN = int(os.environ.get("WLEN", "300"))
CTXLO = int(os.environ.get("CTXLO", "100")); KGROUP = int(os.environ.get("KGROUP", "15"))
INDJSON = os.environ.get("INDJSON", "runs/induction-head/induction.json")
DLAJSON = os.environ.get("DLAJSON", "runs/induction-head/attribution/head_attribution_square_grid.json")
OUTDIR = os.environ.get("OUTDIR", "runs/axes/3_causal/axis_under_ablation")
RNG = np.random.default_rng(0)


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
    for l, h in heads: d.setdefault(l, []).append(h)
    return d


def select_groups(ind, dla, tag, nL, nH):
    gen = np.array(ind.get(tag, {}).get("generic", np.zeros((nL, nH))))
    att = np.array(dla.get(tag, {}).get("head_attr", np.zeros((nL, nH))))
    def topk(mat):
        order = np.argsort(mat, axis=None)[::-1]
        return [(int(i // nH), int(i % nH)) for i in order][:KGROUP]
    induction = topk(gen); writers = topk(att); used = set(induction) | set(writers)
    pool = [(l, h) for l in range(nL) for h in range(nH) if (l, h) not in used]
    rand = [pool[i] for i in RNG.choice(len(pool), min(KGROUP, len(pool)), replace=False)]
    return {"induction": induction, "dla": writers, "random": rand}


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
def node_means(model, tok, blocks, cm, walks, dev, n, abl):
    grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    nL = cm.num_hidden_layers
    caps = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
    handles = ablation_hooks(blocks, cm, dev, abl) if abl else []
    nsum = {L: np.zeros((n, cm.hidden_size)) for L in range(nL)}; ncnt = np.zeros(n)
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; grabbed.clear(); model(input_ids=ids)
            single = [t[-1] for t in spans]; cl = np.arange(1, len(nodes) + 1)
            for L in range(nL):
                rows = grabbed[L][0][single].float().cpu().numpy()
                for s in range(len(nodes)):
                    if cl[s] >= CTXLO:
                        nsum[L][nodes[s]] += rows[s]
                        if L == 0: ncnt[nodes[s]] += 1
    finally:
        for h in caps: h.remove()
        for h in handles: h.remove()
    cn = np.maximum(ncnt, 1)
    return {L: nsum[L] / cn[:, None] for L in range(nL)}


def axis_power(means, cuts, nL):
    out = {c: np.zeros(nL) for c in cuts}
    for L in range(nL):
        H = means[L]; Hc = H - H.mean(0); tot = (Hc ** 2).sum() + 1e-12
        for c, u in cuts.items():
            v = Hc.T @ u; out[c][L] = (v ** 2).sum() / tot
    return out


def main():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    os.makedirs(OUTDIR, exist_ok=True)
    ind = json.load(open(INDJSON))["models"] if os.path.exists(INDJSON) else {}
    dla = json.load(open(DLAJSON))["models"] if os.path.exists(DLAJSON) else {}
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH], n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg); n = graph.n_nodes; coords = np.array(graph.coords, float)
    cuts = {"x": unit(coords[:, 0]), "y": unit(coords[:, 1])}
    par = two_colour(graph)
    if not np.allclose(par, par[0]): cuts["parity"] = unit(par)
    print(f"[{tag}] loading", flush=True)
    model, tok = load_with_fallback(hf, mirror, cfg)
    cm = model.config; blocks = M._decoder_blocks(model)
    nL = cm.num_hidden_layers; nH = getattr(cm, "num_attention_heads", None) or cm.n_head
    walks = G.generate_walks(graph, cfg)
    groups = select_groups(ind, dla, tag, nL, nH)
    conds = {"clean": None, "ablate_induction": by_layer(groups["induction"]),
             "ablate_dla": by_layer(groups["dla"]), "ablate_random": by_layer(groups["random"])}
    out = {"graph": GRAPH, "model": tag, "nL": nL, "kgroup": KGROUP,
           "groups": {k: [list(t) for t in v] for k, v in groups.items()}, "conds": {}}
    for cname, abl in conds.items():
        means = node_means(model, tok, blocks, cm, walks, dev, n, abl)
        ap = axis_power(means, cuts, nL)
        out["conds"][cname] = {c: ap[c].tolist() for c in cuts}
        out["conds"][cname]["_peak"] = {c: float(np.max(ap[c])) for c in cuts}
        np.savez_compressed(f"{OUTDIR}/nodemeans_ablated_{cname}_{GRAPH}.npz",
                            **{f"layer_{L}": means[L].astype(np.float16) for L in range(nL)},
                            adjacency=np.array([[1 if j in graph.adjacency[i] else 0 for j in range(n)] for i in range(n)], np.int8),
                            coords=coords, rows=np.array([GKW[GRAPH].get("grid_rows", 0)]),
                            cols=np.array([GKW[GRAPH].get("grid_cols", 0)]))
        pk = out["conds"][cname]["_peak"]
        print(f"[{tag}/{GRAPH}/{cname}] peak axis power  x={pk['x']:.3f} y={pk['y']:.3f} "
              f"parity={pk.get('parity', float('nan')):.3f}", flush=True)
    del model, tok; gc.collect()
    if torch and torch.cuda.is_available(): torch.cuda.empty_cache()
    json.dump(out, open(f"{OUTDIR}/axis_under_ablation_{GRAPH}.json", "w"), indent=2)
    print(f"DONE -> {OUTDIR}/axis_under_ablation_{GRAPH}.json", flush=True)


if __name__ == "__main__":
    main()
