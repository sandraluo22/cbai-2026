"""Ablate BOTH greedy head sets (coord xy UNION parity) and check whether behaviour collapses fully.
Conditions: clean | coord-only(K) | parity-only(K) | BOTH (union) | random-K2 control (same #heads as
union, drawn from heads in neither set). Also a progressive-union curve: step k ablates coord[:k] U
parity[:k]. Metrics: neighbour_validity(=accuracy), neighbour_mass, parity_validity, parity_mass, plus
the chance floors (neighbour: mean_degree/(n-1); parity: (#opposite)/(n-1)).

Env: GEN_MODEL(Llama) GRAPH(square_grid) NWALKS(16) CTXLO(100) GREEDYJSON OUTDIR DEVICE SEED(0)
Out: <OUTDIR>/ghb_both_<model>_<graph>.json
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

ALLSPEC = {"Llama": ("meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"),
           "Gemma": ("google/gemma-2-9b", "unsloth/gemma-2-9b"), "Qwen": ("Qwen/Qwen3-8B-Base", None)}
GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
GKW = {"square_grid": dict(graph_type="grid", grid_rows=4, grid_cols=4),
       "ring": dict(graph_type="ring", ring_size=16), "hex": dict(graph_type="hex", hex_rows=4, hex_cols=4)}
GRAPH = os.environ.get("GRAPH", "square_grid")
GS = {"square_grid": "grid"}.get(GRAPH, GRAPH)  # short graph token for filenames
NWALKS = int(os.environ.get("NWALKS", "16")); CTXLO = int(os.environ.get("CTXLO", "100"))
SEED = int(os.environ.get("SEED", "0"))
GREEDYJSON = os.environ.get("GREEDYJSON", "runs/axes/4_circuits/greedy_head_set/ghs_Llama_grid.json")
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
def behaviour(model, tok, blocks, cm, graph, cand_t, col, dev, walks, abl):
    handles = ablation_hooks(blocks, cm, dev, abl) if abl else []
    nbr_v = nbr_m = par_v = par_m = 0.0; cnt = 0
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes
            logits = model(input_ids=ids).logits[0]
            for s in range(len(nodes) - 1):
                if s + 1 < CTXLO: continue
                p = torch.softmax(logits[spans[s][-1]][cand_t].float(), 0).cpu().numpy(); p = p / p.sum()
                cur = nodes[s]; nb = graph.neighbors(cur); opp = np.where(col == -col[cur])[0]
                am = int(p.argmax())
                nbr_v += int(am in nb); nbr_m += float(p[nb].sum())
                par_v += int(am in opp); par_m += float(p[opp].sum()); cnt += 1
    finally:
        for h in handles: h.remove()
    c = max(cnt, 1)
    return {"neighbour_validity": nbr_v / c, "neighbour_mass": nbr_m / c,
            "parity_validity": par_v / c, "parity_mass": par_m / c, "n_pred": cnt}


def main():
    dev = os.environ.get("DEVICE", "cuda")
    os.makedirs(OUTDIR, exist_ok=True)
    gj = json.load(open(GREEDYJSON)); tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH], n_walks=NWALKS, walk_length=300, device=dev)
    graph = G.build_graph(cfg); n = graph.n_nodes; col = two_colour(graph)
    words = graph.words[:n]
    coord = [tuple(c["head"]) for c in gj["objectives"]["coord"]["greedy"]]
    parity = [tuple(c["head"]) for c in gj["objectives"]["parity"]["greedy"]]
    union = list(dict.fromkeys(coord + parity))                    # dedup, order-preserving

    # chance floors
    deg = np.array([len(graph.neighbors(i)) for i in range(n)], float)
    nbr_chance = float(deg.mean() / (n - 1))
    opp_counts = np.array([int((col == -col[i]).sum()) for i in range(n)], float)
    par_chance = float(opp_counts.mean() / (n - 1))

    print(f"[{tag}] loading  (union={len(union)} heads; chance nbr={nbr_chance:.2f} par={par_chance:.2f})", flush=True)
    model, tok = load_with_fallback(hf, mirror, cfg)
    cm = model.config; blocks = M._decoder_blocks(model)
    nL, nH = cm.num_hidden_layers, cm.num_attention_heads
    cand_t = torch.tensor([tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in words], device=dev)
    walks = G.generate_walks(graph, cfg)

    # random control: |union| heads drawn from heads in NEITHER greedy set
    rng = np.random.default_rng(SEED); used = set(union)
    allh = [(L, H) for L in range(nL) for H in range(nH) if (L, H) not in used]
    ridx = rng.choice(len(allh), size=len(union), replace=False)
    rand = [allh[i] for i in ridx]

    def run(abl): return behaviour(model, tok, blocks, cm, graph, cand_t, col, dev, walks, abl)

    conds = {"clean": run(None),
             "coord_only": run(by_layer(coord)),
             "parity_only": run(by_layer(parity)),
             "both_union": run(by_layer(union)),
             "random_ctrl": run(by_layer(rand))}
    for k, v in conds.items():
        print(f"[{tag}] {k:12} nbr_v={v['neighbour_validity']:.3f} nbr_m={v['neighbour_mass']:.3f} "
              f"par_v={v['parity_validity']:.3f} par_m={v['parity_mass']:.3f}", flush=True)

    # progressive union: step k ablates coord[:k] U parity[:k]; random baseline = same #heads, averaged
    NR = int(os.environ.get("NRAND", "3"))
    prog = [{"step": 0, **conds["clean"], "rand_nbr_v": conds["clean"]["neighbour_validity"],
             "rand_par_v": conds["clean"]["parity_validity"]}]
    for k in range(1, max(len(coord), len(parity)) + 1):
        S = list(dict.fromkeys(coord[:k] + parity[:k]))
        b = run(by_layer(S))
        # random control: ablate len(S) heads drawn from heads in NEITHER circuit, NR draws averaged
        rn = rp = 0.0
        for _ in range(NR):
            rsel = [allh[i] for i in rng.choice(len(allh), size=len(S), replace=False)]
            rb = run(by_layer(rsel)); rn += rb["neighbour_validity"]; rp += rb["parity_validity"]
        b2 = {"step": k, "n_heads": len(S), **b, "rand_nbr_v": rn / NR, "rand_par_v": rp / NR}
        prog.append(b2)
        print(f"[{tag}] progU k={k} ({len(S)}h): nbr_v={b['neighbour_validity']:.3f} "
              f"par_v={b['parity_validity']:.3f} | rand_nbr={rn/NR:.3f} rand_par={rp/NR:.3f}", flush=True)

    out = {"model": tag, "graph": GRAPH, "n_union": len(union),
           "union_heads": [list(h) for h in union], "random_heads": [list(h) for h in rand],
           "chance": {"neighbour": nbr_chance, "parity": par_chance},
           "conditions": conds, "progressive_union": prog}
    del model, tok; gc.collect()
    if torch and torch.cuda.is_available(): torch.cuda.empty_cache()
    p = f"{OUTDIR}/ghb_both_{tag}_{GS}.json"
    json.dump(out, open(p, "w"), indent=2); print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
