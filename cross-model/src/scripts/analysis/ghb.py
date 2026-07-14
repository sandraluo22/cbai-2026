"""Downstream BEHAVIOUR along each greedy head set. For the coord(xy) and parity greedy sets in
ghs_<model>_<graph>.json, re-ablate the CUMULATIVE head set at each greedy step and
measure next-node behaviour (not just representation power):
  neighbour_validity  argmax next-token is a true graph neighbour (== next-node accuracy)
  neighbour_mass      softmax mass on the current node's true neighbours
  parity_validity     argmax next-token has the OPPOSITE two-colour to the current node
  parity_mass         softmax mass on all opposite-parity nodes
Step 0 = no ablation (clean baseline). Predictions restricted to context positions >= CTXLO,
scored over the 16 node-word candidate tokens.

Env: GEN_MODEL(Llama) GRAPH(square_grid) NWALKS(16) CTXLO(100) TEMP(1.0)
     GREEDYJSON OUTDIR DEVICE
Out: <OUTDIR>/ghb_<model>_<graph>.json
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
TEMP = float(os.environ.get("TEMP", "1.0"))
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
    """One full pass over walks with head set `abl` ablated -> behaviour metrics."""
    handles = ablation_hooks(blocks, cm, dev, abl) if abl else []
    nbr_v = nbr_m = par_v = par_m = 0.0; cnt = 0
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes
            logits = model(input_ids=ids).logits[0]
            for s in range(len(nodes) - 1):
                if s + 1 < CTXLO: continue
                p = torch.softmax(logits[spans[s][-1]][cand_t].float() / TEMP, 0).cpu().numpy()
                p = p / p.sum()
                cur = nodes[s]; nb = graph.neighbors(cur)
                opp = np.where(col == -col[cur])[0]
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
    print(f"[{tag}] loading", flush=True)
    model, tok = load_with_fallback(hf, mirror, cfg)
    cm = model.config; blocks = M._decoder_blocks(model)
    cand_t = torch.tensor([tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in words], device=dev)
    walks = G.generate_walks(graph, cfg)

    base = behaviour(model, tok, blocks, cm, graph, cand_t, col, dev, walks, None)
    print(f"[{tag}] clean  nbr_v={base['neighbour_validity']:.3f} par_v={base['parity_validity']:.3f}", flush=True)

    out = {"model": tag, "graph": GRAPH, "ctxlo": CTXLO, "clean": base, "objectives": {}}
    for oname in ("coord", "parity"):
        greedy = gj["objectives"][oname]["greedy"]
        steps = [{"step": 0, "heads": [], "head_added": None, **base}]
        S = []
        for k, c in enumerate(greedy, 1):
            S.append(tuple(c["head"]))
            b = behaviour(model, tok, blocks, cm, graph, cand_t, col, dev, walks, by_layer(S))
            steps.append({"step": k, "heads": [list(h) for h in S], "head_added": list(c["head"]), **b})
            print(f"[{tag}/{oname}] step {k} +L{c['head'][0]}H{c['head'][1]}: "
                  f"nbr_v={b['neighbour_validity']:.3f} nbr_m={b['neighbour_mass']:.3f} "
                  f"par_v={b['parity_validity']:.3f} par_m={b['parity_mass']:.3f}", flush=True)
        out["objectives"][oname] = steps
    del model, tok; gc.collect()
    if torch and torch.cuda.is_available(): torch.cuda.empty_cache()
    p = f"{OUTDIR}/ghb_{tag}_{GS}.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
