"""ALL 16 combinations of the four head entities {coord, parity, QK(induction), DLA}: mean-ablation
keep-only (keep the union of the selected entities, mean-ablate every other head; MLPs kept clean),
measuring neighbour & parity validity. coord/parity = greedy_head_set circuits; QK = induction heads;
DLA = direct-logit-attribution heads.

Env: GEN_MODEL(Llama) GRAPH(square_grid) NWALKS(16) CTXLO(100) GHSJSON DLAJSON INDJSON OUTDIR DEVICE
Out: <OUTDIR>/mean_circuit_combos_<model>_<G>.json
"""
from __future__ import annotations
import os, json, gc, itertools
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
GKW = {"square_grid": dict(graph_type="grid", grid_rows=4, grid_cols=4)}
GRAPH = os.environ.get("GRAPH", "square_grid"); GS = {"square_grid": "grid"}.get(GRAPH, GRAPH)
NWALKS = int(os.environ.get("NWALKS", "16")); CTXLO = int(os.environ.get("CTXLO", "100"))
GHSJSON = os.environ.get("GHSJSON", "runs/axes/4_circuits/greedy_head_set/greedy_head_set_Llama_square_grid.json")
DLAJSON = os.environ.get("DLAJSON", "runs/induction-head/1_circuits/attribution/head_attribution_square_grid.json")
INDJSON = os.environ.get("INDJSON", "runs/induction-head/1_circuits/induction_heads/induction.json")
OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/mean_circuit")
NDLA = int(os.environ.get("NDLA", "10")); NIND = int(os.environ.get("NIND", "5")); KHEAD = int(os.environ.get("KHEAD", "8"))


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


@torch.no_grad()
def record_means(model, tok, blocks, cm, dev, walks):
    nL = cm.num_hidden_layers; sums = [None] * nL; cnt = [0] * nL; hooks = []
    def mk(L):
        def pre(_m, args):
            x = args[0]; s = x.reshape(-1, x.shape[-1]).sum(0).float().cpu().numpy()
            sums[L] = s if sums[L] is None else sums[L] + s; cnt[L] += x.reshape(-1, x.shape[-1]).shape[0]
        return pre
    for L in range(nL): hooks.append(attn_proj(blocks[L], cm)[0].register_forward_pre_hook(mk(L)))
    try:
        for wk in walks:
            model(input_ids=tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev))
    finally:
        for h in hooks: h.remove()
    return [sums[L] / max(cnt[L], 1) for L in range(nL)]


def mean_ablate_hooks(blocks, cm, dev, by_layer_map, hmean):
    handles = []
    for L, heads in by_layer_map.items():
        proj, hd = attn_proj(blocks[L], cm)
        ct = np.concatenate([np.arange(h * hd, (h + 1) * hd) for h in heads])
        cti = torch.tensor(ct, device=dev, dtype=torch.long); mv = torch.tensor(hmean[L][ct], device=dev)
        def pre(_m, args, cti=cti, mv=mv):
            x = args[0].clone(); x[..., cti] = mv.to(x.dtype)
            return (x,) + tuple(args[1:])
        handles.append(proj.register_forward_pre_hook(pre))
    return handles


@torch.no_grad()
def behaviour(model, tok, blocks, cm, graph, cand_t, col, dev, walks, abl, hmean):
    handles = mean_ablate_hooks(blocks, cm, dev, abl, hmean) if abl else []
    nbr_v = par_v = 0.0; cnt = 0
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes
            logits = model(input_ids=ids).logits[0]
            for s in range(len(nodes) - 1):
                if s + 1 < CTXLO: continue
                p = torch.softmax(logits[spans[s][-1]][cand_t].float(), 0).cpu().numpy(); p = p / p.sum()
                cur = nodes[s]; nb = graph.neighbors(cur); opp = np.where(col == -col[cur])[0]; am = int(p.argmax())
                nbr_v += int(am in nb); par_v += int(am in opp); cnt += 1
    finally:
        for h in handles: h.remove()
    c = max(cnt, 1)
    return {"neighbour_validity": nbr_v / c, "parity_validity": par_v / c}


def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH], n_walks=NWALKS, walk_length=int(os.environ.get("WLEN","300")), device=dev)
    graph = G.build_graph(cfg); n = graph.n_nodes; col = two_colour(graph)
    gj = json.load(open(GHSJSON))
    ENT = {
        "coord": [tuple(c["head"]) for c in gj["objectives"]["coord"]["greedy"][:KHEAD]],
        "parity": [tuple(c["head"]) for c in gj["objectives"]["parity"]["greedy"][:KHEAD]],
        "QK": [(int(t["layer"]), int(t["head"])) for t in json.load(open(INDJSON))["models"][tag]["top_task"][:NIND]],
    }
    ha = np.array(json.load(open(DLAJSON))["models"][tag]["head_attr"])
    ENT["DLA"] = [(int(l), int(h)) for l, h in np.dstack(np.unravel_index(np.argsort(ha, axis=None)[::-1], ha.shape))[0][:NDLA]]
    names = ["coord", "parity", "QK", "DLA"]
    deg = np.array([len(graph.neighbors(i)) for i in range(n)], float)
    nbr_chance = float(deg.mean() / (n - 1)); par_chance = float(np.array([int((col == -col[i]).sum()) for i in range(n)]).mean() / (n - 1))
    print(f"[{tag}] sizes " + " ".join(f"{k}={len(v)}" for k, v in ENT.items()), flush=True)
    model, tok = load_with_fallback(hf, mirror, cfg)
    cm = model.config; blocks = M._decoder_blocks(model); nL, nH = cm.num_hidden_layers, cm.num_attention_heads
    cand_t = torch.tensor([tok(" " + graph.words[i], add_special_tokens=False)["input_ids"][0] for i in range(n)], device=dev)
    walks = G.generate_walks(graph, cfg); hmean = record_means(model, tok, blocks, cm, dev, walks)
    allh = [(L, H) for L in range(nL) for H in range(nH)]
    clean = behaviour(model, tok, blocks, cm, graph, cand_t, col, dev, walks, None, hmean)

    combos = {}
    for r in range(len(names) + 1):
        for sub in itertools.combinations(names, r):
            keep = set().union(*[ENT[s] for s in sub]) if sub else set()
            met = behaviour(model, tok, blocks, cm, graph, cand_t, col, dev, walks,
                            by_layer([h for h in allh if h not in keep]), hmean)
            key = "+".join(sub) if sub else "none"
            combos[key] = {"groups": list(sub), "n_heads": len(keep), **met}
            print(f"[{tag}] {key:22s} ({len(keep):2d}h) nbr_v={met['neighbour_validity']:.3f} par_v={met['parity_validity']:.3f}", flush=True)

    out = {"model": tag, "graph": GRAPH, "chance": {"neighbour": nbr_chance, "parity": par_chance},
           "entities": {k: [list(h) for h in v] for k, v in ENT.items()}, "clean": clean, "combos": combos}
    del model, tok; gc.collect()
    if torch and torch.cuda.is_available(): torch.cuda.empty_cache()
    p = f"{OUTDIR}/mean_circuit_combos_{tag}_{GS}.json"
    json.dump(out, open(p, "w"), indent=2); print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
