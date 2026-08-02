"""Minimum-viable-circuit search with MEAN-ablation (heads only; MLPs kept clean throughout).
(1) mean keep-only M vs the old zero keep-only; (2) additive greedy RESTORE from all-heads-mean-ablated,
adding the head that most recovers neighbour validity until a threshold; (3) named keep-only sets:
M, M+DLA, M+induction, M+DLA+induction. Mean = per-layer mean of the o_proj input over all positions.

Env: GEN_MODEL(Llama) GRAPH(square_grid) NWALKS(16) CTXLO(100) K(18) THRESH(0.9)
     MJSON DLAJSON INDJSON HEIGJSON OUTDIR DEVICE
Out: <OUTDIR>/mean_circuit_<model>_<G>.json
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
GRAPH = os.environ.get("GRAPH", "square_grid"); GS = {"square_grid": "grid"}.get(GRAPH, GRAPH)
NWALKS = int(os.environ.get("NWALKS", "16")); CTXLO = int(os.environ.get("CTXLO", "100"))
K = int(os.environ.get("K", "18")); THRESH = float(os.environ.get("THRESH", "0.9"))
MJSON = os.environ.get("MJSON", "runs/axes/4_circuits/head_eig_greedy_keep/head_eig_greedy_keep_Llama_grid.json")
DLAJSON = os.environ.get("DLAJSON", "runs/induction-head/1_circuits/attribution/head_attribution_square_grid.json")
INDJSON = os.environ.get("INDJSON", "runs/induction-head/1_circuits/induction_heads/induction.json")
HEIGJSON = os.environ.get("HEIGJSON", "runs/axes/4_circuits/head_eig_sweep/head_eig_sweep_Llama_grid.json")
OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/mean_circuit")
NDLA = int(os.environ.get("NDLA", "10")); NIND = int(os.environ.get("NIND", "5")); POOL = int(os.environ.get("POOL", "12"))


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
    """per-layer mean of the o_proj INPUT (concatenated head outputs) over all positions."""
    nL = cm.num_hidden_layers; sums = [None] * nL; cnt = [0] * nL; hooks = []
    def mk(L):
        def pre(_m, args):
            x = args[0]; s = x.reshape(-1, x.shape[-1]).sum(0).float().cpu().numpy()
            sums[L] = s if sums[L] is None else sums[L] + s; cnt[L] += x.reshape(-1, x.shape[-1]).shape[0]
        return pre
    for L in range(nL): hooks.append(attn_proj(blocks[L], cm)[0].register_forward_pre_hook(mk(L)))
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            model(input_ids=ids)
    finally:
        for h in hooks: h.remove()
    return [sums[L] / max(cnt[L], 1) for L in range(nL)]


def mean_ablate_hooks(blocks, cm, dev, by_layer_map, hmean):
    handles = []
    for L, heads in by_layer_map.items():
        proj, hd = attn_proj(blocks[L], cm)
        ct = np.concatenate([np.arange(h * hd, (h + 1) * hd) for h in heads])
        cti = torch.tensor(ct, device=dev, dtype=torch.long)
        mv = torch.tensor(hmean[L][ct], device=dev)
        def pre(_m, args, cti=cti, mv=mv):
            x = args[0].clone(); x[..., cti] = mv.to(x.dtype)
            return (x,) + tuple(args[1:])
        handles.append(proj.register_forward_pre_hook(pre))
    return handles


@torch.no_grad()
def behaviour(model, tok, blocks, cm, graph, cand_t, col, dev, walks, abl_by_layer, hmean):
    handles = mean_ablate_hooks(blocks, cm, dev, abl_by_layer, hmean) if abl_by_layer else []
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
    cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH], n_walks=NWALKS, walk_length=300, device=dev)
    graph = G.build_graph(cfg); n = graph.n_nodes; col = two_colour(graph)
    # head sets
    Mheads = [tuple(h) for h in json.load(open(MJSON))["union_heads"]]
    ha = np.array(json.load(open(DLAJSON))["models"][tag]["head_attr"])
    dla = [(int(l), int(h)) for l, h in np.dstack(np.unravel_index(np.argsort(ha, axis=None)[::-1], ha.shape))[0][:NDLA]]
    ind = [(int(t["layer"]), int(t["head"])) for t in json.load(open(INDJSON))["models"][tag]["top_task"][:NIND]]
    D = np.array(json.load(open(HEIGJSON))["damage"]); cand = set(Mheads) | set(dla) | set(ind)
    for m in range(D.shape[0]):
        for l, h in np.dstack(np.unravel_index(np.argsort(D[m], axis=None)[::-1], D[m].shape))[0][:POOL]:
            cand.add((int(l), int(h)))
    cand = sorted(cand)
    deg = np.array([len(graph.neighbors(i)) for i in range(n)], float)
    nbr_chance = float(deg.mean() / (n - 1)); par_chance = float(np.array([int((col == -col[i]).sum()) for i in range(n)]).mean() / (n - 1))
    print(f"[{tag}] loading (|M|={len(Mheads)} |DLA|={len(dla)} |ind|={len(ind)} |cand|={len(cand)})", flush=True)
    model, tok = load_with_fallback(hf, mirror, cfg)
    cm = model.config; blocks = M._decoder_blocks(model); nL, nH = cm.num_hidden_layers, cm.num_attention_heads
    cand_t = torch.tensor([tok(" " + graph.words[i], add_special_tokens=False)["input_ids"][0] for i in range(n)], device=dev)
    walks = G.generate_walks(graph, cfg)
    hmean = record_means(model, tok, blocks, cm, dev, walks)

    allh = [(L, H) for L in range(nL) for H in range(nH)]
    def keep_only(keep):
        ks = set(keep); return behaviour(model, tok, blocks, cm, graph, cand_t, col, dev, walks,
                                         by_layer([h for h in allh if h not in ks]), hmean)
    clean = behaviour(model, tok, blocks, cm, graph, cand_t, col, dev, walks, None, hmean)
    print(f"[{tag}] clean {clean}", flush=True)

    # (1)+(3) named keep-only sets (mean-ablated complement)
    named = {"clean": clean, "keep_M": keep_only(Mheads), "keep_M+DLA": keep_only(set(Mheads) | set(dla)),
             "keep_M+ind": keep_only(set(Mheads) | set(ind)),
             "keep_M+DLA+ind": keep_only(set(Mheads) | set(dla) | set(ind)),
             "keep_ind_only": keep_only(ind), "keep_DLA_only": keep_only(dla),
             "keep_ind+DLA": keep_only(set(ind) | set(dla)),
             "keep_none": keep_only([])}
    for k, v in named.items():
        print(f"[{tag}] {k:16s} nbr_v={v['neighbour_validity']:.3f} par_v={v['parity_validity']:.3f}", flush=True)

    # (2) additive greedy RESTORE from all-mean-ablated
    restored = []; curve = [{"step": 0, "head": None, **named["keep_none"]}]
    for step in range(1, K + 1):
        best, bestv, bestm = None, -1e9, None
        for hc in cand:
            if hc in restored: continue
            met = keep_only(restored + [hc])
            if met["neighbour_validity"] > bestv: bestv, best, bestm = met["neighbour_validity"], hc, met
        restored.append(best); curve.append({"step": step, "head": list(best), **bestm})
        print(f"[{tag}] restore {step}: +L{best[0]}H{best[1]}  nbr_v={bestm['neighbour_validity']:.3f} par_v={bestm['parity_validity']:.3f}", flush=True)
        if bestv >= THRESH * clean["neighbour_validity"]:
            print(f"[{tag}] reached {THRESH:.0%} of clean at step {step}", flush=True); break

    out = {"model": tag, "graph": GRAPH, "chance": {"neighbour": nbr_chance, "parity": par_chance},
           "M": [list(h) for h in Mheads], "DLA": [list(h) for h in dla], "induction": [list(h) for h in ind],
           "named": named, "restore": curve, "restored_set": [list(h) for h in restored]}
    del model, tok; gc.collect()
    if torch and torch.cuda.is_available(): torch.cuda.empty_cache()
    p = f"{OUTDIR}/mean_circuit_{tag}_{GS}.json"
    json.dump(out, open(p, "w"), indent=2); print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
