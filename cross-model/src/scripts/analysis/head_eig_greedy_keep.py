"""(1) For each CRITICAL normalized-Laplacian eigenmode, greedily select the head set whose ablation
most damages that mode's power (candidates shortlisted from head_eig_sweep; objective = the NORMALIZED
mode's power, so basis mismatch is harmless). (2) KEEP-ONLY test: ablate every head EXCEPT the union
of those greedy sets and measure downstream behaviour, vs matched-size random-keep and keep-none.

Env: GEN_MODEL(Llama) GRAPH(square_grid) NWALKS(16) CTXLO(100) CRIT(2,11,14) POOL(20) K(6) NRAND(3)
     HEIGJSON OUTDIR DEVICE
Out: <OUTDIR>/head_eig_greedy_keep_<model>_<G>.json
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
GS = {"square_grid": "grid"}.get(GRAPH, GRAPH)
NWALKS = int(os.environ.get("NWALKS", "16")); CTXLO = int(os.environ.get("CTXLO", "100"))
CRIT = [int(x) for x in os.environ.get("CRIT", "2,11,14").split(",")]     # gma normalized-mode indices
POOL = int(os.environ.get("POOL", "20")); K = int(os.environ.get("K", "6")); NRAND = int(os.environ.get("NRAND", "3"))
HEIGJSON = os.environ.get("HEIGJSON", "runs/axes/4_circuits/head_eig_sweep/head_eig_sweep_Llama_grid.json")
OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/head_eig_greedy_keep")
LSTAR = int(os.environ.get("LSTAR", "31"))


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
def power_at(model, tok, blocks, cm, graph, dev, walks, n, L, cuts, abl):
    """recompute node-means at layer L under ablation `abl`; return power fraction along each cut."""
    grabbed = {}
    def hh(_m, _i, out): grabbed[0] = (out[0] if isinstance(out, tuple) else out).detach()
    cap = blocks[L].register_forward_hook(hh)
    handles = ablation_hooks(blocks, cm, dev, abl) if abl else []
    nsum = np.zeros((n, cm.hidden_size)); ncnt = np.zeros(n)
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; grabbed.clear(); model(input_ids=ids)
            single = [t[-1] for t in spans]; cl = np.arange(1, len(nodes) + 1)
            rows = grabbed[0][0][single].float().cpu().numpy()
            for s in range(len(nodes)):
                if cl[s] >= CTXLO: nsum[nodes[s]] += rows[s]; ncnt[nodes[s]] += 1
    finally:
        cap.remove()
        for h in handles: h.remove()
    Hc = nsum / np.maximum(ncnt, 1)[:, None]; Hc = Hc - Hc.mean(0); tot = (Hc ** 2).sum() + 1e-12
    return {k: float(((Hc.T @ u) ** 2).sum() / tot) for k, u in cuts.items()}


@torch.no_grad()
def behaviour(model, tok, blocks, cm, graph, cand_t, col, dev, walks, abl):
    handles = ablation_hooks(blocks, cm, dev, abl) if abl else []
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
    A = np.zeros((n, n))
    for a in range(n):
        for b in graph.adjacency[a]: A[a, b] = 1.0
    d = A.sum(1); di = 1 / np.sqrt(d); Ln = np.eye(n) - di[:, None] * A * di[None, :]
    w, U = np.linalg.eigh(Ln)
    cuts = {str(k): unit(U[:, k]) for k in CRIT}

    # candidate shortlist: union of top-POOL heads across ALL head_eig_sweep modes (basis-agnostic)
    he = json.load(open(HEIGJSON)); D = np.array(he["damage"])   # (15, nL, nH) unnormalized-mode damage
    cand = set()
    for m in range(D.shape[0]):
        idx = np.dstack(np.unravel_index(np.argsort(D[m], axis=None)[::-1], D[m].shape))[0][:POOL]
        for l, h in idx: cand.add((int(l), int(h)))
    cand = sorted(cand)
    deg = np.array([len(graph.neighbors(i)) for i in range(n)], float)
    nbr_chance = float(deg.mean() / (n - 1))
    par_chance = float(np.array([int((col == -col[i]).sum()) for i in range(n)]).mean() / (n - 1))
    print(f"[{tag}] loading ({len(cand)} candidate heads; chance nbr={nbr_chance:.2f})", flush=True)
    model, tok = load_with_fallback(hf, mirror, cfg)
    cm = model.config; blocks = M._decoder_blocks(model); nL, nH = cm.num_hidden_layers, cm.num_attention_heads
    cand_t = torch.tensor([tok(" " + graph.words[i], add_special_tokens=False)["input_ids"][0] for i in range(n)], device=dev)
    walks = G.generate_walks(graph, cfg)

    clean_pw = power_at(model, tok, blocks, cm, graph, dev, walks, n, LSTAR, cuts, None)
    clean_beh = behaviour(model, tok, blocks, cm, graph, cand_t, col, dev, walks, None)
    print(f"[{tag}] clean powers {clean_pw}  beh {clean_beh}", flush=True)

    # ---- (1) greedy head set per critical mode (maximize power damage to that mode) ----
    greedy = {}
    for k in CRIT:
        key = str(k); S = []; curve = []
        for step in range(K):
            best, bestval = None, -1e9
            for hc in cand:
                if hc in S: continue
                pw = power_at(model, tok, blocks, cm, graph, dev, walks, n, LSTAR, {key: cuts[key]}, by_layer(S + [hc]))
                dmg = clean_pw[key] - pw[key]
                if dmg > bestval: bestval, best = dmg, hc
            S.append(best); curve.append({"head": list(best), "cum_damage": float(bestval),
                                          "cum_frac": float(bestval / (clean_pw[key] + 1e-9))})
            print(f"[{tag}] mode{k} step{step+1}: +L{best[0]}H{best[1]} dmg {bestval:.3f} ({bestval/clean_pw[key]*100:.0f}%)", flush=True)
        greedy[key] = {"lambda": float(w[k]), "greedy": curve, "heads": [c["head"] for c in curve]}

    M_union = sorted({tuple(h) for k in CRIT for h in greedy[str(k)]["heads"]})
    print(f"[{tag}] union head set M = {len(M_union)} heads: {M_union}", flush=True)

    # ---- (2) KEEP-ONLY: ablate all heads EXCEPT M; controls = keep random |M|, keep none ----
    allh = [(L, H) for L in range(nL) for H in range(nH)]
    def keep_only(keepset):
        compl = [h for h in allh if h not in set(keepset)]
        return behaviour(model, tok, blocks, cm, graph, cand_t, col, dev, walks, by_layer(compl))
    keep_M = keep_only(M_union)
    rng = np.random.default_rng(0); kr_n = kr_p = 0.0
    for _ in range(NRAND):
        rk = [allh[i] for i in rng.choice(len(allh), size=len(M_union), replace=False)]
        b = keep_only(rk); kr_n += b["neighbour_validity"]; kr_p += b["parity_validity"]
    keep_rand = {"neighbour_validity": kr_n / NRAND, "parity_validity": kr_p / NRAND}
    keep_none = keep_only([])
    print(f"[{tag}] KEEP-ONLY  M={keep_M}  rand={keep_rand}  none={keep_none}", flush=True)

    out = {"model": tag, "graph": GRAPH, "crit_modes": CRIT, "Lstar": LSTAR,
           "chance": {"neighbour": nbr_chance, "parity": par_chance},
           "clean_power": clean_pw, "clean_behaviour": clean_beh,
           "greedy_per_mode": greedy, "union_heads": [list(h) for h in M_union],
           "keep_only": {"M": keep_M, "random": keep_rand, "none": keep_none, "clean": clean_beh}}
    del model, tok; gc.collect()
    if torch and torch.cuda.is_available(): torch.cuda.empty_cache()
    p = f"{OUTDIR}/head_eig_greedy_keep_{tag}_{GS}.json"
    json.dump(out, open(p, "w"), indent=2); print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
