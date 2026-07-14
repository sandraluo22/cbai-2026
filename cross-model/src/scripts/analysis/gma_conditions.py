"""Project out FIXED eigenmode sets (each in ISOLATION, not cumulative) and measure downstream
behaviour, for the grid. Conditions:
  clean | low2 {2 lowest = coords} | high1 {highest = parity} | all3 {low2 + high1}
Each also gets a matched-rank RANDOM-projection control (project out the same #random hidden dirs).
Projection is applied at EVERY layer (whole-stack), using each layer's clean-node-mean readout dir.

Env: GEN_MODEL(Llama) GRAPH(square_grid) NWALKS(16) CTXLO(100) NRAND(3) SEED(0) OUTDIR DEVICE
Out: <OUTDIR>/gma_cond_<model>_<graph>.json
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
NRAND = int(os.environ.get("NRAND", "3"))
OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/greedy_mode_ablate")


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


@torch.no_grad()
def forward_collect(model, tok, blocks, cm, graph, cand_t, col, dev, walks, proj_Q=None, grab=False):
    nL = cm.num_hidden_layers; grabbed = {}; hooks = []
    if grab:
        def mk(L):
            def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
            return hh
        hooks += [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
    if proj_Q is not None:
        def mkp(L):
            Q = proj_Q.get(L)
            def hh(_m, _i, out):
                if Q is None: return out
                h = out[0] if isinstance(out, tuple) else out
                hf = h.float(); hf = hf - (hf @ Q) @ Q.T
                h2 = hf.to(h.dtype)
                return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
            return hh
        hooks += [blocks[L].register_forward_hook(mkp(L)) for L in range(nL)]
    n = graph.n_nodes
    nsum = {L: np.zeros((n, cm.hidden_size)) for L in range(nL)} if grab else None
    ncnt = np.zeros(n); nbr_v = par_v = nbr_m = par_m = 0.0; cnt = 0
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; grabbed.clear()
            logits = model(input_ids=ids).logits[0]; cl = np.arange(1, len(nodes) + 1)
            if grab:
                single = [t[-1] for t in spans]
                for L in range(nL):
                    rows = grabbed[L][0][single].float().cpu().numpy()
                    for s in range(len(nodes)):
                        if cl[s] >= CTXLO:
                            nsum[L][nodes[s]] += rows[s]
                            if L == 0: ncnt[nodes[s]] += 1
            for s in range(len(nodes) - 1):
                if s + 1 < CTXLO: continue
                p = torch.softmax(logits[spans[s][-1]][cand_t].float(), 0).cpu().numpy(); p = p / p.sum()
                cur = nodes[s]; nb = graph.neighbors(cur); opp = np.where(col == -col[cur])[0]
                am = int(p.argmax())
                nbr_v += int(am in nb); par_v += int(am in opp)
                nbr_m += float(p[nb].sum()); par_m += float(p[opp].sum()); cnt += 1
    finally:
        for h in hooks: h.remove()
    c = max(cnt, 1)
    met = {"neighbour_validity": nbr_v / c, "parity_validity": par_v / c,
           "neighbour_mass": nbr_m / c, "parity_mass": par_m / c, "n_pred": cnt}
    if grab:
        cn = np.maximum(ncnt, 1)
        return met, {L: nsum[L] / cn[:, None] for L in range(nL)}
    return met, None


def build_Q(Rdir, S, nL, dev, thr=1e-8):
    Q = {}
    for L in range(nL):
        cols = [Rdir[k][L] for k in S if float(np.linalg.norm(Rdir[k][L])) > thr]
        if not cols: Q[L] = None; continue
        Mt = torch.tensor(np.stack(cols, 1), dtype=torch.float32, device=dev)
        q, _ = torch.linalg.qr(Mt, mode="reduced"); Q[L] = q
    return Q


def build_Q_random(rank, nL, dev, hidden, rng):
    Q = {}
    for L in range(nL):
        Mt = torch.tensor(rng.standard_normal((hidden, rank)).astype(np.float32), device=dev)
        q, _ = torch.linalg.qr(Mt, mode="reduced"); Q[L] = q
    return Q


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
    deg = np.array([len(graph.neighbors(i)) for i in range(n)], float)
    nbr_chance = float(deg.mean() / (n - 1))
    par_chance = float(np.array([int((col == -col[i]).sum()) for i in range(n)]).mean() / (n - 1))

    # conditions as eigenmode-index sets
    CONDS = {"low2 {2 lowest = coords}": [1, 2],
             "high1 {highest = parity}": [n - 1],
             "all3 {low2+high1}": [1, 2, n - 1]}

    print(f"[{tag}] loading (chance nbr={nbr_chance:.2f} par={par_chance:.2f})", flush=True)
    model, tok = load_with_fallback(hf, mirror, cfg)
    cm = model.config; blocks = M._decoder_blocks(model); nL = cm.num_hidden_layers
    cand_t = torch.tensor([tok(" " + graph.words[i], add_special_tokens=False)["input_ids"][0] for i in range(n)], device=dev)
    walks = G.generate_walks(graph, cfg)

    base, means = forward_collect(model, tok, blocks, cm, graph, cand_t, col, dev, walks, grab=True)
    print(f"[{tag}] clean nbr_v={base['neighbour_validity']:.3f} par_v={base['parity_validity']:.3f}", flush=True)
    Rdir = {}
    for k in range(1, n):
        Rk = np.zeros((nL, cm.hidden_size))
        for L in range(nL):
            Hc = means[L] - means[L].mean(0); Rk[L] = Hc.T @ U[:, k]
        Rdir[k] = Rk

    rng = np.random.default_rng(int(os.environ.get("SEED", "0")))
    out = {"model": tag, "graph": GRAPH, "chance": {"neighbour": nbr_chance, "parity": par_chance},
           "conditions": {"clean": base}, "rand": {}}
    for name, modes in CONDS.items():
        met, _ = forward_collect(model, tok, blocks, cm, graph, cand_t, col, dev, walks,
                                 proj_Q=build_Q(Rdir, modes, nL, dev))
        rn = rp = rnm = rpm = 0.0
        for _ in range(NRAND):
            mr, _ = forward_collect(model, tok, blocks, cm, graph, cand_t, col, dev, walks,
                                    proj_Q=build_Q_random(len(modes), nL, dev, cm.hidden_size, rng))
            rn += mr["neighbour_validity"]; rp += mr["parity_validity"]
            rnm += mr["neighbour_mass"]; rpm += mr["parity_mass"]
        out["conditions"][name] = {"modes": modes, **met}
        out["rand"][name] = {"neighbour_validity": rn / NRAND, "parity_validity": rp / NRAND,
                             "neighbour_mass": rnm / NRAND, "parity_mass": rpm / NRAND}
        print(f"[{tag}] {name:26s} nbr_v={met['neighbour_validity']:.3f} par_v={met['parity_validity']:.3f}"
              f"  | rand nbr_v={rn/NRAND:.3f} par_v={rp/NRAND:.3f}", flush=True)

    del model, tok; gc.collect()
    if torch and torch.cuda.is_available(): torch.cuda.empty_cache()
    p = f"{OUTDIR}/gma_cond_{tag}_{GS}.json"
    json.dump(out, open(p, "w"), indent=2); print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
