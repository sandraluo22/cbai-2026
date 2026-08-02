"""Experiment 2 -- per-mode single ablation. For each graph-Laplacian eigenmode k, project the
residual stream onto the orthogonal complement of that ONE mode's per-layer readout direction
r_{k,L} = Hc_L^T u_k (at every layer), and measure the drop in neighbour / parity validity. This is
the single-mode variant of gma.py (which greedily ablates SETS): it isolates each mode's individual
causal contribution, so you can see which single mode most damages neighbour prediction vs parity.

A rank-1 RANDOM-direction control per layer gives the "ablate one arbitrary direction" baseline, so a
mode's drop is read against removing a random rank-1 direction of the same size.

Env: GEN_MODEL(Llama) GRAPH(square_grid) NWALKS(16) WLEN(300) CTXLO(100) LAP(norm|unnorm) SEED(0) OUTDIR DEVICE
Out: <OUTDIR>/per_mode_ablate_<model>_<graph>.json
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
       "ring": dict(graph_type="ring", ring_size=16), "hex": dict(graph_type="hex", hex_rows=4, hex_cols=4),
       "prism": dict(graph_type="prism", prism_k=7)}
GRAPH = os.environ.get("GRAPH", "square_grid")
GS = {"square_grid": "grid"}.get(GRAPH, GRAPH)
NWALKS = int(os.environ.get("NWALKS", "16")); WLEN = int(os.environ.get("WLEN", "300"))
CTXLO = int(os.environ.get("CTXLO", "100")); LAP = os.environ.get("LAP", "norm"); SEED = int(os.environ.get("SEED", "0"))
OUTDIR = os.environ.get("OUTDIR", "runs/axes/4_circuits/per_mode_ablate")


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


def laplacian_modes(A, kind):
    d = A.sum(1)
    if kind == "unnorm":
        L = np.diag(d) - A
    else:
        di = 1.0 / np.sqrt(np.maximum(d, 1e-12)); L = np.eye(len(A)) - (di[:, None] * A * di[None, :])
    return np.linalg.eigh(L)


@torch.no_grad()
def forward_collect(model, tok, blocks, cm, graph, cand_t, col, dev, walks, proj_Q=None, grab=False):
    """One pass. grab: accumulate clean per-node last-token residuals per layer. proj_Q: at each layer
    project residual onto orthogonal complement of Q[L] (hidden x r). Measures nbr/parity validity."""
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
                hf = h.float(); hf = hf - (hf @ Q) @ Q.T; h2 = hf.to(h.dtype)
                return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
            return hh
        hooks += [blocks[L].register_forward_hook(mkp(L)) for L in range(nL)]
    n = graph.n_nodes
    nsum = {L: np.zeros((n, cm.hidden_size)) for L in range(nL)} if grab else None
    ncnt = np.zeros(n); nbr_v = par_v = 0.0; cnt = 0
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
                cur = nodes[s]; nb = graph.neighbors(cur); opp = np.where(col == -col[cur])[0]; am = int(p.argmax())
                nbr_v += int(am in nb); par_v += int(am in opp); cnt += 1
    finally:
        for h in hooks: h.remove()
    c = max(cnt, 1)
    metrics = {"neighbour_validity": nbr_v / c, "parity_validity": par_v / c, "n_pred": cnt}
    if grab:
        cn = np.maximum(ncnt, 1); return metrics, {L: nsum[L] / cn[:, None] for L in range(nL)}
    return metrics, None


def build_Q_from_dirs(dirs_by_layer, nL, dev, thr=1e-8):
    """dirs_by_layer[L] = list of hidden-space vectors -> orthonormal Q[L] (hidden x r)."""
    Q = {}
    for L in range(nL):
        cols = [v for v in dirs_by_layer[L] if float(np.linalg.norm(v)) > thr]
        if not cols: Q[L] = None; continue
        Mt = torch.tensor(np.stack(cols, 1), dtype=torch.float32, device=dev)
        q, _ = torch.linalg.qr(Mt, mode="reduced"); Q[L] = q
    return Q


def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH], n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg); n = graph.n_nodes; col = two_colour(graph)
    A = np.zeros((n, n))
    for a in range(n):
        for b in graph.adjacency[a]: A[a, b] = 1.0
    w, U = laplacian_modes(A, LAP)
    coords = np.array(graph.coords, float)

    def corr(a, b):
        a = a - a.mean(); b = b - b.mean(); return float(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    def label(k):
        cx = max(corr(U[:, k], coords[:, 0]), corr(U[:, k], coords[:, 1])); cp = corr(U[:, k], col)
        return "parity" if cp > 0.9 else ("coord" if cx > 0.7 else "other")

    deg = np.array([len(graph.neighbors(i)) for i in range(n)], float); nbr_chance = float(deg.mean() / (n - 1))
    opp = np.array([int((col == -col[i]).sum()) for i in range(n)], float); par_chance = float(opp.mean() / (n - 1))

    print(f"[{tag}] loading (chance nbr={nbr_chance:.2f} par={par_chance:.2f}; LAP={LAP}, {GS} n={n})", flush=True)
    model, tok = load_with_fallback(hf, mirror, cfg)
    cm = model.config; blocks = M._decoder_blocks(model); nL = cm.num_hidden_layers
    cand_t = torch.tensor([tok(" " + graph.words[i], add_special_tokens=False)["input_ids"][0] for i in range(n)], device=dev)
    walks = G.generate_walks(graph, cfg)

    base, means = forward_collect(model, tok, blocks, cm, graph, cand_t, col, dev, walks, grab=True)
    print(f"[{tag}] clean nbr_v={base['neighbour_validity']:.3f} par_v={base['parity_validity']:.3f}", flush=True)

    # per-layer readout direction for each mode: r_{k,L} = Hc_L^T u_k
    Rdir = {k: np.stack([(means[L] - means[L].mean(0)).T @ U[:, k] for L in range(nL)]) for k in range(1, n)}

    rng = np.random.default_rng(SEED)
    results = []
    for k in range(1, n):                                        # skip trivial mode 0
        Q = build_Q_from_dirs([[Rdir[k][L]] for L in range(nL)], nL, dev)
        m, _ = forward_collect(model, tok, blocks, cm, graph, cand_t, col, dev, walks, proj_Q=Q)
        results.append({"mode": k, "eigenvalue": float(w[k]), "label": label(k),
                        "neighbour_validity": m["neighbour_validity"], "parity_validity": m["parity_validity"],
                        "d_nbr": base["neighbour_validity"] - m["neighbour_validity"],
                        "d_par": base["parity_validity"] - m["parity_validity"]})
        print(f"  ablate m{k:<2} ({results[-1]['label']:<6} λ={w[k]:.2f}): "
              f"nbr {base['neighbour_validity']:.3f}->{m['neighbour_validity']:.3f} "
              f"par {base['parity_validity']:.3f}->{m['parity_validity']:.3f}", flush=True)

    # random rank-1 control: mean over a few random per-layer directions
    rand = []
    for _ in range(3):
        dirs = [[rng.standard_normal(cm.hidden_size)] for _ in range(nL)]
        Q = build_Q_from_dirs(dirs, nL, dev)
        m, _ = forward_collect(model, tok, blocks, cm, graph, cand_t, col, dev, walks, proj_Q=Q)
        rand.append(m)
    rctrl = {"neighbour_validity": float(np.mean([r["neighbour_validity"] for r in rand])),
             "parity_validity": float(np.mean([r["parity_validity"] for r in rand]))}
    print(f"  random rank-1 control: nbr {rctrl['neighbour_validity']:.3f} par {rctrl['parity_validity']:.3f}", flush=True)

    out = {"model": tag, "graph": GRAPH, "lap": LAP, "walk_length": WLEN, "ctxlo": CTXLO,
           "chance": {"neighbour": nbr_chance, "parity": par_chance}, "baseline": base,
           "random_rank1": rctrl, "modes": results}
    del model, tok; gc.collect()
    if torch and torch.cuda.is_available(): torch.cuda.empty_cache()
    p = f"{OUTDIR}/per_mode_ablate_{tag}_{GS}.json"
    json.dump(out, open(p, "w"), indent=2); print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
