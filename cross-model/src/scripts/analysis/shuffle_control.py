"""Shuffle control: compare REAL random-walk context vs a SHUFFLED walk (same token multiset, order
permuted so the graph transition structure is destroyed). If parity/coord/RSA are genuine in-context
graph signatures they should collapse; behaviour should drop to chance. Grid, per-layer.

Env: GEN_MODEL(Llama) GRAPH(square_grid) NWALKS(24) WLEN(300) CTXLO(100) SEED(0) OUTDIR DEVICE
Out: <OUTDIR>/shuffle_control_<model>_<G>.json
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
from graph import Walk
import models as M
from models import resolve_token_spans

ALLSPEC = {"Llama": ("meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"),
           "Gemma": ("google/gemma-2-9b", "unsloth/gemma-2-9b"), "Qwen": ("Qwen/Qwen3-8B-Base", None)}
GEN_MODEL = os.environ.get("GEN_MODEL", "Llama")
GKW = {"square_grid": dict(graph_type="grid", grid_rows=4, grid_cols=4)}
GRAPH = os.environ.get("GRAPH", "square_grid"); GS = {"square_grid": "grid"}.get(GRAPH, GRAPH)
NWALKS = int(os.environ.get("NWALKS", "24")); WLEN = int(os.environ.get("WLEN", "300"))
CTXLO = int(os.environ.get("CTXLO", "100")); SEED = int(os.environ.get("SEED", "0"))
OUTDIR = os.environ.get("OUTDIR", "runs/axes/1_decomposition/shuffle")


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


def shuffle_walks(walks, rng):
    out = []
    for wk in walks:
        idx = rng.permutation(len(wk.nodes))
        nodes = [wk.nodes[i] for i in idx]; words = [wk.words[i] for i in idx]
        out.append(Walk(walk_id=wk.walk_id, nodes=nodes, words=words))
    return out


@torch.no_grad()
def run(model, tok, blocks, cm, graph, cand_t, col, dev, walks, n):
    nL = cm.num_hidden_layers; grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    caps = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
    nsum = {L: np.zeros((n, cm.hidden_size)) for L in range(nL)}; ncnt = np.zeros(n)
    nbr_v = par_v = 0.0; cnt = 0
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; grabbed.clear()
            logits = model(input_ids=ids).logits[0]
            single = [t[-1] for t in spans]; cl = np.arange(1, len(nodes) + 1)
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
        for h in caps: h.remove()
    cn = np.maximum(ncnt, 1)[:, None]; c = max(cnt, 1)
    means = [nsum[L] / cn for L in range(nL)]
    return means, {"neighbour_validity": nbr_v / c, "parity_validity": par_v / c}


def main():
    dev = os.environ.get("DEVICE", "cuda"); os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH], n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg); n = graph.n_nodes; col = two_colour(graph)
    coords = np.array(graph.coords, float); Gc = coords - coords.mean(0)
    GD = np.abs(coords[:, None] - coords[None]).sum(-1)[np.triu_indices(n, 1)]
    A = np.zeros((n, n))
    for a in range(n):
        for b in graph.adjacency[a]: A[a, b] = 1.0
    w, V = np.linalg.eigh(np.diag(A.sum(1)) - A)                  # unnormalized eigenmodes
    par_i, co_i = int(np.argmax(w)), [1, 2]                       # parity = top mode; coords = modes 1,2 (1-based in p[1:]? see below)
    nbr_chance = float(np.mean([len(graph.neighbors(i)) for i in range(n)]) / (n - 1))

    def sp(a, b): return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
    def rdm(H): iu = np.triu_indices(n, 1); return np.linalg.norm(H[:, None] - H[None], axis=2)[iu]
    def best2d(H):
        Hc = H - H.mean(0); U, S, Vh = np.linalg.svd(Hc, full_matrices=False)
        Z = U[:, :6] * S[:6]; Wm = np.linalg.lstsq(Z, Gc, rcond=None)[0]; return Z @ Wm
    def metrics(means):
        out = {"parity_pow": [], "coord_pow": [], "best2d_rsa": [], "full_rsa": [], "pc2_rsa": []}
        for H in means:
            Hc = H - H.mean(0); c = V.T @ Hc; p = (c ** 2).sum(1); p[0] = 0; p = p / (p.sum() + 1e-12)
            out["parity_pow"].append(float(p[par_i])); out["coord_pow"].append(float(p[1] + p[2]))
            out["best2d_rsa"].append(sp(rdm(best2d(H)), GD)); out["full_rsa"].append(sp(rdm(Hc), GD))
            _, _, Vh = np.linalg.svd(Hc, full_matrices=False)                 # raw top-2 PC grid RSA (unsupervised)
            out["pc2_rsa"].append(sp(rdm(Hc @ Vh[:2].T), GD))
        return out

    print(f"[{tag}] loading", flush=True)
    model, tok = load_with_fallback(hf, mirror, cfg)
    cm = model.config; blocks = M._decoder_blocks(model)
    cand_t = torch.tensor([tok(" " + graph.words[i], add_special_tokens=False)["input_ids"][0] for i in range(n)], device=dev)
    rng = np.random.default_rng(SEED)
    real = G.generate_walks(graph, cfg); shuf = shuffle_walks(real, rng)

    res = {}
    for name, walks in [("real", real), ("shuffled", shuf)]:
        means, beh = run(model, tok, blocks, cm, graph, cand_t, col, dev, walks, n)
        mt = metrics(means); res[name] = {"behaviour": beh, **mt}
        print(f"[{tag}/{name:8}] nbr_v={beh['neighbour_validity']:.3f} par_v={beh['parity_validity']:.3f} "
              f"| peak parity_pow={max(mt['parity_pow']):.3f} coord_pow={max(mt['coord_pow']):.3f} "
              f"best2d_rsa={max(mt['best2d_rsa']):.3f} raw-PC2_rsa={max(mt['pc2_rsa']):.3f}", flush=True)

    par_chance = float(len(np.where(col == -col[0])[0]) / (n - 1))
    out = {"model": tag, "graph": GRAPH, "walk_length": WLEN, "ctxlo": CTXLO,
           "chance": {"neighbour": nbr_chance, "parity": par_chance}, "results": res}
    del model, tok; gc.collect()
    if torch and torch.cuda.is_available(): torch.cuda.empty_cache()
    p = f"{OUTDIR}/shuffle_control_{tag}_{GS}.json"
    json.dump(out, open(p, "w"), indent=2); print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
