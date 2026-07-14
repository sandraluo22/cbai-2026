"""Greedy EIGENMODE ablation: which graph-Laplacian modes, projected out of the residual stream at
every layer, drive next-node (neighbour) validity to its chance floor?

Each Laplacian eigenmode u_k (16-dim node vector) has a per-layer readout direction r_{k,L} =
Hc_L^T u_k in hidden space (from clean node-means). Ablating a mode = projecting the residual stream
onto the orthogonal complement of {r_{k,L}} at EVERY layer. Greedy: at each step add the mode whose
joint ablation most lowers neighbour validity; stop at the chance floor.

Env: GEN_MODEL(Llama) GRAPH(square_grid) NWALKS(16) CTXLO(100) MAXK(8) LAP(norm|unnorm) OUTDIR DEVICE
Out: <OUTDIR>/gma_<model>_<graph>.json
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
MAXK = int(os.environ.get("MAXK", "8")); LAP = os.environ.get("LAP", "norm")
MODE_BASIS = os.environ.get("MODE_BASIS", "eig")     # "eig" (Laplacian modes) | "cuts" (x/y/diag/antidiag/parity)
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


def laplacian_modes(A, kind):
    d = A.sum(1)
    if kind == "unnorm":
        L = np.diag(d) - A
    else:
        di = 1.0 / np.sqrt(np.maximum(d, 1e-12))
        L = np.eye(len(A)) - (di[:, None] * A * di[None, :])
    w, U = np.linalg.eigh(L)                        # ascending
    return w, U


@torch.no_grad()
def forward_collect(model, tok, blocks, cm, graph, cand_t, col, dev, walks, proj_Q=None, grab=False):
    """One pass. If grab: accumulate per-node last-token residuals per layer (clean node-means).
    If proj_Q: at each layer project residual onto orthogonal complement of Q[L] (hidden x r).
    Always measure neighbour/parity validity. Returns (metrics, node_means_or_None)."""
    nL = cm.num_hidden_layers
    grabbed = {}; hooks = []
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
                hf = h.float()
                hf = hf - (hf @ Q) @ Q.T
                h2 = hf.to(h.dtype)
                return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
            return hh
        hooks += [blocks[L].register_forward_hook(mkp(L)) for L in range(nL)]

    n = graph.n_nodes
    nsum = {L: np.zeros((n, cm.hidden_size)) for L in range(nL)} if grab else None
    ncnt = np.zeros(n)
    nbr_v = par_v = 0.0; cnt = 0
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; grabbed.clear()
            logits = model(input_ids=ids).logits[0]
            cl = np.arange(1, len(nodes) + 1)
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
                nbr_v += int(am in nb); par_v += int(am in opp); cnt += 1
    finally:
        for h in hooks: h.remove()
    c = max(cnt, 1)
    metrics = {"neighbour_validity": nbr_v / c, "parity_validity": par_v / c, "n_pred": cnt}
    if grab:
        cn = np.maximum(ncnt, 1)
        means = {L: nsum[L] / cn[:, None] for L in range(nL)}
        return metrics, means
    return metrics, None


def build_Q(Rdir, S, nL, dev, thr=1e-8):
    """Per-layer orthonormal basis (hidden x r) of the selected modes' readout directions."""
    Q = {}
    for L in range(nL):
        cols = [Rdir[k][L] for k in S if float(np.linalg.norm(Rdir[k][L])) > thr]
        if not cols:
            Q[L] = None; continue
        Mt = torch.tensor(np.stack(cols, 1), dtype=torch.float32, device=dev)   # hidden x r
        q, _ = torch.linalg.qr(Mt, mode="reduced")
        Q[L] = q
    return Q


def build_Q_random(rank, nL, dev, hidden, rng):
    """Per-layer orthonormal basis of `rank` RANDOM hidden directions (projection-out control)."""
    Q = {}
    for L in range(nL):
        Mt = torch.tensor(rng.standard_normal((hidden, rank)).astype(np.float32), device=dev)
        q, _ = torch.linalg.qr(Mt, mode="reduced")
        Q[L] = q
    return Q


def main():
    dev = os.environ.get("DEVICE", "cuda")
    os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH], n_walks=NWALKS, walk_length=300, device=dev)
    graph = G.build_graph(cfg); n = graph.n_nodes; col = two_colour(graph)
    A = np.zeros((n, n))
    for a in range(n):
        for b in graph.adjacency[a]: A[a, b] = 1.0
    coords = np.array(graph.coords, float)
    def corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        return float(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    def unit(v): v = v - v.mean(); return v / (np.linalg.norm(v) + 1e-9)

    # build the ablation basis: each item = (name, node-vector u, lambda-or-None, label)
    items = []
    if MODE_BASIS == "cuts":
        x, y = coords[:, 0], coords[:, 1]
        raw = {"x": x, "y": y, "diagonal": x + y, "anti-diagonal": x - y, "parity": col}
        for name, r in raw.items():
            items.append({"name": name, "u": unit(r), "lam": None, "label": name})
    else:
        w, U = laplacian_modes(A, LAP)
        for k in range(1, n):
            cx = max(corr(U[:, k], coords[:, 0]), corr(U[:, k], coords[:, 1]))
            cp = corr(U[:, k], col)
            lab = ("parity" if cp > 0.9 else ("coord" if cx > 0.7 else "other"))
            items.append({"name": str(k), "u": U[:, k], "lam": float(w[k]), "label": lab})

    deg = np.array([len(graph.neighbors(i)) for i in range(n)], float)
    nbr_chance = float(deg.mean() / (n - 1))
    opp = np.array([int((col == -col[i]).sum()) for i in range(n)], float)
    par_chance = float(opp.mean() / (n - 1))

    print(f"[{tag}] loading (chance nbr={nbr_chance:.2f} par={par_chance:.2f}; LAP={LAP})", flush=True)
    model, tok = load_with_fallback(hf, mirror, cfg)
    cm = model.config; blocks = M._decoder_blocks(model); nL = cm.num_hidden_layers
    cand_t = torch.tensor([tok(" " + graph.words[i], add_special_tokens=False)["input_ids"][0] for i in range(n)], device=dev)
    walks = G.generate_walks(graph, cfg)

    base, means = forward_collect(model, tok, blocks, cm, graph, cand_t, col, dev, walks, grab=True)
    print(f"[{tag}] clean nbr_v={base['neighbour_validity']:.3f} par_v={base['parity_validity']:.3f}", flush=True)

    # per-layer readout direction r_{it,L} = Hc_L^T u_it  (un-normalized; QR handles scale)
    Rdir = {}
    for it in items:
        Rk = np.zeros((nL, cm.hidden_size))
        for L in range(nL):
            Hc = means[L] - means[L].mean(0)
            Rk[L] = Hc.T @ it["u"]
        Rdir[it["name"]] = Rk
    meta = {it["name"]: {"lam": it["lam"], "label": it["label"]} for it in items}

    candidates = [it["name"] for it in items]
    # optional forced SEED modes ablated first (e.g. lowest-2 + highest eigenmode), then greedy.
    sm = os.environ.get("SEED_MODES", "").strip()
    if sm == "lo2hi":
        seed_names = ["1", "2", str(n - 1)]           # 2 lowest non-trivial + highest eigenmode
    elif sm:
        seed_names = [s.strip() for s in sm.split(",")]
    else:
        seed_names = []

    NRAND = int(os.environ.get("NRAND", "3")); rng = np.random.default_rng(int(os.environ.get("SEED", "0")))

    def rand_baseline(rank):
        """project out `rank` random hidden directions, NRAND draws averaged -> (nbr_v, par_v)."""
        if NRAND <= 0 or rank == 0:
            return base["neighbour_validity"], base["parity_validity"]
        rn = rp = 0.0
        for _ in range(NRAND):
            mr, _ = forward_collect(model, tok, blocks, cm, graph, cand_t, col, dev, walks,
                                    proj_Q=build_Q_random(rank, nL, dev, cm.hidden_size, rng))
            rn += mr["neighbour_validity"]; rp += mr["parity_validity"]
        return rn / NRAND, rp / NRAND

    rb0 = rand_baseline(0)
    chosen = []; history = [{"step": 0, "mode": None, "seeded": False,
                             "rand_nbr_v": rb0[0], "rand_par_v": rb0[1], **base}]
    step = 0

    def record(name, met, seeded):
        lam = meta[name]["lam"]; lab = meta[name]["label"]
        rn, rp = rand_baseline(len(chosen))
        history.append({"step": step, "mode": name, "lambda": lam, "label": lab, "seeded": seeded,
                        "rand_nbr_v": rn, "rand_par_v": rp, **met})
        ls = f"λ={lam:.2f}," if lam is not None else ""
        tagx = "SEED" if seeded else "grdy"
        print(f"[{tag}] {tagx} step {step}: +{name} ({ls}{lab})  "
              f"nbr_v={met['neighbour_validity']:.3f} par_v={met['parity_validity']:.3f}  "
              f"| rand_nbr={rn:.3f}", flush=True)

    for name in seed_names:                            # forced seed steps
        if name in chosen: continue
        step += 1
        met, _ = forward_collect(model, tok, blocks, cm, graph, cand_t, col, dev, walks,
                                 proj_Q=build_Q(Rdir, chosen + [name], nL, dev))
        chosen.append(name); record(name, met, True)

    while step < MAXK:                                 # greedy steps
        best_c, best_nbr, best_m = None, 1e9, None
        for c in candidates:
            if c in chosen: continue
            met, _ = forward_collect(model, tok, blocks, cm, graph, cand_t, col, dev, walks,
                                     proj_Q=build_Q(Rdir, chosen + [c], nL, dev))
            if met["neighbour_validity"] < best_nbr:
                best_nbr, best_c, best_m = met["neighbour_validity"], c, met
        if best_c is None:
            print(f"[{tag}] no candidates left at step {step}", flush=True); break
        step += 1; chosen.append(best_c); record(best_c, best_m, False)
        if best_nbr <= nbr_chance + 0.02:
            print(f"[{tag}] reached chance floor at step {step}", flush=True); break

    out = {"model": tag, "graph": GRAPH, "basis": MODE_BASIS, "lap": LAP,
           "chance": {"neighbour": nbr_chance, "parity": par_chance},
           "mode_meta": meta, "greedy": history, "chosen_modes": chosen}
    del model, tok; gc.collect()
    if torch and torch.cuda.is_available(): torch.cuda.empty_cache()
    suffix = ("cuts" if MODE_BASIS == "cuts" else LAP) + ("seed" if seed_names else "")
    p = f"{OUTDIR}/gma_{suffix}_{tag}_{GS}.json"
    json.dump(out, open(p, "w"), indent=2); print(f"DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
