"""cut_sweep_summary analogue for DATA-DRIVEN directions: additive-steer along PC1, PC2 (top node-mean
PCs) and best2d-1, best2d-2 (the 2-D plane that best recovers the grid), measured the same way as
axis_cut_sweep -- mass moved onto each direction's + side vs dose -- with x/y/parity as reference.

Directions are node-cuts u (16-vec) defined from the clean node-means at the best grid-recovery layer
L*; the per-layer steer readout is v_L = normalise(Hc_L^T u), pushed additively (h += dose*std*v_L).

Env: PRESET GEN_MODEL(Llama) GRAPH(square_grid) XCTX(150) NWALK_TF(12) CTXLO(100) DOSES(0.25,0.5,1,2,4) OUTDIR DEVICE
Out: <OUTDIR>/axis_pc_sweep_<model>_<graph>.json
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
XCTX = int(os.environ.get("XCTX", "150")); NWALK_TF = int(os.environ.get("NWALK_TF", "12"))
CTXLO = int(os.environ.get("CTXLO", "100")); TEMP = float(os.environ.get("TEMP", "1.0"))
DOSES = [float(x) for x in os.environ.get("DOSES", "0.25,0.5,1,2,4").split(",")]
OUTDIR = os.environ.get("OUTDIR", "runs/axes/3_causal/axis_pc_sweep")


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


def best2d_plane(H, Gc):
    Hc = H - H.mean(0); U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    k = min(6, Vt.shape[0]); Z = U[:, :k] * S[:k]
    W = np.linalg.lstsq(Z, Gc - Gc.mean(0), rcond=None)[0]
    B = Vt[:k].T @ W
    Q, _ = np.linalg.qr(B)
    return Q                                              # (d,2) orthonormal residual plane


def grid_rsa(P, Gc):
    def pdist(X):
        D = np.linalg.norm(X[:, None] - X[None], axis=2); return D[np.triu_indices(len(X), 1)]
    a, b = pdist(P), pdist(Gc)
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def clean_node_means(model, tok, blocks, cm, walks, dev, n):
    grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    nL = cm.num_hidden_layers; hs = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
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
        for h in hs: h.remove()
    cn = np.maximum(ncnt, 1)
    return {L: nsum[L] / cn[:, None] for L in range(nL)}


def steer_vecs(means, u, dev, nL):
    out = {}
    for L in range(nL):
        H = means[L]; Hc = H - H.mean(0); v = Hc.T @ u; v = v / (np.linalg.norm(v) + 1e-9)
        std = float((Hc @ v).std()); out[L] = torch.tensor(std * v, device=dev, dtype=torch.float32)
    return out


def steer_hooks(blocks, layers_vec):
    handles = []
    for L, vec in layers_vec.items():
        def hk(_m, _i, out, vec=vec):
            h = out[0] if isinstance(out, tuple) else out
            h = h + vec.to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        handles.append(blocks[L].register_forward_hook(hk))
    return handles


@torch.no_grad()
def measure(model, tok, blocks, cm, graph, cand_t, dev, walks, layers_vec, plus, minus):
    handles = steer_hooks(blocks, layers_vec) if layers_vec else []
    mp = mm = nbr = 0.0; val = cnt = 0
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes
            logits = model(input_ids=ids).logits[0]
            for s in range(len(nodes) - 1):
                if s + 1 < CTXLO: continue
                p = torch.softmax(logits[spans[s][-1]][cand_t].float() / TEMP, 0).cpu().numpy(); p = p / p.sum()
                nb = graph.neighbors(nodes[s])
                mp += float(p[plus].sum()); mm += float(p[minus].sum()); nbr += float(p[nb].sum())
                val += int(int(p.argmax()) in nb); cnt += 1
    finally:
        for h in handles: h.remove()
    c = max(cnt, 1)
    return {"mass_plus": mp / c, "mass_minus": mm / c, "nbr": nbr / c, "val": val / c}


def main():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    os.makedirs(OUTDIR, exist_ok=True)
    tag = GEN_MODEL; hf, mirror = ALLSPEC[tag]
    cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH], n_walks=max(NWALK_TF, 24), walk_length=XCTX, device=dev)
    graph = G.build_graph(cfg); n = graph.n_nodes; coords = np.array(graph.coords, float)
    print(f"[{tag}] loading", flush=True)
    model, tok = load_with_fallback(hf, mirror, cfg)
    cm = model.config; blocks = M._decoder_blocks(model); nL = cm.num_hidden_layers
    cand_t = torch.tensor([tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in graph.words], device=dev)
    walks = G.generate_walks(graph, cfg); tf = walks[:NWALK_TF]
    means = clean_node_means(model, tok, blocks, cm, walks, dev, n)

    # L* = best grid-recovery layer (max best-2d RSA)
    rsa = {}
    for L in range(nL):
        B = best2d_plane(means[L], coords); rsa[L] = grid_rsa((means[L] - means[L].mean(0)) @ B, coords)
    Lstar = max(rsa, key=rsa.get); Hc = means[Lstar] - means[Lstar].mean(0)
    U, S, Vt = np.linalg.svd(Hc, full_matrices=False); B = best2d_plane(means[Lstar], coords); Pb = Hc @ B
    print(f"[{tag}] L*={Lstar} best2d-RSA={rsa[Lstar]:.2f}", flush=True)

    cuts_raw = {"PC1": U[:, 0], "PC2": U[:, 1], "best2d_1": Pb[:, 0], "best2d_2": Pb[:, 1],
                "x": coords[:, 0], "y": coords[:, 1], "parity": two_colour(graph)}
    cuts = {}
    for k, r in cuts_raw.items():
        u = unit(r); med = np.median(r)
        cuts[k] = {"u": u, "plus": np.where(r > med)[0].tolist(), "minus": np.where(r <= med)[0].tolist()}

    out = {"graph": GRAPH, "model": tag, "Lstar": Lstar, "best2d_rsa": rsa[Lstar], "doses": DOSES, "cuts": {}}
    for cn, cd in cuts.items():
        sv = steer_vecs(means, cd["u"], dev, nL)
        base = measure(model, tok, blocks, cm, graph, cand_t, dev, tf, None, cd["plus"], cd["minus"])
        rows = {"clean": base}
        for d in DOSES:
            rows[f"{d:g}"] = measure(model, tok, blocks, cm, graph, cand_t, dev, tf, {L: d * sv[L] for L in range(nL)}, cd["plus"], cd["minus"])
        out["cuts"][cn] = {"plus": cd["plus"], "minus": cd["minus"], "sweep": rows}
        ctl = rows[f"{DOSES[-1]:g}"]["mass_plus"] - base["mass_plus"]
        print(f"[{tag}/{cn:9s}] +side clean {base['mass_plus']:.2f} -> dose{DOSES[-1]:g} {rows[f'{DOSES[-1]:g}']['mass_plus']:.2f}  "
              f"(control {ctl:+.2f}) val {base['val']:.2f}->{rows[f'{DOSES[-1]:g}']['val']:.2f}", flush=True)
    del model, tok; gc.collect()
    if torch and torch.cuda.is_available(): torch.cuda.empty_cache()
    json.dump(out, open(f"{OUTDIR}/axis_pc_sweep_{tag}_{GRAPH}.json", "w"), indent=2)
    print(f"DONE -> {OUTDIR}/axis_pc_sweep_{tag}_{GRAPH}.json", flush=True)


if __name__ == "__main__":
    main()
