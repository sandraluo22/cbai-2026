"""Cross-model similarity of the induced grid geometry, several ways, all leakage-guarded:
  RSA(A,B)   : Spearman of the two node-RDMs (symmetric, basis-free; but dominated by shared graph)
  CKA(A,B)   : linear centered-kernel alignment (basis-free, handles different dims)
  (1) ridge  : leave-one-node-out ridge A -> B(top-6 PC); held-out R2 = linear predictability of
               B's geometry from A. Directional (report both ways).
  (2) coords : decode each model to grid (row,col) via leave-one-node-out, then compare the two
               models' decoded configs in the shared coordinate frame (per-axis corr, mean dist).

Node-means taken at each model's best-2D-RSA peak layer (16 grid nodes, same indices across models).

Env: PRESET GRAPH(square_grid) NWALKS(24) WLEN(300) CTXLO(100) OUTDIR DEVICE
Out: <OUTDIR>/cross_model_sim_<graph>.json
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

MODELS = [("Llama", "meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"),
          ("Gemma", "google/gemma-2-9b", "unsloth/gemma-2-9b"),
          ("Qwen",  "Qwen/Qwen3-8B-Base", None)]
if os.environ.get("PRESET") == "smoke":
    MODELS = [("distilgpt2", "distilgpt2", None), ("distilgpt2b", "distilgpt2", None)]
_mf = os.environ.get("MODELS_FILTER")
if _mf:
    MODELS = [m for m in MODELS if m[0] in set(_mf.split(","))]
MODE = os.environ.get("MODE", "full")   # capture (save means npz) | combine (load & compare) | full
GRAPH = os.environ.get("GRAPH", "square_grid")
NWALKS = int(os.environ.get("NWALKS", "24"))
WLEN = int(os.environ.get("WLEN", "300"))
CTXLO = int(os.environ.get("CTXLO", "100"))
ALPHAS = [1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6]
OUTDIR = os.environ.get("OUTDIR", "/workspace/cross-model/runs/induction-head/cross_model_sim")


def load_with_fallback(hf, mirror, cfg):
    try:
        return M.load_model(hf, cfg)
    except Exception:
        return M.load_model(mirror, cfg)


def sp(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def best2d_rsa(H, Gc, GDu, iu):
    Hc = H - H.mean(0); U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    k = min(6, Vt.shape[0]); Z = U[:, :k] * S[:k]
    W = np.linalg.lstsq(Z, Gc - Gc.mean(0), rcond=None)[0]
    P = Hc @ (Vt[:k].T @ W)
    return sp(np.linalg.norm(P[:, None] - P[None], axis=2)[iu], GDu)


def rdm(H):
    Hc = H - H.mean(0)
    return np.linalg.norm(Hc[:, None] - Hc[None], axis=2)


def cka(X, Y):
    Xc = X - X.mean(0); Yc = Y - Y.mean(0)
    hsic = np.linalg.norm(Yc.T @ Xc, "fro") ** 2
    return float(hsic / (np.linalg.norm(Xc.T @ Xc, "fro") * np.linalg.norm(Yc.T @ Yc, "fro") + 1e-12))


def ridge_pred(Xtr, ytr_c, xk, a):
    U, S, Vt = np.linalg.svd(Xtr, full_matrices=False)
    return xk @ (Vt.T @ ((S / (S ** 2 + a))[:, None] * (U.T @ ytr_c)))


def loo_map_r2(A, B, k=6):
    """leave-one-node-out ridge A -> top-k PC of B; variance-weighted held-out R2 (best alpha on LOO)."""
    n = A.shape[0]
    Bc = B - B.mean(0); Ub, Sb, Vtb = np.linalg.svd(Bc, full_matrices=False)
    kk = min(k, Vtb.shape[0]); Bpc = Ub[:, :kk] * Sb[:kk]
    best = -9.0
    for a in ALPHAS:
        pred = np.zeros((n, kk))
        for h in range(n):
            idx = [i for i in range(n) if i != h]
            Xtr = A[idx]; ytr = Bpc[idx]; xk = A[h:h + 1]
            mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
            ymu = ytr.mean(0)
            pred[h] = ridge_pred((Xtr - mu) / sd, ytr - ymu, (xk - mu) / sd, a) + ymu
        ssr = ((Bpc - pred) ** 2).sum(); sst = ((Bpc - Bpc.mean(0)) ** 2).sum()
        r = 1 - ssr / sst
        if r > best:
            best = r
    return float(best)


def loo_coords(H, coords):
    """leave-one-node-out decoded (row,col) per node (best alpha on LOO)."""
    n = H.shape[0]
    best = (-9.0, None)
    for a in ALPHAS:
        pred = np.zeros((n, 2))
        for h in range(n):
            idx = [i for i in range(n) if i != h]
            Xtr = H[idx]; ytr = coords[idx]; xk = H[h:h + 1]
            mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6; ymu = ytr.mean(0)
            pred[h] = ridge_pred((Xtr - mu) / sd, ytr - ymu, (xk - mu) / sd, a) + ymu
        sc = 0.5 * (sp(coords[:, 0], pred[:, 0]) + sp(coords[:, 1], pred[:, 1]))
        if sc > best[0]:
            best = (sc, pred)
    return best[1]


@torch.no_grad()
def peak_means(hf, mirror, cfg, graph, dev, Gc, GDu, iu):
    model, tok = load_with_fallback(hf, mirror, cfg)
    cm = model.config; blocks = M._decoder_blocks(model); nL = cm.num_hidden_layers; n = graph.n_nodes
    grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    hs = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
    nsum = {L: np.zeros((n, cm.hidden_size)) for L in range(nL)}; ncnt = np.zeros(n)
    for wk in G.generate_walks(graph, cfg):
        ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
        spans = resolve_token_spans(tok, wk); nodes = wk.nodes; cl = np.arange(1, len(nodes) + 1); grabbed.clear()
        model(input_ids=ids)
        for L in range(nL):
            for s in range(len(nodes)):
                if cl[s] >= CTXLO:
                    nsum[L][nodes[s]] += grabbed[L][0, spans[s][-1]].float().cpu().numpy()
                    if L == 0: ncnt[nodes[s]] += 1
    for h in hs: h.remove()
    cn = np.maximum(ncnt, 1); means = {L: nsum[L] / cn[:, None] for L in range(nL)}
    peak = int(np.argmax([best2d_rsa(means[L], Gc, GDu, iu) for L in range(nL)]))
    out = means[peak].copy()
    del model, tok; gc.collect()
    if torch and torch.cuda.is_available(): torch.cuda.empty_cache()
    return out, peak


def main():
    dev = os.environ.get("DEVICE", "cpu" if os.environ.get("PRESET") == "smoke" else "cuda")
    os.makedirs(OUTDIR, exist_ok=True)
    cfg = replace(get_config("gemma_qwen"), graph_type="grid", grid_rows=4, grid_cols=4,
                  n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg); n = graph.n_nodes
    iu = np.triu_indices(n, 1); GDu = graph.distance_matrix()[iu]; Gc = np.array(graph.coords, float)
    coords = np.array([(i // 4, i % 4) for i in range(n)], float)
    if MODE in ("capture", "full"):
        for tag, hf, mirror in MODELS:
            print(f"[{tag}] capturing", flush=True)
            mn, pk = peak_means(hf, mirror, cfg, graph, dev, Gc, GDu, iu)
            np.savez(f"{OUTDIR}/cm_{tag}.npz", means=mn.astype(np.float32), peak=pk, decoded=loo_coords(mn, coords))
            print(f"[{tag}] peak L{pk} -> cm_{tag}.npz", flush=True)
        if MODE == "capture":
            print("CAPTURE_DONE", flush=True); return
    # combine: load every cm_*.npz present
    means = {}; peaks = {}; decoded = {}
    import glob
    for f in sorted(glob.glob(f"{OUTDIR}/cm_*.npz")):
        tag = os.path.basename(f)[3:-4]; z = np.load(f)
        means[tag] = z["means"].astype(float); peaks[tag] = int(z["peak"]); decoded[tag] = z["decoded"]
    tags = list(means.keys())
    print(f"combining {tags}", flush=True)
    out = {"graph": GRAPH, "peaks": peaks, "pairs": {}}
    for a, b in itertools.combinations(tags, 2):
        rsa_ab = sp(rdm(means[a])[iu], rdm(means[b])[iu])
        cka_ab = cka(means[a], means[b])
        r2_ab = loo_map_r2(means[a], means[b]); r2_ba = loo_map_r2(means[b], means[a])
        cr = 0.5 * (sp(decoded[a][:, 0], decoded[b][:, 0]) + sp(decoded[a][:, 1], decoded[b][:, 1]))
        dist = float(np.linalg.norm(decoded[a] - decoded[b], axis=1).mean())
        out["pairs"][f"{a}-{b}"] = {"RSA": rsa_ab, "CKA": cka_ab,
                                    "ridge_LOO_R2_AtoB": r2_ab, "ridge_LOO_R2_BtoA": r2_ba,
                                    "coord_corr": cr, "coord_meandist": dist}
        print(f"[{a}-{b}] RSA={rsa_ab:.2f} CKA={cka_ab:.2f} | ridge LOO {a}->{b}={r2_ab:.2f} {b}->{a}={r2_ba:.2f} "
              f"| coord corr={cr:.2f} dist={dist:.2f}", flush=True)
    json.dump(out, open(f"{OUTDIR}/cross_model_sim_{GRAPH}.json", "w"), indent=2)
    print(f"DONE -> {OUTDIR}/cross_model_sim_{GRAPH}.json", flush=True)


if __name__ == "__main__":
    main()
