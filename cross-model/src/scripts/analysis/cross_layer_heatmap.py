"""Layer x layer cross-model alignment R^2 heatmaps. For each model pair (A,B) and each
(layer_i of A, layer_j of B), fit ridge A_i -> top-6 PC of B_j and score R^2:
  - in-sample (full fit, no holdout)         [inflated -- shows leakage baseline]
  - leave-one-node-out (all 16 holdouts)     [honest -- real cross-layer correspondence]
Slideshow: page 1 = full-fit R^2 (3 pair heatmaps), page 2 = 16-holdout LOO R^2 (3 pair heatmaps).

MODE=capture saves per-model all-layer node-means (respects two-pass HF_HOME); MODE=combine
loads them and builds the slideshow.

Env: PRESET MODELS_FILTER MODE(capture|combine|full) GRAPH(square_grid) NWALKS(24) WLEN(300)
     CTXLO(100) ALPHA(1000) OUTDIR DEVICE
Out: <OUTDIR>/cross_layer_heatmap_<graph>.pdf + .json
"""
from __future__ import annotations
import os, json, gc, glob, itertools
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
    MODELS = [("Aa", "distilgpt2", None), ("Bb", "distilgpt2", None), ("Cc", "distilgpt2", None)]
_mf = os.environ.get("MODELS_FILTER")
if _mf:
    MODELS = [m for m in MODELS if m[0] in set(_mf.split(","))]
MODE = os.environ.get("MODE", "full")
GRAPH = os.environ.get("GRAPH", "square_grid")
GKW = {"square_grid": dict(graph_type="grid", grid_rows=4, grid_cols=4),
       "hex": dict(graph_type="hex", hex_rows=4, hex_cols=4),
       "ring": dict(graph_type="ring", ring_size=16),
       "days": dict(graph_type="ring", ring_size=7, word_set="days")}
NWALKS = int(os.environ.get("NWALKS", "24"))
WLEN = int(os.environ.get("WLEN", "300"))
CTXLO = int(os.environ.get("CTXLO", "100"))
ALPHA = float(os.environ.get("ALPHA", "1000"))
NPERM = int(os.environ.get("NPERM", "200"))   # node-correspondence shuffle null on the pooled R2 (0 disables)
KPC = 6
OUTDIR = os.environ.get("OUTDIR", "/workspace/cross-model/runs/induction-head/cross_layer_heatmap")


def load_with_fallback(hf, mirror, cfg):
    try:
        return M.load_model(hf, cfg)
    except Exception:
        return M.load_model(mirror, cfg)


@torch.no_grad()
def all_layer_means(hf, mirror, cfg, graph, dev):
    model, tok = load_with_fallback(hf, mirror, cfg)
    cm = model.config; blocks = M._decoder_blocks(model); nL = cm.num_hidden_layers; n = graph.n_nodes
    grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    hs = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
    nsum = np.zeros((nL, n, cm.hidden_size)); ncnt = np.zeros(n)
    for wk in G.generate_walks(graph, cfg):
        ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
        spans = resolve_token_spans(tok, wk); nodes = wk.nodes; cl = np.arange(1, len(nodes) + 1); grabbed.clear()
        model(input_ids=ids)
        for L in range(nL):
            for s in range(len(nodes)):
                if cl[s] >= CTXLO:
                    nsum[L, nodes[s]] += grabbed[L][0, spans[s][-1]].float().cpu().numpy()
                    if L == 0: ncnt[nodes[s]] += 1
    for h in hs: h.remove()
    cn = np.maximum(ncnt, 1)
    means = nsum / cn[None, :, None]
    del model, tok; gc.collect()
    if torch and torch.cuda.is_available(): torch.cuda.empty_cache()
    return means.astype(np.float32)


def pca(B, k):
    Bc = B - B.mean(0); U, S, Vt = np.linalg.svd(Bc, full_matrices=False)
    kk = min(k, Vt.shape[0]); return (U[:, :kk] * S[:kk])


def prep_A(Ai):
    """per-fold standardized SVD of A layer (label/target independent)."""
    n = Ai.shape[0]; folds = []
    for kf in range(n):
        idx = [i for i in range(n) if i != kf]
        Xtr = Ai[idx]; xk = Ai[kf:kf + 1]
        mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
        U, S, Vt = np.linalg.svd((Xtr - mu) / sd, full_matrices=False)
        folds.append((np.array(idx), ((xk - mu) / sd @ Vt.T).ravel(), U.T.copy(), S))
    # full-fit standardized SVD
    mu = Ai.mean(0); sd = Ai.std(0) + 1e-6
    Uf, Sf, Vtf = np.linalg.svd((Ai - mu) / sd, full_matrices=False)
    return folds, (mu, sd, Uf, Sf, Vtf)


def r2(y, yh):
    ss = ((y - yh) ** 2).sum(); tot = ((y - y.mean(0)) ** 2).sum()
    return float(1 - ss / tot) if tot > 0 else float("nan")


def spear(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def coord_loo(H, coords):
    """leave-one-node-out decoded (row,col) + mean R2 (best alpha)."""
    folds = prep_A(H)[0]; best = (-9.0, None)
    for a in [1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6]:
        pred = np.zeros((H.shape[0], 2))
        for kf, (idx, proj, UT, S) in enumerate(folds):
            ytr = coords[idx]; ymu = ytr.mean(0)
            pred[kf] = proj @ ((S / (S ** 2 + a))[:, None] * (UT @ (ytr - ymu))) + ymu
        sc = 0.5 * (spear(coords[:, 0], pred[:, 0]) + spear(coords[:, 1], pred[:, 1]))
        if sc > best[0]:
            best = (sc, pred)
    return best[1], best[0]


def loo_full(Aprep, Bpc, a):
    """returns (pooled R2, per-node R2 array[16]). per-node R2 = 1 - ||pred_k-true_k||^2 /
    ||true_k - train_mean||^2 : how much better than the centroid that held-out node is predicted."""
    folds = Aprep[0]; n = len(folds); ssr = np.zeros(n); sst = np.zeros(n)
    for kf, (idx, proj, UT, S) in enumerate(folds):
        ytr = Bpc[idx]; ymu = ytr.mean(0)
        pred = proj @ ((S / (S ** 2 + a))[:, None] * (UT @ (ytr - ymu))) + ymu
        true = Bpc[kf]
        ssr[kf] = ((pred - true) ** 2).sum(); sst[kf] = ((true - ymu) ** 2).sum()
    pooled = float(1 - ssr.sum() / sst.sum()) if sst.sum() > 0 else float("nan")
    return pooled, 1 - ssr / np.maximum(sst, 1e-12)


def full_r2(Aprep, Bpc, a):
    mu, sd, Uf, Sf, Vtf = Aprep[1]; ymu = Bpc.mean(0)
    coef = Vtf.T @ ((Sf / (Sf ** 2 + a))[:, None] * (Uf.T @ (Bpc - ymu)))
    Xs = None  # standardized A already used in Uf; reconstruct pred via Uf S Vt? use full X
    # pred = Xs @ coef ; Xs = U S Vt  => pred = U (S) Vt Vt.T ... simpler: pred = (Uf*Sf) @ (Vtf@coef)
    pred = (Uf * Sf) @ (Vtf @ coef) + ymu
    return r2(Bpc, pred)


def main():
    dev = os.environ.get("DEVICE", "cpu" if os.environ.get("PRESET") == "smoke" else "cuda")
    os.makedirs(OUTDIR, exist_ok=True)
    cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH], n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg)
    if MODE in ("capture", "full"):
        for tag, hf, mirror in MODELS:
            print(f"[{tag}] capturing all layers ({GRAPH})", flush=True)
            mn = all_layer_means(hf, mirror, cfg, graph, dev)
            np.savez(f"{OUTDIR}/allmeans_{GRAPH}_{tag}.npz", means=mn)
            print(f"[{tag}] {mn.shape} -> allmeans_{GRAPH}_{tag}.npz", flush=True)
        if MODE == "capture":
            print("CAPTURE_DONE", flush=True); return
    means = {}; _pre = f"allmeans_{GRAPH}_"
    for f in sorted(glob.glob(f"{OUTDIR}/{_pre}*.npz")):
        tag = os.path.basename(f)[len(_pre):-4]; means[tag] = np.load(f)["means"].astype(float)
    tags = list(means.keys()); print(f"combining {tags} [{GRAPH}]", flush=True)
    upairs = list(itertools.combinations(tags, 2))          # unordered (columns)
    nnodes = int(next(iter(means.values())).shape[1])
    coords = np.array(graph.coords, float)
    pooled = {}; pernode = {}                               # keyed by "X->Y" (directional)
    def do_dir(X, Y):                                       # predict Y from X
        MX = means[X]; MY = means[Y]; nX = MX.shape[0]; nY = MY.shape[0]
        Ypcs = [pca(MY[j], KPC) for j in range(nY)]; Xpreps = [prep_A(MX[i]) for i in range(nX)]
        Rp = np.zeros((nX, nY)); Rn = np.zeros((nnodes, nX, nY))
        for i in range(nX):
            for j in range(nY):
                pl, pn = loo_full(Xpreps[i], Ypcs[j], ALPHA)
                Rp[i, j] = pl; Rn[:, i, j] = pn
        pooled[f"{X}->{Y}"] = Rp.tolist(); pernode[f"{X}->{Y}"] = Rn.tolist()
        print(f"[{X}->{Y}] pooled max={Rp.max():.2f} per-node max={Rn.max():.2f}", flush=True)
    for A, B in upairs:                                     # BOTH directions
        do_dir(A, B); do_dir(B, A)
    # permutation null: shuffle the node<->node correspondence between the two models (break
    # geometry matching), re-fit the ridge at each direction's BEST layer-pair cell, and ask
    # whether the real alignment beats random node-matching. Xprep depends only on X (unpermuted),
    # so we permute Y's node rows. Fixed at the real argmax cell -> one number + p per direction.
    nullinfo = {}
    if NPERM > 0:
        rng = np.random.default_rng(0)
        for key, Rp in pooled.items():
            X, Y = key.split("->"); Rp = np.array(Rp)
            i0, j0 = np.unravel_index(int(np.argmax(Rp)), Rp.shape); real = float(Rp[i0, j0])
            Xprep = prep_A(means[X][i0])
            nulls = np.array([loo_full(Xprep, pca(means[Y][j0][rng.permutation(nnodes)], KPC), ALPHA)[0]
                              for _ in range(NPERM)])
            p = float((np.sum(nulls >= real) + 1) / (NPERM + 1))
            nullinfo[key] = {"cell": [int(i0), int(j0)], "real": real, "n_perm": NPERM,
                             "null_mean": float(nulls.mean()), "null_std": float(nulls.std()),
                             "null_p95": float(np.percentile(nulls, 95)), "p": p}
            print(f"[null {key}] best cell ({i0},{j0}) real={real:.3f} "
                  f"null={nulls.mean():+.3f}±{nulls.std():.3f} p={p:.4f}", flush=True)
    json.dump({"graph": GRAPH, "alpha": ALPHA, "nnodes": nnodes, "upairs": [list(p) for p in upairs],
               "pooled_loo": pooled, "pernode_loo": pernode, "null_pooled": nullinfo},
              open(f"{OUTDIR}/cross_layer_heatmap_{GRAPH}.json", "w"))
    make_fig(upairs, pooled, pernode, nnodes, coords, f"{OUTDIR}/cross_layer_heatmap_{GRAPH}.pdf", nullinfo)
    print(f"DONE -> {OUTDIR}/cross_layer_heatmap_{GRAPH}.pdf", flush=True)


def _panel(ax, R, X, Y, cmap, vmin, vmax):
    im = ax.imshow(np.clip(R, vmin, vmax), aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel(f"{Y} layer"); ax.set_ylabel(f"{X} layer")
    ax.set_title(f"predict {Y} from {X}  (max {R.max():.2f})", fontsize=8)
    return im


def make_fig(upairs, pooled, pernode, nnodes, coords, path, nullinfo=None):
    nullinfo = nullinfo or {}
    def pnote(ax, key):                                    # annotate best-cell perm-null p under the panel title
        ni = nullinfo.get(key)
        if ni:
            ax.text(0.5, 1.14, f"best-cell null {ni['null_mean']:+.2f}±{ni['null_std']:.2f}, p={ni['p']:.3f}",
                    transform=ax.transAxes, ha="center", va="bottom", fontsize=7, color="0.3")
    with PdfPages(path) as pdf:
        # page 1: pooled -- 2x3 (cols=pairs, row0 = A->B, row1 = B->A reverse)
        fig, ax = plt.subplots(2, len(upairs), figsize=(5.6 * len(upairs), 9), squeeze=False)
        for j, (A, B) in enumerate(upairs):
            im = _panel(ax[0, j], np.array(pooled[f"{A}->{B}"]), A, B, "viridis", 0, 1); fig.colorbar(im, ax=ax[0, j], fraction=.046); pnote(ax[0, j], f"{A}->{B}")
            im = _panel(ax[1, j], np.array(pooled[f"{B}->{A}"]), B, A, "viridis", 0, 1); fig.colorbar(im, ax=ax[1, j], fraction=.046); pnote(ax[1, j], f"{B}->{A}")
        fig.suptitle(f"POOLED leave-one-node-out R² — both directions (top: A→B, bottom: B→A; ridge top-6 PC, α={ALPHA:g}; "
                     f"best-cell node-shuffle null, {NPERM} perms)", fontsize=11)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        # one page per held-out node: 2x3 both directions
        for k in range(nnodes):
            fig, ax = plt.subplots(2, len(upairs), figsize=(5.6 * len(upairs), 9), squeeze=False)
            for j, (A, B) in enumerate(upairs):
                im = _panel(ax[0, j], np.array(pernode[f"{A}->{B}"])[k], A, B, "RdBu_r", -1, 1); fig.colorbar(im, ax=ax[0, j], fraction=.046)
                im = _panel(ax[1, j], np.array(pernode[f"{B}->{A}"])[k], B, A, "RdBu_r", -1, 1); fig.colorbar(im, ax=ax[1, j], fraction=.046)
            cx = tuple(round(float(v), 2) for v in coords[k])
            fig.suptitle(f"[{path.split('_')[-1][:-4]}] HELD-OUT node {k} @ {cx} — per-node R² layer×layer, both directions "
                         "(red=well predicted, blue=worse than centroid)", fontsize=10)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
