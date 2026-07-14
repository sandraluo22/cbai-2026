"""Context-length x LAYER leave-one-node-out coord-probe R^2 heatmap.

coord_decode gives probe R^2 vs LAYER (context collapsed to ctx>=CTXLO node-means). This resolves
the SECOND axis: for each context-length bin AND each layer, bin the per-occurrence activations by
context length, take per-node means WITHIN the bin, and run the same leave-one-node-out ridge coord
probe (2-D target = graph coords, best-alpha, matches coord_decode). The result is a (context x
layer) heatmap of geometric decodability -- where and when the in-context grid becomes linearly
readable.

Reads a per-occurrence acts_sub npz (layer_* + meta_node + meta_context_length). No model / GPU:
the activations are already captured. Graph coords are rebuilt from GRAPH for the probe target.

Env: ACTS TAG GRAPH NBINS(12) WINFRAC(0.15) WINMIN(15) OUTDIR
Out: <OUTDIR>/context_layer_probe_<TAG>_<graph>.json + .pdf
"""
from __future__ import annotations
import os, json
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from config import get_config
import graph as G

ACTS = os.environ.get("ACTS", "runs/v2/square_grid/Llama_acts_sub.npz")
TAG = os.environ.get("TAG", "Llama"); GRAPH = os.environ.get("GRAPH", "square_grid")
NBINS = int(os.environ.get("NBINS", "12"))
WINFRAC = float(os.environ.get("WINFRAC", "0.15")); WINMIN = int(os.environ.get("WINMIN", "15"))
OUTDIR = os.environ.get("OUTDIR", "runs/axes/1_decomposition/context_layer_probe")
GKW = {"square_grid": dict(graph_type="grid", grid_rows=4, grid_cols=4),
       "ring": dict(graph_type="ring", ring_size=16), "hex": dict(graph_type="hex", hex_rows=4, hex_cols=4),
       "days": dict(graph_type="ring", ring_size=7, word_set="days")}
ALPHAS = [1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6]


def r2(y, yh):
    tot = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ((y - yh) ** 2).sum() / tot) if tot > 0 else float("nan")


def prep_folds(Mn):
    n = Mn.shape[0]; folds = []
    for k in range(n):
        idx = [i for i in range(n) if i != k]
        Xtr = Mn[idx]; xk = Mn[k:k + 1]
        mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
        Xs = (Xtr - mu) / sd; xks = (xk - mu) / sd
        U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
        folds.append((np.array(idx), (xks @ Vt.T).ravel(), U.T.copy(), S))
    return folds


def loo_r2_bestalpha(folds, y):
    n = len(folds); best = -18.0
    for a in ALPHAS:
        pred = np.zeros((n, 2))
        for k, (idx, proj, UT, S) in enumerate(folds):
            ytr = y[idx]; ymu = ytr.mean(0)
            coef = (S / (S ** 2 + a))[:, None] * (UT @ (ytr - ymu))
            pred[k] = proj @ coef + ymu
        s = r2(y[:, 0], pred[:, 0]) + r2(y[:, 1], pred[:, 1])
        if s > best: best = s; keep = (r2(y[:, 0], pred[:, 0]), r2(y[:, 1], pred[:, 1]))
    return 0.5 * (keep[0] + keep[1])


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    npz = np.load(ACTS, allow_pickle=True)
    node = np.asarray(npz["meta_node"]); ctx = np.asarray(npz["meta_context_length"])
    nL = sum(1 for k in npz.files if k.startswith("layer_"))
    cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH])
    graph = G.build_graph(cfg); n = graph.n_nodes; coords = np.array(graph.coords, float)

    cmax = int(np.percentile(ctx, 99)); cmin = max(2, int(ctx.min()))
    centers = np.unique(np.linspace(cmin, cmax, NBINS).astype(int))
    # precompute occurrence indices per (bin) and their nodes
    bin_idx = []
    for c in centers:
        w = max(WINMIN, int(WINFRAC * c))
        sel = np.where(np.abs(ctx - c) <= w)[0]
        bin_idx.append(sel)

    R = np.full((len(centers), nL), np.nan)
    for L in range(nL):
        H = npz[f"layer_{L}"].astype(np.float32)
        for bi, sel in enumerate(bin_idx):
            if len(sel) < n: continue
            nd = node[sel]; means = np.zeros((n, H.shape[1])); ok = True
            for j in range(n):
                m = sel[nd == j]
                if len(m) == 0: ok = False; break
                means[j] = H[m].mean(0)
            if not ok: continue
            folds = prep_folds(means)
            R[bi, L] = loo_r2_bestalpha(folds, coords)
        if L % 8 == 0: print(f"[{TAG}/{GRAPH}] layer {L}/{nL}", flush=True)

    out = {"tag": TAG, "graph": GRAPH, "n": n, "nL": nL, "centers": centers.tolist(),
           "R2": np.where(np.isfinite(R), R, None).tolist(),
           "peak": float(np.nanmax(R)), "peak_at": [int(x) for x in np.unravel_index(np.nanargmax(R), R.shape)]}
    json.dump(out, open(f"{OUTDIR}/context_layer_probe_{TAG}_{GRAPH}.json", "w"), indent=2)
    make_fig(out, R, centers, f"{OUTDIR}/context_layer_probe_{TAG}_{GRAPH}.pdf")
    pk = out["peak_at"]
    print(f"[{TAG}/{GRAPH}] DONE peak R2={out['peak']:.2f} at ctx~{centers[pk[0]]} layer {pk[1]} "
          f"-> {OUTDIR}/context_layer_probe_{TAG}_{GRAPH}.pdf", flush=True)


def make_fig(out, R, centers, path):
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(R, aspect="auto", origin="lower", cmap="viridis", vmin=-0.2, vmax=1.0,
                       extent=[0, out["nL"], 0, len(centers)])
        ax.set_yticks(np.arange(len(centers)) + 0.5); ax.set_yticklabels(centers, fontsize=7)
        ax.set_xlabel("layer"); ax.set_ylabel("context length (node step)")
        ax.set_title(f"{out['tag']} {out['graph']}: leave-one-node-out coord-probe R²  (context × layer)\n"
                     f"peak {out['peak']:.2f} at ctx~{centers[out['peak_at'][0]]}, layer {out['peak_at'][1]}", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=.046, label="probe R²")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
