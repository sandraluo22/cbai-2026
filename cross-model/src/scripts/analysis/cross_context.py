"""Probe geometry as a function of CONTEXT LENGTH.

We bin every node occurrence by its context position (how far into the walk it appeared) and
accumulate per-bin, per-layer node-means. Then:
  1-model  : coord-probe leave-one-node-out R² for every (context-bin x layer) -> heatmap per
             model. Shows WHERE (layer) and WHEN (context length) the map crystallizes.
  cross-model: at each model's peak layer, ridge-align A's per-bin means to B's per-bin means
             (LOO) for every context bin -> alignment R² vs context length, per pair.

MODE=capture saves per-model [nbins,nL,n,d] means (respects two-pass HF_HOME); MODE=combine
loads them and builds the figures locally (no GPU).

Env: PRESET MODELS_FILTER MODE(capture|combine|full) GRAPH(square_grid) NWALKS(24) WLEN(300)
     NBINS(8) ALPHA(1000) OUTDIR DEVICE
Out: <OUTDIR>/cross_context_<graph>.pdf + .json
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
    MODELS = [("Aa", "distilgpt2", None), ("Bb", "distilgpt2", None)]
_mf = os.environ.get("MODELS_FILTER")
if _mf:
    MODELS = [m for m in MODELS if m[0] in set(_mf.split(","))]
MODE = os.environ.get("MODE", "full")
GKW = {"square_grid": dict(graph_type="grid", grid_rows=4, grid_cols=4),
       "ring": dict(graph_type="ring", ring_size=16),
       "hex": dict(graph_type="hex", hex_rows=4, hex_cols=4),
       "days": dict(graph_type="ring", ring_size=7, word_set="days")}
GRAPH = os.environ.get("GRAPH", "square_grid")
NWALKS = int(os.environ.get("NWALKS", "24"))
WLEN = int(os.environ.get("WLEN", "300"))
NBINS = int(os.environ.get("NBINS", "8"))
ALPHA = float(os.environ.get("ALPHA", "1000"))
KPC = 6
ALPHAS = [1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6]
OUTDIR = os.environ.get("OUTDIR", "/workspace/cross-model/runs/induction-head/cross_context")


def load_with_fallback(hf, mirror, cfg):
    try:
        return M.load_model(hf, cfg)
    except Exception:
        return M.load_model(mirror, cfg)


def bin_edges():
    return np.linspace(1, WLEN, NBINS + 1)


@torch.no_grad()
def ctx_binned_means(hf, mirror, cfg, graph, dev):
    """[nbins, nL, n, d] per-bin per-layer node-mean, and [nbins, n] counts."""
    model, tok = load_with_fallback(hf, mirror, cfg)
    cm = model.config; blocks = M._decoder_blocks(model); nL = cm.num_hidden_layers; n = graph.n_nodes
    edges = bin_edges()
    grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    hs = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
    nsum = np.zeros((NBINS, nL, n, cm.hidden_size), np.float32); ncnt = np.zeros((NBINS, n))
    for wk in G.generate_walks(graph, cfg):
        ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
        spans = resolve_token_spans(tok, wk); nodes = wk.nodes; cl = np.arange(1, len(nodes) + 1); grabbed.clear()
        model(input_ids=ids); single = [t[-1] for t in spans]
        b_of = np.clip(np.digitize(cl, edges) - 1, 0, NBINS - 1)
        for L in range(nL):
            rows = grabbed[L][0][single].float().cpu().numpy()
            for s in range(len(nodes)):
                nsum[b_of[s], L, nodes[s]] += rows[s]
                if L == 0: ncnt[b_of[s], nodes[s]] += 1
    for h in hs: h.remove()
    del model, tok; gc.collect()
    if torch and torch.cuda.is_available(): torch.cuda.empty_cache()
    return nsum, ncnt


def _r2(y, yh):
    tot = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ((y - yh) ** 2).sum() / tot) if tot > 0 else float("nan")


def coord_loo_r2(H, coords):
    n = H.shape[0]
    if not np.isfinite(H).all(): return float("nan")
    mu = H.mean(0); sd = H.std(0) + 1e-6; Xs = (H - mu) / sd; Yc = coords - coords.mean(0)
    folds = []
    for k in range(n):
        idx = [i for i in range(n) if i != k]
        U, S, Vt = np.linalg.svd(Xs[idx], full_matrices=False)
        folds.append((np.array(idx), (Xs[k] @ Vt.T), U.T.copy(), S))
    best = -9.0
    for a in ALPHAS:
        pred = np.zeros((n, 2))
        for k, (idx, proj, UT, S) in enumerate(folds):
            ytr = Yc[idx]; ymu = ytr.mean(0)
            pred[k] = proj @ ((S / (S ** 2 + a))[:, None] * (UT @ (ytr - ymu))) + ymu
        best = max(best, 0.5 * (_r2(Yc[:, 0], pred[:, 0]) + _r2(Yc[:, 1], pred[:, 1])))
    return float(best)


def pca(B, k):
    Bc = B - B.mean(0); U, S, Vt = np.linalg.svd(Bc, full_matrices=False)
    kk = min(k, Vt.shape[0]); return U[:, :kk] * S[:kk]


def align_loo(A, Bpc, a):
    """LOO ridge pooled R²: predict Bpc from A (both n x .)."""
    n = A.shape[0]; mu = A.mean(0); sd = A.std(0) + 1e-6; Xs = (A - mu) / sd
    ssr = 0.0; sst = 0.0
    for k in range(n):
        idx = [i for i in range(n) if i != k]
        U, S, Vt = np.linalg.svd(Xs[idx], full_matrices=False)
        proj = Xs[k] @ Vt.T; ytr = Bpc[idx]; ymu = ytr.mean(0)
        pred = proj @ ((S / (S ** 2 + a))[:, None] * (U.T @ (ytr - ymu))) + ymu
        ssr += ((pred - Bpc[k]) ** 2).sum(); sst += ((Bpc[k] - ymu) ** 2).sum()
    return float(1 - ssr / sst) if sst > 0 else float("nan")


def main():
    dev = os.environ.get("DEVICE", "cpu" if os.environ.get("PRESET") == "smoke" else "cuda")
    os.makedirs(OUTDIR, exist_ok=True)
    cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH], n_walks=NWALKS, walk_length=WLEN, device=dev)
    graph = G.build_graph(cfg); coords = np.array(graph.coords, float)
    if MODE in ("capture", "full"):
        for tag, hf, mirror in MODELS:
            print(f"[{tag}] ctx-binned capture ({GRAPH})", flush=True)
            nsum, ncnt = ctx_binned_means(hf, mirror, cfg, graph, dev)
            np.savez(f"{OUTDIR}/ctxmeans_{GRAPH}_{tag}.npz", nsum=nsum, ncnt=ncnt)
            print(f"[{tag}] {nsum.shape} -> ctxmeans_{GRAPH}_{tag}.npz", flush=True)
        if MODE == "capture":
            print("CAPTURE_DONE", flush=True); return
    # ---- combine ----
    pre = f"ctxmeans_{GRAPH}_"; means = {}; cnts = {}
    for f in sorted(glob.glob(f"{OUTDIR}/{pre}*.npz")):
        tag = os.path.basename(f)[len(pre):-4]; d = np.load(f)
        means[tag] = d["nsum"].astype(float); cnts[tag] = d["ncnt"]
    tags = list(means.keys())
    edges = bin_edges(); bmid = 0.5 * (edges[:-1] + edges[1:])
    # per-model (bin x layer) coord-probe R² heatmap (each model has its own layer count)
    heat = {}; peakL = {}
    for tag in tags:
        H4 = means[tag]; C = cnts[tag]                              # [nbins,nLt,n,d], [nbins,n]
        nLt = H4.shape[1]
        Hm = np.where(C[:, None, :, None] > 0, H4 / np.maximum(C[:, None, :, None], 1), np.nan)
        R = np.full((NBINS, nLt), np.nan)
        for b in range(NBINS):
            for L in range(nLt):
                R[b, L] = coord_loo_r2(Hm[b, L], coords)
        heat[tag] = R
        last = np.where(np.isfinite(R[-1]), R[-1], -9)              # peak layer from full-context bin
        peakL[tag] = int(np.argmax(last))
        print(f"[{tag}] peak layer (full ctx) L{peakL[tag]} R²={R[-1, peakL[tag]]:.2f}", flush=True)
    # cross-model per-bin alignment at peak layers
    cross = {}
    for A, B in itertools.combinations(tags, 2):
        for (X, Y) in [(A, B), (B, A)]:
            HmX = means[X]; CX = cnts[X]; HmY = means[Y]; CY = cnts[Y]
            lx, ly = peakL[X], peakL[Y]; curve = []
            for b in range(NBINS):
                hx = np.where(CX[b][:, None] > 0, HmX[b, lx] / np.maximum(CX[b][:, None], 1), np.nan)
                hy = np.where(CY[b][:, None] > 0, HmY[b, ly] / np.maximum(CY[b][:, None], 1), np.nan)
                ok = np.isfinite(hx).all(1) & np.isfinite(hy).all(1)
                curve.append(align_loo(hx[ok], pca(hy[ok], KPC), ALPHA) if ok.sum() >= 6 else float("nan"))
            cross[f"{X}->{Y}"] = curve
            print(f"[{X}->{Y}] align R² over ctx: {[round(v,2) for v in curve]}", flush=True)
    json.dump({"graph": GRAPH, "nbins": NBINS, "bin_mid": bmid.tolist(), "peakL": peakL,
               "heat": {t: heat[t].tolist() for t in tags}, "cross": cross},
              open(f"{OUTDIR}/cross_context_{GRAPH}.json", "w"))
    make_fig(tags, heat, cross, bmid, peakL, f"{OUTDIR}/cross_context_{GRAPH}.pdf")
    print(f"DONE -> {OUTDIR}/cross_context_{GRAPH}.pdf", flush=True)


def make_fig(tags, heat, cross, bmid, peakL, path):
    order = ["Llama", "Gemma", "Qwen"]; tags = [t for t in order if t in tags] + [t for t in tags if t not in order]
    with PdfPages(path) as pdf:
        # page 1: per-model heatmap bin x layer
        fig, ax = plt.subplots(1, len(tags), figsize=(5.2 * len(tags), 4.8), squeeze=False)
        for j, t in enumerate(tags):
            R = np.array(heat[t])
            im = ax[0, j].imshow(np.clip(R, 0, 1), aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1,
                                 extent=[0, R.shape[1], bmid[0], bmid[-1]])
            fig.colorbar(im, ax=ax[0, j], fraction=.046)
            ax[0, j].set_xlabel("layer"); ax[0, j].set_ylabel("context length (bin mid)")
            ax[0, j].set_title(f"{t}  coord-probe R² (peakL={peakL[t]})", fontsize=9)
        fig.suptitle(f"coord-probe R² vs (context length x layer) — where/when the map crystallizes", fontsize=11)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        # page 2: cross-model alignment vs context
        fig, axx = plt.subplots(1, 1, figsize=(7, 5))
        for key, curve in cross.items():
            axx.plot(bmid, curve, "-o", ms=4, label=key)
        axx.set_xlabel("context length (bin mid)"); axx.set_ylabel("cross-model align R² (peak layers)")
        axx.set_ylim(-0.3, 1.0); axx.axhline(0, color=".7", lw=.6); axx.legend(fontsize=8)
        axx.set_title("cross-model probe alignment vs context length", fontsize=11)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
