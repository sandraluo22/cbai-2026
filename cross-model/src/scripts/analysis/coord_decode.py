"""Linear decodability of grid coordinates, per layer, WITHOUT node-identity leakage.

Probe target: given a node's in-context mean representation, predict its grid (row, col).
The naive "held out by walk" probe LEAKS -- every node appears in training, so the map just
memorizes each node's identity->coordinate (16 points are trivially linearly fittable in high
dim -> R2=1, meaningless). Instead we do LEAVE-ONE-NODE-OUT: train a ridge map on 15 node-means,
predict the 16th node's coordinate. A held-out node lands correctly ONLY if the geometry
genuinely constrains its position -- so LOO R2 measures real geometric structure, per axis.
Both axes high => 2D structure (a 1D recency code can place at most one axis).

Env: PRESET MODELS_FILTER GRAPH(square_grid) NWALKS(24) WLEN(300) CTXLO(100) OUTDIR DEVICE
Out: <OUTDIR>/coord_decode_<graph>.json + .pdf
"""
from __future__ import annotations
import os, json, gc
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

PRESET = os.environ.get("PRESET", "gemma_qwen")
if PRESET == "smoke":
    MODELS = [("distilgpt2", "distilgpt2", None)]
else:
    MODELS = [("Llama", "meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"),
              ("Gemma", "google/gemma-2-9b", "unsloth/gemma-2-9b"),
              ("Qwen",  "Qwen/Qwen3-8B-Base", None)]
_mf = os.environ.get("MODELS_FILTER")
if _mf:
    MODELS = [m for m in MODELS if m[0] in set(_mf.split(","))]
GKW = {"square_grid": dict(graph_type="grid", grid_rows=4, grid_cols=4),
       "ring": dict(graph_type="ring", ring_size=16),
       "hex": dict(graph_type="hex", hex_rows=4, hex_cols=4),
       "days": dict(graph_type="ring", ring_size=7, word_set="days")}
GRAPH = os.environ.get("GRAPH", "square_grid")
NWALKS = int(os.environ.get("NWALKS", "24"))
WLEN = int(os.environ.get("WLEN", "300"))
CTXLO = int(os.environ.get("CTXLO", "100"))
NPERM = int(os.environ.get("NPERM", "200"))
LABELS = os.environ.get("LABELS", "auto")   # auto | gridlabel (force row,col) | shuffle (permute labels)
SUFFIX = os.environ.get("SUFFIX", "")        # output filename suffix (so controls don't clobber)
ALPHAS = [1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6]
OUTDIR = os.environ.get("OUTDIR", "/workspace/cross-model/runs/induction-head/coord_decode")


def load_with_fallback(tag, hf, mirror, cfg):
    try:
        return M.load_model(hf, cfg)
    except Exception:
        return M.load_model(mirror, cfg)


def r2(y, yh):
    tot = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ((y - yh) ** 2).sum() / tot) if tot > 0 else float("nan")


def ridge_pred(Xtr, ytr_c, xk, a):
    U, S, Vt = np.linalg.svd(Xtr, full_matrices=False)
    coef = Vt.T @ ((S / (S ** 2 + a))[:, None] * (U.T @ ytr_c))
    return xk @ coef


def prep_folds(Mn):
    """Precompute per-fold SVD (label-independent) so many label-permutations are cheap."""
    n = Mn.shape[0]; folds = []
    for k in range(n):
        idx = [i for i in range(n) if i != k]
        Xtr = Mn[idx]; xk = Mn[k:k + 1]
        mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
        Xs = (Xtr - mu) / sd; xks = (xk - mu) / sd
        U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
        folds.append((np.array(idx), (xks @ Vt.T).ravel(), U.T.copy(), S))   # idx, proj[15], U.T[15,15], S[15]
    return folds


def loo_r2_bestalpha(folds, y):
    """LOO predictions per axis, choosing alpha (on the grid) that maximizes mean LOO R2."""
    n = len(folds); best = (-9.0, -9.0)
    for a in ALPHAS:
        pred = np.zeros((n, 2))
        for k, (idx, proj, UT, S) in enumerate(folds):
            ytr = y[idx]; ymu = ytr.mean(0)
            coef = (S / (S ** 2 + a))[:, None] * (UT @ (ytr - ymu))          # [15,2]
            pred[k] = proj @ coef + ymu
        rr = r2(y[:, 0], pred[:, 0]); rc = r2(y[:, 1], pred[:, 1])
        if rr + rc > best[0] + best[1]:
            best = (rr, rc)
    return best


@torch.no_grad()
def node_means(model, tok, blocks, cm, walks, n, dev):
    nL = cm.num_hidden_layers; grabbed = {}
    def mk(L):
        def hh(_m, _i, out): grabbed[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hh
    hs = [blocks[L].register_forward_hook(mk(L)) for L in range(nL)]
    nsum = {L: np.zeros((n, cm.hidden_size)) for L in range(nL)}; ncnt = np.zeros(n)
    try:
        for wk in walks:
            ids = tok(wk.text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            spans = resolve_token_spans(tok, wk); nodes = wk.nodes; cl = np.arange(1, len(nodes) + 1); grabbed.clear()
            model(input_ids=ids)
            for L in range(nL):
                for s in range(len(nodes)):
                    if cl[s] >= CTXLO:
                        nsum[L][nodes[s]] += grabbed[L][0, spans[s][-1]].float().cpu().numpy()
                        if L == 0: ncnt[nodes[s]] += 1
    finally:
        for h in hs: h.remove()
    cn = np.maximum(ncnt, 1)
    return {L: nsum[L] / cn[:, None] for L in range(nL)}


def main():
    dev = os.environ.get("DEVICE", "cpu" if PRESET == "smoke" else "cuda")
    os.makedirs(OUTDIR, exist_ok=True)
    out = {"graph": GRAPH, "models": {}}
    for tag, hf, mirror in MODELS:
        cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH], n_walks=NWALKS, walk_length=WLEN, device=dev)
        graph = G.build_graph(cfg); n = graph.n_nodes; walks = G.generate_walks(graph, cfg)
        if GRAPH == "ring":                                     # circular manifold -> (cos, sin) target
            th = 2 * np.pi * np.arange(n) / n
            coords = np.stack([np.cos(th), np.sin(th)], 1); axes = ("cos", "sin")
        else:
            coords = np.array([(i // 4, i % 4) for i in range(n)], float); axes = ("row", "col")
        if LABELS == "gridlabel":                               # force GRID (row,col) labels (e.g. grid-probe on ring acts)
            coords = np.array([(i // 4, i % 4) for i in range(n)], float); axes = ("row", "col")
        elif LABELS == "shuffle":                               # scramble node->label (specificity control; should fail)
            perm = np.random.default_rng(7).permutation(n); coords = coords[perm]; axes = (axes[0] + "_shuf", axes[1] + "_shuf")
        print(f"[{tag}] loading", flush=True)
        model, tok = load_with_fallback(tag, hf, mirror, cfg)
        cm = model.config; blocks = M._decoder_blocks(model); nL = cm.num_hidden_layers
        Mn = node_means(model, tok, blocks, cm, walks, n, dev)
        del model, tok; gc.collect()
        if torch and torch.cuda.is_available(): torch.cuda.empty_cache()
        rng = np.random.default_rng(0)
        r2row = []; r2col = []; nr_mean = []; nr_std = []; nc_mean = []; nc_std = []; p_row = []; p_col = []
        for L in range(nL):
            folds = prep_folds(Mn[L])
            rr, rc = loo_r2_bestalpha(folds, coords)                 # real (same alpha-on-LOO procedure)
            nulls = np.array([loo_r2_bestalpha(folds, coords[rng.permutation(n)]) for _ in range(NPERM)])
            r2row.append(rr); r2col.append(rc)
            nr_mean.append(float(nulls[:, 0].mean())); nr_std.append(float(nulls[:, 0].std()))
            nc_mean.append(float(nulls[:, 1].mean())); nc_std.append(float(nulls[:, 1].std()))
            p_row.append(float((np.sum(nulls[:, 0] >= rr) + 1) / (NPERM + 1)))
            p_col.append(float((np.sum(nulls[:, 1] >= rc) + 1) / (NPERM + 1)))
        peak = int(np.nanargmax([(a + b) / 2 for a, b in zip(r2row, r2col)]))
        rec = {"n_layers": nL, "n_perm": NPERM, "axes": list(axes), "graph": GRAPH, "r2_row": r2row, "r2_col": r2col,
               "null_row_mean": nr_mean, "null_row_std": nr_std, "null_col_mean": nc_mean, "null_col_std": nc_std,
               "p_row": p_row, "p_col": p_col, "peak_layer": peak,
               "peak_r2_row": r2row[peak], "peak_r2_col": r2col[peak],
               "peak_null_row": nr_mean[peak], "peak_null_col": nc_mean[peak],
               "peak_p_row": p_row[peak], "peak_p_col": p_col[peak],
               "note": "leave-one-node-out; permutation null shuffles node->coord labels (same alpha-on-LOO procedure)"}
        out["models"][tag] = rec
        print(f"[{tag}/{GRAPH}] peak L{peak}: {axes[0]}={r2row[peak]:.3f} (null {nr_mean[peak]:+.3f}±{nr_std[peak]:.3f}, p={p_row[peak]:.3f}) "
              f"{axes[1]}={r2col[peak]:.3f} (null {nc_mean[peak]:+.3f}±{nc_std[peak]:.3f}, p={p_col[peak]:.3f})", flush=True)
        del Mn; gc.collect()

    prev = f"{OUTDIR}/coord_decode_{GRAPH}{SUFFIX}.json"
    if os.path.exists(prev):
        p = json.load(open(prev)).get("models", {}); p.update(out["models"]); out["models"] = p
    json.dump(out, open(prev, "w"), indent=2)
    make_fig(out, f"{OUTDIR}/coord_decode_{GRAPH}{SUFFIX}.pdf")
    print(f"DONE -> {prev}", flush=True)


def make_fig(out, path):
    order = ["Llama", "Gemma", "Qwen"]; models = [m for m in order if m in out["models"]] + [m for m in out["models"] if m not in order]
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.6), squeeze=False)
        for j, m in enumerate(models):
            r = out["models"][m]; L = list(range(r["n_layers"])); axn = r.get("axes", ["row", "col"])
            ax[0, j].plot(L, r["r2_row"], "-o", ms=3, label=f"LOO R² {axn[0]}", color="tab:blue")
            ax[0, j].plot(L, r["r2_col"], "-o", ms=3, label=f"LOO R² {axn[1]}", color="tab:red")
            if "null_row_mean" in r:
                nm = np.array(r["null_row_mean"]); ns = np.array(r["null_row_std"])
                ax[0, j].fill_between(L, nm - 2 * ns, nm + 2 * ns, color="gray", alpha=.3, label="perm null (row) ±2σ")
                cm_ = np.array(r["null_col_mean"]); cs = np.array(r["null_col_std"])
                ax[0, j].fill_between(L, cm_ - 2 * cs, cm_ + 2 * cs, color="orange", alpha=.2, label="perm null (col) ±2σ")
            ax[0, j].axhline(0, color=".7", lw=.6); ax[0, j].set_ylim(-0.6, 1.0)
            ax[0, j].set_xlabel("layer"); ax[0, j].set_ylabel("leave-one-node-out R²")
            ax[0, j].set_title(f"{m} (peak L{r['peak_layer']}: row={r['peak_r2_row']:.2f} col={r['peak_r2_col']:.2f})", fontsize=8)
            ax[0, j].legend(fontsize=8)
        fig.suptitle(f"[{out['graph']}] leave-one-node-out linear decodability of grid coordinates per layer", fontsize=11)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
