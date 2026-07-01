"""What is the massive-activation outlier dimension about, and which distance metric
de-confounds it?

For each (model, graph), per layer (node means over ctx>=CTXLO):
  OUTLIER CHARACTERISATION
    - top-1 / top-3 dims' share of total across-occurrence variance
    - the dominant dim's index, its mean value (bias magnitude in units of the typical
      dim's std), and its NODE-SPREAD RATIO = std(per-node means of that dim) /
      std(that dim over all occurrences). ~0 => the big dim is a (near-)constant bias
      that carries no node identity (a massive activation), so it only inflates Euclidean
      distance; >~1 would mean it actually separates nodes.
  METRIC COMPARISON (node-mean RSA vs graph distance)
    raw (Euclid) | zscore (per-dim standardise) | drop1 / drop10 (remove top-var dims) | cosine

Reads saved activations only (no models/GPU). Version-aware (CM_VERSION).
Env: CM_VERSION GRAPHS(days,square_grid) MODELS_FILTER CTXLO(100) OUTDIR
Out: <OUTDIR>/outlier.json , <OUTDIR>/outlier_metrics.pdf
"""
from __future__ import annotations
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from dataclasses import replace
from config import get_config
import graph as G
import paths as P

GKW = {"days": dict(graph_type="ring", ring_size=7, word_set="days"),
       "square_grid": dict(graph_type="grid", grid_rows=4, grid_cols=4),
       "ring": dict(graph_type="ring", ring_size=16),
       "hex": dict(graph_type="hex", hex_rows=4, hex_cols=4)}
GRAPHS = os.environ.get("GRAPHS", "days,square_grid").split(",")
_mf = os.environ.get("MODELS_FILTER")
MODELS = [m for m in P.MODELS if not _mf or m in set(_mf.split(","))]
CTXLO = int(os.environ.get("CTXLO", "100"))
OUTDIR = os.environ.get("OUTDIR", "/workspace/cross-model/runs/induction-head/outlier")
METRICS = ["raw", "zscore", "drop1", "drop10", "cosine"]


def sp(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def node_means(X, node, mask, n):
    return np.stack([X[mask & (node == k)].mean(0) for k in range(n)])


def rsa(H, GD, iu, metric, drop=None):
    if metric == "cosine":
        Hn = H / np.clip(np.linalg.norm(H, axis=1, keepdims=True), 1e-9, None)
        D = 1 - Hn @ Hn.T
        return sp(D[iu], GD)
    Hc = H.copy()
    if drop is not None and len(drop):
        Hc[:, drop] = 0.0
    R = np.linalg.norm(Hc[:, None] - Hc[None], axis=2)[iu]
    return sp(R, GD)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    out = {"ctxlo": CTXLO, "models": {}}
    for m in MODELS:
        out["models"].setdefault(m, {})
        for gname in GRAPHS:
            try:
                z = np.load(P.acts_path(gname, m), allow_pickle=False)
            except FileNotFoundError:
                print(f"skip {m}/{gname}: no acts", flush=True); continue
            cfg = replace(get_config("gemma_qwen"), **GKW[gname])
            gr = G.build_graph(cfg); n = gr.n_nodes
            iu = np.triu_indices(n, 1); GD = gr.distance_matrix()[iu]
            node = z["meta_node"]; mask = z["meta_context_length"] >= CTXLO
            layers = sorted(int(l) for l in z["_layers"])
            rec = {"layers": layers, "rsa": {k: [] for k in METRICS}, "outlier": {}}
            # standardise stats per layer; node-mean RSA under each metric
            for L in layers:
                X = z[f"layer_{L}"].astype(np.float64)
                Xm = X[mask]
                v = Xm.var(0); order = np.argsort(v)[::-1]
                d1, d10 = order[:1], order[:10]
                # outlier characterisation
                H = node_means(X, node, mask, n)
                top = int(order[0]); typ_sd = float(np.sqrt(np.median(v)))
                node_spread = float(H[:, top].std())
                rec["outlier"][L] = {
                    "dim": top,
                    "var_frac_top1": float(v[order[0]] / v.sum()),
                    "var_frac_top3": float(v[order[:3]].sum() / v.sum()),
                    "mean_over_typical_sd": float(Xm[:, top].mean() / max(typ_sd, 1e-9)),
                    "node_spread_ratio": float(node_spread / max(np.sqrt(v[top]), 1e-9)),
                }
                # metrics
                rec["rsa"]["raw"].append(rsa(H, GD, iu, "raw"))
                mu, sd = Xm.mean(0), Xm.std(0)
                Hz = node_means((X - mu) / np.clip(sd, 1e-9, None), node, mask, n)
                rec["rsa"]["zscore"].append(rsa(Hz, GD, iu, "raw"))
                rec["rsa"]["drop1"].append(rsa(H, GD, iu, "raw", drop=d1))
                rec["rsa"]["drop10"].append(rsa(H, GD, iu, "raw", drop=d10))
                rec["rsa"]["cosine"].append(rsa(H, GD, iu, "cosine"))
            out["models"][m][gname] = rec
            # report the peak-raw layer's outlier
            praw = int(np.nanargmax(rec["rsa"]["raw"]))
            best = {k: float(np.nanmax(rec["rsa"][k])) for k in METRICS}
            ol = rec["outlier"][layers[len(layers)//2]]
            print(f"[{m}/{gname}] peakRSA  raw={best['raw']:+.2f} zscore={best['zscore']:+.2f} "
                  f"drop1={best['drop1']:+.2f} drop10={best['drop10']:+.2f} cos={best['cosine']:+.2f} "
                  f"| mid-layer outlier dim{ol['dim']} var%={ol['var_frac_top1']*100:.0f} "
                  f"mean/sd={ol['mean_over_typical_sd']:+.0f} node_spread={ol['node_spread_ratio']:.2f}", flush=True)
    json.dump(out, open(f"{OUTDIR}/outlier.json", "w"), indent=2)
    make_fig(out, f"{OUTDIR}/outlier_metrics.pdf")
    print(f"DONE -> {OUTDIR}/outlier.json + outlier_metrics.pdf", flush=True)


def make_fig(out, path):
    order = ["Llama", "Gemma", "Qwen"]
    models = [m for m in order if m in out["models"]] + [m for m in out["models"] if m not in order]
    rows = [(m, g) for m in models for g in out["models"][m]]
    colors = {"raw": "k", "zscore": "tab:green", "drop1": "tab:orange", "drop10": "tab:red", "cosine": "tab:blue"}
    with PdfPages(path) as pdf:
        for m, g in rows:
            r = out["models"][m][g]; Ls = r["layers"]
            fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
            for k in METRICS:
                ax[0].plot(Ls, r["rsa"][k], "-o", ms=3, color=colors[k], label=k)
            ax[0].axhline(0, color=".85", lw=.6); ax[0].set_xlabel("layer"); ax[0].set_ylabel("node-mean RSA")
            ax[0].set_title(f"{m}/{g}: RSA by metric", fontsize=10); ax[0].legend(fontsize=7)
            vf = [r["outlier"][str(L)]["var_frac_top1"] if str(L) in r["outlier"] else r["outlier"][L]["var_frac_top1"] for L in Ls]
            ns = [r["outlier"][str(L)]["node_spread_ratio"] if str(L) in r["outlier"] else r["outlier"][L]["node_spread_ratio"] for L in Ls]
            ax[1].plot(Ls, vf, "-o", ms=3, color="purple", label="top-1 dim variance share")
            ax[1].plot(Ls, ns, "-o", ms=3, color="gray", label="top dim node-spread ratio")
            ax[1].axhline(0, color=".85", lw=.6); ax[1].set_xlabel("layer"); ax[1].set_ylim(-0.05, 1.05)
            ax[1].set_title(f"{m}/{g}: outlier dim — dominance & whether it codes node identity", fontsize=9)
            ax[1].legend(fontsize=7)
            fig.suptitle(f"{m} / {g}: massive-activation outlier and metric de-confounding", fontsize=11)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
