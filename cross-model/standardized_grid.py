"""Does Qwen's in-context grid surface once the massive-activation dimension is
demoted? Re-measure grid structure per layer on the saved all-layer subsample,
comparing RAW vs STANDARDIZED activations.

Metric: RSA -- Spearman correlation between each model's 16-node representational
dissimilarity matrix (pairwise Euclidean distance of per-node-mean vectors) and
the true grid (Manhattan) distance matrix. Positive => graph geometry encoded.
Preprocessings: raw | z-score per dim | drop top-5 highest-variance dims.
Runs locally from runs/gemma_qwen_all/acts_sub_*.npz (no GPU).
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import get_config
import graph as G
from reproduce import grid_recovery_score

RUN = "runs/gemma_qwen_all"
CFG = get_config("gemma_qwen")
GRAPH = G.build_grid_graph(CFG)
IU = np.triu_indices(16, 1)
GD = GRAPH.grid_distance_matrix()[IU]            # true grid dissimilarities


def rankdata(x):
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(len(x))
    return ranks


def spearman(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def node_means(X, node):
    return np.stack([X[node == k].mean(0) for k in range(16)])


def rdm(M):
    D = np.linalg.norm(M[:, None, :] - M[None, :, :], axis=2)
    return D[IU]


def analyze(npz_path, layers):
    z = np.load(npz_path, allow_pickle=False)
    node = z["meta_node"]
    rows = []
    for L in layers:
        X = z[f"layer_{L}"].astype(np.float32)
        mu, sd = X.mean(0), X.std(0)
        sd = np.where(sd < 1e-6, 1.0, sd)
        Xz = (X - mu) / sd
        keep = np.argsort(X.var(0))[:-5]                     # drop top-5 var dims
        Xd = X[:, keep]
        rows.append({
            "layer": int(L),
            "rsa_raw":  spearman(rdm(node_means(X, node)), GD),
            "rsa_z":    spearman(rdm(node_means(Xz, node)), GD),
            "rsa_drop5": spearman(rdm(node_means(Xd, node)), GD),
            "pca_raw":  grid_recovery_score(node_means(X, node), GRAPH)["distance_corr"],
            "pca_z":    grid_recovery_score(node_means(Xz, node), GRAPH)["distance_corr"],
        })
    return rows


def load_layers(p):
    z = np.load(p, allow_pickle=False)
    return [int(l) for l in z["_layers"]]


def main():
    g = analyze(f"{RUN}/acts_sub_gemma.npz", load_layers(f"{RUN}/acts_sub_gemma.npz"))
    q = analyze(f"{RUN}/acts_sub_qwen.npz", load_layers(f"{RUN}/acts_sub_qwen.npz"))
    json.dump({"gemma": g, "qwen": q}, open(f"{RUN}/standardized_grid.json", "w"), indent=2)

    def peak(rows, key):
        b = max(rows, key=lambda r: r[key])
        return b["layer"], b[key]

    print("            raw-RSA peak     z-scored-RSA peak    drop5-RSA peak")
    for tag, rows in (("Gemma", g), ("Qwen", q)):
        lr, vr = peak(rows, "rsa_raw"); lz, vz = peak(rows, "rsa_z"); ld, vd = peak(rows, "rsa_drop5")
        print(f"  {tag:5}   L{lr:>2}={vr:+.3f}        L{lz:>2}={vz:+.3f}         L{ld:>2}={vd:+.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, tag, rows in ((axes[0], "Gemma", g), (axes[1], "Qwen", q)):
        L = [r["layer"] for r in rows]
        ax.plot(L, [r["rsa_raw"] for r in rows], "-o", ms=3, label="raw")
        ax.plot(L, [r["rsa_z"] for r in rows], "-o", ms=3, label="z-scored")
        ax.plot(L, [r["rsa_drop5"] for r in rows], "-o", ms=3, label="drop top-5 var dims")
        ax.axhline(0, color="0.7", lw=.8)
        ax.set_title(f"{tag}: grid RSA vs layer")
        ax.set_xlabel("layer"); ax.legend(fontsize=8)
    axes[0].set_ylabel("Spearman(node-RDM, grid distance)")
    fig.suptitle("Does the in-context grid surface after demoting massive activations?")
    fig.tight_layout()
    fig.savefig(f"{RUN}/standardized_grid.png", dpi=140)
    print("wrote", f"{RUN}/standardized_grid.png")


if __name__ == "__main__":
    main()
