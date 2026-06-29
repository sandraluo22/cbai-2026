"""Paper-faithful redo (Park, Lee, Lubana et al., ICLR 2025), offline.

Their representation protocol (verified from arXiv:2501.00070):
  - At context position t, take a SLIDING window of Nw=50 preceding tokens
    (or all if t<50), and average each node-token's activations within it
    -> one vector per node, h_tau(t).
  - Quantify graph structure with Dirichlet energy  E = sum_ij A_ij ||h_i-h_j||^2
    and watch it as context length t grows.
  - PCA of the per-node means recovers the grid at high context.

Differences we are correcting vs. our earlier runs: we had used whole-walk
cumulative means (over-averaging -> saturation). Here we use the 50-token
sliding window and study emergence vs context t, then redo PCA grid-recovery
and the cross-model similarity on these paper-faithful representations.

Runs from runs/v1/square_grid/gemma_qwen/acts_model_{a,b}.npz (200k occ, per-step tags).
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

CFG = get_config("gemma_qwen")
GRAPH = G.build_grid_graph(CFG)
WORDS = CFG.words()
NW = 50                                            # paper's window size
CTX = [5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
GEMMA_LAYER, QWEN_LAYER = 32, 28                   # deep layers we captured
IU = np.triu_indices(16, 1)
GRIDD = GRAPH.grid_distance_matrix()
A = np.zeros((16, 16))
for n in range(16):
    for m in GRAPH.neighbors(n):
        A[n, m] = 1.0


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def rdm(H):
    return np.linalg.norm(H[:, None, :] - H[None, :, :], axis=2)[IU]


def energy_ratio(H):
    """Normalized Dirichlet energy: mean squared distance on EDGES divided by
    mean over ALL pairs. <1 => graph-adjacent nodes closer than average (grid
    structure); ~1 => no structure. Scale-free (paper's standardized variant)."""
    D2 = ((H[:, None, :] - H[None, :, :]) ** 2).sum(-1)
    return float(D2[A > 0].mean() / D2[IU].mean())


def windowed_means(acts, step, node, t):
    """Pooled per-node means over the Nw=50 window ending at context t."""
    lo = max(0, t - NW)
    win = (step >= lo) & (step < t)
    H = np.full((16, acts.shape[1]), np.nan, np.float32)
    for k in range(16):
        m = win & (node == k)
        if m.any():
            H[k] = acts[m].mean(0)
    return H


def per_model(npz, layer):
    z = np.load(npz, allow_pickle=False)
    acts = z[f"layer_{layer}"].astype(np.float32)
    step, node = z["meta_step"], z["meta_node"]
    rows, Hs = [], {}
    for t in CTX:
        H = windowed_means(acts, step, node, t)
        Hs[t] = H
        rows.append({"ctx": t,
                     "energy_ratio": energy_ratio(H),
                     "pca_gridcorr": grid_recovery_score(H, GRAPH)["distance_corr"],
                     "rsa": spearman(rdm(H), GRIDD[IU])})
    return rows, Hs


def main():
    g_rows, g_H = per_model("runs/v1/square_grid/gemma_qwen/acts_model_a.npz", GEMMA_LAYER)
    q_rows, q_H = per_model("runs/v1/square_grid/gemma_qwen/acts_model_b.npz", QWEN_LAYER)

    # cross-model similarity on the paper-faithful per-node reps, per context
    cross = []
    for t in CTX:
        cross.append({"ctx": t, "cross_rsa": spearman(rdm(g_H[t]), rdm(q_H[t]))})

    json.dump({"gemma_layer": GEMMA_LAYER, "qwen_layer": QWEN_LAYER, "Nw": NW,
               "gemma": g_rows, "qwen": q_rows, "cross": cross},
              open("runs/v1/square_grid/gemma_qwen/paper_faithful.json", "w"), indent=2)

    for tag, rows in (("Gemma", g_rows), ("Qwen", q_rows)):
        b = max(rows, key=lambda r: r["rsa"])
        lo = min(rows, key=lambda r: r["energy_ratio"])
        print(f"{tag}: best grid RSA={b['rsa']:.3f}@ctx{b['ctx']}, "
              f"min energy-ratio={lo['energy_ratio']:.3f}@ctx{lo['ctx']}")

    # ---- emergence curves ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    cx = CTX
    ax[0].plot(cx, [r["energy_ratio"] for r in g_rows], "-o", ms=3, label="Gemma")
    ax[0].plot(cx, [r["energy_ratio"] for r in q_rows], "-o", ms=3, label="Qwen")
    ax[0].axhline(1.0, color="0.7", lw=.8, ls="--")
    ax[0].set_title("Normalized Dirichlet energy (lower=grid)"); ax[0].set_xscale("log")
    ax[1].plot(cx, [r["rsa"] for r in g_rows], "-o", ms=3, label="Gemma")
    ax[1].plot(cx, [r["rsa"] for r in q_rows], "-o", ms=3, label="Qwen")
    ax[1].set_title("Grid RSA (Spearman node-RDM vs grid)"); ax[1].set_xscale("log")
    ax[2].plot(cx, [c["cross_rsa"] for c in cross], "-o", ms=3, color="purple")
    ax[2].set_title("Cross-model RSA (Gemma node-RDM vs Qwen)"); ax[2].set_xscale("log")
    for a in ax:
        a.set_xlabel("context length (tokens)"); a.axhline(0, color="0.85", lw=.6)
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=8)
    fig.suptitle(f"Paper-faithful (Nw=50 window): structure vs context "
                 f"(Gemma L{GEMMA_LAYER}, Qwen L{QWEN_LAYER})")
    fig.tight_layout(); fig.savefig("runs/v1/square_grid/gemma_qwen/paper_faithful_emergence.png", dpi=140)

    # ---- PCA grid recovery at high context ----
    fig2, ax2 = plt.subplots(1, 2, figsize=(11, 5.3))
    for a, tag, H in ((ax2[0], "Gemma", g_H[1000]), (ax2[1], "Qwen", q_H[1000])):
        sc = grid_recovery_score(H, GRAPH)
        c2 = sc["coords2d"]
        for n in range(16):
            for m in GRAPH.neighbors(n):
                if m > n and not (np.isnan(c2[n]).any() or np.isnan(c2[m]).any()):
                    a.plot([c2[n, 0], c2[m, 0]], [c2[n, 1], c2[m, 1]], color="0.8", zorder=1)
        for n in range(16):
            if not np.isnan(c2[n]).any():
                a.scatter(*c2[n], zorder=2); a.annotate(WORDS[n], c2[n], fontsize=8)
        a.set_title(f"{tag}: PCA of Nw=50 node means @ctx1000 (corr={sc['distance_corr']:.2f})")
    fig2.tight_layout(); fig2.savefig("runs/v1/square_grid/gemma_qwen/paper_faithful_pca.png", dpi=140)
    print("wrote paper_faithful_emergence.png, paper_faithful_pca.png, paper_faithful.json")


if __name__ == "__main__":
    main()
