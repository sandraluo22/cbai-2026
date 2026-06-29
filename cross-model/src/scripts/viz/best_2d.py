"""'Best 2 directions' for each topology: top row = PCA top-2 (max variance),
bottom row = supervised 2-D (top-6 PCs regressed onto the graph's layout coords).
Shows the grid is ~2-D-linear but hidden in low-variance directions that PCA
misses. -> runs/v1/overview/best_2d_projection.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import replace
from config import get_config
import graph as G


def sp(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def rdm(H):
    iu = np.triu_indices(H.shape[0], 1)
    return np.linalg.norm(H[:, None] - H[None], axis=2)[iu]


SPECS = [("square_grid", dict(graph_type="grid", grid_rows=4, grid_cols=4), "runs/v1/square_grid/acts_sub_gemma.npz", 40),
         ("ring", dict(graph_type="ring", ring_size=16), "runs/v1/ring/Gemma_acts_sub.npz", 39),
         ("hex", dict(graph_type="hex", hex_rows=4, hex_cols=4), "runs/v1/hex/Gemma_acts_sub.npz", 39)]


def main():
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    for col, (g, kw, npz, L) in enumerate(SPECS):
        gr = G.build_graph(replace(get_config("gemma_qwen"), **kw))
        n = gr.n_nodes; GD = gr.distance_matrix()[np.triu_indices(n, 1)]
        Gc = np.array(gr.coords, float); Gc = Gc - Gc.mean(0)
        z = np.load(npz); node = z["meta_node"]; mask = z["meta_context_length"] >= 300
        X = z[f"layer_{L}"].astype(np.float64)
        H = np.stack([X[mask & (node == k)].mean(0) for k in range(n)]); Hc = H - H.mean(0)
        U, S, _ = np.linalg.svd(Hc, full_matrices=False)
        pca2 = U[:, :2] * S[:2]
        Z = U[:, :6] * S[:6]; W = np.linalg.lstsq(Z, Gc, rcond=None)[0]; best = Z @ W
        for r, (P, title) in enumerate([(pca2, f"{g} PCA-2D (RSA {sp(rdm(pca2), GD):.2f})"),
                                        (best, f"{g} best-2D (RSA {sp(rdm(best), GD):.2f})")]):
            a = ax[r, col]
            for i in range(n):
                for j in gr.neighbors(i):
                    if j > i:
                        a.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], color="0.8", zorder=1)
            for i in range(n):
                a.scatter(*P[i], zorder=2); a.annotate(gr.words[i], P[i], fontsize=7)
            a.set_title(title, fontsize=9); a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Top: PCA top-2 (variance)   |   Bottom: best-2D (top-6 PCs -> graph coords)")
    fig.tight_layout(); fig.savefig("runs/v1/overview/best_2d_projection.png", dpi=140)
    print("wrote runs/v1/overview/best_2d_projection.png")


if __name__ == "__main__":
    main()
