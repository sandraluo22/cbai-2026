"""'Best 2 directions' per topology, for EACH model (Gemma / Qwen / Llama).

Generalization of best_2d.py (which only did Gemma) to all three models.
For every (model, topology):
  top row    = PCA top-2 (max-variance directions)
  bottom row = supervised best-2D (top-6 PCs regressed onto the graph's
               ground-truth layout coords).
Each model uses its OWN grid-peak layer (argmax full-dim grid RSA), since the
models have different depths (Gemma 42, Qwen 36, Llama 32 layers).

Shows whether the grid is ~2-D-linear but hidden in low-variance directions
that raw PCA misses. -> runs/overview/best_2d_projection_<model>.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import replace
from config import get_config
import graph as G
import paths as P


def sp(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def rdm(H):
    iu = np.triu_indices(H.shape[0], 1)
    return np.linalg.norm(H[:, None] - H[None], axis=2)[iu]


# topology -> graph config kwargs
GRAPHS = [("square_grid", dict(graph_type="grid", grid_rows=4, grid_cols=4)),
          ("ring", dict(graph_type="ring", ring_size=16)),
          ("hex", dict(graph_type="hex", hex_rows=4, hex_cols=4))]

# model -> {topology: (npz_path, grid-peak layer)}.  Layers are each model's
# layer per (model, graph) is each model's grid-peak layer via paths.peak_layer.


def projections(npz, L, gr):
    """Return (pca2, best2) node layouts and the graph distance RDM."""
    n = gr.n_nodes
    GD = gr.distance_matrix()[np.triu_indices(n, 1)]
    Gc = np.array(gr.coords, float); Gc = Gc - Gc.mean(0)
    z = np.load(npz)
    node = z["meta_node"]; mask = z["meta_context_length"] >= 300
    X = z[f"layer_{L}"].astype(np.float64)
    H = np.stack([X[mask & (node == k)].mean(0) for k in range(n)]); Hc = H - H.mean(0)
    U, S, _ = np.linalg.svd(Hc, full_matrices=False)
    pca2 = U[:, :2] * S[:2]
    Z = U[:, :6] * S[:6]; W = np.linalg.lstsq(Z, Gc, rcond=None)[0]; best = Z @ W
    return pca2, best, GD


def draw(ax, P, gr, title):
    n = gr.n_nodes
    for i in range(n):
        for j in gr.neighbors(i):
            if j > i:
                ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], color="0.8", zorder=1)
    for i in range(n):
        ax.scatter(*P[i], zorder=2); ax.annotate(gr.words[i], P[i], fontsize=7)
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])


def main():
    for model in P.MODELS:
        fig, ax = plt.subplots(2, 3, figsize=(15, 9))
        for col, (g, kw) in enumerate(GRAPHS):
            npz, L = P.acts_path(g, model), P.peak_layer(g, model)
            gr = G.build_graph(replace(get_config("gemma_qwen"), **kw))
            pca2, best, GD = projections(npz, L, gr)
            draw(ax[0, col], pca2, gr, f"{g} L{L} PCA-2D (RSA {sp(rdm(pca2), GD):.2f})")
            draw(ax[1, col], best, gr, f"{g} L{L} best-2D (RSA {sp(rdm(best), GD):.2f})")
        fig.suptitle(f"{model}   Top: PCA top-2 (variance)   |   "
                     f"Bottom: best-2D (top-6 PCs -> graph coords)")
        fig.tight_layout()
        out = f"{P.overview()}/best_2d_projection_{model.lower()}.png"
        fig.savefig(out, dpi=140); plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
