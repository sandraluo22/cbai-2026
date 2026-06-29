"""Per-occurrence cloud on the best-2D directions.

best_2d.py fits two directions (top-6 node-mean PCs regressed onto graph
coords) and plots only the 16 node MEANS on them. Here we recover those two
directions in the full activation space and project EVERY occurrence onto them,
so each node is a cloud of points rather than a single dot.

direction recovery:
  node-mean PCA:  Hc = U S Vh           (Hc is 16 x d, centered node means)
  top-6 PC axes:  Vk = Vh[:6].T          (d x 6)
  scores:         Z  = Hc @ Vk = U[:,:6]*S[:6]
  fit:            W  = lstsq(Z, Gc)      (6 x 2)
  best dirs:      D  = Vk @ W            (d x 2)   <- the two axes, in d-space
  node means:     (H - mu) @ D == best   (exactly)
  occurrences:    (X_occ - mu) @ D        (same 2-D frame)

-> runs/overview/best_2d_peroccurrence_<model>.png
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


GRAPHS = [("square_grid", dict(graph_type="grid", grid_rows=4, grid_cols=4)),
          ("ring", dict(graph_type="ring", ring_size=16)),
          ("hex", dict(graph_type="hex", hex_rows=4, hex_cols=4))]

# layer per (model, graph) is each model's grid-peak layer via paths.peak_layer.


def main():
    for model in P.MODELS:
        fig, ax = plt.subplots(1, 3, figsize=(16, 5.5))
        for col, (g, kw) in enumerate(GRAPHS):
            npz, L = P.acts_path(g, model), P.peak_layer(g, model)
            gr = G.build_graph(replace(get_config("gemma_qwen"), **kw))
            n = gr.n_nodes
            GD = gr.distance_matrix()[np.triu_indices(n, 1)]
            Gc = np.array(gr.coords, float); Gc = Gc - Gc.mean(0)

            z = np.load(npz)
            node = z["meta_node"]; mask = z["meta_context_length"] >= 300
            X = z[f"layer_{L}"].astype(np.float64)[mask]; nd = node[mask]

            mu = np.stack([X[nd == k].mean(0) for k in range(n)])           # node means (n x d)
            gmu = mu.mean(0)                                                # global mean of node means
            Hc = mu - gmu
            U, S, Vh = np.linalg.svd(Hc, full_matrices=False)
            Vk = Vh[:6].T                                                  # d x 6
            Z = Hc @ Vk
            W = np.linalg.lstsq(Z, Gc, rcond=None)[0]                      # 6 x 2
            D = Vk @ W                                                      # d x 2  (the two axes)

            P_node = Hc @ D                                                 # == best, n x 2
            P_occ = (X - gmu) @ D                                           # every occurrence, m x 2

            a = ax[col]
            cmap = plt.cm.tab20(np.linspace(0, 1, n))
            a.scatter(P_occ[:, 0], P_occ[:, 1], c=cmap[nd], s=3, alpha=0.25, linewidths=0)
            for i in range(n):
                for j in gr.neighbors(i):
                    if j > i:
                        a.plot([P_node[i, 0], P_node[j, 0]], [P_node[i, 1], P_node[j, 1]],
                               color="0.3", lw=1, zorder=3)
            for i in range(n):
                a.scatter(*P_node[i], color=cmap[i], edgecolor="k", s=70, zorder=4)
                a.annotate(gr.words[i], P_node[i], fontsize=8, zorder=5)
            a.set_title(f"{g} L{L}  best-2D  (node-mean RSA {sp(rdm(P_node), GD):.2f})", fontsize=10)
            a.set_xticks([]); a.set_yticks([])
        fig.suptitle(f"{model} — every occurrence projected onto the two best-2D directions "
                     f"(dots=occurrences, big markers=node means)")
        fig.tight_layout()
        out = f"{P.overview()}/best_2d_peroccurrence_{model.lower()}.png"
        fig.savefig(out, dpi=140); plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
