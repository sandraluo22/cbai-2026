"""Per-occurrence (point-cloud) PCA slideshows for every graph, 3 models each.
PCA over all occurrences of a layer; plot the cloud coloured by node + per-node
centroids + word labels, axes = PC % variance. One PDF per graph in slides/.

Run:  PYTHONPATH=src python src/scripts/viz/perocc_slideshows.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from dataclasses import replace

from config import get_config
import graph as G
from make_pca_pdf import top2          # randomized top-2 PCA + % variance (n-agnostic)

MODELS = ["Llama", "Gemma", "Qwen"]
PLOT_N = 2500
GRAPHS = {
    "square_grid": dict(graph_type="grid", grid_rows=4, grid_cols=4, word_set="concepts"),
    "ring": dict(graph_type="ring", ring_size=16, word_set="concepts"),
    "hex":  dict(graph_type="hex", hex_rows=4, hex_cols=4, word_set="concepts"),
    "days": dict(graph_type="ring", ring_size=7, word_set="days"),
}


def sub_path(g, m):
    if g == "square_grid":
        return {"Llama": "runs/square_grid/llama/acts_sub_llama.npz",
                "Gemma": "runs/square_grid/acts_sub_gemma.npz",
                "Qwen":  "runs/square_grid/acts_sub_qwen.npz"}[m]
    return f"runs/{g}/{m}_acts_sub.npz"


def draw_graph_page(pdf, graph, words):
    fig, ax = plt.subplots(figsize=(8, 8))
    xy = np.array(graph.coords)
    for i in range(graph.n_nodes):
        for j in graph.neighbors(i):
            if j > i:
                ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], color="0.7", lw=1.3, zorder=1)
    for i in range(graph.n_nodes):
        ax.scatter(*xy[i], s=700, color="#cfe3ff", edgecolors="#3576c4", zorder=2)
        ax.text(xy[i, 0], xy[i, 1], words[i], ha="center", va="center", fontsize=8, zorder=3)
    ax.set_title(f"Graph: {graph.n_nodes} nodes (per-occurrence PCA follows)")
    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def panel(ax, X, node, words, n, pidx, tag, L):
    scores, v1, v2 = top2(X)
    ax.scatter(scores[pidx, 0], scores[pidx, 1], c=node[pidx], cmap="tab20",
               s=4, alpha=0.35, linewidths=0, rasterized=True)
    for k in range(n):                                  # per-node centroids + words
        m = node == k
        if m.any():
            cx, cy = scores[m, 0].mean(), scores[m, 1].mean()
            ax.scatter([cx], [cy], c="k", s=12, zorder=3)
            ax.text(cx, cy, words[k], fontsize=6, zorder=4)
    ax.set_xlabel(f"PC1 ({v1:.1f}% var)", fontsize=7)
    ax.set_ylabel(f"PC2 ({v2:.1f}% var)", fontsize=7)
    ax.set_title(f"{tag} L{L}", fontsize=8); ax.tick_params(labelsize=5)


def main():
    for gname, gkw in GRAPHS.items():
        cfg = replace(get_config("gemma_qwen"), **gkw)
        graph = G.build_graph(cfg); n, words = graph.n_nodes, graph.words
        data = {}
        for m in MODELS:
            z = np.load(sub_path(gname, m), allow_pickle=False)
            layers = [int(l) for l in z["_layers"]]
            node = z["meta_node"]
            pidx = np.random.default_rng(0).choice(len(node), min(PLOT_N, len(node)), replace=False)
            data[m] = {"z": z, "layers": layers, "node": node, "pidx": pidx}

        os.makedirs(f"runs/{gname}/slides", exist_ok=True)
        out = f"runs/{gname}/slides/pca_per_layer_perocc_3models.pdf"
        with PdfPages(out) as pdf:
            draw_graph_page(pdf, graph, words)
            for i in range(max(len(data[m]["layers"]) for m in MODELS)):
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                for col, m in enumerate(MODELS):
                    d = data[m]; Ls = d["layers"]
                    if i < len(Ls):
                        panel(ax[col], d["z"][f"layer_{Ls[i]}"], d["node"], words, n,
                              d["pidx"], m, Ls[i])
                    else:
                        ax[col].axis("off")
                fig.suptitle(f"{gname}: per-occurrence PCA — page {i + 1}", fontsize=9)
                fig.tight_layout(rect=[0, 0, 1, 0.96])
                pdf.savefig(fig); plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
