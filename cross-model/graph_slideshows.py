"""Per-layer per-node-mean PCA slideshows for ring / hex / days, 3 models each
(Llama | Gemma | Qwen), with each graph's own edges drawn. One PDF per graph.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from dataclasses import replace

from config import get_config
import graph as G
from reproduce import grid_recovery_score

GRAPHS = {
    "ring": dict(graph_type="ring", ring_size=16, word_set="concepts"),
    "hex":  dict(graph_type="hex", hex_rows=4, hex_cols=4, word_set="concepts"),
    "days": dict(graph_type="ring", ring_size=7, word_set="days"),
}
MODELS = ["Llama", "Gemma", "Qwen"]


def spearman(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def node_means(z, L, node, mask, n):
    X = z[f"layer_{L}"].astype(np.float32)
    H = np.full((n, X.shape[1]), np.nan, np.float32)
    for k in range(n):
        m = mask & (node == k)
        if m.any():
            H[k] = X[m].mean(0)
    return H


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
    ax.set_title(f"Graph: {graph.n_nodes} nodes"); ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def panel(ax, info, graph, words, tag, L):
    c2 = info["coords"]
    for i in range(graph.n_nodes):
        for j in graph.neighbors(i):
            if j > i and not (np.isnan(c2[i]).any() or np.isnan(c2[j]).any()):
                ax.plot([c2[i, 0], c2[j, 0]], [c2[i, 1], c2[j, 1]], color="0.8", zorder=1)
    for i in range(graph.n_nodes):
        if not np.isnan(c2[i]).any():
            ax.scatter(*c2[i], zorder=2); ax.annotate(words[i], c2[i], fontsize=7)
    ax.set_title(f"{tag} L{L}  (RSA={info['rsa']:+.2f})", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


for gname, gkw in GRAPHS.items():
    cfg = replace(get_config("gemma_qwen"), **gkw)
    graph = G.build_graph(cfg)
    n, words = graph.n_nodes, graph.words
    iu = np.triu_indices(n, 1)
    GD = graph.distance_matrix()[iu]

    data = {}
    for m in MODELS:
        z = np.load(f"runs/{gname}/{m}_acts_sub.npz", allow_pickle=False)
        layers = [int(l) for l in z["_layers"]]
        node = z["meta_node"]; mask = z["meta_context_length"] >= 300
        info = {}
        for L in layers:
            H = node_means(z, L, node, mask, n)
            sc = grid_recovery_score(H, graph)
            d = np.linalg.norm(H[:, None, :] - H[None, :, :], axis=2)[iu]
            info[L] = {"coords": sc["coords2d"], "rsa": spearman(d, GD)}
        data[m] = {"layers": layers, "info": info}

    out = f"runs/{gname}/pca_per_layer_3models.pdf"
    with PdfPages(out) as pdf:
        draw_graph_page(pdf, graph, words)
        maxL = max(len(data[m]["layers"]) for m in MODELS)
        for i in range(maxL):
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            for col, m in enumerate(MODELS):
                Ls = data[m]["layers"]
                if i < len(Ls):
                    panel(ax[col], data[m]["info"][Ls[i]], graph, words, m, Ls[i])
                else:
                    ax[col].axis("off")
            fig.suptitle(f"{gname}: per-node-mean PCA + edges — page {i + 1}", fontsize=9)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig); plt.close(fig)
    print(f"wrote {out}")
