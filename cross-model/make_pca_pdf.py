"""PCA of each layer's activations -> multipage PDF, Gemma vs Qwen.

For every layer we PCA the per-occurrence activations (15k-occurrence all-layer
subsample), plot PC1 x PC2 with the % variance explained on each axis, colour
points by graph node, and overlay each node's centroid + word so any in-context
graph structure is visible. One page per layer index (Gemma left, Qwen right).
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from config import get_config
import graph as G

CFG = get_config("gemma_qwen")
RUN = "runs/square_grid"
WORDS = CFG.words()                                # 16 node words
PLOT_N = 2500                                      # points drawn per panel


def draw_grid_page(pdf):
    """Title page: the actual graph (4x4 grid) with concept words on nodes."""
    graph = G.build_grid_graph(CFG)
    fig, ax = plt.subplots(figsize=(8, 8))
    xy = {n: (c, -r) for n, (r, c) in enumerate(graph.coords)}   # row 0 on top
    for n in range(graph.n_nodes):                               # edges
        for m in graph.neighbors(n):
            if m > n:
                (x0, y0), (x1, y1) = xy[n], xy[m]
                ax.plot([x0, x1], [y0, y1], color="0.7", lw=1.5, zorder=1)
    for n in range(graph.n_nodes):                               # nodes + words
        x, y = xy[n]
        ax.scatter([x], [y], s=900, color="#cfe3ff", edgecolors="#3576c4", zorder=2)
        ax.text(x, y, WORDS[n], ha="center", va="center", fontsize=10, zorder=3)
    ax.set_title(f"In-context graph: {CFG.grid_rows}x{CFG.grid_cols} grid "
                 f"({graph.n_nodes} nodes), plain random walk\n"
                 f"nodes = semantically unrelated words; edges = orthogonal neighbours",
                 fontsize=12)
    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout()
    pdf.savefig(fig, dpi=120)
    plt.close(fig)


def top2(X: np.ndarray):
    """Top-2 PCs via randomized SVD; return (scores[n,2], %var1, %var2)."""
    X = X.astype(np.float32)
    mean = X.mean(0); Xc = X - mean
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(Xc @ rng.standard_normal((Xc.shape[1], 12)).astype(np.float32))
    for _ in range(3):
        Q, _ = np.linalg.qr(Xc.T @ Q); Q, _ = np.linalg.qr(Xc @ Q)
    _, _, Vt = np.linalg.svd(Q.T @ Xc, full_matrices=False)
    scores = Xc @ Vt[:2].T
    tot = float((Xc ** 2).sum())
    return scores, 100 * float((scores[:, 0] ** 2).sum()) / tot, \
                   100 * float((scores[:, 1] ** 2).sum()) / tot


def panel(ax, X, node, pidx, title):
    scores, v1, v2 = top2(X)
    ax.scatter(scores[pidx, 0], scores[pidx, 1], c=node[pidx], cmap="tab20",
               s=4, alpha=0.35, linewidths=0, rasterized=True)
    for n in range(16):                            # node centroids + words
        m = node == n
        if m.any():
            cx, cy = scores[m, 0].mean(), scores[m, 1].mean()
            ax.scatter([cx], [cy], c="k", s=12, zorder=3)
            ax.text(cx, cy, WORDS[n], fontsize=6, zorder=4)
    ax.set_xlabel(f"PC1 ({v1:.1f}% var)")
    ax.set_ylabel(f"PC2 ({v2:.1f}% var)")
    ax.set_title(title, fontsize=10)


def load(npz):
    z = np.load(npz, allow_pickle=False)
    return z, [int(l) for l in z["_layers"]]


def main():
    zg, gl = load(f"{RUN}/acts_sub_gemma.npz")
    zq, ql = load(f"{RUN}/acts_sub_qwen.npz")
    node = zg["meta_node"]
    pidx = np.random.default_rng(0).choice(len(node), min(PLOT_N, len(node)),
                                           replace=False)

    out = f"{RUN}/pca_per_layer.pdf"
    with PdfPages(out) as pdf:
        draw_grid_page(pdf)                        # title page: the graph itself
        for i in range(max(len(gl), len(ql))):
            fig, axes = plt.subplots(1, 2, figsize=(11, 5.3))
            if i < len(gl):
                panel(axes[0], zg[f"layer_{gl[i]}"], node, pidx, f"Gemma  L{gl[i]}")
            else:
                axes[0].axis("off")
            if i < len(ql):
                panel(axes[1], zq[f"layer_{ql[i]}"], node, pidx, f"Qwen  L{ql[i]}")
            else:
                axes[1].axis("off")
            fig.suptitle(f"PCA of layer activations — page {i + 1}", fontsize=9)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig, dpi=120)
            plt.close(fig)
            print(f"  page {i + 1}/{max(len(gl), len(ql))} done", flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
