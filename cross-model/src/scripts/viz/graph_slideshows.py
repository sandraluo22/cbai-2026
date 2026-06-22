"""For each graph (square_grid, ring, hex, days): per-node-mean PCA slideshow
(3 models, axes labelled with PC % variance) + the three pairwise cross-model
RSA heatmaps. All local from the saved subsamples.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from dataclasses import replace

from config import get_config
import graph as G

MODELS = ["Llama", "Gemma", "Qwen"]
GRAPHS = {
    "square_grid": dict(graph_type="grid", grid_rows=4, grid_cols=4, word_set="concepts"),
    "ring": dict(graph_type="ring", ring_size=16, word_set="concepts"),
    "hex":  dict(graph_type="hex", hex_rows=4, hex_cols=4, word_set="concepts"),
    "days": dict(graph_type="ring", ring_size=7, word_set="days"),
}


def sub_path(g, m):
    if g == "square_grid":
        return {"Llama": "runs/llama/acts_sub_llama.npz",
                "Gemma": "runs/square_grid/acts_sub_gemma.npz",
                "Qwen":  "runs/square_grid/acts_sub_qwen.npz"}[m]
    return f"runs/{g}/{m}_acts_sub.npz"


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


def pca2(H):
    valid = ~np.isnan(H).any(1)
    X = H[valid].astype(np.float64)
    coords = np.full((H.shape[0], 2), np.nan)
    if X.shape[0] < 2:
        return coords, 0.0, 0.0
    Xc = X - X.mean(0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords[valid] = Xc @ Vt[:2].T
    tot = float((S ** 2).sum()) or 1.0
    return coords, 100 * S[0] ** 2 / tot, 100 * S[1] ** 2 / tot


def rdm(H, iu):
    return np.linalg.norm(H[:, None, :] - H[None, :, :], axis=2)[iu]


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
    ax.set_xlabel(f"PC1 ({info['v1']:.1f}% var)", fontsize=7)
    ax.set_ylabel(f"PC2 ({info['v2']:.1f}% var)", fontsize=7)
    ax.set_title(f"{tag} L{L}  (grid RSA={info['rsa']:+.2f})", fontsize=8)
    ax.tick_params(labelsize=5)


def main():
    for gname, gkw in GRAPHS.items():
        cfg = replace(get_config("gemma_qwen"), **gkw)
        graph = G.build_graph(cfg)
        n, words = graph.n_nodes, graph.words
        iu = np.triu_indices(n, 1)
        GD = graph.distance_matrix()[iu]

        data = {}
        for m in MODELS:
            z = np.load(sub_path(gname, m), allow_pickle=False)
            layers = [int(l) for l in z["_layers"]]
            node = z["meta_node"]; mask = z["meta_context_length"] >= 300
            info = {}
            for L in layers:
                H = node_means(z, L, node, mask, n)
                coords, v1, v2 = pca2(H)
                r = rdm(H, iu)
                info[L] = {"coords": coords, "v1": v1, "v2": v2,
                           "rdm": r, "rsa": spearman(r, GD)}
            data[m] = {"layers": layers, "info": info}

        # ---- slideshow (axes labelled) ----
        out = f"runs/{gname}/pca_per_layer_3models.pdf"
        with PdfPages(out) as pdf:
            draw_graph_page(pdf, graph, words)
            for i in range(max(len(data[m]["layers"]) for m in MODELS)):
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

        # ---- pairwise cross-model RSA heatmaps ----
        pairs = [("Gemma", "Qwen"), ("Gemma", "Llama"), ("Qwen", "Llama")]
        with PdfPages(f"runs/{gname}/cross_model_rsa_heatmaps.pdf") as pdf:
            for A, B in pairs:
                La, Lb = data[A]["layers"], data[B]["layers"]
                Hm = np.array([[spearman(data[A]["info"][a]["rdm"], data[B]["info"][b]["rdm"])
                                for b in Lb] for a in La])
                fig, ax = plt.subplots(figsize=(8, 7))
                im = ax.imshow(Hm, origin="lower", aspect="auto", cmap="viridis", vmin=-0.1,
                               vmax=1, extent=[Lb[0]-.5, Lb[-1]+.5, La[0]-.5, La[-1]+.5])
                bi, bj = np.unravel_index(int(np.nanargmax(Hm)), Hm.shape)
                ax.plot(Lb[bj], La[bi], "r*", ms=12)
                ax.set_xlabel(f"{B} layer"); ax.set_ylabel(f"{A} layer")
                ax.set_title(f"{gname}: {A} vs {B} cross-model RSA  "
                             f"(max {Hm[bi, bj]:.2f} @ {A} L{La[bi]} / {B} L{Lb[bj]})", fontsize=9)
                fig.colorbar(im, label="cross-model RSA")
                fig.tight_layout()
                fig.savefig(f"runs/{gname}/rsa_{A}_{B}.png", dpi=130)
                pdf.savefig(fig); plt.close(fig)
        print(f"{gname}: slideshow + 3 heatmaps written")


if __name__ == "__main__":
    main()
