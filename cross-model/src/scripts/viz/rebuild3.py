"""Three-model versions of the per-layer PCA slideshow and cross-model RSA
heatmaps, from the saved all-layer subsamples (Gemma, Qwen, Llama).

Outputs (runs/square_grid/):
  pca_per_layer_3models.pdf        -- per-node-mean PCA + grid edges, 3 columns
  cross_model_rsa_<A>_<B>.png      -- pairwise RSA heatmaps (3 pairs)
  cross_model_rsa_heatmaps.pdf     -- the 3 heatmaps compiled into one slideshow
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from config import get_config
import graph as G
from reproduce import grid_recovery_score
from make_pca_pdf import draw_grid_page

CFG = get_config("gemma_qwen")
GRAPH = G.build_grid_graph(CFG)
WORDS = CFG.words()
IU = np.triu_indices(16, 1)
GRIDD = GRAPH.grid_distance_matrix()[IU]
HICTX = 300
OUT = "runs/square_grid"
SPECS = [("Gemma", "runs/square_grid/acts_sub_gemma.npz"),
         ("Qwen",  "runs/square_grid/acts_sub_qwen.npz"),
         ("Llama", "runs/square_grid/llama/acts_sub_llama.npz")]


def spearman(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def rdm(H):
    return np.linalg.norm(H[:, None, :] - H[None, :, :], axis=2)[IU]


def node_means(z, L, node, mask):
    X = z[f"layer_{L}"].astype(np.float32)
    H = np.full((16, X.shape[1]), np.nan, np.float32)
    for k in range(16):
        m = mask & (node == k)
        if m.any():
            H[k] = X[m].mean(0)
    return H


def precompute():
    models = {}
    for name, path in SPECS:
        z = np.load(path, allow_pickle=False)
        layers = [int(l) for l in z["_layers"]]
        node = z["meta_node"]; mask = z["meta_context_length"] >= HICTX
        info = {}
        for L in layers:
            H = node_means(z, L, node, mask)
            sc = grid_recovery_score(H, GRAPH)
            info[L] = {"coords": sc["coords2d"], "gridrsa": spearman(rdm(H), GRIDD),
                       "rdm": rdm(H)}
        models[name] = {"layers": layers, "info": info}
        print(f"  {name}: {len(layers)} layers, grid-RSA peak "
              f"{max(info[L]['gridrsa'] for L in layers):.2f}")
    return models


def panel(ax, info, tag, L):
    c2 = info["coords"]
    for n in range(16):
        for m in GRAPH.neighbors(n):
            if m > n and not (np.isnan(c2[n]).any() or np.isnan(c2[m]).any()):
                ax.plot([c2[n, 0], c2[m, 0]], [c2[n, 1], c2[m, 1]], color="0.8", zorder=1)
    for n in range(16):
        if not np.isnan(c2[n]).any():
            ax.scatter(*c2[n], zorder=2); ax.annotate(WORDS[n], c2[n], fontsize=7)
    ax.set_title(f"{tag} L{L}  (RSA={info['gridrsa']:.2f})", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def slideshow(models):
    names = ["Gemma", "Qwen", "Llama"]
    maxL = max(len(models[n]["layers"]) for n in names)
    with PdfPages(f"{OUT}/pca_per_layer_3models.pdf") as pdf:
        draw_grid_page(pdf)
        for i in range(maxL):
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            for col, nm in enumerate(names):
                Ls = models[nm]["layers"]
                if i < len(Ls):
                    panel(ax[col], models[nm]["info"][Ls[i]], nm, Ls[i])
                else:
                    ax[col].axis("off")
            fig.suptitle(f"Per-node-mean PCA + grid edges — page {i + 1}", fontsize=9)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig); plt.close(fig)
    print(f"  wrote {OUT}/pca_per_layer_3models.pdf")


def heatmap_fig(models, A, B):
    La, Lb = models[A]["layers"], models[B]["layers"]
    H = np.array([[spearman(models[A]["info"][a]["rdm"], models[B]["info"][b]["rdm"])
                   for b in Lb] for a in La])
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(H, origin="lower", aspect="auto", cmap="viridis", vmin=-0.1, vmax=1,
                   extent=[Lb[0] - .5, Lb[-1] + .5, La[0] - .5, La[-1] + .5])
    bi, bj = np.unravel_index(int(np.nanargmax(H)), H.shape)
    ax.plot(Lb[bj], La[bi], "r*", ms=13)
    ax.set_xlabel(f"{B} layer"); ax.set_ylabel(f"{A} layer")
    ax.set_title(f"{A} vs {B}  cross-model RSA  "
                 f"(max {H[bi, bj]:.2f} @ {A} L{La[bi]} / {B} L{Lb[bj]})", fontsize=10)
    fig.colorbar(im, label="cross-model RSA")
    fig.tight_layout()
    return fig


def heatmaps(models):
    pairs = [("Gemma", "Qwen"), ("Gemma", "Llama"), ("Qwen", "Llama")]
    with PdfPages(f"{OUT}/cross_model_rsa_heatmaps.pdf") as pdf:
        for A, B in pairs:
            fig = heatmap_fig(models, A, B)
            fig.savefig(f"{OUT}/cross_model_rsa_{A}_{B}.png", dpi=140)
            pdf.savefig(fig); plt.close(fig)
            print(f"  wrote cross_model_rsa_{A}_{B}.png")
    print(f"  wrote {OUT}/cross_model_rsa_heatmaps.pdf")


def main():
    print("precomputing per-node geometries...")
    models = precompute()
    slideshow(models)
    heatmaps(models)


if __name__ == "__main__":
    main()
