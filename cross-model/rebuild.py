"""Rebuild the per-layer PCA slideshow and the cross-model similarity heatmap
the paper-faithful way: PER-NODE MEAN representations (not per-occurrence), and
RSA (not CKA) for the cross-model comparison.

From the all-layer 15k subsample, using high-context occurrences for the means.
Outputs:
  pca_per_layer_nodemean.pdf  -- 16 nodes + grid edges per layer, both models
  cross_model_rsa_heatmap.png -- Gemma-layer x Qwen-layer RSA (node-RDM corr)
  rebuild_grid_rsa.json       -- per-layer grid RSA for both models
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from config import get_config
import graph as G
from reproduce import grid_recovery_score
from make_pca_pdf import draw_grid_page

RUN = "runs/gemma_qwen_all"
CFG = get_config("gemma_qwen")
GRAPH = G.build_grid_graph(CFG)
WORDS = CFG.words()
IU = np.triu_indices(16, 1)
GRIDD = GRAPH.grid_distance_matrix()[IU]
HICTX = 300                                        # use high-context occurrences


def spearman(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def rdm(H):
    return np.linalg.norm(H[:, None, :] - H[None, :, :], axis=2)[IU]


def node_means(z, layer, node, mask):
    X = z[f"layer_{layer}"].astype(np.float32)
    H = np.full((16, X.shape[1]), np.nan, np.float32)
    for k in range(16):
        m = mask & (node == k)
        if m.any():
            H[k] = X[m].mean(0)
    return H


def load(p):
    z = np.load(p, allow_pickle=False)
    return z, [int(l) for l in z["_layers"]]


def precompute(z, layers, node, mask):
    out = {}
    for L in layers:
        H = node_means(z, L, node, mask)
        sc = grid_recovery_score(H, GRAPH)
        out[L] = {"coords": sc["coords2d"], "gridcorr": sc["distance_corr"],
                  "rdm": rdm(H), "gridrsa": spearman(rdm(H), GRIDD)}
    return out


def panel(ax, info, tag, L):
    c2 = info["coords"]
    for n in range(16):
        for m in GRAPH.neighbors(n):
            if m > n and not (np.isnan(c2[n]).any() or np.isnan(c2[m]).any()):
                ax.plot([c2[n, 0], c2[m, 0]], [c2[n, 1], c2[m, 1]], color="0.8", zorder=1)
    for n in range(16):
        if not np.isnan(c2[n]).any():
            ax.scatter(*c2[n], zorder=2)
            ax.annotate(WORDS[n], c2[n], fontsize=8)
    ax.set_title(f"{tag} L{L}  (grid RSA={info['gridrsa']:.2f}, PCA corr={info['gridcorr']:.2f})",
                 fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    zg, gl = load(f"{RUN}/acts_sub_gemma.npz")
    zq, ql = load(f"{RUN}/acts_sub_qwen.npz")
    maskg = zg["meta_context_length"] >= HICTX
    maskq = zq["meta_context_length"] >= HICTX
    Gd = precompute(zg, gl, zg["meta_node"], maskg)
    Qd = precompute(zq, ql, zq["meta_node"], maskq)

    json.dump({"gemma": {int(L): Gd[L]["gridrsa"] for L in gl},
               "qwen": {int(L): Qd[L]["gridrsa"] for L in ql}},
              open(f"{RUN}/rebuild_grid_rsa.json", "w"), indent=2)

    # ---- PCA slideshow (per-node means) ----
    with PdfPages(f"{RUN}/pca_per_layer_nodemean.pdf") as pdf:
        draw_grid_page(pdf)
        for i in range(max(len(gl), len(ql))):
            fig, ax = plt.subplots(1, 2, figsize=(11, 5.3))
            if i < len(gl):
                panel(ax[0], Gd[gl[i]], "Gemma", gl[i])
            else:
                ax[0].axis("off")
            if i < len(ql):
                panel(ax[1], Qd[ql[i]], "Qwen", ql[i])
            else:
                ax[1].axis("off")
            fig.suptitle(f"Per-node-mean PCA (ctx>={HICTX}) + grid edges — page {i+1}",
                         fontsize=9)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig); plt.close(fig)

    # ---- cross-model RSA heatmap ----
    H = np.array([[spearman(Gd[Lg]["rdm"], Qd[Lq]["rdm"]) for Lq in ql] for Lg in gl])
    np.save(f"{RUN}/cross_model_rsa.npy", H)
    g_grid = np.array([Gd[L]["gridrsa"] for L in gl])
    q_grid = np.array([Qd[L]["gridrsa"] for L in ql])
    bi, bj = np.unravel_index(int(np.nanargmax(H)), H.shape)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(H, origin="lower", aspect="auto", cmap="viridis", vmin=-0.1, vmax=1,
                   extent=[ql[0]-.5, ql[-1]+.5, gl[0]-.5, gl[-1]+.5])
    ax.axhline(gl[int(g_grid.argmax())], color="w", ls="--", lw=.8, alpha=.7)
    ax.axvline(ql[int(q_grid.argmax())], color="w", ls="--", lw=.8, alpha=.7)
    ax.plot(ql[bj], gl[bi], "r*", ms=14)
    ax.set_xlabel("Qwen layer"); ax.set_ylabel("Gemma layer")
    ax.set_title("Cross-model RSA (node-RDM Spearman, per-node means)\n"
                 f"max {H[bi,bj]:.2f} @ G{gl[bi]}/Q{ql[bj]}; dashed = each model's grid-RSA peak "
                 f"(G{gl[int(g_grid.argmax())]}, Q{ql[int(q_grid.argmax())]})", fontsize=9)
    fig.colorbar(im, label="cross-model RSA")
    fig.tight_layout(); fig.savefig(f"{RUN}/cross_model_rsa_heatmap.png", dpi=140)

    print(f"Gemma grid-RSA peak: L{gl[int(g_grid.argmax())]}={g_grid.max():.3f}")
    print(f"Qwen  grid-RSA peak: L{ql[int(q_grid.argmax())]}={q_grid.max():.3f}")
    print(f"cross-model RSA max: {H[bi,bj]:.3f} @ Gemma L{gl[bi]} / Qwen L{ql[bj]}")
    print("wrote pca_per_layer_nodemean.pdf, cross_model_rsa_heatmap.png, "
          "cross_model_rsa.npy, rebuild_grid_rsa.json")


if __name__ == "__main__":
    main()
