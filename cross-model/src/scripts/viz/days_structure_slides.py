"""PC3-4 and best-2D projections of the in-context ring structure for the days
(semantic-prior) condition, as per-layer slideshows -- in NODE-MEAN and
PER-OCCURRENCE forms, for all three models.

For days the top PCs are dominated by the pretrained weekday cycle; the IN-CONTEXT
ring lives in lower components. So we show two structure-bearing 2-D frames:
  - PC3-4 : the 3rd/4th principal components (node-mean PCA for the node-mean slides,
            per-occurrence PCA for the per-occ slides)
  - best-2D : the supervised plane (top-6 node-mean PCs regressed onto the ring's
            layout coords, as in best_2d.py); occurrences projected onto the same
            node-mean-derived axes (as in best_2d_peroccurrence.py)
In-context ring edges (purple) are drawn; titles report context-RSA (projected
geometry vs the in-context ring distance).

One slide per layer (paged by relative depth). Version-aware (CM_VERSION).
Env: CM_VERSION GRAPH(days) MODELS_FILTER(e.g. "Llama")
Out: runs/<ver>/<graph>/slides/<graph>_struct_nodemean.pdf
     runs/<ver>/<graph>/slides/<graph>_struct_perocc.pdf
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from dataclasses import replace
from config import get_config
import graph as G
import paths as P
from align import _pca

GRAPH = os.environ.get("GRAPH", "days")
GKW = {"days": dict(graph_type="ring", ring_size=7, word_set="days"),
       "square_grid": dict(graph_type="grid", grid_rows=4, grid_cols=4),
       "ring": dict(graph_type="ring", ring_size=16),
       "hex": dict(graph_type="hex", hex_rows=4, hex_cols=4)}
_filt = os.environ.get("MODELS_FILTER")
MODELS = [m for m in P.MODELS if not _filt or m in set(_filt.split(","))]
PLOT_PTS = 2500            # per-occ points drawn per panel (PCA/RSA use all)
RNG = np.random.default_rng(0)


def sp(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def rdm(H, iu):
    return np.linalg.norm(H[:, None] - H[None], axis=2)[iu]


def load(m):
    z = np.load(P.acts_path(GRAPH, m), allow_pickle=False)
    layers = sorted(int(l) for l in z["_layers"])
    return z, layers, z["meta_node"], z["meta_context_length"] >= P.CTX_LO


def node_means(X, nd, n):
    return np.stack([X[nd == k].mean(0) if (nd == k).any()
                     else np.full(X.shape[1], np.nan) for k in range(n)])


def panel(ax, P_cloud, nd_cloud, P_node, gr, words, cmap, title, iu, CTX_D):
    """Draw a 2-D projection: optional occurrence cloud + in-context ring + node pts."""
    n = P_node.shape[0]
    if P_cloud is not None:
        ax.scatter(P_cloud[:, 0], P_cloud[:, 1], c=cmap[nd_cloud], s=3, alpha=.22, linewidths=0)
    for i in range(n):
        for j in gr.neighbors(i):
            if j > i:
                ax.plot([P_node[i, 0], P_node[j, 0]], [P_node[i, 1], P_node[j, 1]],
                        color="purple", lw=1.3, alpha=.6, zorder=3)
    for i in range(n):
        ax.scatter(*P_node[i], color=cmap[i], edgecolor="k", s=70, zorder=4)
        ax.annotate(words[i], P_node[i], fontsize=7, zorder=5)
    r = sp(rdm(P_node, iu), CTX_D) if not np.isnan(P_node).any() else float("nan")
    ax.set_title(f"{title}  (ctx-RSA {r:+.2f})", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH])
    gr = G.build_graph(cfg); n = gr.n_nodes
    iu = np.triu_indices(n, 1); CTX_D = gr.distance_matrix()[iu]
    Gc = np.array(gr.coords, float); Gc = Gc - Gc.mean(0)
    words = gr.words
    cmap = plt.cm.tab20(np.linspace(0, 1, n))

    data = {m: load(m) for m in MODELS}
    Npg = max(len(data[m][1]) for m in MODELS)

    def layer_at(m, p):
        ls = data[m][1]; return ls[round(p / (Npg - 1) * (len(ls) - 1))] if Npg > 1 else ls[0]

    slides = f"{P.gdir(GRAPH)}/slides"; os.makedirs(slides, exist_ok=True)
    nm_pdf = f"{slides}/{GRAPH}_struct_nodemean.pdf"
    po_pdf = f"{slides}/{GRAPH}_struct_perocc.pdf"
    with PdfPages(nm_pdf) as p_nm, PdfPages(po_pdf) as p_po:
        for p in range(Npg):
            f_nm, ax_nm = plt.subplots(2, len(MODELS), figsize=(5 * len(MODELS), 9), squeeze=False)
            f_po, ax_po = plt.subplots(2, len(MODELS), figsize=(5 * len(MODELS), 9), squeeze=False)
            for col, m in enumerate(MODELS):
                z, layers, node, mask = data[m]
                L = layer_at(m, p)
                X = z[f"layer_{L}"].astype(np.float64)[mask]; nd = node[mask]
                H = node_means(X, nd, n)
                gmu = H.mean(0); Hc = H - gmu
                _, _, Vh = np.linalg.svd(Hc, full_matrices=False)
                # node-mean PC3-4 axes and best-2D axes (from node means)
                pc34 = Vh[[2, 3]].T if Vh.shape[0] > 3 else Vh[:2].T
                Vk = Vh[:min(6, Vh.shape[0])].T
                D = Vk @ np.linalg.lstsq(Hc @ Vk, Gc, rcond=None)[0]          # d x 2

                # ---- node-mean slides ----
                panel(ax_nm[0, col], None, None, Hc @ pc34, gr, words, cmap,
                      f"{m} L{L}  PC3-4 (node-mean)", iu, CTX_D)
                panel(ax_nm[1, col], None, None, Hc @ D, gr, words, cmap,
                      f"{m} L{L}  best-2D (node-mean)", iu, CTX_D)

                # ---- per-occurrence slides ----
                sel = RNG.choice(X.shape[0], min(PLOT_PTS, X.shape[0]), replace=False)
                # PC3-4 from the OCCURRENCE distribution
                mean_o, comps = _pca(X, 6)
                ax34 = comps[[2, 3]] if comps.shape[0] > 3 else comps[:2]      # 2 x d
                Po_occ = (X[sel] - mean_o) @ ax34.T
                Po_node = (H - mean_o) @ ax34.T
                panel(ax_po[0, col], Po_occ, nd[sel], Po_node, gr, words, cmap,
                      f"{m} L{L}  PC3-4 (per-occ)", iu, CTX_D)
                # best-2D: node-mean axes, occurrences projected (best_2d_peroccurrence style)
                Pb_occ = (X[sel] - gmu) @ D
                panel(ax_po[1, col], Pb_occ, nd[sel], Hc @ D, gr, words, cmap,
                      f"{m} L{L}  best-2D (per-occ)", iu, CTX_D)

            rd = p / (Npg - 1) if Npg > 1 else 0.0
            f_nm.suptitle(f"{GRAPH} [{P.VERSION}] node-mean structure — relative depth {rd:.2f}\n"
                          f"top: PC3-4   bottom: best-2D   (purple = in-context ring)", fontsize=11)
            f_po.suptitle(f"{GRAPH} [{P.VERSION}] per-occurrence structure — relative depth {rd:.2f}\n"
                          f"top: PC3-4   bottom: best-2D   (dots = occurrences, big = node means)", fontsize=11)
            f_nm.tight_layout(); p_nm.savefig(f_nm); plt.close(f_nm)
            f_po.tight_layout(); p_po.savefig(f_po); plt.close(f_po)
    print(f"wrote {nm_pdf} and {po_pdf}  ({Npg} pages, models={MODELS})", flush=True)


if __name__ == "__main__":
    main()
