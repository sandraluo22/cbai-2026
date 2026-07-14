"""EXACTLY best_2d_slideshows.py (per-occurrence + node-mean slideshows, all three models per slide,
relative-depth paging) EXCEPT the axes are raw PC1 / PC2 (top-2 node-mean principal components) instead
of the supervised best-2D directions. Version-aware (CM_VERSION).

Per graph -> runs/<version>/<graph>/slides/pca12_slideshow_peroccurrence.pdf
             runs/<version>/<graph>/slides/pca12_slideshow_nodemean.pdf
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
import paths as P

GRAPHS = [("square_grid", dict(graph_type="grid", grid_rows=4, grid_cols=4)),
          ("ring", dict(graph_type="ring", ring_size=16)),
          ("hex", dict(graph_type="hex", hex_rows=4, hex_cols=4))]


def sp(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def rdm(H):
    iu = np.triu_indices(H.shape[0], 1)
    return np.linalg.norm(H[:, None] - H[None], axis=2)[iu]


def ve(M, u):
    Mc = M - M.mean(0)
    s = Mc @ u
    return float((s @ s) / (Mc * Mc).sum())


def build_cache(npz, gr):
    n = gr.n_nodes
    GD = gr.distance_matrix()[np.triu_indices(n, 1)]
    z = np.load(npz); node = z["meta_node"]; cl = z["meta_context_length"]
    mask = (cl >= P.CTX_LO) & (cl <= P.CTX_HI)
    nd = node[mask]
    layers = sorted(int(k.split("_")[1]) for k in z.files if k.startswith("layer_"))
    cache = {}
    for L in layers:
        X = z[f"layer_{L}"].astype(np.float64)[mask]
        H = np.stack([X[nd == k].mean(0) for k in range(n)])
        gmu = H.mean(0); Hc = H - gmu
        U, S, Vh = np.linalg.svd(Hc, full_matrices=False)
        D = Vh[:2].T                       # PC1, PC2 directions in hidden space
        u1, u2 = D[:, 0], D[:, 1]          # already unit norm (rows of Vh)
        P_node = Hc @ D
        cache[L] = dict(P_occ=((X - gmu) @ D).astype(np.float32), P_node=P_node,
                        rsa=sp(rdm(P_node), GD),
                        ve_occ=(ve(X, u1), ve(X, u2)), ve_node=(ve(H, u1), ve(H, u2)))
        del X
    return dict(layers=layers, nd=nd, cache=cache)


def axis_labels(ax, ve1, ve2):
    ax.set_xlabel(f"PC1 ({ve1:.0%} var)", fontsize=8)
    ax.set_ylabel(f"PC2 ({ve2:.0%} var)", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def run_graph(gname, kw):
    gr = G.build_graph(replace(get_config("gemma_qwen"), **kw))
    n = gr.n_nodes
    cmap = plt.cm.tab20(np.linspace(0, 1, n))
    edges = [(i, j) for i in range(n) for j in gr.neighbors(i) if j > i]

    data = {m: build_cache(P.acts_path(gname, m), gr) for m in P.MODELS}
    Npg = max(len(data[m]["layers"]) for m in P.MODELS)
    def layer_at(m, p):
        ls = data[m]["layers"]
        return ls[round(p / (Npg - 1) * (len(ls) - 1))]

    sld = f"{P.gdir(gname)}/slides"; os.makedirs(sld, exist_ok=True)
    occ_pdf = f"{sld}/pca12_slideshow_peroccurrence.pdf"
    with PdfPages(occ_pdf) as pdf:
        for p in range(Npg):
            fig, ax = plt.subplots(1, 3, figsize=(15, 5.5))
            for col, m in enumerate(P.MODELS):
                L = layer_at(m, p); c = data[m]["cache"][L]; nd = data[m]["nd"]
                Po = c["P_occ"]; Pn = c["P_node"]; a = ax[col]
                a.scatter(Po[:, 0], Po[:, 1], c=cmap[nd], s=3, alpha=0.22, linewidths=0, rasterized=True)
                for i, j in edges:
                    a.plot([Pn[i, 0], Pn[j, 0]], [Pn[i, 1], Pn[j, 1]], color="0.3", lw=0.8, zorder=3)
                for i in range(n):
                    a.scatter(*Pn[i], color=cmap[i], edgecolor="k", s=55, zorder=4)
                    a.annotate(gr.words[i], Pn[i], fontsize=7, zorder=5)
                a.set_title(f"{m}  L{L}  (RSA {c['rsa']:.2f})", fontsize=10)
                axis_labels(a, *c["ve_occ"])
            fig.suptitle(f"{gname} [{P.VERSION}] per-occurrence on PC1/PC2 axes — relative depth {p/(Npg-1):.2f}")
            fig.tight_layout(); pdf.savefig(fig, dpi=110); plt.close(fig)

    nm_pdf = f"{sld}/pca12_slideshow_nodemean.pdf"
    with PdfPages(nm_pdf) as pdf:
        for p in range(Npg):
            fig, ax = plt.subplots(1, 3, figsize=(15, 5.5))
            for col, m in enumerate(P.MODELS):
                L = layer_at(m, p); c = data[m]["cache"][L]; Pn = c["P_node"]; a = ax[col]
                for i, j in edges:
                    a.plot([Pn[i, 0], Pn[j, 0]], [Pn[i, 1], Pn[j, 1]], color="0.8", zorder=1)
                for i in range(n):
                    a.scatter(*Pn[i], color=cmap[i], zorder=2)
                    a.annotate(gr.words[i], Pn[i], fontsize=7, zorder=3)
                a.set_title(f"{m}  L{L}  (RSA {c['rsa']:.2f})", fontsize=10)
                axis_labels(a, *c["ve_node"])
            fig.suptitle(f"{gname} [{P.VERSION}] node means on PC1/PC2 axes — relative depth {p/(Npg-1):.2f}")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(f"wrote {occ_pdf} + {nm_pdf}  ({Npg} pages)", flush=True)


def main():
    only = os.environ.get("GRAPHS")
    graphs = [(g, kw) for g, kw in GRAPHS if (not only or g in only.split(","))]
    for gname, kw in graphs:
        run_graph(gname, kw)


if __name__ == "__main__":
    main()
