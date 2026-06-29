"""Per-layer NODE-MEAN standard PCA slideshow, three models per slide, per graph.

The node-mean analogue of perocc_slideshows.py: for each layer, take per-node
mean activations over the version's context window [CTX_LO, CTX_HI], project onto
the top-2 *variance* PCs (unsupervised), and plot the 16 nodes + graph edges with
each axis labelled by % variance. Contrast with best_2d_slideshow_nodemean
(which uses the supervised best-2D axes).

Pages indexed by relative depth; three model panels per slide. Version-aware.

Per graph -> runs/<version>/<graph>/slides/pca_per_layer_nodemean_3models.pdf
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
        H = np.stack([X[nd == k].mean(0) for k in range(n)]); del X
        Hc = H - H.mean(0)
        U, S, _ = np.linalg.svd(Hc, full_matrices=False)
        coords = U[:, :2] * S[:2]
        v = (S ** 2) / (S ** 2).sum()
        cache[L] = dict(coords=coords, v1=float(v[0]), v2=float(v[1]),
                        rsa=sp(rdm(coords), GD))
    return dict(layers=layers, cache=cache)


def main():
    for gname, kw in GRAPHS:
        if not all(os.path.exists(P.acts_path(gname, m)) for m in P.MODELS):
            print(f"skip {gname}: no acts for {P.VERSION}", flush=True); continue
        gr = G.build_graph(replace(get_config("gemma_qwen"), **kw))
        n = gr.n_nodes
        cmap = plt.cm.tab20(np.linspace(0, 1, n))
        edges = [(i, j) for i in range(n) for j in gr.neighbors(i) if j > i]
        data = {m: build_cache(P.acts_path(gname, m), gr) for m in P.MODELS}
        Npg = max(len(data[m]["layers"]) for m in P.MODELS)
        def layer_at(m, p):
            ls = data[m]["layers"]; return ls[round(p / (Npg - 1) * (len(ls) - 1))]

        sld = f"{P.gdir(gname)}/slides"; os.makedirs(sld, exist_ok=True)
        out = f"{sld}/pca_per_layer_nodemean_3models.pdf"
        with PdfPages(out) as pdf:
            for p in range(Npg):
                fig, ax = plt.subplots(1, 3, figsize=(15, 5.5))
                for col, m in enumerate(P.MODELS):
                    L = layer_at(m, p); c = data[m]["cache"][L]; Pc = c["coords"]; a = ax[col]
                    for i, j in edges:
                        a.plot([Pc[i, 0], Pc[j, 0]], [Pc[i, 1], Pc[j, 1]], color="0.8", zorder=1)
                    for i in range(n):
                        a.scatter(*Pc[i], color=cmap[i], zorder=2)
                        a.annotate(gr.words[i], Pc[i], fontsize=7, zorder=3)
                    a.set_title(f"{m}  L{L}  (RSA {c['rsa']:.2f})", fontsize=10)
                    a.set_xlabel(f"PC1 ({c['v1']:.0%} var)", fontsize=8)
                    a.set_ylabel(f"PC2 ({c['v2']:.0%} var)", fontsize=8)
                    a.set_xticks([]); a.set_yticks([])
                fig.suptitle(f"{gname} [{P.VERSION}] node-mean PCA (top-2 variance) — "
                             f"relative depth {p/(Npg-1):.2f}")
                fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        print(f"wrote {out}  ({Npg} pages)", flush=True)


if __name__ == "__main__":
    main()
