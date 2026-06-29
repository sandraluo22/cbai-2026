"""Within-model context-length RSA: how stable is a layer's node-geometry across
context length?

For one (model, graph, layer): at each context-length center c, take per-node
mean activations over occurrences within +/-max(20%,20) of c, build the node RDM
at that context, then RSA (Spearman of RDMs) between every pair of context
centers. The result is a (context x context) heatmap per layer; high off-diagonal
= the geometry has stabilised, the rising approach to the bottom-right = the
in-context grid forming.

CONTROL: a node-label permutation null. For each context pair, shuffle node
correspondence B times and recompute RSA; the per-cell 95th percentile is the
chance ceiling. Each slide is a 2x3 grid: top row = observed RSA (3 models),
bottom row = the matching permutation-null95 (same colour scale) -- structure is
real only where the top heatmap is clearly brighter than the one below it.

One slide per layer (paged by relative depth). Version-aware (CM_VERSION);
context centers extend to CTX_HI.

Per graph -> runs/<version>/<graph>/slides/context_rsa_by_layer.pdf
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.backends.backend_pdf import PdfPages
from dataclasses import replace
from config import get_config
import graph as G
import paths as P

GRAPHS = [("square_grid", dict(graph_type="grid", grid_rows=4, grid_cols=4)),
          ("ring", dict(graph_type="ring", ring_size=16)),
          ("hex", dict(graph_type="hex", hex_rows=4, hex_cols=4))]
# fine-grained centers; capped at the version's max context
BASE_CENTERS = [10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 650, 800, 1000,
                1250, 1500, 1750, 2000]
B = 200                     # permutations for the null
RNG = np.random.default_rng(0)


def sp(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def centers():
    return [c for c in BASE_CENTERS if c <= P.CTX_HI]


def build_cache(npz, gr, C):
    """Per layer: observed (k x k) RSA matrix and the perm-null 95th-pct matrix."""
    n = gr.n_nodes
    iu = np.triu_indices(n, 1)
    z = np.load(npz); node = z["meta_node"]; cl = z["meta_context_length"]
    layers = sorted(int(k.split("_")[1]) for k in z.files if k.startswith("layer_"))
    masks = [(np.abs(cl - c) <= max(0.2 * c, 20)) for c in C]
    counts = [int(m.sum()) for m in masks]
    perms = [RNG.permutation(n) for _ in range(B)]
    k = len(C)
    out = {}
    for L in layers:
        X = z[f"layer_{L}"].astype(np.float64)
        # full n x n node-RDM per context center
        Dm = []
        for m in masks:
            ndm = node[m]; Xm = X[m]
            H = np.stack([Xm[ndm == q].mean(0) if (ndm == q).any()
                          else np.full(X.shape[1], np.nan) for q in range(n)])
            Dm.append(np.linalg.norm(H[:, None] - H[None], axis=2))
        del X
        tri = [D[iu] for D in Dm]
        obs = np.eye(k); nl95 = np.zeros((k, k))
        for i in range(k):
            # diagonal null: geometry vs its own node-shuffle
            di = [sp(tri[i], Dm[i][np.ix_(p, p)][iu]) for p in perms]
            nl95[i, i] = np.percentile(di, 95)
            for j in range(i + 1, k):
                obs[i, j] = obs[j, i] = sp(tri[i], tri[j])
                nv = [sp(tri[i], Dm[j][np.ix_(p, p)][iu]) for p in perms]
                nl95[i, j] = nl95[j, i] = np.percentile(nv, 95)
        out[L] = (obs, nl95)
    return dict(layers=layers, obs_null=out, counts=counts)


def main():
    C = centers(); lab = [str(c) for c in C]
    for gname, kw in GRAPHS:
        gr = G.build_graph(replace(get_config("gemma_qwen"), **kw))
        data = {m: build_cache(P.acts_path(gname, m), gr, C) for m in P.MODELS}
        Npg = max(len(data[m]["layers"]) for m in P.MODELS)
        def layer_at(m, p):
            ls = data[m]["layers"]; return ls[round(p / (Npg - 1) * (len(ls) - 1))]

        slides = f"{P.gdir(gname)}/slides"; os.makedirs(slides, exist_ok=True)
        out_pdf = f"{slides}/context_rsa_by_layer.pdf"
        with PdfPages(out_pdf) as pdf:
            for p in range(Npg):
                fig, ax = plt.subplots(2, 3, figsize=(15, 9))
                for col, m in enumerate(P.MODELS):
                    L = layer_at(m, p); obs, nl95 = data[m]["obs_null"][L]
                    for row, (Mtx, tag) in enumerate([(obs, f"{m}  L{L}"),
                                                      (nl95, f"{m}  L{L}  perm-null 95%")]):
                        a = ax[row, col]
                        im = a.imshow(Mtx, vmin=-0.2, vmax=1, cmap="viridis", origin="lower")
                        a.set_xticks(range(len(C))); a.set_xticklabels(lab, rotation=90, fontsize=5)
                        a.set_yticks(range(len(C))); a.set_yticklabels(lab, fontsize=5)
                        a.set_xlabel("context length"); a.set_ylabel("context length")
                        a.set_title(tag, fontsize=10)
                        fig.colorbar(im, ax=a, fraction=0.046, label="RSA")
                fig.suptitle(f"{gname} [{P.VERSION}] context-vs-context node-geometry RSA — "
                             f"relative depth {p/(Npg-1):.2f}\n"
                             f"top: observed   |   bottom: node-label permutation null (95th pct)")
                fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        print(f"wrote {out_pdf}  ({Npg} pages, {len(C)} centers, B={B})", flush=True)

        # significance: observed RSA - perm-null95 (>0 = above chance), like the
        # cross-model significance heatmaps. Diagonal masked (trivially 1).
        sig_pdf = f"{slides}/context_rsa_significance.pdf"
        with PdfPages(sig_pdf) as pdf:
            for p in range(Npg):
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                for col, m in enumerate(P.MODELS):
                    L = layer_at(m, p); obs, nl95 = data[m]["obs_null"][L]
                    S = obs - nl95
                    np.fill_diagonal(S, np.nan)
                    vmax = max(0.05, float(np.nanmax(np.abs(S))))
                    a = ax[col]
                    im = a.imshow(S, origin="lower", cmap="RdBu_r",
                                  norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax))
                    a.set_xticks(range(len(C))); a.set_xticklabels(lab, rotation=90, fontsize=5)
                    a.set_yticks(range(len(C))); a.set_yticklabels(lab, fontsize=5)
                    a.set_xlabel("context length"); a.set_ylabel("context length")
                    frac = float(np.nansum(S > 0)) / float(np.sum(~np.isnan(S))) * 100
                    a.set_title(f"{m}  L{L}  ({frac:.0f}% > null)", fontsize=10)
                    fig.colorbar(im, ax=a, fraction=0.046, label="RSA − null95")
                fig.suptitle(f"{gname} [{P.VERSION}] context-RSA significance (observed − perm-null95) "
                             f"— relative depth {p/(Npg-1):.2f}\nred = above chance (real)")
                fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        print(f"wrote {sig_pdf}", flush=True)


if __name__ == "__main__":
    main()
