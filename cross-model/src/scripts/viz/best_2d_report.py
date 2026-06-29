"""Best-2D report across all three models, with a label-permutation null, as a
multi-page PDF.

For each model (Gemma / Qwen / Llama) x topology (square_grid / ring / hex), at
that model's grid-peak layer:
  - PCA-2D  : top-2 max-variance node-mean directions (unsupervised)
  - best-2D : top-6 node-mean PCs regressed onto the graph's true coords
  - per-occ : every occurrence projected onto the best-2D axes (D = Vk @ W)

NULL (label permutation, through the fit): keep the real top-6 PC scores Z,
permute node labels pi, refit W = lstsq(Z, Gc[pi]), recompute RSA against the
consistently-permuted graph distances. B draws -> null distribution. This is the
honest baseline for the *supervised* best-2D RSA (chance ~0.47, not the 0.16 of
the random-representation null in rsa_null.py, which is correct only for
unsupervised RSA).

-> runs/overview/best_2d_report.pdf  (+ keeps the per-model PNGs)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from dataclasses import replace
from config import get_config
import graph as G
import paths as P

B = 3000
RNG = np.random.default_rng(0)


def sp(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def rdm(H):
    iu = np.triu_indices(H.shape[0], 1)
    return np.linalg.norm(H[:, None] - H[None], axis=2)[iu]


GRAPHS = [("square_grid", dict(graph_type="grid", grid_rows=4, grid_cols=4)),
          ("ring", dict(graph_type="ring", ring_size=16)),
          ("hex", dict(graph_type="hex", hex_rows=4, hex_cols=4))]



def compute(npz, L, gr):
    """Return everything one (model, graph) cell needs."""
    n = gr.n_nodes
    Dfull = gr.distance_matrix(); iu = np.triu_indices(n, 1)
    GD = Dfull[iu]
    Gc = np.array(gr.coords, float); Gc = Gc - Gc.mean(0)

    z = np.load(npz); node = z["meta_node"]; mask = z["meta_context_length"] >= 300
    X = z[f"layer_{L}"].astype(np.float64)[mask]; nd = node[mask]
    mu_nodes = np.stack([X[nd == k].mean(0) for k in range(n)])
    gmu = mu_nodes.mean(0); Hc = mu_nodes - gmu
    U, S, Vh = np.linalg.svd(Hc, full_matrices=False)

    pca2 = U[:, :2] * S[:2]
    Z = U[:, :6] * S[:6]
    W = np.linalg.lstsq(Z, Gc, rcond=None)[0]
    best = Z @ W
    D = Vh[:6].T @ W                       # best axes in activation space
    P_occ = (X - gmu) @ D                  # every occurrence on best-2D axes

    # label-permutation null through the fit
    def stat(perm):
        Wp = np.linalg.lstsq(Z, Gc[perm], rcond=None)[0]
        GDp = Dfull[np.ix_(perm, perm)][iu]
        return sp(rdm(Z @ Wp), GDp)
    nul = np.array([stat(RNG.permutation(n)) for _ in range(B)])
    obs_best = sp(rdm(best), GD)
    p = (1 + int(np.sum(nul >= obs_best))) / (B + 1)

    return dict(n=n, GD=GD, pca2=pca2, best=best, P_occ=P_occ, node=nd,
                pca_rsa=sp(rdm(pca2), GD), best_rsa=obs_best,
                null95=float(np.percentile(nul, 95)), null99=float(np.percentile(nul, 99)),
                p=p)


def draw_layout(ax, P, gr, title):
    n = gr.n_nodes
    for i in range(n):
        for j in gr.neighbors(i):
            if j > i:
                ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], color="0.8", zorder=1)
    for i in range(n):
        ax.scatter(*P[i], zorder=2); ax.annotate(gr.words[i], P[i], fontsize=7)
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])


def sig(p):
    return "p<.001" if p < 1e-3 else f"p={p:.3f}"


def main():
    grs = {g: G.build_graph(replace(get_config("gemma_qwen"), **kw)) for g, kw in GRAPHS}
    R = {m: {g: compute(P.acts_path(g, m), P.peak_layer(g, m), grs[g]) for g, _ in GRAPHS}
         for m in P.MODELS}

    with PdfPages(f"{P.overview()}/best_2d_report.pdf") as pdf:
        # pages 1-3: PCA-2D vs best-2D, null in the title
        for m in P.MODELS:
            fig, ax = plt.subplots(2, 3, figsize=(15, 9))
            for col, (g, _) in enumerate(GRAPHS):
                d = R[m][g]; gr = grs[g]
                draw_layout(ax[0, col], d["pca2"], gr, f"{g} PCA-2D (RSA {d['pca_rsa']:.2f})")
                draw_layout(ax[1, col], d["best"], gr,
                            f"{g} best-2D (RSA {d['best_rsa']:.2f} | null95 {d['null95']:.2f}, {sig(d['p'])})")
            fig.suptitle(f"{m}   Top: PCA top-2 (variance)   |   Bottom: best-2D (top-6 PCs -> coords); "
                         f"null = label-permutation through the fit")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # pages 4-6: per-occurrence clouds on the best-2D axes
        for m in P.MODELS:
            fig, ax = plt.subplots(1, 3, figsize=(16, 5.5))
            for col, (g, _) in enumerate(GRAPHS):
                d = R[m][g]; gr = grs[g]; n = d["n"]
                cmap = plt.cm.tab20(np.linspace(0, 1, n)); Pn = d["best"]
                ax[col].scatter(d["P_occ"][:, 0], d["P_occ"][:, 1], c=cmap[d["node"]],
                                s=3, alpha=0.25, linewidths=0)
                for i in range(n):
                    for j in gr.neighbors(i):
                        if j > i:
                            ax[col].plot([Pn[i, 0], Pn[j, 0]], [Pn[i, 1], Pn[j, 1]], color="0.3", lw=1, zorder=3)
                for i in range(n):
                    ax[col].scatter(*Pn[i], color=cmap[i], edgecolor="k", s=70, zorder=4)
                    ax[col].annotate(gr.words[i], Pn[i], fontsize=8, zorder=5)
                ax[col].set_title(f"{g} best-2D (RSA {d['best_rsa']:.2f}, {sig(d['p'])})", fontsize=10)
                ax[col].set_xticks([]); ax[col].set_yticks([])
            fig.suptitle(f"{m} — every occurrence on the two best-2D axes (dots=occurrences, markers=node means)")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # page 7: summary bars, best-2D RSA vs the permutation null
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        for ax, (g, _) in zip(axs, GRAPHS):
            ms = list(P.MODELS); x = np.arange(len(ms))
            obs = [R[m][g]["best_rsa"] for m in ms]
            n99 = [R[m][g]["null99"] for m in ms]
            n95 = [R[m][g]["null95"] for m in ms]
            ax.bar(x, n99, color="0.85", label="null ≤99th pct")
            ax.bar(x, obs, width=0.5, color="steelblue", label="best-2D RSA")
            ax.hlines(n95, x - 0.4, x + 0.4, color="red", lw=1.5, label="null 95th pct")
            for xi, m in zip(x, ms):
                ax.text(xi, R[m][g]["best_rsa"] + 0.02, sig(R[m][g]["p"]), ha="center", fontsize=8)
            ax.set_title(g); ax.set_xticks(x); ax.set_xticklabels(ms); ax.set_ylim(0, 1.05)
        axs[0].set_ylabel("RSA"); axs[0].legend(fontsize=8, loc="lower right")
        fig.suptitle("best-2D RSA vs label-permutation null (gray = null ≤99th pct, red = null 95th pct)")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    print(f"wrote {P.overview()}/best_2d_report.pdf")
    print(f"{'model':6}{'graph':12}{'PCA':>6}{'best':>6}{'null95':>8}{'p':>9}")
    for m in P.MODELS:
        for g, _ in GRAPHS:
            d = R[m][g]
            print(f"{m:6}{g:12}{d['pca_rsa']:>6.2f}{d['best_rsa']:>6.2f}{d['null95']:>8.2f}{sig(d['p']):>9}")


if __name__ == "__main__":
    main()
