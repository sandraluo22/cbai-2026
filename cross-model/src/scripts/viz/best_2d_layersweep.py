"""Best-2D across ALL layers (not just the grid-peak), with a per-layer
label-permutation null.

For every model x topology x layer:
  - PCA-2D RSA  : top-2 max-variance node-mean directions
  - best-2D RSA : top-6 node-mean PCs regressed onto true graph coords
  - null        : label-permutation through the fit (B draws), per layer

Output: a curve of RSA vs layer for each (model, graph), with the chance band
(null 5th-95th pct) shaded and the 95th-pct line drawn, so you can see where the
2-D grid emerges with depth and where best-2D clears chance.

-> runs/overview/best_2d_layersweep.pdf  +  runs/overview/best_2d_layersweep.json
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from dataclasses import replace
from config import get_config
import graph as G
import paths as P

B = 1000
RNG = np.random.default_rng(0)


def sp(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def rdm(H):
    iu = np.triu_indices(H.shape[0], 1)
    return np.linalg.norm(H[:, None] - H[None], axis=2)[iu]


GRAPHS = [("square_grid", dict(graph_type="grid", grid_rows=4, grid_cols=4)),
          ("ring", dict(graph_type="ring", ring_size=16)),
          ("hex", dict(graph_type="hex", hex_rows=4, hex_cols=4))]



def sweep(npz, gr):
    n = gr.n_nodes
    Dfull = gr.distance_matrix(); iu = np.triu_indices(n, 1)
    GD = Dfull[iu]
    Gc = np.array(gr.coords, float); Gc = Gc - Gc.mean(0)
    z = np.load(npz)
    node = z["meta_node"]; mask = z["meta_context_length"] >= 300
    nd = node[mask]
    layers = sorted((int(k.split("_")[1]) for k in z.files if k.startswith("layer_")))

    # pre-draw permutations once (shared across layers) + their permuted graph RDMs
    perms = [RNG.permutation(n) for _ in range(B)]
    GDp = [Dfull[np.ix_(p, p)][iu] for p in perms]

    out = {"layers": layers, "pca": [], "best": [], "null95": [], "null05": [], "null99": []}
    for L in layers:
        X = z[f"layer_{L}"].astype(np.float64)[mask]
        H = np.stack([X[nd == k].mean(0) for k in range(n)]); Hc = H - H.mean(0)
        del X
        U, S, _ = np.linalg.svd(Hc, full_matrices=False)
        pca2 = U[:, :2] * S[:2]
        Z = U[:, :6] * S[:6]
        best = Z @ np.linalg.lstsq(Z, Gc, rcond=None)[0]
        nul = np.array([sp(rdm(Z @ np.linalg.lstsq(Z, Gc[p], rcond=None)[0]), gp)
                        for p, gp in zip(perms, GDp)])
        out["pca"].append(sp(rdm(pca2), GD))
        out["best"].append(sp(rdm(best), GD))
        out["null05"].append(float(np.percentile(nul, 5)))
        out["null95"].append(float(np.percentile(nul, 95)))
        out["null99"].append(float(np.percentile(nul, 99)))
    return out


def main():
    grs = {g: G.build_graph(replace(get_config("gemma_qwen"), **kw)) for g, kw in GRAPHS}
    R = {}
    for m in P.MODELS:
        R[m] = {}
        for g, _ in GRAPHS:
            R[m][g] = sweep(P.acts_path(g, m), grs[g])
            print(f"done {m} {g}: {len(R[m][g]['layers'])} layers", flush=True)

    json.dump(R, open(f"{P.overview()}/best_2d_layersweep.json", "w"), indent=1)

    with PdfPages(f"{P.overview()}/best_2d_layersweep.pdf") as pdf:
        for m in P.MODELS:
            fig, ax = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
            for col, (g, _) in enumerate(GRAPHS):
                d = R[m][g]; x = d["layers"]; a = ax[col]
                a.fill_between(x, d["null05"], d["null95"], color="0.85", label="null 5-95th pct")
                a.plot(x, d["null99"], color="red", lw=0.8, ls=":", label="null 99th pct")
                a.plot(x, d["pca"], color="gray", marker=".", ms=3, label="PCA-2D")
                a.plot(x, d["best"], color="steelblue", marker=".", ms=3, label="best-2D")
                pk = x[int(np.argmax(d["best"]))]
                a.axvline(pk, color="steelblue", lw=0.6, ls="--")
                a.set_title(f"{g} (best-2D peak L{pk})"); a.set_xlabel("layer")
                a.set_ylim(-0.2, 1.05)
            ax[0].set_ylabel("RSA vs graph distance"); ax[0].legend(fontsize=8, loc="lower right")
            fig.suptitle(f"{m} — PCA-2D & best-2D RSA across all layers (shaded = label-permutation null)")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # cross-model overlay of best-2D, one panel per graph
        fig, ax = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        col_for = {"Gemma": "tab:green", "Qwen": "tab:orange", "Llama": "tab:blue"}
        for col, (g, _) in enumerate(GRAPHS):
            for m in P.MODELS:
                d = R[m][g]
                # x as fraction of depth so models of different depth overlay
                xf = np.array(d["layers"]) / max(d["layers"])
                ax[col].plot(xf, d["best"], color=col_for[m], marker=".", ms=3, label=m)
            ax[col].set_title(g); ax[col].set_xlabel("relative depth"); ax[col].set_ylim(-0.2, 1.05)
        ax[0].set_ylabel("best-2D RSA"); ax[0].legend(fontsize=9)
        fig.suptitle("best-2D RSA across relative depth — models overlaid")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    print(f"wrote {P.overview()}/best_2d_layersweep.pdf + .json")


if __name__ == "__main__":
    main()
