"""Cross-model RSA heatmaps with the PERMUTATION null on REAL activations drawn
on each colorbar. For each model-pair, the null is computed from the two models'
actual deep-layer RDMs by shuffling node correspondence (the standard RSA
permutation test) -- conditions on each model's true geometry. Per-heatmap
thresholds. Writes PDFs into <graph>/slides/ and recompiles the combined PDF.
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

MODELS = ["Llama", "Gemma", "Qwen"]
PAIRS = [("Gemma", "Qwen"), ("Gemma", "Llama"), ("Qwen", "Llama")]
GRAPHS = {
    "square_grid": dict(graph_type="grid", grid_rows=4, grid_cols=4, word_set="concepts"),
    "ring": dict(graph_type="ring", ring_size=16, word_set="concepts"),
    "hex":  dict(graph_type="hex", hex_rows=4, hex_cols=4, word_set="concepts"),
    "days": dict(graph_type="ring", ring_size=7, word_set="days"),
}
rng = np.random.default_rng(1)


def sub_path(g, m):
    if g == "square_grid":
        return {"Llama": "runs/square_grid/llama/acts_sub_llama.npz",
                "Gemma": "runs/square_grid/acts_sub_gemma.npz",
                "Qwen":  "runs/square_grid/acts_sub_qwen.npz"}[m]
    return f"runs/{g}/{m}_acts_sub.npz"


def sp(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def node_means(z, L, node, mask, n):
    X = z[f"layer_{L}"].astype(np.float32)
    H = np.full((n, X.shape[1]), np.nan, np.float32)
    for k in range(n):
        m = mask & (node == k)
        if m.any():
            H[k] = X[m].mean(0)
    return H


def full_rdm(H):
    return np.linalg.norm(H[:, None, :] - H[None, :, :], axis=2)


def perm_null(RA, RB, n, T=5000):
    """Shuffle node correspondence of RB; correlate with RA. Real-RDM null."""
    iu = np.triu_indices(n, 1); a = RA[iu]; v = np.empty(T)
    for t in range(T):
        p = rng.permutation(n)
        v[t] = sp(a, RB[np.ix_(p, p)][iu])
    return float(np.percentile(v, 95)), float(np.percentile(v, 99))


def png_to_pdf(pngs, out):
    with PdfPages(out) as pdf:
        for f in pngs:
            img = plt.imread(f); h, w = img.shape[:2]
            fig = plt.figure(figsize=(w / 120, h / 120)); ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(img); ax.axis("off"); pdf.savefig(fig, dpi=120); plt.close(fig)


def main():
    all_pngs = []
    for gname, gkw in GRAPHS.items():
        cfg = replace(get_config("gemma_qwen"), **gkw)
        graph = G.build_graph(cfg); n = graph.n_nodes; iu = np.triu_indices(n, 1)
        data = {}
        for m in MODELS:
            z = np.load(sub_path(gname, m), allow_pickle=False)
            layers = [int(l) for l in z["_layers"]]
            node = z["meta_node"]; mask = z["meta_context_length"] >= 300
            rdms = {L: full_rdm(node_means(z, L, node, mask, n)) for L in layers}
            deep = min(layers, key=lambda L: abs(L - 0.8 * max(layers)))
            data[m] = {"layers": layers, "rdms": rdms, "deep": deep}

        os.makedirs(f"runs/{gname}/slides", exist_ok=True)
        with PdfPages(f"runs/{gname}/slides/cross_model_rsa_heatmaps.pdf") as pdf:
            for A, B in PAIRS:
                La, Lb = data[A]["layers"], data[B]["layers"]
                H = np.array([[sp(data[A]["rdms"][a][iu], data[B]["rdms"][b][iu])
                               for b in Lb] for a in La])
                t95, t99 = perm_null(data[A]["rdms"][data[A]["deep"]],
                                     data[B]["rdms"][data[B]["deep"]], n)
                fig, ax = plt.subplots(figsize=(8, 7))
                im = ax.imshow(H, origin="lower", aspect="auto", cmap="viridis", vmin=-0.1,
                               vmax=1, extent=[Lb[0]-.5, Lb[-1]+.5, La[0]-.5, La[-1]+.5])
                bi, bj = np.unravel_index(int(np.nanargmax(H)), H.shape)
                ax.plot(Lb[bj], La[bi], "r*", ms=12)
                ax.set_xlabel(f"{B} layer"); ax.set_ylabel(f"{A} layer")
                ax.set_title(f"{gname} (n={n}): {A} vs {B} cross-model RSA  (max {H[bi, bj]:.2f};  "
                             f"perm-null95={t95:.2f}, 99={t99:.2f})", fontsize=9)
                cbar = fig.colorbar(im, label="cross-model RSA")
                cbar.ax.axhline(t95, color="red", lw=1.6)
                cbar.ax.axhline(t99, color="red", lw=1.0, ls=":")
                cbar.ax.text(1.6, t95, " perm-null 95%", color="red", fontsize=7, va="center",
                             transform=cbar.ax.get_yaxis_transform())
                fig.tight_layout()
                png = f"runs/{gname}/rsa_{A}_{B}.png"
                fig.savefig(png, dpi=130); all_pngs.append(png)
                pdf.savefig(fig); plt.close(fig)
                print(f"{gname} {A}x{B}: perm-null95={t95:.2f}, 99={t99:.2f}")

    os.makedirs("runs/slides", exist_ok=True)
    png_to_pdf(all_pngs, "runs/slides/all_cross_model_rsa.pdf")
    print("recompiled runs/slides/all_cross_model_rsa.pdf")


if __name__ == "__main__":
    main()
