"""Significance heatmaps: cross-model RSA MINUS the per-pair permutation null95,
so 0 is the significance boundary (>0 = above-chance, real; <0 = noise).
Diverging colormap centered at 0. Writes <graph>/slides/cross_model_rsa_significance.pdf
and runs/slides/all_cross_model_rsa_significance.pdf.
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
from heatmaps_with_null import (sub_path, sp, node_means, full_rdm, perm_null,
                                png_to_pdf, MODELS, PAIRS, GRAPHS)


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
        with PdfPages(f"runs/{gname}/slides/cross_model_rsa_significance.pdf") as pdf:
            for A, B in PAIRS:
                La, Lb = data[A]["layers"], data[B]["layers"]
                H = np.array([[sp(data[A]["rdms"][a][iu], data[B]["rdms"][b][iu])
                               for b in Lb] for a in La])
                t95, _ = perm_null(data[A]["rdms"][data[A]["deep"]],
                                   data[B]["rdms"][data[B]["deep"]], n)
                S = H - t95                                    # significance margin
                vmax = max(0.05, float(np.nanmax(np.abs(S))))
                norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
                fig, ax = plt.subplots(figsize=(8, 7))
                im = ax.imshow(S, origin="lower", aspect="auto", cmap="RdBu_r", norm=norm,
                               extent=[Lb[0]-.5, Lb[-1]+.5, La[0]-.5, La[-1]+.5])
                bi, bj = np.unravel_index(int(np.nanargmax(S)), S.shape)
                ax.plot(Lb[bj], La[bi], "k*", ms=12)
                ax.set_xlabel(f"{B} layer"); ax.set_ylabel(f"{A} layer")
                frac = float((S > 0).mean()) * 100
                ax.set_title(f"{gname} (n={n}): {A} vs {B}  RSA − perm-null95 ({t95:.2f})\n"
                             f"red = above chance (significant); {frac:.0f}% of cells significant",
                             fontsize=9)
                fig.colorbar(im, label="RSA − null95   (>0 significant)")
                fig.tight_layout()
                png = f"runs/{gname}/rsa_sig_{A}_{B}.png"
                fig.savefig(png, dpi=130); all_pngs.append(png)
                pdf.savefig(fig); plt.close(fig)
                print(f"{gname} {A}x{B}: null95={t95:.2f}, {frac:.0f}% significant")

    os.makedirs("runs/slides", exist_ok=True)
    png_to_pdf(all_pngs, "runs/slides/all_cross_model_rsa_significance.pdf")
    print("recompiled runs/slides/all_cross_model_rsa_significance.pdf")


if __name__ == "__main__":
    main()
