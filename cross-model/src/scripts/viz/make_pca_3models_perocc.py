"""Per-occurrence PCA slideshow (the ORIGINAL pca_per_layer.pdf style) with all
three models: Gemma | Qwen | Llama. Reuses make_pca_pdf.panel exactly -- PCA
over every occurrence, clouds coloured by node + centroids, % variance on axes,
no grid edges.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from make_pca_pdf import panel, draw_grid_page, load, PLOT_N

OUT = "runs/square_grid/pca_per_layer_perocc_3models.pdf"
SPECS = [("Gemma", "runs/square_grid/acts_sub_gemma.npz"),
         ("Qwen",  "runs/square_grid/acts_sub_qwen.npz"),
         ("Llama", "runs/square_grid/llama/acts_sub_llama.npz")]


def main():
    data = {}
    for name, path in SPECS:
        z, layers = load(path)
        node = z["meta_node"]
        pidx = np.random.default_rng(0).choice(len(node), min(PLOT_N, len(node)), replace=False)
        data[name] = {"z": z, "layers": layers, "node": node, "pidx": pidx}

    names = ["Gemma", "Qwen", "Llama"]
    maxL = max(len(data[n]["layers"]) for n in names)
    with PdfPages(OUT) as pdf:
        draw_grid_page(pdf)
        for i in range(maxL):
            fig, ax = plt.subplots(1, 3, figsize=(16, 5.3))
            for col, nm in enumerate(names):
                d = data[nm]; Ls = d["layers"]
                if i < len(Ls):
                    L = Ls[i]
                    panel(ax[col], d["z"][f"layer_{L}"], d["node"], d["pidx"], f"{nm}  L{L}")
                else:
                    ax[col].axis("off")
            fig.suptitle(f"PCA of per-occurrence activations — page {i + 1}", fontsize=9)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig, dpi=120)
            plt.close(fig)
            print(f"  page {i + 1}/{maxL}", flush=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
