"""head_eig_sweep-style attention-damage heatmaps, but for EVERY eigenmode at EVERY measurement layer:
one page per measurement layer, each page = a grid of per-eigenmode (ablation-layer x head) damage
heatmaps. Reads the cached head_mode_bylayer damage tensor damage[mode, meas_layer, abl_layer, head].
Each mode uses a fixed color scale across pages so you can watch its circuit emerge with depth.

Env: FAM(grid) TAG(Llama) LC(dir with <TAG>_<FAM>_damage.npz) OUT
Out: <OUT>/mode_layer_heatmaps_<TAG>_<FAM>.pdf  (nL pages)
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

FAM = os.environ.get("FAM", "grid"); TAG = os.environ.get("TAG", "Llama")
LC = os.environ.get("LC", "runs/axes/4_circuits/head_mode_bylayer")
OUT = os.environ.get("OUT", LC)


def main():
    d = np.load(f"{LC}/{TAG}_{FAM}_damage.npz")
    D = d["damage"]; w = d["eigenvalues"]; cpow = d["clean_pow"]; readout = int(d["readout"])
    N, nL, nAbl, nH = D.shape                                  # [mode, meas_layer, abl_layer, head]
    modes = list(range(1, N))                                  # skip constant mode 0
    vlim = {k: (np.abs(D[k]).max() + 1e-9) for k in modes}     # per-mode fixed scale across layers
    ncol = 5; nrow = int(np.ceil(len(modes) / ncol))
    with PdfPages(f"{OUT}/mode_layer_heatmaps_{TAG}_{FAM}.pdf") as pdf:
        for mL in range(nL):
            fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.4 * nrow))
            for ax, k in zip(np.array(axes).flat, modes):
                ax.imshow(D[k, mL], aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vlim[k], vmax=vlim[k])
                ax.set_title(f"m{k} λ{w[k]:.1f}  p{cpow[mL, k]:.2f}", fontsize=7)
                ax.set_xticks([]); ax.set_yticks([])
            for ax in np.array(axes).flat[len(modes):]:
                ax.axis("off")
            mark = "  <-- readout" if mL == readout else ""
            fig.suptitle(f"{TAG} {FAM} — eigenmode damage heatmaps (ablation-layer[y] x head[x]) "
                         f"measured at LAYER {mL}{mark}\nred = ablating that head weakens the mode at this layer; per-mode fixed scale",
                         fontsize=10)
            fig.supxlabel("ablation head", fontsize=8); fig.supylabel("ablation layer", fontsize=8)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(f"[{TAG}/{FAM}] wrote {OUT}/mode_layer_heatmaps_{TAG}_{FAM}.pdf  ({nL} pages, {len(modes)} modes each)", flush=True)


if __name__ == "__main__":
    main()
