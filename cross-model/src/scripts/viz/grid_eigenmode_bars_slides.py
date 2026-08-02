"""Slideshow (one page per grid 3x3..8x8) of the MODEL's projection onto EVERY Laplacian eigenmode, each drawn
as its OWN BAR CHART: for eigenmode k, bar height at node i = Hc[i] . (Hc^T u_k) (the model's per-node
projection along that mode's readout direction), sign-aligned to the ground-truth eigenvector u_k, bars
coloured by the node's parity class. n bar charts are tiled per n-node grid, ordered low→high spatial
frequency; the parity (top-eigenvalue checkerboard) mode's panel is outlined red. A clean two-block/alternating
bar pattern means the model represents that mode; flat/noisy means it does not.
Reads grid_parity_compare_<model>.npz (needs znode + eigU).
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

DIR = os.environ.get("DIR", "runs/axes/4_circuits/parity"); MODEL = os.environ.get("MODEL", "Llama")
d = json.load(open(f"{DIR}/grid_parity_compare_{MODEL}.json")); z = np.load(f"{DIR}/grid_parity_compare_{MODEL}.npz")
grids = d["grids"]

out = f"{DIR}/grid_eigenmode_bars_slides_{MODEL}.pdf"
with PdfPages(out) as pdf:
    for k in grids:
        znode = z[f"{k}_znode"].astype(float); U = z[f"{k}_eigU"].astype(float); w = z[f"{k}_eigw"]
        sp = z[f"{k}_spectrum"]; col = z[f"{k}_col"]; coords = z[f"{k}_coords"].astype(int)
        n = len(col); R = int(coords[:, 0].max() + 1); C = int(coords[:, 1].max() + 1); pm = int(np.argmax(w))
        Hc = znode - znode.mean(0)
        # per-mode model projection over nodes, sign-aligned to the ground-truth eigenvector
        proj = np.zeros((n, n))
        for m in range(n):
            dirm = Hc.T @ U[:, m]                                  # readout direction in head-output space
            sc = Hc @ dirm                                         # per-node projection
            if np.corrcoef(sc, U[:, m])[0, 1] < 0: sc = -sc
            proj[m] = sc
        bar_col = np.where(col > 0, "#C2410C", "#1D4ED8")          # colour bars by parity class
        ncol = C; nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(1.55 * ncol, 1.5 * nrow), squeeze=False)
        pg = d["per_grid"][k]
        for m in range(nrow * ncol):
            ax = axes[m // ncol][m % ncol]
            if m >= n:
                ax.axis("off"); continue
            ax.bar(range(n), proj[m], color=bar_col, width=1.0)
            ax.axhline(0, color="k", lw=0.4)
            ax.set_xticks([]); ax.set_yticks([])
            ttl = f"m{m} λ{w[m]:.2f}\np={sp[m]:.2f}"
            is_par = (m == pm)
            ax.set_title(ttl, fontsize=6.2, color="#C2410C" if is_par else "#374151",
                         fontweight="bold" if is_par else "normal", pad=1.5)
            for s in ax.spines.values():
                s.set_edgecolor("#C2410C" if is_par else "#D1D5DB"); s.set_linewidth(1.8 if is_par else 0.5)
        fig.suptitle(f"{k} grid — model projection onto EVERY eigenmode (bars = per-node projection, coloured by parity class)\n"
                     f"n={pg['n']}, balance {pg['balance']}  ·  parity mode = m{pm} (red, λ=2.0), write p={pg['parity_mode_power']:.3f}  ·  low→high frequency, left→right/top→bottom",
                     fontsize=10.5, y=1.0)
        fig.tight_layout(rect=[0, 0, 1, 0.96 if nrow > 3 else 0.9])
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
print("wrote", out, f"({len(grids)} slides)")
