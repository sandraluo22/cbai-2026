"""Slideshow (one page per grid 3x3..8x8) of L14H26's eigenmode WRITE SPECTRUM: how much the head writes
each normalized-Laplacian eigenmode of the grid, with the dominant modes rendered as grid patterns and the
parity (top-eigenvalue checkerboard) mode highlighted. Shows the small grids concentrate write power on the
parity checkerboard while large grids spread it over low-frequency coordinate modes.
Reads grid_parity_compare_<model>.{json,npz}; eigenvectors recomputed locally from the saved coords.
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


def eigmodes(coords):
    n = len(coords); A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if abs(coords[i, 0] - coords[j, 0]) + abs(coords[i, 1] - coords[j, 1]) == 1: A[i, j] = 1.0
    dg = A.sum(1); di = 1 / np.sqrt(np.maximum(dg, 1e-12)); Lap = np.eye(n) - di[:, None] * A * di[None, :]
    return np.linalg.eigh(Lap)                                     # ascending eigenvalue, matches saved spectrum order


def gimg(vec, coords):
    R = int(coords[:, 0].max() + 1); C = int(coords[:, 1].max() + 1); img = np.full((R, C), np.nan)
    for i in range(len(coords)): img[coords[i, 0], coords[i, 1]] = vec[i]
    return img


out = f"{DIR}/grid_eigenspectrum_slides_{MODEL}.pdf"
with PdfPages(out) as pdf:
    for k in grids:
        coords = z[f"{k}_coords"].astype(int); sp = z[f"{k}_spectrum"]; n = len(sp)
        w, U = eigmodes(coords); pm = int(np.argmax(w))
        pg = d["per_grid"][k]
        # top write modes (for the mode-pattern thumbnails) + always show the parity mode
        top = list(np.argsort(sp)[::-1][:3]); show = top + ([pm] if pm not in top else [])
        fig = plt.figure(figsize=(11.5, 7.2))
        gs = fig.add_gridspec(2, len(show), height_ratios=[1.55, 1.0], hspace=0.42, wspace=0.28)
        # ---- spectrum bar ----
        axb = fig.add_subplot(gs[0, :])
        cols = ["#9CA3AF"] * n; cols[pm] = "#C2410C"
        axb.bar(range(n), sp, color=cols, width=0.9)
        for t in top:
            axb.annotate(f"m{t}", (t, sp[t]), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8, color="#1D4ED8")
        axb.annotate("parity", (pm, sp[pm]), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8, color="#C2410C")
        axb.set_xlabel("eigenmode index  (ascending eigenvalue = low → high spatial frequency)")
        axb.set_ylabel("L14H26 write power")
        axb.set_title(f"{k} grid — n={pg['n']} nodes, parity balance {pg['balance']}\n"
                      f"parity-mode write = {pg['parity_mode_power']:.3f}   ·   parity sep = {pg['sep_own']:.3f}   ·   axis↔checkerboard = {pg['parity_axis_eig_corr']}",
                      fontsize=11)
        axb.spines[["top", "right"]].set_visible(False)
        # ---- dominant-mode patterns ----
        for j, t in enumerate(show):
            ax = fig.add_subplot(gs[1, j]); ax.imshow(gimg(U[:, t], coords), cmap="RdBu", vmin=-abs(U[:, t]).max(), vmax=abs(U[:, t]).max())
            lab = f"mode {t}\n$\\lambda$={w[t]:.2f}  p={sp[t]:.2f}"
            if t == pm: lab += "\n(PARITY)"
            ax.set_title(lab, fontsize=8.5, color="#C2410C" if t == pm else "#111827")
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"Eigenmode write-spectrum — {k} grid ({MODEL}, L14H26)", fontsize=12.5, y=1.0)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
print("wrote", out, f"({len(grids)} slides)")
