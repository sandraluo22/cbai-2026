"""One bar chart PER GRID of the model's projection power onto EVERY Laplacian eigenmode (spectrum = the
per-mode projection of L14H26's node activations, ||Hc^T u_k||^2 normalized). x = eigenmode index ordered
low->high spatial frequency; parity (top-eigenvalue checkerboard) mode in red. Shows the parity bar shrinking
as grid size grows. Reads grid_parity_compare_<model>.{json,npz}.

Env: MODEL(Llama) DIR NCOL(2)
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = os.environ.get("DIR", "runs/axes/4_circuits/parity"); MODEL = os.environ.get("MODEL", "Llama")
NCOL = int(os.environ.get("NCOL", "2"))
d = json.load(open(f"{DIR}/grid_parity_compare_{MODEL}.json")); z = np.load(f"{DIR}/grid_parity_compare_{MODEL}.npz")
grids = d["grids"]; ng = len(grids); nrow = int(np.ceil(ng / NCOL))

fig, axes = plt.subplots(nrow, NCOL, figsize=(7.2 * NCOL, 1.9 * nrow), squeeze=False)
for i, k in enumerate(grids):
    ax = axes[i // NCOL][i % NCOL]
    sp = z[f"{k}_spectrum"]; ew = z[f"{k}_eigw"]; n = len(sp); pm = int(np.argmax(ew))
    cols = ["#9CA3AF"] * n; cols[pm] = "#C2410C"
    ax.bar(range(n), sp, color=cols, width=1.0)
    ax.annotate("parity", (pm, sp[pm]), textcoords="offset points", xytext=(0, 2), ha="right",
                fontsize=7.5, color="#C2410C", fontweight="bold")
    pg = d["per_grid"][k]
    ax.set_title(f"{k}  (n={pg['n']}, balance {pg['balance']}, parity p={pg['parity_mode_power']:.3f})", fontsize=9.5)
    ax.set_ylabel("proj. power", fontsize=8); ax.set_xlabel("eigenmode (low→high freq)", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=7)
for j in range(ng, nrow * NCOL): axes[j // NCOL][j % NCOL].axis("off")
fig.suptitle(f"Model projection power onto each eigenmode, per grid 3x3–{grids[-1]} ({MODEL}, L14H26) — parity mode in red", fontsize=13, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.985])
out = f"{DIR}/grid_eigenmode_projection_spectrum_{MODEL}.pdf"
fig.savefig(out, bbox_inches="tight"); print("wrote", out)
# also print the parity-power trend
print("parity projection power by grid:")
for k in grids: print(f"  {k:6} n={d['per_grid'][k]['n']:<4} parity_p={d['per_grid'][k]['parity_mode_power']:.4f}")
