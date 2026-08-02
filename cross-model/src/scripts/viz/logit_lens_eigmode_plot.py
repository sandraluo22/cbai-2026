"""Plot experiment 1 (eigenmode logit lens). Left: per mode, the signed correlation of its logit-lens
readout dla(c_k) with the mode's own node pattern u_k -- COORD modes come out positive (neighbours share
a coordinate -> promote same-sign nodes) while PARITY/product modes come out negative (neighbours are the
opposite colour -> promote opposite-sign nodes); the sign IS the adjacency rule. Right: for a few selected
modes, the mode pattern u_k vs its logit-lens readout, both laid on the 4x4 grid.
Reads logit_lens_eigmode_<model>_<graph>.json.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

DIR = os.environ.get("DIR", "runs/axes/1_decomposition/logit_lens_eigmode")
MODEL = os.environ.get("MODEL", "Llama"); GRAPH = os.environ.get("GRAPH", "square_grid")
GS = {"square_grid": "grid"}.get(GRAPH, GRAPH)
IDX = {1: "coord", 2: "coord", 3: "coord", 4: "fold", 5: "fold", 6: "product", 7: "parity×coord",
       8: "product", 9: "parity×coord", 10: "parity×coord", 11: "parity×coord", 12: "fold",
       13: "parity×fold", 14: "parity×fold", 15: "parity"}
CMAP = {"parity": "#C2410C", "parity×fold": "#EA580C", "parity×coord": "#F59E0B",
        "coord": "#1D4ED8", "fold": "#6B7280", "product": "#9CA3AF"}
SEL = [int(s) for s in os.environ.get("SEL", "2,15,14").split(",")]     # modes to show as grids

d = json.load(open(f"{DIR}/logit_lens_eigmode_{MODEL}_{GS}.json"))
U = np.array(d["eigenvectors"]); M = np.array(d["logit_lens_matrix"]); coords = np.array(d["coords"], int)
modes = {m["mode"]: m for m in d["modes"]}
rows = int(coords[:, 0].max()) + 1; cols = int(coords[:, 1].max()) + 1

def grid(v):
    g = np.full((rows, cols), np.nan)
    for i, (r, c) in enumerate(coords): g[r, c] = v[i]
    return g

fig = plt.figure(figsize=(6.2 + 3.0 * len(SEL), 4.6))
gs = fig.add_gridspec(2, 1 + len(SEL), width_ratios=[1.5] + [1] * len(SEL))

# ---- left: signed LL·u_k bar ----
axL = fig.add_subplot(gs[:, 0])
ks = sorted(modes); vals = [modes[k]["corr_selfpattern"] for k in ks]
cols_ = [CMAP.get(IDX.get(k, ""), "#9CA3AF") for k in ks]
axL.barh([f"m{k}" for k in ks], vals, color=cols_)
axL.axvline(0, color="k", lw=0.8); axL.invert_yaxis()
axL.set_xlabel("corr( logit-lens , mode pattern u_k )"); axL.set_xlim(-1, 1)
axL.set_title("does the mode's write decode\nto its own pattern? (sign = adjacency rule)", fontsize=9)
axL.tick_params(labelsize=7); axL.spines[["top", "right"]].set_visible(False)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CMAP.values()]
axL.legend(handles, list(CMAP), fontsize=6.5, frameon=False, loc="lower right")

# ---- right: selected modes, u_k vs logit-lens on the grid ----
for j, k in enumerate(SEL):
    for row, (dat, ti) in enumerate([(U[:, k], f"m{k} pattern u_k"),
                                     (M[k], f"m{k} logit-lens")]):
        ax = fig.add_subplot(gs[row, 1 + j])
        vmax = np.nanmax(np.abs(dat))
        ax.imshow(grid(dat), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        lab = IDX.get(k, ""); ll = modes[k]
        sub = f"LL·u={ll['corr_selfpattern']:+.2f}" if row == 1 else lab
        ax.set_title(f"{ti}\n{sub}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

fig.suptitle(f"Eigenmode logit lens — {MODEL}, {GS} (layer {d['layer']}/{d['n_layers']}, 2nd-to-last)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
for ext in ("pdf",):
    out = f"{DIR}/logit_lens_eigmode_{MODEL}_{GS}.{ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
