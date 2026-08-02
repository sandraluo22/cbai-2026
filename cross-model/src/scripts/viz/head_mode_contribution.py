"""How much does each circuit head write to each eigenmode? For the parity and coord head circuits,
a heatmap of head (row) x eigenmode (col), colored by that head's contribution to the mode's power
(= power damage when the head is ablated; + = builds the mode, - = suppresses it). Reads
head_eig_sweep damage[mode, layer, head] and the greedy_head_set parity/coord circuits.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HE = json.load(open("runs/axes/4_circuits/head_eig_sweep/head_eig_sweep_Llama_grid.json"))
D = np.array(HE["damage"])            # (15 modes, nL, nH)  damage = clean_pow - ablated_pow
w = np.array(HE["eigenvalues"])       # (15,) unnormalized-Laplacian eigenvalues
GS = json.load(open("runs/axes/4_circuits/greedy_head_set/ghs_Llama_grid.json"))
CIRC = {"parity": [tuple(c["head"]) for c in GS["objectives"]["parity"]["greedy"]],
        "coord":  [tuple(c["head"]) for c in GS["objectives"]["coord"]["greedy"]]}
# mode-type label from eigenvalue: lowest two = coords, highest = parity
def mtype(k):                          # k is 1-based mode index
    if k in (1, 2): return "coord"
    if k == 15: return "parity"
    return ""

vmax = float(np.abs([D[:, l, h] for c in CIRC.values() for (l, h) in c]).max())
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
for ax, (name, heads) in zip(axes, CIRC.items()):
    heads = sorted(heads)                                # by layer
    Mm = np.array([D[:, l, h] for (l, h) in heads])      # (n_heads, 15)
    im = ax.imshow(Mm, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(heads))); ax.set_yticklabels([f"L{l}H{h}" for l, h in heads], fontsize=8)
    ax.set_xticks(range(15)); ax.set_xticklabels(
        [f"{k+1}\n{mtype(k+1)}" if mtype(k+1) else f"{k+1}" for k in range(15)], fontsize=7)
    ax.set_xlabel("eigenmode (low→high freq)"); ax.set_title(f"{name} circuit heads", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="writes to mode (+build / −suppress)")
fig.suptitle("How much each circuit head writes to each eigenmode (Llama-8B, grid)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
os.makedirs("runs/axes/4_circuits/head_eig_sweep", exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"runs/axes/4_circuits/head_eig_sweep/head_mode_contribution_Llama_grid.{ext}", dpi=150, bbox_inches="tight")
print("wrote head_mode_contribution_Llama_grid.png")
