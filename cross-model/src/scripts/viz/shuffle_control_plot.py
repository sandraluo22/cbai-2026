"""Plot the shuffle control (real random-walk context vs order-shuffled walk, same tokens): per-layer
parity power, coord power, raw top-2-PC grid RSA (unsupervised), best-2D grid RSA, plus behaviour bars.
The raw-PC2 panel is the ctx-2000 payoff: at long context the grid rises INTO the top PCs for the real
walk (matching pca_context_sweep) but collapses to noise when the walk order is destroyed.
Reads shuffle_control_<model>_<graph>.json.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

DIR = os.environ.get("DIR", "runs/axes/1_decomposition/shuffle")
MODEL = os.environ.get("MODEL", "Llama"); GRAPH = os.environ.get("GRAPH", "square_grid")
GS = {"square_grid": "grid"}.get(GRAPH, GRAPH)
BLUE, ORANGE = "#1D4ED8", "#C2410C"

d = json.load(open(f"{DIR}/shuffle_control_{MODEL}_{GS}.json"))
real, shuf = d["results"]["real"], d["results"]["shuffled"]
nL = len(real["parity_pow"]); depth = np.arange(nL) / max(nL - 1, 1)

CURVES = [("parity_pow", "parity power (mode 15)"),
          ("coord_pow", "coord power (modes 1+2)"),
          ("pc2_rsa", "raw top-2-PC grid RSA"),
          ("best2d_rsa", "best-2D grid RSA")]
fig, axes = plt.subplots(1, len(CURVES) + 1, figsize=(4.6 * (len(CURVES) + 1), 4.2))

for ax, (key, title) in zip(axes, CURVES):
    ax.plot(depth, real[key], color=BLUE, lw=2, label="real")
    ax.plot(depth, shuf[key], color=ORANGE, lw=2, label="shuffled")
    ax.set_title(title, fontsize=11); ax.set_xlabel("relative depth")
    ax.set_ylim(bottom=0); ax.grid(axis="y", color="#EEEEEE", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=9, frameon=False, loc="upper left")

# behaviour bars
ax = axes[-1]
nch = d["chance"]["neighbour"]; pch = d["chance"].get("parity", 0.5)
x = np.array([0, 1]); w = 0.38
nb = [real["behaviour"]["neighbour_validity"], shuf["behaviour"]["neighbour_validity"]]
pv = [real["behaviour"]["parity_validity"], shuf["behaviour"]["parity_validity"]]
ax.bar(x - w / 2, nb, w, color=BLUE, label="neighbour validity")
ax.bar(x + w / 2, pv, w, color=ORANGE, label="parity validity")
for xi, (a, b) in zip(x, zip(nb, pv)):
    ax.text(xi - w / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
    ax.text(xi + w / 2, b + 0.01, f"{b:.2f}", ha="center", fontsize=8)
ax.axhline(nch, ls=":", color=BLUE, lw=1.2, label=f"nbr chance ({nch:.2f})")
ax.axhline(pch, ls=":", color=ORANGE, lw=1.2, label=f"par chance ({pch:.2f})")
ax.set_xticks(x); ax.set_xticklabels(["real", "shuffled"]); ax.set_ylim(0, 1.1)
ax.set_title("behaviour", fontsize=11); ax.legend(fontsize=8, frameon=False, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)

wl = d.get("walk_length", "?"); cl = d.get("ctxlo", "?")
fig.suptitle(f"Shuffle control ({MODEL}-8B, {GS}): real vs order-shuffled walk (same tokens) "
             f"— walk_len={wl}, ctx≥{cl}", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
for ext in ("pdf",):
    out = f"{DIR}/shuffle_control_{MODEL}_{GS}.{ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
