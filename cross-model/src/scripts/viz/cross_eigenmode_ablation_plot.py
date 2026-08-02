"""Plot cross-structure eigenmode ablation: project out the SQUARE-GRID PARITY direction or the RING COORD
direction and measure the effect on the five geometry families (fit RSA) and on the torus / grid / ring
(neighbour validity). Structure-specific ablation validates the directions (grid-parity kills grid, ring-
coord kills ring); the torus is hit by BOTH — most by ring-coord — because it is two coupled rings.
Reads cross_eigenmode_ablation_<model>.json.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

DIR = os.environ.get("DIR", "runs/axes/6_geometry"); MODEL = os.environ.get("MODEL", "Llama")
d = json.load(open(f"{DIR}/cross_eigenmode_ablation_{MODEL}.json"))
CC = {"baseline": "#111827", "ablate_gridparity": "#C2410C", "ablate_ringcoord": "#1D4ED8"}
LAB = {"baseline": "baseline", "ablate_gridparity": "ablate grid-parity", "ablate_ringcoord": "ablate ring-coord"}
conds = ["baseline", "ablate_gridparity", "ablate_ringcoord"]

fig, ax = plt.subplots(1, 2, figsize=(16, 5.2), gridspec_kw={"width_ratios": [2.1, 1]})

# ---- families: fit RSA ----
fams = ["arc", "simplex", "tree", "product", "helix"]
order = [n for f in fams for n in d["families"] if d["families"][n]["family"] == f]
x = np.arange(len(order)); w = 0.26
for j, c in enumerate(conds):
    ax[0].bar(x + (j - 1) * w, [d["families"][n][c] for n in order], w, color=CC[c], label=LAB[c])
ax[0].set_xticks(x); ax[0].set_xticklabels([f"{n}\n[{d['families'][n]['family']}]" for n in order], fontsize=7, rotation=90)
ax[0].set_ylabel("fit to own geometry (RSA / equidist)"); ax[0].set_title("five families — geometry fit under ablation", fontsize=10)
ax[0].legend(fontsize=8, frameon=False); ax[0].axhline(0, color="k", lw=.5); ax[0].spines[["top", "right"]].set_visible(False)
# family boundaries
b = 0
for f in fams[:-1]:
    b += sum(1 for n in order if d["families"][n]["family"] == f); ax[0].axvline(b - 0.5, color="#DDD", lw=1)

# ---- torus + controls + real cycles: neighbour validity / arithmetic accuracy ----
tor = list(d["tori"]) + [f"({k})" for k in d["controls"]] + [f"{k} QA" for k in d.get("cyclic", {})]
vals = {**d["tori"], **{f"({k})": v for k, v in d["controls"].items()},
        **{f"{k} QA": v for k, v in d.get("cyclic", {}).items()}}
xt = np.arange(len(tor))
for j, c in enumerate(conds):
    ax[1].bar(xt + (j - 1) * w, [vals[t][c] for t in tor], w, color=CC[c])
    for xi, t in zip(xt + (j - 1) * w, tor):
        ax[1].text(xi, vals[t][c] + 0.01, f"{vals[t][c]:.2f}", ha="center", fontsize=6, rotation=90)
ax[1].set_xticks(xt); ax[1].set_xticklabels(tor, fontsize=8, rotation=30, ha="right")
ax[1].set_ylabel("neighbour validity  /  months-days arithmetic acc"); ax[1].set_ylim(0, 1.12)
ax[1].set_title("torus & controls (nbr validity) + real cycles (arithmetic acc)\ntorus re-uses ring-coord; months/days do NOT", fontsize=9)
ax[1].axhline(0.2, ls=":", color="k", lw=1); ax[1].spines[["top", "right"]].set_visible(False)

fig.suptitle(f"Cross-structure eigenmode ablation ({MODEL}): does each structure re-use the grid-parity / ring-coord direction?", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
for ext in ("pdf",):
    out = f"{DIR}/cross_eigenmode_ablation_{MODEL}.{ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
