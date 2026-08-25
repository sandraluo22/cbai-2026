"""Heatmap of the cosine matrix (fitted / optim / story variants), L45 + L52.
Diverging map centered at 0 (cosine is signed), annotated cells. No GPU."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
d = json.load(open(os.path.join(OUT, "cos_matrix.json")))

LABEL = {"FITTED trust": "FITTED trust", "optim": "optim",
         "story_comb": "story (combined)", "story_trust": "story (normal)",
         "story_trust@acctnb": "story (acct, no ban)",
         "story_trust@storynb": "story (bare, no ban)"}

fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.4))
for ax, lk in zip(axes, ("L45", "L52")):
    names = d[lk]["names"]
    M = np.array(d[lk]["M"])
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    labs = [LABEL.get(n, n) for n in names]
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labs, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(labs if lk == "L45" else [""] * len(labs), fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(M[i, j]) > 0.55 else "black")
    ax.set_title(f"{lk} — cosine between unit directions", fontsize=10)
cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
cb.set_label("cosine similarity")
p = os.path.join(OUT, "cos_heatmap.png")
fig.savefig(p, dpi=160, bbox_inches="tight")
print("wrote", p)
