"""MAIN_2 slide 2.1: are the literature's 15 kinds of trust 15 different
directions in the model, or a few? Cosine heatmap with references."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
d = json.load(open(os.path.join(OUT, "typology_cos.json")))
names, M = d["names"], np.array(d["M"])
PRETTY = {"story_trust": "trust (stories)", "FITTED trust": "trust (fitted)",
          "story_warmth": "warmth control", "story_comp": "competence control",
          "optim": "trust (optimized)", "random": "random control"}
labs = [PRETTY.get(n, n) for n in names]
fig, ax = plt.subplots(figsize=(12.8, 11))
im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
for i in range(len(names)):
    for j in range(len(names)):
        ax.text(j, i, f"{M[i,j]:.2f}".replace("0.", "."), ha="center", va="center",
                fontsize=6.5, color="white" if abs(M[i, j]) > 0.55 else "black")
ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=8.5)
ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=8.5)
ax.axhline(14.5, color="k", lw=1.4); ax.axvline(14.5, color="k", lw=1.4)
cb = fig.colorbar(im, fraction=0.035, pad=0.02)
cb.set_label("similarity of directions (1 = same, 0 = unrelated)")
ax.set_title("Fifteen kinds of trust from the research literature, each turned into a direction\n"
             "in the model from 100 story pairs. How similar are they to each other,\n"
             "and to the earlier trust vectors and controls (bottom/right block)?", fontsize=11)
fig.tight_layout()
p = os.path.join(OUT, "typology_heatmap.png")
fig.savefig(p, dpi=160); print("wrote", p)
