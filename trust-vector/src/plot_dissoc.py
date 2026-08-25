"""Heatmap figure for the dissociation vignettes. out/dissoc_summary.png"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
d = json.load(open(os.path.join(OUT, "dissoc200.json")))
DIRS = ["FITTED trust", "story_trust", "optim", "story_comp", "story_warmth",
        "warmth_b", "random"]
PROBES = ["trust", "comp", "like", "task"]
TITLES = {"comp_malice": "competent, but wants\nto harm you",
          "like_incomp": "likeable, but bad\nat the task",
          "values": "endorses lying/cheating;\nfriendly to you",
          "immoral": "plainly immoral;\nno relationship info"}

fig, axes = plt.subplots(1, 4, figsize=(18.5, 4.6))
for ax, st in zip(axes, TITLES):
    M = np.array([[d["steer"][f"{st}|{p}|{dn}|a0.5"][0] for dn in DIRS]
                  for p in PROBES])
    im = ax.imshow(M, cmap="RdBu_r", vmin=-8, vmax=8, aspect="auto")
    for i in range(len(PROBES)):
        for j in range(len(DIRS)):
            v = M[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8.5,
                    fontweight="bold" if abs(v) > 2 else "normal",
                    color="white" if abs(v) > 4.5 else "black")
    labels = [f"{p}\n(base {d['base'][f'{st}|{p}'][0]:+.0f})" for p in PROBES]
    ax.set_yticks(range(len(PROBES)))
    ax.set_yticklabels(labels if st == "comp_malice" else
                       [f"{p}\n({d['base'][f'{st}|{p}'][0]:+.0f})" for p in PROBES],
                       fontsize=8.5)
    ax.set_xticks(range(len(DIRS)))
    ax.set_xticklabels(DIRS, rotation=40, ha="right", fontsize=8.5)
    ax.set_title(TITLES[st], fontsize=10)
    for j, dn in enumerate(DIRS):
        if dn in ("story_warmth", "story_comp", "warmth_b", "random"):
            ax.add_patch(plt.Rectangle((j - .5, -.5), 1, len(PROBES), fill=False,
                                       hatch="///", edgecolor="gray", lw=0))
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01)
cb.set_label("Δ margin, (+v)−(−v)  [logits], α=0.5")
fig.suptitle("Dissociation vignettes: text pins attributes apart; which probe does each vector move?\n"
             "rows = probe (unsteered baseline in parens); hatched columns = decoys/floor; |Δ|<2 ≈ random-floor territory",
             fontsize=10, y=1.04)
p = os.path.join(OUT, "dissoc_summary.png")
fig.savefig(p, dpi=160, bbox_inches="tight")
print("wrote", p)
