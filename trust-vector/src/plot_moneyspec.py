"""Two-panel summary of the money-trust spectrum. Reads moneyspec{,_steeronly}.json."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
ms = json.load(open(os.path.join(OUT, "moneyspec.json")))
so = json.load(open(os.path.join(OUT, "moneyspec_steeronly.json")))
LOGY = np.arange(6)
YLAB = ms["ylab"]

fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))

# --- A: the spectrum -------------------------------------------------------
ax = axes[0]
groups = [("story", "pos", "#2ca02c", "-"), ("story", "neu", "#7f7f7f", "-"),
          ("story", "mix", "#1f77b4", "-"), ("story", "neg", "#d62728", "-"),
          ("famous", "trusted", "#2ca02c", "--"), ("famous", "mid", "#7f7f7f", "--"),
          ("famous", "distrusted", "#d62728", "--")]
for kind, cell, col, ls in groups:
    M = np.array([c["margins"] for c in ms["contexts"]
                  if c["kind"] == kind and c["cell"] == cell])
    ax.errorbar(LOGY, M.mean(0), yerr=M.std(0) / np.sqrt(len(M)), color=col, ls=ls,
                marker="o", ms=3, lw=1.6, capsize=2,
                label=f"{kind} {cell} (n={len(M)})")
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(LOGY)
ax.set_xticklabels(YLAB, fontsize=8)
ax.set_xlabel("amount")
ax.set_ylabel("margin logit(yes)−logit(no)")
ax.set_title("A  'Would you trust {n} with $y of your own money?'\n"
             "solid = story contexts, dashed = famous people (Qwen3-32B)",
             fontsize=10)
ax.legend(fontsize=7, frameon=False)

# --- B: steering the threshold --------------------------------------------
ax = axes[1]
S = so["steer"]
order = ["optim", "optim_like", "FITTED trust", "story_trust", "story_warmth",
         "story_comb", "warmth_b", "nomfame", "random"]
HATCH = ("optim_like", "warmth_b", "story_warmth", "random")
w = 0.38
for j, (kind, col) in enumerate([("story", "#1f77b4"), ("famous", "#ff7f0e")]):
    m = [S[f"{k}|{kind}"][0] for k in order]
    se = [S[f"{k}|{kind}"][1] for k in order]
    ax.bar(np.arange(len(order)) + (j - 0.5) * w, m, w, yerr=se, capsize=2,
           color=col, hatch=["//" if k in HATCH else "" for k in order],
           label="mix stories (n=64)" if kind == "story" else "ambiguous famous (n=6)")
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("Δ threshold, (+v)−(−v)  [log10 $ decades]")
ax.set_title("B  steering the max trusted amount, α=0.35 at name tokens, L45\n"
             "hatched = decoys/floor; optim_like ≈ optim → not trust-specific",
             fontsize=10)
ax.legend(fontsize=8, frameon=False)

fig.tight_layout()
p = os.path.join(OUT, "moneyspec_summary.png")
fig.savefig(p, dpi=160)
print("wrote", p)
