"""Object-severity spectrum figure: ladder curves, money-vs-object scatter,
steering bars. Reads moneyspec.json + moneyspec_objects.json."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
mo = json.load(open(os.path.join(OUT, "moneyspec_objects.json")))
mm = json.load(open(os.path.join(OUT, "moneyspec.json")))
X = np.arange(len(mo["ylab"]))

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))

# --- A: ladder curves ------------------------------------------------------
ax = axes[0]
groups = [("story", "pos", "#2ca02c", "-"), ("story", "neu", "#7f7f7f", "-"),
          ("story", "mix", "#1f77b4", "-"), ("story", "neg", "#d62728", "-"),
          ("famous", "trusted", "#2ca02c", "--"), ("famous", "mid", "#7f7f7f", "--"),
          ("famous", "distrusted", "#d62728", "--")]
for kind, cell, col, ls in groups:
    M = np.array([c["margins"] for c in mo["contexts"]
                  if c["kind"] == kind and c["cell"] == cell])
    ax.errorbar(X, M.mean(0), yerr=M.std(0) / np.sqrt(len(M)), color=col, ls=ls,
                marker="o", ms=3, lw=1.6, capsize=2, label=f"{kind} {cell}")
man = next(c for c in mo["contexts"] if c["name"] == "Nelson Mandela")
ax.plot(X, man["margins"], color="#2ca02c", lw=0.9, alpha=0.6, marker=".",
        ls=":", label="Mandela (single)")
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(X)
ax.set_xticklabels(mo["ylab"], rotation=35, ha="right", fontsize=8)
ax.set_ylabel("margin logit(yes)−logit(no)")
ax.set_title("A  'Would you trust {n} with ___?'\n"
             "even Mandela drops to ~0 at the bomb", fontsize=10)
ax.legend(fontsize=7, frameon=False)

# --- B: money threshold vs object threshold --------------------------------
ax = axes[1]
key = lambda c: (c["kind"], c["cell"], c["name"])  # noqa: E731
om, seen = {}, {}
for c in mm["contexts"]:
    om.setdefault(key(c), []).append(c["thr"])
CC = {"pos": "#2ca02c", "neu": "#7f7f7f", "mix": "#1f77b4", "neg": "#d62728",
      "trusted": "#2ca02c", "mid": "#7f7f7f", "distrusted": "#d62728"}
for c in mo["contexts"]:
    k = key(c)
    j = seen.get(k, 0)
    if k in om and j < len(om[k]):
        ax.scatter(om[k][j], c["thr"], s=14 if c["kind"] == "story" else 34,
                   marker="o" if c["kind"] == "story" else "^",
                   color=CC[c["cell"]], alpha=0.55, linewidths=0)
        seen[k] = j + 1
ax.set_xlabel("money threshold  [log10 $]")
ax.set_ylabel("object threshold  [ladder rank]")
ax.set_title("B  the two spectra read the same quantity\n"
             "within-cell r: mix +0.91, neu +0.94, pos +0.92, famous-mid +0.82",
             fontsize=10)
hs = [plt.Line2D([], [], color=CC[c], marker="o", ls="", label=c)
      for c in ("pos", "neu", "mix", "neg")]
hs.append(plt.Line2D([], [], color="#7f7f7f", marker="^", ls="", label="famous"))
ax.legend(handles=hs, fontsize=7, frameon=False)

# --- C: steering -----------------------------------------------------------
ax = axes[2]
S = json.load(open(os.path.join(OUT, "moneyspec_objects_steeronly.json")))["steer"]
order = ["optim", "optim_like", "FITTED trust", "story_trust", "story_warmth", "story_comb", "warmth_b", "nomfame", "random"]
HATCH = ("optim_like", "warmth_b", "story_warmth", "random")
w = 0.38
for j, (kind, col) in enumerate([("story", "#1f77b4"), ("famous", "#ff7f0e")]):
    m = [S[f"{k}|{kind}"][0] for k in order]
    se = [S[f"{k}|{kind}"][1] for k in order]
    ax.bar(np.arange(len(order)) + (j - 0.5) * w, m, w, yerr=se, capsize=2,
           color=col, hatch=["//" if k in HATCH else "" for k in order],
           label="mix stories (n=32)" if kind == "story" else "ambiguous famous (n=6)")
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("Δ threshold  [ladder ranks]")
ax.set_title("C  steering the object threshold, α=0.35, L45\n"
             "optim_like ≥ optim again — not trust-specific", fontsize=10)
ax.legend(fontsize=8, frameon=False)

fig.tight_layout()
p = os.path.join(OUT, "objects_summary.png")
fig.savefig(p, dpi=160)
print("wrote", p)
