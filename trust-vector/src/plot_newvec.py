"""Summary figure for the 2026-08-14 batch: four panels, one per estimand.

  A  stated-trust dose-response, mean-difference-style new directions vs refs
  B  stated-trust dose-response, optimized vector vs its like-optimized decoy
  C  push-pull entity differential (position-cancelled), all directions
  D  advisor battery, per position and bed (the position artifact stays visible)

FINAL n=200 rebuild: all four panels from *_final JSONs. Writes out/newvec_summary.png.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))

s1 = json.load(open(os.path.join(OUT, "newvec_sweep_final.json")))
s2 = s1
s3 = s1
s4 = s1
pp4 = None
pp = json.load(open(os.path.join(OUT, "pushpull_final.json")))
pp3 = None
pp7 = None
adv7 = None
adv = json.load(open(os.path.join(OUT, "newvec_advisor_final.json")))
adv3 = None
AL = s1["alphas"]

COL0 = {"story_comp": "#dbdb8d"}
COL = {**COL0, "storyend": "#1f77b4", "storyend_x": "#17becf", "nominate": "#9467bd",
       "avg_all": "#2ca02c", "optim": "#d62728", "optim_like": "#ff9896",
       "story_comb": "#98df8a", "story_combx": "#2f6f2f", "story_trust": "#7f7f7f", "FITTED trust": "#404040",
       "direct_b": "#bcbd22", "nomfame": "#8c564b", "story_warmth": "#c49c94", "story_comp_": "#dbdb8d", "warmth_b": "#e377c2", "random": "#c7c7c7"}


def sweep_curve(src, name):
    e = src[f"L45|{name}"]["eff"]
    return np.array([x[0] for x in e]), np.array([x[1] for x in e])


def panel_sweep(ax, names_src, title):
    for name, src, ls in names_src:
        m, se = sweep_curve(src, name)
        ax.errorbar(AL, m, yerr=se, label=name, color=COL[name], ls=ls,
                    marker="o", ms=3, lw=1.6, capsize=2)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("alpha (injection scale, x resid norm)")
    ax.set_ylabel("Δ stated trust, (+v)−(−v)  [logits]")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, frameon=False)


fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))
panel_sweep(axes[0, 0],
            [("storyend", s1, "-"), ("storyend_x", s3, "-"),
             ("story_comb", s4, "-"), ("story_combx", s4, "-"),
             ("nominate", s1, "-"), ("avg_all", s1, "-"),
             ("story_trust", s1, "--"), ("FITTED trust", s1, "--"),
             ("warmth_b", s1, ":"), ("story_warmth", s1, ":"), ("random", s1, ":")],
            "A  stated trust: 'Do you trust {n}?' at L45, inject at name tokens\n"
            "new mean-diff-style directions (solid) vs refs (dashed) & controls (dotted)")
panel_sweep(axes[0, 1],
            [("optim", s1, "-"), ("optim_like", s2, "-"),
             ("FITTED trust", s1, "--"), ("random", s1, ":")],
            "B  same probe: optimized vector vs like/dislike-optimized DECOY\n"
            "(equal through α=0.5 — probe does not discriminate)")

# --- C push-pull ---------------------------------------------------------
order = ["optim", "optim_like", "story_warmth", "story_comb", "story_combx", "story_trust",
         "avg_all", "FITTED trust", "story_comp", "warmth_b", "direct_b", "storyend_x",
         "storyend", "nomfame", "nominate", "random"]
src_pp = pp
vals = [src_pp[k][0] for k in order]
ses = [src_pp[k][1] for k in order]
ax = axes[1, 0]
ax.bar(range(len(order)), vals, yerr=ses, capsize=3,
       color=[COL.get(k, "#888") for k in order],
       hatch=["//" if k in ("optim_like", "warmth_b", "story_warmth", "story_comp", "random") else "" for k in order])
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=40, ha="right", fontsize=8)
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("margin toward +v adviser's pick  [logits]")
ax.set_title("C  push-pull: +v one adviser, −v the other (position-cancelled)\n"
             "α=0.35, L45, n=32; hatched = decoys/floor", fontsize=10)

# --- D advisor battery, per position -------------------------------------
show = ["storyend", "storyend_x", "nominate", "nomfame", "avg_all", "optim",
        "story_trust", "story_warmth", "warmth_b", "random"]
srcs = adv
ax = axes[1, 1]
w = 0.2
for j, (bed, person, col, lab) in enumerate([
        ("plain", "Ana", "#1f77b4", "plain, Ana (listed 1st)"),
        ("plain", "Bob", "#ff7f0e", "plain, Bruno (listed 2nd)"),
        ("conditional", "Ana", "#aec7e8", "conditional, Ana"),
        ("conditional", "Bob", "#ffbb78", "conditional, Bruno")]):
    m = [srcs[f"{bed}|L45|{k}"][person][0] for k in show]
    se = [srcs[f"{bed}|L45|{k}"][person][1] for k in show]
    ax.bar(np.arange(len(show)) + (j - 1.5) * w, m, w, yerr=se, capsize=2,
           color=col, label=lab)
ax.set_xticks(range(len(show)))
ax.set_xticklabels(show, rotation=40, ha="right", fontsize=8)
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("margin toward own pick, (+v)−(−v)  [logits]")
ax.set_title("D  advisor battery, single injection at one name, α=0.5, L45\n"
             "second-listed position inflates every direction (known artifact)",
             fontsize=10)
ax.legend(fontsize=7, frameon=False)

fig.tight_layout()
p = os.path.join(OUT, "newvec_summary.png")
fig.savefig(p, dpi=160)
print("wrote", p)
