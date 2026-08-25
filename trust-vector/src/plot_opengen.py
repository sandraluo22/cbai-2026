"""Endorsement-rate figure for the expanded open-ended experiment."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
d = json.load(open(os.path.join(OUT, "opengen2_final.json")))
CONDS = d["conds"]
CL = [{"none": "unsteered", "text+": "text+ ('deeply\ntrustworthy')"}.get(c, c.replace(" trust", "")) for c in CONDS]

rows, labs, seps = [], [], []
for c in d["cases"]:
    rows.append([np.mean([o["m"] > 0 for o in c["gens"][k]]) for k in CONDS])
    labs.append(f"{c['name']}  ({c['grp']})")
R = np.array(rows)

fig = plt.figure(figsize=(14.2, 6.4))
ax = fig.add_axes([0.20, 0.16, 0.44, 0.76])
im = ax.imshow(R, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(CONDS)))
ax.set_xticklabels(CL, rotation=40, ha="right", fontsize=7.5)
ax.set_yticks(range(len(labs)))
ax.set_yticklabels(labs, fontsize=7.5)
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        ax.text(j, i, f"{R[i, j]:.2f}".rstrip("0").rstrip("."), ha="center",
                va="center", fontsize=6.5,
                color="black")
for y in (7.5, 9.5, 11.5):
    ax.axhline(y, color="k", lw=1.2)
ax.set_title("endorsement rate: judge says the response is willing to trust {n}\n"
             "(K=4 sampled generations per cell, temp 0.8, α=0.5 at name tokens, L45)",
             fontsize=9.5)

axb = fig.add_axes([0.72, 0.16, 0.26, 0.76])
bl = [c for c in d["cases"] if c["grp"] == "borderline"]
m, se = [], []
for k in CONDS:
    v = np.array([o["m"] > 0 for c in bl for o in c["gens"][k]], float)
    m.append(v.mean())
    se.append(v.std(ddof=1) / np.sqrt(len(v)))
def _col(c):
    if c == "none": return "#7f7f7f"
    if c == "text+": return "#a1d99b"
    if c.startswith("FITTED"): return "#08519c"
    if c.startswith("optim ") or c in ("optim+", "optim-"): return "#d62728"
    if c.startswith("story_trust"): return "#2ca02c"
    return "#bdbdbd"
def _hat(c):
    if c.endswith("-"): return "\\\\"
    if c.split("+")[0].rstrip("-") in ("optim_like", "warmth_b", "story_warmth", "random"): return "//"
    return ""
cols = [_col(c) for c in CONDS]
axb.bar(range(len(CONDS)), m, yerr=se, capsize=3, color=cols,
        hatch=[_hat(c) for c in CONDS])
axb.axhline(m[0], color="k", lw=0.8, ls=":")
axb.set_xticks(range(len(CONDS)))
axb.set_xticklabels(CL, rotation=40, ha="right", fontsize=7.5)
axb.set_ylabel("endorsement rate")
axb.set_title("borderline-infamous group (8 people × 4 samples)\n"
              "text+ ≈ baseline; +injections raise, optim− suppresses;\n"
              "decoys (hatched) near the trust vectors on the + side",
              fontsize=9)
fig.savefig(os.path.join(OUT, "opengen_summary.png"), dpi=160)
print("wrote opengen_summary.png")
