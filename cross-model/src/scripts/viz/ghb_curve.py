"""Plot downstream behaviour vs greedy head-ablation step, for the coord(xy) and parity greedy sets.
Reads ghb_<model>_<graph>.json. Two panels (coord | parity); each shows neighbour
validity (=next-node accuracy), neighbour mass, parity validity, parity mass vs # heads ablated,
with the head added labelled at each step and the parity chance line at 0.5.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

JSON = os.environ.get("JSON", "runs/axes/4_circuits/greedy_head_set/ghb_Llama_grid.json")
d = json.load(open(JSON)); OUTDIR = os.environ.get("OUTDIR", os.path.dirname(JSON))
model, graph = d["model"], {"square_grid":"grid"}.get(d["graph"], d["graph"])

METRICS = [("neighbour_validity", "#1D4ED8", "o", "neighbour validity (accuracy)"),
           ("neighbour_mass",     "#60A5FA", "s", "neighbour mass"),
           ("parity_validity",    "#C2410C", "o", "parity validity"),
           ("parity_mass",        "#FB923C", "s", "parity mass")]
TITLE = {"coord": "ablate COORDINATE (x+y) greedy head set",
         "parity": "ablate PARITY greedy head set"}

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
for ax, oname in zip(axes, ("coord", "parity")):
    steps = d["objectives"][oname]
    x = [s["step"] for s in steps]
    labels = ["clean"] + [f"+L{s['head_added'][0]}H{s['head_added'][1]}" for s in steps[1:]]
    for key, col, mk, lab in METRICS:
        y = [s[key] for s in steps]
        ax.plot(x, y, color=col, marker=mk, markersize=5, linewidth=1.8, label=lab, zorder=3)
    ax.axhline(0.5, color="#9CA3AF", ls=":", lw=1, zorder=1, label="parity chance (0.5)")
    ax.set_title(f"{TITLE[oname]}", fontsize=10)
    ax.set_xlabel("greedy step  (# heads jointly ablated)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0.3, 1.02); ax.grid(axis="y", color="#EEEEEE", lw=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("behaviour metric")
axes[0].legend(frameon=False, fontsize=8, loc="lower left")
fig.suptitle(f"Downstream behaviour along greedy head sets — {model}, {graph}", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
for ext in ("pdf", "png"):
    out = f"{OUTDIR}/ghb_{model}_{graph}.{ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
