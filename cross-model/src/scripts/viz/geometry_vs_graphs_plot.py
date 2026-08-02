"""Does each concept-space FAMILY resemble the abstract graph structures (ring / grid / hex)? For every
space we plot its unsupervised RSA to ring, grid, hex (whichever its coordinates allow to build) alongside
its own hypothesised geometry. 1D families (arcs) can be laid on a ring; 2D families (products, helices)
on grid/hex; simplices & trees have no lattice layout. All RSA is unsupervised (fixed structural distance
matrices) so nothing here can overfit. Reads geometry_fit_<model>.json.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

DIR = os.environ.get("DIR", "runs/axes/6_geometry"); MODEL = os.environ.get("MODEL", "Llama")
d = json.load(open(f"{DIR}/geometry_fit_{MODEL}.json"))["spaces"]
GCOL = {"ring": "#059669", "grid": "#C2410C", "hex": "#7C3AED", "own": "#111827"}
order = [n for fam in ["arc", "helix", "product", "simplex", "tree"] for n in d if d[n]["family"] == fam]

fig, ax = plt.subplots(figsize=(11, 6))
y = np.arange(len(order)); h = 0.2
for yi, n in zip(y, order):
    r = d[n]; g = r.get("graph_rsa", {})
    own = r["equidistance"] if r["family"] == "simplex" else (r["intended_rsa"] or 0)
    bars = [("own\n(" + r["intended"] + ")", own, GCOL["own"])] + [(k, g[k], GCOL[k]) for k in ("ring", "grid", "hex") if k in g]
    for j, (lab, val, c) in enumerate(bars):
        ax.barh(yi + (1.5 - j) * h, val, h, color=c)
    # mark best graph structure
    if g:
        bg = max(g, key=g.get)
        ax.text(1.01, yi, f"→ {bg}" if g[bg] > 0.3 else "→ none", va="center", fontsize=7.5,
                color=GCOL.get(bg, "#111827") if g[bg] > 0.3 else "#9CA3AF")
    else:
        ax.text(1.01, yi, "→ none (equidist/tree)", va="center", fontsize=7.5, color="#9CA3AF")
ax.set_yticks(y); ax.set_yticklabels([f"{n} [{d[n]['family']}]" for n in order], fontsize=8); ax.invert_yaxis()
ax.set_xlabel("unsupervised RSA (rep distances vs structure — cannot overfit)"); ax.set_xlim(0, 1.15)
ax.axvline(0, color="k", lw=.5)
handles = [plt.Rectangle((0, 0), 1, 1, color=GCOL[k]) for k in ["own", "ring", "grid", "hex"]]
ax.legend(handles, ["own geometry", "ring", "grid", "hex"], fontsize=8, frameon=False, loc="lower right")
ax.set_title("Products & helices → GRID;  some arcs → RING;  simplices & trees → none of ring/grid/hex", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"Does each concept family resemble the ring / grid / hex graph structures? ({MODEL})", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
for ext in ("pdf",):
    out = f"{DIR}/geometry_vs_graphs_{MODEL}.{ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
