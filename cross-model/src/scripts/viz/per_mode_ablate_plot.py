"""Plot experiment 2 (per-mode single ablation): for each eigenmode, the drop in neighbour and parity
validity when that ONE mode's readout direction is projected out at every layer, vs the random rank-1
control. Bars coloured by the mode's index label from the eigmode index (parity / coord / product).
Reads per_mode_ablate_<model>_<graph>.json.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

DIR = os.environ.get("DIR", "runs/axes/4_circuits/per_mode_ablate")
MODEL = os.environ.get("MODEL", "Llama"); GRAPH = os.environ.get("GRAPH", "square_grid")
GS = {"square_grid": "grid"}.get(GRAPH, GRAPH)
# semantic label per grid mode index (from eigmode_index exp 3); overrides gma's coarse label
IDX = {1: "coord", 2: "coord", 3: "coord", 4: "fold", 5: "fold", 6: "product", 7: "parity×coord",
       8: "product", 9: "parity×coord", 10: "parity×coord", 11: "parity×coord", 12: "fold",
       13: "parity×fold", 14: "parity×fold", 15: "parity"}
CMAP = {"parity": "#C2410C", "parity×fold": "#EA580C", "parity×coord": "#F59E0B",
        "coord": "#1D4ED8", "fold": "#6B7280", "product": "#9CA3AF"}

d = json.load(open(f"{DIR}/per_mode_ablate_{MODEL}_{GS}.json"))
modes = sorted(d["modes"], key=lambda x: x["mode"])
x = np.array([m["mode"] for m in modes])
labs = [IDX.get(m["mode"], m["label"]) for m in modes]
cols = [CMAP.get(l, "#9CA3AF") for l in labs]
base = d["baseline"]; rc = d["random_rank1"]

fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
for ax, key, title, bkey in [(axes[0], "d_nbr", "Δ neighbour validity (baseline − ablated)", "neighbour_validity"),
                             (axes[1], "d_par", "Δ parity validity (baseline − ablated)", "parity_validity")]:
    ax.bar(x, [m[key] for m in modes], color=cols)
    rand_drop = base[bkey] - rc[bkey]
    ax.axhline(rand_drop, ls=":", color="k", lw=1.2, label=f"random rank-1 (Δ={rand_drop:+.3f})")
    for xi, m, l in zip(x, modes, labs):
        if m[key] > 0.05: ax.text(xi, m[key] + 0.005, f"m{m['mode']}", ha="center", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels([f"m{i}" for i in x], fontsize=6, rotation=90)
    ax.set_xlabel("eigenmode"); ax.set_title(title, fontsize=10)
    ax.grid(axis="y", color="#EEEEEE", lw=0.6); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CMAP.values()]
axes[0].legend(handles + [plt.Line2D([0], [0], ls=":", color="k")], list(CMAP) + ["random rank-1"],
               fontsize=7.5, frameon=False, loc="upper left", ncol=2)
fig.suptitle(f"Per-mode single ablation — {MODEL}, {GS} (baseline nbr={base['neighbour_validity']:.2f}, "
             f"chance={d['chance']['neighbour']:.2f})", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
for ext in ("pdf",):
    out = f"{DIR}/per_mode_ablate_{MODEL}_{GS}.{ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
