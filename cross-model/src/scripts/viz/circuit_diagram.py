"""Circuit diagram (Llama-8B, grid), four head entities. Top: heads placed by LAYER in four lanes
(coord / parity / QK-induction / DLA). Bottom: ALL 16 combinations of the four entities — neighbour
validity under mean-ablation keep-only (keep the union, mean-ablate the rest; MLPs clean).
Reads mean_circuit_combos_<model>_<G>.json.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

J = json.load(open("runs/axes/4_circuits/mean_circuit/mean_circuit_combos_Llama_grid.json"))
E = J["entities"]; nL = 32
LANES = [("DLA — write next-node logits", "DLA", "#7C3AED", 3),
         ("QK — induction (read edges from context)", "QK", "#059669", 2),
         ("parity — colour", "parity", "#C2410C", 1),
         ("coord — position", "coord", "#1D4ED8", 0)]
ECOL = {"coord": "#1D4ED8", "parity": "#C2410C", "QK": "#059669", "DLA": "#7C3AED"}

fig = plt.figure(figsize=(15, 9)); gs = fig.add_gridspec(2, 1, height_ratios=[1.7, 1.4], hspace=0.34)

# --- top: head map by layer, four lanes ---
ax = fig.add_subplot(gs[0])
for label, key, colr, y in LANES:
    for (l, h) in E[key]:
        ax.scatter(l, y, s=230, color=colr, edgecolors="white", linewidths=1.2, zorder=3)
        ax.annotate(f"{l}·{h}", (l, y), fontsize=5.5, color="white", ha="center", va="center", zorder=4)
    ax.text(-1.4, y, label, ha="right", va="center", fontsize=9, color=colr, fontweight="bold")
ax.text(-1.4, -0.85, "context →", ha="right", fontsize=9, color="0.4")
ax.text(nL - 0.5, -0.85, "→ logits", ha="left", fontsize=9, color="0.4")
ax.set_xlim(-11, nL + 1); ax.set_ylim(-1.1, 3.6); ax.set_yticks([])
ax.set_xlabel("layer (depth →)", fontsize=10); ax.set_xticks(range(0, nL, 2))
ax.set_title("Four head entities by depth (Llama-8B, grid)", fontsize=12)
for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)

# --- bottom: all 16 combinations, sorted by neighbour validity ---
axb = fig.add_subplot(gs[1])
combos = J["combos"]; items = sorted(combos.items(), key=lambda kv: kv[1]["neighbour_validity"])
x = np.arange(len(items)); vals = [v["neighbour_validity"] for _, v in items]
# color each bar by a stack of its entities (dominant = highest-order group present)
def barcol(groups):
    if not groups: return "0.7"
    for g in ["QK", "parity", "DLA", "coord"]:
        if g in groups: return ECOL[g]
    return "0.7"
cols = [barcol(v["groups"]) for _, v in items]
axb.bar(x, vals, 0.7, color=cols, edgecolor="white")
axb.axhline(combos["none"]["neighbour_validity"], ls="--", color="0.5", lw=1.2, label="all-mean-ablated (0.39)")
axb.axhline(J["chance"]["neighbour"], ls=":", color="#1D4ED8", lw=1, label="chance (0.20)")
axb.axhline(J["clean"]["neighbour_validity"], ls=":", color="0.3", lw=1, label="clean (0.99)")
for xi, (_, v) in zip(x, items): axb.text(xi, v["neighbour_validity"] + 0.015, f"{v['neighbour_validity']:.2f}", ha="center", fontsize=7)
labs = [k.replace("+", "+\n") if len(k) > 10 else k for k, _ in items]
axb.set_xticks(x); axb.set_xticklabels([k for k, _ in items], fontsize=7, rotation=40, ha="right")
axb.set_ylim(0, 1.06); axb.set_ylabel("neighbour validity")
axb.set_title("All 16 combinations of {coord, parity, QK, DLA} — the essential pair is parity+QK; "
              "coord is nearly dispensable", fontsize=10.5)
axb.legend(fontsize=8, frameon=False, loc="upper left"); axb.spines[["top", "right"]].set_visible(False)
axb.grid(axis="y", color="#EEE", lw=0.6)

fig.suptitle("Head-circuit for in-context graph tracing — entities & all combinations", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
for ext in ("pdf", "png"):
    fig.savefig(f"runs/axes/4_circuits/mean_circuit/circuit_diagram_Llama_grid.{ext}", dpi=150, bbox_inches="tight")
print("wrote circuit_diagram_Llama_grid.png")
