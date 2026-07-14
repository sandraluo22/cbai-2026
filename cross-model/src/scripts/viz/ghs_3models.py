"""Greedy head-set saturation for all 3 models on ONE slide (one subplot per model).
Each subplot: cumulative damage (% of total) vs greedy step, for the coord(xy) and parity head sets,
with the head added labelled at each step. Reads ghs_<model>_<graph>.json.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = os.environ.get("DIR", "runs/axes/4_circuits/greedy_head_set")
GRAPH = os.environ.get("GRAPH", "square_grid")
GS = {"square_grid": "grid"}.get(GRAPH, GRAPH)  # short graph token for filenames
MODELS = os.environ.get("MODELS", "Llama,Gemma,Qwen").split(",")
OUT = os.environ.get("OUT", f"{DIR}/ghs_3models_{GS}.pdf")
STYLE = {"parity": dict(color="#C2410C", marker="o", label="parity"),
         "coord":  dict(color="#1D4ED8", marker="s", label="coordinates")}

fig, axes = plt.subplots(1, len(MODELS), figsize=(5.3 * len(MODELS), 4.8), sharey=True)
for ax, m in zip(np.atleast_1d(axes), MODELS):
    fp = f"{DIR}/ghs_{m}_{GS}.json"
    if not os.path.exists(fp): ax.axis("off"); ax.set_title(f"{m} (missing)"); continue
    d = json.load(open(fp))
    for oname in ("parity", "coord"):
        curve = d["objectives"][oname]["greedy"]; steps = np.arange(1, len(curve) + 1)
        pct = np.array([c["cum_frac"] * 100 for c in curve]); st = STYLE[oname]
        ax.plot(steps, pct, color=st["color"], marker=st["marker"], ms=5, lw=1.8, label=st["label"])
        for s, p, c in zip(steps, pct, curve):
            l, h = c["head"]
            ax.annotate(f"L{l}H{h}", (s, p), textcoords="offset points",
                        xytext=(0, -12 if oname == "coord" else 7), ha="center", fontsize=6, color=st["color"])
    ax.axhline(100, color="#9CA3AF", ls=":", lw=1)
    ax.set_title(m, fontsize=11); ax.set_xlabel("greedy step (# heads jointly ablated)")
    ax.set_ylim(0, 108); ax.set_xticks(range(1, 9)); ax.grid(axis="y", color="#EEEEEE", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("cumulative damage (% of total)")
axes[0].legend(frameon=False, loc="lower right")
fig.suptitle(f"Greedy joint head-set saturation — 3 models ({GS})", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT); fig.savefig(OUT.replace(".pdf", ".png"), dpi=140)
print("wrote", OUT)
