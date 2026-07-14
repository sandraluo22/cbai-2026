"""Plot greedy head-set saturation curves: step (# heads jointly ablated) vs % of total damage.
Reads ghs_<model>_<graph>.json; writes ghs_curve_<model>_<graph>.pdf/.png.
Env: JSON (path) OUTDIR (defaults alongside JSON).
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

JSON = os.environ.get("JSON", "runs/axes/4_circuits/greedy_head_set/ghs_Llama_grid.json")
d = json.load(open(JSON))
OUTDIR = os.environ.get("OUTDIR", os.path.dirname(JSON))
model, graph = d["model"], {"square_grid":"grid"}.get(d["graph"], d["graph"])

# brand-neutral, colorblind-safe: parity (concentrated) vs coordinates (distributed)
STYLE = {
    "parity": dict(color="#C2410C", marker="o", label="parity"),
    "coord":  dict(color="#1D4ED8", marker="s", label="coordinates"),
}

fig, ax = plt.subplots(figsize=(6.4, 4.4))
for oname in ("parity", "coord"):
    obj = d["objectives"][oname]
    curve = obj["greedy"]
    steps = np.arange(1, len(curve) + 1)
    pct = np.array([c["cum_frac"] * 100 for c in curve])
    st = STYLE[oname]
    ax.plot(steps, pct, color=st["color"], marker=st["marker"], markersize=6,
            linewidth=2, label=st["label"], zorder=3)
    # annotate the head added at each step
    for s, p, c in zip(steps, pct, curve):
        l, h = c["head"]
        ax.annotate(f"L{l}H{h}", (s, p), textcoords="offset points",
                    xytext=(0, -13 if oname == "coord" else 8),
                    ha="center", fontsize=6.5, color=st["color"])

ax.axhline(100, color="#9CA3AF", linestyle=":", linewidth=1, zorder=1)
ax.set_xlabel("greedy step  (# heads jointly ablated)")
ax.set_ylabel("cumulative damage  (% of total)")
ax.set_title(f"Greedy joint head-set saturation — {model}, {graph}")
ax.set_ylim(0, 105)
ax.set_xticks(range(1, 9))
ax.grid(axis="y", color="#E5E7EB", linewidth=0.6, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()

for ext in ("pdf", "png"):
    out = f"{OUTDIR}/ghs_curve_{model}_{graph}.{ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
