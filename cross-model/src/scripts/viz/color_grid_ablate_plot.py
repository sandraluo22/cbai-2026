"""Plot the colour-grid causal ablation (double-dissociation attempt). Grouped bars: lightness task and hue
task accuracy under baseline / ablate-lightness / ablate-hue / ablate-random. A clean 2D grid with
orthogonal, causally-used axes would show a double dissociation (each ablation breaks only its own task).
Caveat annotated: the lightness comparison is lexically trivial (the words dark/light are in the prompt),
so its arm is inconclusive; hue breaks under both axis ablations => the axes are non-orthogonal.
Reads color_grid_ablate_<model>.json.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

DIR = os.environ.get("DIR", "runs/axes/6_geometry"); MODEL = os.environ.get("MODEL", "Llama")
d = json.load(open(f"{DIR}/color_grid_ablate_{MODEL}.json"))
conds = ["baseline", "ablate_lightness", "ablate_hue", "ablate_random"]
CC = {"baseline": "#111827", "ablate_lightness": "#6B7280", "ablate_hue": "#EA580C", "ablate_random": "#9CA3AF"}

fig, ax = plt.subplots(figsize=(8.5, 5))
x = np.arange(2); w = 0.2
for j, c in enumerate(conds):
    vals = [d["results"][c]["lightness_task"], d["results"][c]["hue_task"]]
    ax.bar(x + (j - 1.5) * w, vals, w, color=CC[c], label=c)
    for xi, v in zip(x + (j - 1.5) * w, vals): ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(["lightness task\n(which is darker)", "hue task\n(closer to red)"])
ax.set_ylabel("task accuracy"); ax.set_ylim(0, 1.12); ax.axhline(0.5, ls=":", color="k", lw=1, label="chance (2-choice)")
ax.set_title(f"Colour-grid causal ablation ({MODEL})  —  LOO decode R²: lightness={d['loo_decode']['lightness']:.2f}, hue={d['loo_decode']['hue']:.2f}\n"
             "hue breaks under BOTH axis ablations (not random) ⇒ axes non-orthogonal; lightness task is lexically trivial", fontsize=9)
ax.legend(fontsize=8, frameon=False, ncol=2); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
for ext in ("pdf",):
    out = f"{DIR}/color_grid_ablate_{MODEL}.{ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
