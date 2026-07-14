"""Plot greedy eigenmode ablation: neighbour & parity validity vs # modes projected out, for the
normalized and unnormalized Laplacian bases. Each point labelled with the mode added (index, band).
Reads gma_{norm,unnorm}_<model>_<graph>.json.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

DIR = os.environ.get("DIR", "runs/axes/4_circuits/greedy_mode_ablate")
MODEL = os.environ.get("MODEL", "Llama"); GRAPH = os.environ.get("GRAPH", "square_grid")
GS = {"square_grid": "grid"}.get(GRAPH, GRAPH)  # short graph token for filenames

PANELS = [("norm", "normalized Laplacian eigenmodes"),
          ("unnorm", "unnormalized Laplacian eigenmodes"),
          ("cuts", "named cuts {x, y, diag, anti-diag, parity}")]
fig, axes = plt.subplots(1, len(PANELS), figsize=(6.2 * len(PANELS), 5.0), sharey=True)


def lab_for(s):
    if s["mode"] is None: return "clean"
    lam = s.get("lambda")
    return f"+{s['mode']}\n(λ{lam:.1f},{s['label']})" if lam is not None else f"+{s['mode']}"


for ax, (basis, title) in zip(axes, PANELS):
    fp = f"{DIR}/gma_{basis}_{MODEL}_{GS}.json"
    if not os.path.exists(fp): ax.axis("off"); continue
    d = json.load(open(fp))
    g = d["greedy"]; x = list(range(len(g)))
    nbr = [s["neighbour_validity"] for s in g]; par = [s["parity_validity"] for s in g]
    ax.plot(x, nbr, "o-", color="#1D4ED8", lw=2, label="neighbour validity (accuracy)")
    ax.plot(x, par, "o-", color="#C2410C", lw=2, label="parity validity")
    nch, pch = d["chance"]["neighbour"], d["chance"]["parity"]
    ax.axhline(nch, ls=":", color="#1D4ED8", lw=1.2, label=f"neighbour chance ({nch:.2f})")
    ax.axhline(pch, ls=":", color="#C2410C", lw=1.2, label=f"parity chance ({pch:.2f})")
    ax.set_xticks(x); ax.set_xticklabels([lab_for(s) for s in g], fontsize=7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("directions projected out (greedy)")
    ax.set_ylim(0, 1.03); ax.grid(axis="y", color="#EEEEEE", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("validity"); axes[0].legend(fontsize=7.5, frameon=False, loc="upper right")
fig.suptitle(f"Greedy eigenmode ablation → neighbour-validity floor — {MODEL}, {GS}", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
for ext in ("pdf", "png"):
    out = f"{DIR}/gma_curve_{MODEL}_{GS}.{ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
