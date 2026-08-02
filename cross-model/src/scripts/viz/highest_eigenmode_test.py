"""Highest-eigenmode test across graph geometries (grid, hex, ring). For each graph we plot the per-mode
neighbour-prediction importance (Δ neighbour validity when that mode is projected out) against the mode's
normalized-Laplacian eigenvalue. The HIGH-frequency (top-eigenvalue) modes dominate neighbour prediction
everywhere -- predicting the immediate neighbour needs fine spatial resolution. The top eigenvalue itself
separates geometries: BIPARTITE graphs (grid, ring) top out at λ=2 with a clean parity 2-colouring the model
represents (parity-validity ≈ 1); the FRUSTRATED hex (triangular, non-bipartite) has no λ=2 parity mode
(λ_max<2) and low parity-validity. Reads per_mode_ablate_<model>_{grid,ring,hex}.json.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

DIR = os.environ.get("DIR", "runs/axes/4_circuits/per_mode_ablate"); MODEL = os.environ.get("MODEL", "Llama")
GRAPHS = os.environ.get("GRAPHS", "grid,hex,ring").split(",")

fig, axes = plt.subplots(1, len(GRAPHS), figsize=(5.0 * len(GRAPHS), 4.4), sharey=True)
for ax, g in zip(axes, GRAPHS):
    d = json.load(open(f"{DIR}/per_mode_ablate_{MODEL}_{g}.json"))
    ms = sorted(d["modes"], key=lambda x: x["eigenvalue"])
    lam = np.array([m["eigenvalue"] for m in ms]); dnb = np.array([m["d_nbr"] for m in ms])
    lam_max = lam.max(); bip = lam_max > 1.97
    # colour by frequency band; star the highest-eigenvalue mode
    ax.bar(lam, dnb, width=0.06, color=["#C2410C" if l > 1.97 else "#1D4ED8" if l < 0.5 else "#6B7280" for l in lam])
    top = ms[int(np.argmax(lam))]
    ax.scatter([top["eigenvalue"]], [top["d_nbr"]], marker="*", s=180, color="#EA580C", zorder=5,
               label=f"highest mode m{top['mode']} (λ={top['eigenvalue']:.2f})")
    rand = d["baseline"]["neighbour_validity"] - d["random_rank1"]["neighbour_validity"]
    ax.axhline(rand, ls=":", color="k", lw=1, label=f"random rank-1 (Δ={rand:+.02f})")
    ax.set_title(f"{g}   λ_max={lam_max:.2f}  {'bipartite→parity' if bip else 'frustrated: no parity'}\n"
                 f"parity-validity={d['baseline']['parity_validity']:.2f}", fontsize=10)
    ax.set_xlabel("normalized-Laplacian eigenvalue (freq →)")
    ax.grid(axis="y", color="#EEE", lw=.6); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
axes[0].set_ylabel("Δ neighbour validity (importance)")
fig.suptitle(f"Highest-eigenmode test ({MODEL}): high-freq modes dominate neighbour prediction; "
             f"λ_max=2 parity exists only for bipartite grid/ring, not frustrated hex", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
for ext in ("pdf",):
    out = f"{DIR}/highest_eigenmode_test_{MODEL}.{ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
