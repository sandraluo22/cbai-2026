"""Greedy eigenmode ablation for all 3 models on ONE slide (one subplot per model).
Each subplot: neighbour & parity validity vs ablation step. The first steps are the forced SEED modes
(2 lowest + highest eigenmode, shaded); later steps are greedy. Chance floors dashed.
Reads gma_normseed_<model>_<graph>.json.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = os.environ.get("DIR", "runs/axes/4_circuits/greedy_mode_ablate")
GRAPH = os.environ.get("GRAPH", "square_grid")
GS = {"square_grid": "grid"}.get(GRAPH, GRAPH)  # short graph token for filenames
MODELS = os.environ.get("MODELS", "Llama,Gemma,Qwen").split(",")
SUFFIX = os.environ.get("SUFFIX", "normseed")
OUT = os.environ.get("OUT", f"{DIR}/gma_3models_{GS}.pdf")

fig, axes = plt.subplots(1, len(MODELS), figsize=(5.3 * len(MODELS), 4.8), sharey=True)
for ax, m in zip(np.atleast_1d(axes), MODELS):
    fp = f"{DIR}/gma_{SUFFIX}_{m}_{GS}.json"
    if not os.path.exists(fp): ax.axis("off"); ax.set_title(f"{m} (missing)"); continue
    d = json.load(open(fp)); g = d["greedy"]; x = list(range(len(g)))
    nbr = [s["neighbour_validity"] for s in g]; par = [s["parity_validity"] for s in g]
    # shade the seeded region
    seed_steps = [s["step"] for s in g if s.get("seeded")]
    if seed_steps:
        ax.axvspan(0.5, max(seed_steps) + 0.5, color="#FDE68A", alpha=0.35, lw=0, label="seed (lo,lo2,hi)")
    ax.plot(x, nbr, "o-", color="#1D4ED8", lw=2, label="neighbour validity")
    ax.plot(x, par, "o-", color="#C2410C", lw=2, label="parity validity")
    rnbr = [s.get("rand_nbr_v") for s in g]; rpar = [s.get("rand_par_v") for s in g]
    if all(v is not None for v in rnbr):
        ax.plot(x, rnbr, "--", color="#1D4ED8", lw=1.4, alpha=0.6, label="neighbour — random projection")
        ax.plot(x, rpar, "--", color="#C2410C", lw=1.4, alpha=0.6, label="parity — random projection")
    nch, pch = d["chance"]["neighbour"], d["chance"]["parity"]
    ax.axhline(nch, ls=":", color="#1D4ED8", lw=1.2); ax.axhline(pch, ls=":", color="#C2410C", lw=1.2)
    labs = ["clean"] + [f"+{s['mode']}\n({'S' if s.get('seeded') else 'g'})" for s in g[1:]]
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=7)
    ax.set_title(f"{m}  (nbr chance {nch:.2f})", fontsize=11)
    ax.set_xlabel("modes projected out  (S=seed, g=greedy)")
    ax.set_ylim(0, 1.03); ax.grid(axis="y", color="#EEEEEE", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("validity"); axes[0].legend(fontsize=8, frameon=False, loc="upper right")
fig.suptitle(f"Greedy eigenmode ablation (seed: 2 lowest + highest, then greedy) — 3 models ({GS})", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT); fig.savefig(OUT.replace(".pdf", ".png"), dpi=140)
print("wrote", OUT)
