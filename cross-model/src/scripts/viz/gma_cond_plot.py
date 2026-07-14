"""Isolated eigenmode-projection conditions, 1x3 across models. Per model: grouped bars for
neighbour & parity validity under projecting out {2 lowest}, {highest}, {all 3} (each in isolation),
with matched-rank random-projection controls (lighter bars). Chance floors dotted.
Reads gma_cond_<model>_<graph>.json.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = os.environ.get("DIR", "runs/axes/4_circuits/greedy_mode_ablate")
GRAPH = os.environ.get("GRAPH", "grid")
MODELS = os.environ.get("MODELS", "Llama,Gemma,Qwen").split(",")
OUT = os.environ.get("OUT", f"{DIR}/gma_cond_3models_{GRAPH}.pdf")

# (json condition key, short label)
CONDS = [("clean", "clean"),
         ("low2 {2 lowest = coords}", "2 lowest\n(coords)"),
         ("high1 {highest = parity}", "highest\n(parity)"),
         ("all3 {low2+high1}", "all 3")]

fig, axes = plt.subplots(1, len(MODELS), figsize=(5.2 * len(MODELS), 4.8), sharey=True)
for ax, m in zip(np.atleast_1d(axes), MODELS):
    fp = f"{DIR}/gma_cond_{m}_{GRAPH}.json"
    if not os.path.exists(fp): ax.axis("off"); ax.set_title(f"{m} (missing)"); continue
    d = json.load(open(fp)); C = d["conditions"]; R = d.get("rand", {})
    x = np.arange(len(CONDS)); bw = 0.2
    nbr = [C[k]["neighbour_validity"] for k, _ in CONDS]
    par = [C[k]["parity_validity"] for k, _ in CONDS]
    rnbr = [R.get(k, {}).get("neighbour_validity", np.nan) for k, _ in CONDS]
    rpar = [R.get(k, {}).get("parity_validity", np.nan) for k, _ in CONDS]
    ax.bar(x - 1.5 * bw, nbr, bw, color="#1D4ED8", label="neighbour validity")
    ax.bar(x - 0.5 * bw, rnbr, bw, color="#93C5FD", label="neighbour — random proj")
    ax.bar(x + 0.5 * bw, par, bw, color="#C2410C", label="parity validity")
    ax.bar(x + 1.5 * bw, rpar, bw, color="#FDBA74", label="parity — random proj")
    nch, pch = d["chance"]["neighbour"], d["chance"]["parity"]
    ax.axhline(nch, ls=":", color="#1D4ED8", lw=1.2); ax.axhline(pch, ls=":", color="#C2410C", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in CONDS], fontsize=8)
    ax.set_title(f"{m}  (nbr chance {nch:.2f}, par {pch:.2f})", fontsize=10)
    ax.set_ylim(0, 1.05); ax.grid(axis="y", color="#EEEEEE", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("validity"); axes[0].legend(fontsize=7.5, frameon=False, loc="lower left", ncol=1)
fig.suptitle("Project out eigenmodes IN ISOLATION — {2 lowest} vs {highest} vs {all 3} — 3 models "
             "(lighter = matched-rank random-projection control)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT); fig.savefig(OUT.replace(".pdf", ".png"), dpi=140)
print("wrote", OUT)
