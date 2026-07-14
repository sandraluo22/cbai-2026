"""Progressive joint head-ablation (the RIGHT panel of ghb_both), 1x3 across models, with a
random-ablation baseline overlaid. At each step k the union coord[:k] U parity[:k] is ablated; the
random baseline ablates the SAME number of heads (drawn from neither circuit, averaged). Solid = real
union, dashed = random control. Reads ghb_both_<model>_<graph>.json.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = os.environ.get("DIR", "runs/axes/4_circuits/greedy_head_set")
GRAPH = os.environ.get("GRAPH", "grid")
MODELS = os.environ.get("MODELS", "Llama,Gemma,Qwen").split(",")
OUT = os.environ.get("OUT", f"{DIR}/ghb_both_prog_3models_{GRAPH}.pdf")

fig, axes = plt.subplots(1, len(MODELS), figsize=(5.3 * len(MODELS), 4.8), sharey=True)
for ax, m in zip(np.atleast_1d(axes), MODELS):
    fp = f"{DIR}/ghb_both_{m}_{GRAPH}.json"
    if not os.path.exists(fp): ax.axis("off"); ax.set_title(f"{m} (missing)"); continue
    d = json.load(open(fp)); prog = d["progressive_union"]
    x = [p["step"] for p in prog]
    nbr = [p["neighbour_validity"] for p in prog]; par = [p["parity_validity"] for p in prog]
    rnbr = [p.get("rand_nbr_v") for p in prog]; rpar = [p.get("rand_par_v") for p in prog]
    ax.plot(x, nbr, "o-", color="#1D4ED8", lw=2, label="neighbour validity")
    ax.plot(x, par, "o-", color="#C2410C", lw=2, label="parity validity")
    if all(v is not None for v in rnbr):
        ax.plot(x, rnbr, "--", color="#1D4ED8", lw=1.4, alpha=0.6, label="neighbour — random ablation")
        ax.plot(x, rpar, "--", color="#C2410C", lw=1.4, alpha=0.6, label="parity — random ablation")
    nch, pch = d["chance"]["neighbour"], d["chance"]["parity"]
    ax.axhline(nch, ls=":", color="#1D4ED8", lw=1); ax.axhline(pch, ls=":", color="#C2410C", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(x, fontsize=8)
    ax.set_title(f"{m}  (nbr chance {nch:.2f})", fontsize=11)
    ax.set_xlabel("progressive union step  (ablate coord[:k] ∪ parity[:k])")
    ax.set_ylim(0, 1.03); ax.grid(axis="y", color="#EEEEEE", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("validity"); axes[0].legend(fontsize=7.5, frameon=False, loc="lower left")
fig.suptitle("Progressive joint ablation of coordinate + parity head circuits — 3 models "
             "(dashed = random ablation of same #heads)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT); fig.savefig(OUT.replace(".pdf", ".png"), dpi=140)
print("wrote", OUT)
