"""(left) greedy head-set saturation per critical eigenmode: which heads build m2 / m11 / m14.
(right) keep-only test: ablate all heads except the union M vs a matched random keep vs keep-none.
Reads head_eig_greedy_keep_<model>_<G>.json.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JSON = os.environ.get("JSON", "runs/axes/4_circuits/head_eig_greedy_keep/head_eig_greedy_keep_Llama_grid.json")
d = json.load(open(JSON)); OUTDIR = os.environ.get("OUTDIR", os.path.dirname(JSON)); m = d["model"]
MODECOL = {"2": "#1D4ED8", "11": "#7C3AED", "14": "#C2410C"}
MODELAB = {"2": "m2 (coord)", "11": "m11 (parity×coord)", "14": "m14 (parity×fold)"}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1.4, 1]})

# --- left: per-mode greedy saturation ---
for key, gd in d["greedy_per_mode"].items():
    curve = gd["greedy"]; x = np.arange(1, len(curve) + 1)
    frac = [c["cum_frac"] * 100 for c in curve]
    axL.plot(x, frac, "o-", color=MODECOL[key], lw=2, label=MODELAB[key])
    for s, f, c in zip(x, frac, curve):
        l, h = c["head"]
        axL.annotate(f"L{l}H{h}", (s, f), textcoords="offset points", xytext=(0, 6),
                     ha="center", fontsize=6, color=MODECOL[key], rotation=0)
axL.set_xlabel("greedy step (# heads ablated)"); axL.set_ylabel("power damage to mode (% of clean)")
axL.set_title(f"{m}: which heads build each critical eigenmode", fontsize=10)
axL.set_ylim(0, 100); axL.set_xticks(range(1, len(x) + 1)); axL.grid(axis="y", color="#EEE", lw=0.6)
axL.spines[["top", "right"]].set_visible(False); axL.legend(frameon=False, loc="lower right", fontsize=9)

# --- right: keep-only bars ---
ko = d["keep_only"]; conds = [("clean", "clean"), ("M", f"keep only M\n({len(d['union_heads'])}h)"),
                              ("random", f"keep random\n({len(d['union_heads'])}h)"), ("none", "keep none")]
x = np.arange(len(conds)); bw = 0.35
nbr = [ko[k]["neighbour_validity"] for k, _ in conds]
par = [ko[k]["parity_validity"] for k, _ in conds]
axR.bar(x - bw / 2, nbr, bw, color="#1D4ED8", label="neighbour validity")
axR.bar(x + bw / 2, par, bw, color="#C2410C", label="parity validity")
axR.axhline(d["chance"]["neighbour"], ls=":", color="#1D4ED8", lw=1.2, label=f"nbr chance ({d['chance']['neighbour']:.2f})")
axR.axhline(d["chance"]["parity"], ls=":", color="#C2410C", lw=1.2, label=f"par chance ({d['chance']['parity']:.2f})")
axR.set_xticks(x); axR.set_xticklabels([lab for _, lab in conds], fontsize=8)
axR.set_ylim(0, 1.05); axR.set_ylabel("validity")
axR.set_title("keep-only: ablate all heads except M", fontsize=10)
axR.legend(frameon=False, fontsize=7.5, loc="upper right"); axR.spines[["top", "right"]].set_visible(False)
axR.grid(axis="y", color="#EEE", lw=0.6)

fig.suptitle(f"Eigenmode → head circuit ({m}, grid): each critical mode's builder-heads, "
             "and keep-only sufficiency", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
for ext in ("pdf", "png"):
    fig.savefig(f"{OUTDIR}/head_eig_greedy_keep_{m}_grid.{ext}", dpi=150, bbox_inches="tight")
print("wrote", f"{OUTDIR}/head_eig_greedy_keep_{m}_grid.pdf")
