"""Plot the BOTH-ablation test: does behaviour fully collapse when the coord AND parity greedy sets
are ablated together? Panel A = grouped bars over conditions (clean / coord-only / parity-only /
both-union / random control); Panel B = progressive-union curve. Chance floors drawn as dashed lines.
Reads ghb_both_<model>_<graph>.json.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

JSON = os.environ.get("JSON", "runs/axes/4_circuits/greedy_head_set/ghb_both_Llama_grid.json")
d = json.load(open(JSON)); OUTDIR = os.environ.get("OUTDIR", os.path.dirname(JSON))
model, graph = d["model"], {"square_grid":"grid"}.get(d["graph"], d["graph"])
nch, pch = d["chance"]["neighbour"], d["chance"]["parity"]

METRICS = [("neighbour_validity", "#1D4ED8", "neighbour validity (accuracy)"),
           ("neighbour_mass",     "#60A5FA", "neighbour mass"),
           ("parity_validity",    "#C2410C", "parity validity"),
           ("parity_mass",        "#FB923C", "parity mass")]
CONDS = [("clean", "clean"), ("coord_only", "coord only\n(8h)"), ("parity_only", "parity only\n(8h)"),
         ("both_union", f"BOTH union\n({d['n_union']}h)"), ("random_ctrl", f"random ctrl\n({d['n_union']}h)")]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})

# ---- Panel A: grouped bars over conditions ----
x = np.arange(len(CONDS)); bw = 0.2
for i, (key, col, lab) in enumerate(METRICS):
    vals = [d["conditions"][c][key] for c, _ in CONDS]
    axA.bar(x + (i - 1.5) * bw, vals, bw, color=col, label=lab)
axA.axhline(nch, color="#1D4ED8", ls=":", lw=1.2, label=f"neighbour chance ({nch:.2f})")
axA.axhline(pch, color="#C2410C", ls=":", lw=1.2, label=f"parity chance ({pch:.2f})")
axA.set_xticks(x); axA.set_xticklabels([lab for _, lab in CONDS], fontsize=8)
axA.set_ylim(0, 1.05); axA.set_ylabel("behaviour metric")
axA.set_title("Ablating BOTH circuits vs each alone vs random control", fontsize=10)
axA.legend(fontsize=7, ncol=2, loc="upper center", frameon=False)
axA.spines[["top", "right"]].set_visible(False)

# ---- Panel B: progressive union curve ----
prog = d["progressive_union"]
xs = [p["step"] for p in prog]
axB.plot(xs, [p["neighbour_validity"] for p in prog], "o-", color="#1D4ED8", label="neighbour validity")
axB.plot(xs, [p["parity_validity"] for p in prog], "o-", color="#C2410C", label="parity validity")
axB.axhline(nch, color="#1D4ED8", ls=":", lw=1.2); axB.axhline(pch, color="#C2410C", ls=":", lw=1.2)
axB.set_xlabel("progressive union step  (ablate coord[:k] ∪ parity[:k])")
axB.set_ylabel("validity"); axB.set_ylim(0, 1.05); axB.set_xticks(xs)
axB.set_title("Progressive joint ablation", fontsize=10)
axB.legend(fontsize=8, frameon=False, loc="upper right")
axB.spines[["top", "right"]].set_visible(False)
axB.grid(axis="y", color="#EEEEEE", lw=0.6)

fig.suptitle(f"Joint ablation of coordinate + parity circuits — {model}, {graph}", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
for ext in ("pdf", "png"):
    out = f"{OUTDIR}/ghb_both_{model}_{graph}.{ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
