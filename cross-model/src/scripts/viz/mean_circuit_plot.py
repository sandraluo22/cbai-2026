"""(left) named mean-ablation circuits: clean / keep-none / M / M+DLA / M+ind / M+DLA+ind.
(right) additive greedy RESTORE curve from all-heads-mean-ablated. Reads mean_circuit_<model>_<G>.json.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JSON = os.environ.get("JSON", "runs/axes/4_circuits/mean_circuit/mean_circuit_Llama_grid.json")
d = json.load(open(JSON)); OUTDIR = os.environ.get("OUTDIR", os.path.dirname(JSON)); m = d["model"]
nch, pch = d["chance"]["neighbour"], d["chance"]["parity"]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})

# --- left: named circuits ---
order = [("keep_none", f"none\n(all mean-abl)"), ("keep_M", f"M\n({len(d['M'])}h)"),
         ("keep_M+DLA", f"M+DLA\n({len(d['M'])+len(d['DLA'])}h)"),
         ("keep_M+ind", f"M+ind\n({len(d['M'])+len(d['induction'])}h)"),
         ("keep_M+DLA+ind", f"M+DLA+ind\n({len(d['M'])+len(d['DLA'])+len(d['induction'])}h)"),
         ("clean", "clean")]
x = np.arange(len(order)); bw = 0.36
nbr = [d["named"][k]["neighbour_validity"] for k, _ in order]
par = [d["named"][k]["parity_validity"] for k, _ in order]
axL.bar(x - bw/2, nbr, bw, color="#1D4ED8", label="neighbour validity")
axL.bar(x + bw/2, par, bw, color="#C2410C", label="parity validity")
axL.axhline(nch, ls=":", color="#1D4ED8", lw=1.2); axL.axhline(pch, ls=":", color="#C2410C", lw=1.2)
axL.set_xticks(x); axL.set_xticklabels([lab for _, lab in order], fontsize=8)
axL.set_ylim(0, 1.05); axL.set_ylabel("validity"); axL.legend(frameon=False, fontsize=8, loc="upper left")
axL.set_title("Mean-ablation keep-only circuits (MLPs kept clean)", fontsize=10)
axL.spines[["top", "right"]].set_visible(False); axL.grid(axis="y", color="#EEE", lw=0.6)

# --- right: greedy restore curve ---
r = d["restore"]; xs = [s["step"] for s in r]
axR.plot(xs, [s["neighbour_validity"] for s in r], "o-", color="#1D4ED8", lw=2, label="neighbour validity")
axR.plot(xs, [s["parity_validity"] for s in r], "o-", color="#C2410C", lw=2, label="parity validity")
axR.axhline(d["named"]["keep_M+DLA+ind"]["neighbour_validity"], ls="--", color="#16A34A", lw=1.4,
            label="M+DLA+ind (0.97)")
axR.axhline(nch, ls=":", color="#1D4ED8", lw=1); axR.axhline(d["named"]["clean"]["neighbour_validity"], ls=":", color="0.5", lw=1)
# label a few heads
DLA = {tuple(h) for h in d["DLA"]}; IND = {tuple(h) for h in d["induction"]}; MM = {tuple(h) for h in d["M"]}
for s in r[1:]:
    h = tuple(s["head"]); grp = "D" if h in DLA else ("i" if h in IND else ("M" if h in MM else ""))
    axR.annotate(f"L{h[0]}H{h[1]}{('·'+grp) if grp else ''}", (s["step"], s["neighbour_validity"]),
                 textcoords="offset points", xytext=(0, 6), ha="center", fontsize=5.5, rotation=90, color="0.3")
axR.set_xlabel("restore step (# heads restored to clean)"); axR.set_ylabel("validity")
axR.set_title("Additive greedy restore from all-mean-ablated", fontsize=10)
axR.set_ylim(0, 1.05); axR.set_xticks(xs[::2]); axR.legend(frameon=False, fontsize=8, loc="lower right")
axR.spines[["top", "right"]].set_visible(False); axR.grid(axis="y", color="#EEE", lw=0.6)

fig.suptitle(f"Minimum viable circuit ({m}, grid): builders (M) + movers (induction) + readers (DLA)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
for ext in ("pdf", "png"):
    fig.savefig(f"{OUTDIR}/mean_circuit_{m}_grid.{ext}", dpi=150, bbox_inches="tight")
print("wrote", f"{OUTDIR}/mean_circuit_{m}_grid.pdf")
