"""Plot experiment 4 (causal head interchange) and cross-check vs the known circuit. For a SOURCE arm
(antiprism=parity, ring=coord) show: (1) the per-head swap map -- how much injecting that head's
source-run output makes the GRID run predict the SOURCE graph's neighbours; (2) the ablation-derived
circuit write-map for the matching variable (parity = top-eigenvalue mode, coord = two lowest modes,
from head_eig_sweep 'damage'); (3) a scatter of swap vs circuit-write per head with correlation and the
top swap heads labelled. High correlation => the causal interchange confirms the parity/coord circuit.

Env: MODEL(Llama) SOURCE(antiprism|ring) METRIC(d_src_nbr|d_src_par) DIR SWEEPDIR OUTDIR
Reads interchange_<model>_<source>.json + head_eig_sweep_<model>_grid.json.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

MODEL = os.environ.get("MODEL", "Llama"); SOURCE = os.environ.get("SOURCE", "antiprism")
METRIC = os.environ.get("METRIC", "d_src_nbr")
DIR = os.environ.get("DIR", "runs/axes/4_circuits/interchange")
SWEEPDIR = os.environ.get("SWEEPDIR", "runs/axes/4_circuits/head_eig_sweep")
OUTDIR = os.environ.get("OUTDIR", DIR)
VAR = "parity" if SOURCE in ("antiprism", "prism") else "coord"

d = json.load(open(f"{DIR}/interchange_{MODEL}_{SOURCE}.json"))
swap = np.array(d[METRIC])                                        # (nL, nH)
nL, nH = swap.shape

sw = json.load(open(f"{SWEEPDIR}/head_eig_sweep_{MODEL}_grid.json"))
dmg = np.array(sw["damage"]); eig = np.array(sw["eigenvalues"])   # (15, nL, nH), (15,)
if VAR == "parity":
    write = dmg[int(np.argmax(eig))]; wlab = "parity-mode build (damage, top-eig)"
else:
    lo = np.argsort(eig)[:2]; write = dmg[lo[0]] + dmg[lo[1]]; wlab = "coord-mode build (damage, 2 lowest-eig)"

r = float(np.corrcoef(swap.flatten(), write.flatten())[0, 1])
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for a, (mat, ti) in zip(ax[:2], [(swap, f"{SOURCE}→grid swap: Δ {METRIC}"), (write, wlab)]):
    v = max(1e-6, float(np.nanpercentile(np.abs(mat), 99)))
    im = a.imshow(mat, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
    a.set_xlabel("head"); a.set_ylabel("layer"); a.set_title(ti, fontsize=9); fig.colorbar(im, ax=a, fraction=.046)

ax[2].scatter(write.flatten(), swap.flatten(), s=10, alpha=0.5, color="#6B7280")
top = np.argsort(swap, axis=None)[::-1][:8]
for i in top:
    L, h = int(i // nH), int(i % nH)
    ax[2].scatter(write[L, h], swap[L, h], s=40, color="#C2410C")
    ax[2].annotate(f"L{L}H{h}", (write[L, h], swap[L, h]), fontsize=7, color="#C2410C")
ax[2].set_xlabel(wlab); ax[2].set_ylabel(f"swap Δ {METRIC}")
ax[2].set_title(f"swap vs circuit-write  (r = {r:.2f})", fontsize=10)
ax[2].axhline(0, color="k", lw=0.5); ax[2].spines[["top", "right"]].set_visible(False)

fig.suptitle(f"Exp 4 — inject {SOURCE} head-output into GRID: which heads swap {VAR} behaviour, "
             f"and do they match the {VAR} circuit? ({MODEL})", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
for ext in ("pdf",):
    out = f"{OUTDIR}/interchange_{MODEL}_{SOURCE}.{ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
print(f"[{MODEL}/{SOURCE}] corr(swap {METRIC}, {VAR} circuit-write) = {r:.3f}")
print("top swap heads:", ", ".join(f"L{int(i//nH)}H{int(i%nH)}" for i in top))
