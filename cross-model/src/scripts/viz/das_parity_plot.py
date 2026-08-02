"""Plot experiment 5 (DAS): parity-interchange margin vs aligned-subspace dimension r for a single head.
The trained r-dim subspace is compared to (a) r=0 (no patch, baseline), (b) r=full-head (the most this
head can express), and (c) an untrained RANDOM r-dim subspace. We also show the FRACTION of the full-head
parity effect captured: (margin_r - margin_0) / (margin_full - margin_0). If a small trained r captures
most of the effect while the random subspace captures ~none, parity is carried by a sparse within-head
subspace. Reads das_parity_<model>_L<layer>H<head>.json.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

DIR = os.environ.get("DIR", "runs/axes/4_circuits/das")
MODEL = os.environ.get("MODEL", "Llama"); HEAD = os.environ.get("HEAD", "L14H26")
VAR = os.environ.get("VAR", "parity")

d = json.load(open(f"{DIR}/das_{VAR}_{MODEL}_{HEAD}.json"))
res = d["results"]; hd = d["hd"]
rs = sorted(int(k) for k in res)
inner = [r for r in rs if 0 < r < hd]
# no-handle baseline = mean random-subspace margin (a random r-dim subspace is no parity handle == r=0)
m0 = float(np.mean([res[str(r)]["margin_random_subspace"] for r in inner]))
mfull = res[str(hd)]["margin"] if str(hd) in res else res[str(max(rs))]["margin"]
denom = (mfull - m0) if abs(mfull - m0) > 1e-9 else 1.0

def frac(m): return (m - m0) / denom
tr_m = [res[str(r)]["margin"] for r in inner]
rd_m = [res[str(r)]["margin_random_subspace"] for r in inner]

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
# panel 1: raw margin
ax[0].plot(inner, tr_m, "o-", color="#C2410C", lw=2, label="trained subspace")
ax[0].plot(inner, rd_m, "o--", color="#9CA3AF", lw=2, label="random subspace")
ax[0].axhline(m0, ls=":", color="k", lw=1.2, label=f"no-handle baseline (random, {m0:+.2f})")
ax[0].axhline(mfull, ls=":", color="#1D4ED8", lw=1.2, label=f"full head r={hd} ({mfull:+.2f})")
ax[0].set_xscale("log", base=2); ax[0].set_xticks(inner); ax[0].set_xticklabels(inner)
ax[0].set_xlabel("aligned subspace dim r"); ax[0].set_ylabel(f"{VAR} margin  logΣtarget − logΣother")
ax[0].set_title("interchange margin vs subspace dim", fontsize=10)
ax[0].legend(fontsize=8, frameon=False); ax[0].grid(color="#EEE", lw=.6); ax[0].spines[["top", "right"]].set_visible(False)

# panel 2: fraction of full-head effect captured
ax[1].plot(inner, [frac(m) for m in tr_m], "o-", color="#C2410C", lw=2, label="trained subspace")
ax[1].plot(inner, [frac(m) for m in rd_m], "o--", color="#9CA3AF", lw=2, label="random subspace")
ax[1].axhline(1.0, ls=":", color="#1D4ED8", lw=1.2, label="full head")
ax[1].axhline(0.0, ls=":", color="k", lw=1)
ax[1].set_xscale("log", base=2); ax[1].set_xticks(inner); ax[1].set_xticklabels(inner)
ax[1].set_xlabel("aligned subspace dim r"); ax[1].set_ylabel(f"fraction of full-head {VAR} effect")
ax[1].set_title(f"sparse subspace captures the head's {VAR}", fontsize=10)
ax[1].legend(fontsize=8, frameon=False); ax[1].grid(color="#EEE", lw=.6); ax[1].spines[["top", "right"]].set_visible(False)

fig.suptitle(f"Exp 5 — DAS inside head {HEAD} ({MODEL}): is {VAR} in a sparse subspace? "
             f"(multi-position prototype interchange)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
for ext in ("pdf",):
    out = f"{DIR}/das_{VAR}_{MODEL}_{HEAD}.{ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
print("trained frac:", {r: round(frac(m), 2) for r, m in zip(inner, tr_m)})
print("random  frac:", {r: round(frac(m), 2) for r, m in zip(inner, rd_m)})
