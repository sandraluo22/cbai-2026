"""family_spectra first-slide barchart (node-mean variance vs Laplacian eigenmode, per family),
but with THREE grouped bars per eigenmode -- one per model (Llama / Gemma / Qwen).

Same convention as family_spectra.py: unnormalized Laplacian L = D - A (identical graph across
models), per model pick the layer that maximises the low-band (w == w[1]) power, then the bar
height for eigenmode k is that layer's variance fraction P[k] (P[0]:=0, sum_k>=1 P = 1).

Reads nodemeans_<MODEL>_<fam>.npz (markov_families). CPU-only.
Env: MFDIR MODELS OUTDIR
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MFDIR = os.environ.get("MFDIR", "runs/axes/1_decomposition/markov_families")
MODELS = os.environ.get("MODELS", "Llama,Gemma,Qwen").split(",")
OUTDIR = os.environ.get("OUTDIR", "runs/axes/1_decomposition")
FAMS = os.environ.get("FAMS", "grid,ring,tree,smallworld,sbm4,er_random,sbm2").split(",")
COLORS = {"Llama": "#1D4ED8", "Gemma": "#C2410C", "Qwen": "#059669", "Qwen32": "#7C3AED"}


def power_spectrum(fam, model):
    """Best-low-band-layer variance fraction per eigenmode, matching family_spectra.py.
    Returns (w eigenvalues, P over modes 1..n-1)."""
    z = np.load(f"{MFDIR}/nodemeans_{model}_{fam}.npz", allow_pickle=True)
    A = np.array(z["adjacency"], float)
    L = np.diag(A.sum(1)) - A
    w, V = np.linalg.eigh(L)
    nL = sum(1 for k in z.files if k.startswith("layer_"))
    best, P = -1.0, None
    for l in range(nL):
        H = z[f"layer_{l}"].astype(float); Hc = H - H.mean(0)
        c = V.T @ Hc; p = (c ** 2).sum(1); p[0] = 0; p /= p.sum() + 1e-12
        lb = p[np.round(w, 3) == np.round(w[1], 3)].sum()   # low-band power
        if lb > best:
            best, P = lb, p
    return w, P


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    ncol = 4; nrow = int(np.ceil(len(FAMS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow))
    axflat = np.array(axes).flat
    nm = len(MODELS)
    for ax, fam in zip(axflat, FAMS):
        specs = {m: power_spectrum(fam, m) for m in MODELS}
        w = specs[MODELS[0]][0]; k = np.arange(1, len(w))       # modes 1..n-1 (low→high freq)
        bw = 0.8 / nm
        for i, m in enumerate(MODELS):
            P = specs[m][1]
            off = (i - (nm - 1) / 2) * bw
            ax.bar(k + off, P[1:], bw, color=COLORS.get(m, None), label=m, edgecolor="none")
        ax.set_title(fam, fontsize=9)
        ax.set_xlabel("eigenmode (low→high freq)", fontsize=8)
        ax.set_ylabel("variance frac", fontsize=8)
        ax.set_ylim(0, 0.42)
        ax.set_xticks(k)                                   # label EVERY eigenmode
        ax.tick_params(labelsize=6)
    for ax in axflat[len(FAMS):]:
        ax.axis("off")
    # single shared legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m]) for m in MODELS]
    fig.legend(handles, MODELS, loc="lower right", bbox_to_anchor=(0.99, 0.02),
               frameon=False, fontsize=11, title="model")
    fig.suptitle("Node-mean variance vs Laplacian eigenmode, per family — " + " / ".join(MODELS),
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUTDIR}/family_spectra_3models.{ext}", dpi=150, bbox_inches="tight")
    print(f"wrote {OUTDIR}/family_spectra_3models.pdf/.png")


if __name__ == "__main__":
    main()
