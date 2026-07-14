"""2x3 RSA figures for the patch-swap sweeps: top row = raw per-head RSA score, bottom row =
ΔRSA (restoration), columns = the three models. One page per direction.

  toward ORIGINAL grid : from patch_swap_metrics_<a>_<b>.json
      raw RSA_patch = restore_rsa * (RSA_O - RSA_S) + RSA_S      (RSA vs original layout)
  toward SWAPPED grid  : from patch_toswap_<a>_<b>.json
      raw RSA_patch = restore_rsa * (RSAsw_S - RSAsw_O) + RSAsw_O (RSA vs swapped layout)

Raw row = viridis (sequential), delta row = RdBu (diverging, centred 0).

Usage: PYTHONPATH=src python src/scripts/viz/patch_rsa_grid.py [dir] [a] [b]
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

DIR = sys.argv[1] if len(sys.argv) > 1 else "runs/induction-head/patch_swap"
A = sys.argv[2] if len(sys.argv) > 2 else "12"
B = sys.argv[3] if len(sys.argv) > 3 else "15"
ORDER = ["Llama", "Gemma", "Qwen"]


def page(pdf, jpath, raw_from, title):
    d = json.load(open(jpath))["models"]
    models = [m for m in ORDER if m in d] + [m for m in d if m not in ORDER]
    delta = {m: np.array(d[m]["restore_rsa"]) for m in models}
    raw = {m: raw_from(d[m]) for m in models}
    rlo = min(np.nanmin(raw[m]) for m in models); rhi = max(np.nanmax(raw[m]) for m in models)
    dv = max(0.15, max(float(np.nanpercentile(np.abs(delta[m]), 99)) for m in models))
    fig, ax = plt.subplots(2, len(models), figsize=(5 * len(models), 8.4), squeeze=False)
    for j, m in enumerate(models):
        im0 = ax[0, j].imshow(raw[m], aspect="auto", origin="lower", cmap="viridis", vmin=rlo, vmax=rhi)
        ax[0, j].set_title(f"{m}: raw RSA_patch (per head)", fontsize=9)
        ax[0, j].set_xlabel("head"); ax[0, j].set_ylabel("layer"); fig.colorbar(im0, ax=ax[0, j], fraction=.046)
        im1 = ax[1, j].imshow(delta[m], aspect="auto", origin="lower", cmap="RdBu_r", vmin=-dv, vmax=dv)
        ax[1, j].set_title(f"{m}: ΔRSA (restoration)", fontsize=9)
        ax[1, j].set_xlabel("head"); ax[1, j].set_ylabel("layer"); fig.colorbar(im1, ax=ax[1, j], fraction=.046)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def main():
    out = f"{DIR}/patch_rsa_grid_{A}_{B}.pdf"
    with PdfPages(out) as pdf:
        p1 = f"{DIR}/patch_swap_metrics_{A}_{B}.json"
        if os.path.exists(p1):
            page(pdf, p1,
                 lambda r: np.array(r["restore_rsa"]) * (r["RSA_O"] - r["RSA_S"]) + r["RSA_S"],
                 f"Patch sweep — RSA to the ORIGINAL grid (denoise, swap {A}<->{B})   "
                 "top: raw RSA_patch vs original   bottom: ΔRSA restoration")
        p2 = f"{DIR}/patch_toswap_{A}_{B}.json"
        if os.path.exists(p2):
            page(pdf, p2,
                 lambda r: np.array(r["restore_rsa"]) * (r["RSAsw_S"] - r["RSAsw_O"]) + r["RSAsw_O"],
                 f"Patch sweep — RSA to the SWAPPED grid (impose-swap, swap {A}<->{B})   "
                 "top: raw RSA_patch vs swapped   bottom: ΔRSA restoration")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
