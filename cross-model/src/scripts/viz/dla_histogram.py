"""Histogram of per-head direct-logit-attribution (DLA) to the correct next node, per model,
from attribution/head_attribution_square_grid.json. Shows the bulk of heads near 0 and the
thin tail of 'writer' heads that actually push the next-node logit.

Usage: PYTHONPATH=src python src/scripts/viz/dla_histogram.py [head_attribution_*.json] [out.pdf]
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

JP = sys.argv[1] if len(sys.argv) > 1 else "runs/induction-head/attribution/head_attribution_square_grid.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "runs/induction-head/attribution/dla_histogram.pdf"
ORDER = ["Llama", "Gemma", "Qwen"]
COL = {"Llama": "tab:blue", "Gemma": "tab:green", "Qwen": "tab:orange"}


def main():
    d = json.load(open(JP))["models"]
    models = [m for m in ORDER if m in d] + [m for m in d if m not in ORDER]
    with PdfPages(OUT) as pdf:
        fig, ax = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.6), squeeze=False)
        for j, m in enumerate(models):
            a = np.array(d[m]["head_attr"]).flatten(); nH = np.array(d[m]["head_attr"]).shape[1]
            lim = np.percentile(np.abs(a), 99.5)
            bins = np.linspace(-lim, lim, 61)
            ax[0, j].hist(np.clip(a, -lim, lim), bins=bins, color=COL.get(m, "gray"), alpha=.85)
            ax[0, j].set_yscale("log"); ax[0, j].axvline(0, color=".6", lw=.7)
            top = np.argsort(a)[::-1][:5]
            for i in top:
                ax[0, j].axvline(a[i], color="red", lw=.5, alpha=.5)
            ax[0, j].set_title(f"{m}  ({a.size} heads; mean {a.mean():+.3f}, max {a.max():+.2f}, min {a.min():+.2f})", fontsize=8)
            ax[0, j].set_xlabel("per-head DLA to correct next node")
            if j == 0:
                ax[0, j].set_ylabel("# heads (log)")
        fig.suptitle("Per-head direct-logit-attribution to the correct next node (log y) — "
                     "bulk near 0, thin tail of 'writer' heads (red)", fontsize=11)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # overlaid normalized (compare shapes)
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        for m in models:
            a = np.array(d[m]["head_attr"]).flatten()
            lim = np.percentile(np.abs(a), 99)
            ax.hist(np.clip(a, -lim, lim), bins=np.linspace(-lim, lim, 61), density=True,
                    histtype="step", lw=1.8, color=COL.get(m, "gray"), label=m)
        ax.set_yscale("log"); ax.axvline(0, color=".7", lw=.7); ax.legend(fontsize=8)
        ax.set_xlabel("per-head DLA to correct next node"); ax.set_ylabel("density (log)")
        ax.set_title("DLA distributions, overlaid (density, log y)", fontsize=10)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
