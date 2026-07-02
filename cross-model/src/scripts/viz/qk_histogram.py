"""Histogram of the per-head QK induction-score distribution for each model, from
induction-head/induction.json (the 'generic' matrix). Shows the bulk of heads near 0
and the thin tail of prefix-matching (induction) heads; marks the QK thresholds used
for ablation.

Usage: PYTHONPATH=src python src/scripts/viz/qk_histogram.py [induction.json] [out.pdf]
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

JP = sys.argv[1] if len(sys.argv) > 1 else "runs/induction-head/induction.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "runs/induction-head/qk_histogram.pdf"
ORDER = ["Llama", "Gemma", "Qwen"]
COL = {"Llama": "tab:blue", "Gemma": "tab:green", "Qwen": "tab:orange"}
THRESH = [0.2, 0.5]


def main():
    d = json.load(open(JP))["models"]
    models = [m for m in ORDER if m in d] + [m for m in d if m not in ORDER]
    bins = np.linspace(-0.2, 1.0, 49)
    with PdfPages(OUT) as pdf:
        # page 1: per-model panels (log y so the induction tail is visible)
        fig, ax = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.6), squeeze=False, sharey=True)
        for j, m in enumerate(models):
            g = np.array(d[m]["generic"]).flatten(); nH = np.array(d[m]["generic"]).shape[1]
            a = ax[0, j]
            a.hist(g, bins=bins, color=COL.get(m, "gray"), alpha=.85)
            a.set_yscale("log")
            for t in THRESH:
                a.axvline(t, color="red", ls="--", lw=1)
                a.text(t, 0.7, f" >{t}: {(g > t).sum()}", color="red", fontsize=7, rotation=90, va="bottom", transform=a.get_xaxis_transform())
            a.set_title(f"{m}  ({g.size} heads; median {np.median(g):.3f}, max {g.max():.2f})", fontsize=9)
            a.set_xlabel("QK induction score")
            if j == 0:
                a.set_ylabel("# heads (log)")
        fig.suptitle("Per-head QK induction-score distribution (log y) — bulk near 0, thin prefix-matching tail\n"
                     "red dashed = ablation thresholds", fontsize=11)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # page 2: overlaid normalized (to compare shapes / tails across models)
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        for m in models:
            g = np.array(d[m]["generic"]).flatten()
            ax.hist(g, bins=bins, density=True, histtype="step", lw=1.8, color=COL.get(m, "gray"),
                    label=f"{m} (max {g.max():.2f})")
        ax.set_yscale("log"); ax.set_xlabel("QK induction score"); ax.set_ylabel("density (log)")
        for t in THRESH:
            ax.axvline(t, color=".6", ls="--", lw=.8)
        ax.legend(fontsize=8)
        ax.set_title("QK induction-score distributions, overlaid (density, log y)", fontsize=10)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
