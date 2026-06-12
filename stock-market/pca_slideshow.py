"""Per-layer PCA of the marker activations, as a slideshow PDF.

One page per layer (embedding + 32 blocks = 33). Each page has three panels of the
SAME 2-D PCA projection, colored by: (1) tau, (2) the target's private-implied
estimate p, (3) the target's social-implied estimate s. Also prints the per-layer
probe R^2 (private/social/expected-reward) from analysis.json.

    python pca_slideshow.py --out results/scoped_market
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def pca2(X):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pc = Xc @ Vt[:2].T
    ev = (S[:2] ** 2) / (S ** 2).sum()
    return pc, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/scoped_market")
    args = ap.parse_args()
    out = Path(args.out)
    rows = [json.loads(l) for l in (out / "trials.jsonl").read_text().splitlines()]

    A, tau, pv, sv = [], [], [], []
    for r in rows:
        A.append(np.load(r["acts_path"]))          # (33, hidden) marker activations
        t = r["target"]
        tau.append(r["tau"]); pv.append(r["private_implied"][t]); sv.append(r["social_implied"][t])
    A = np.stack(A); tau = np.array(tau); pv = np.array(pv); sv = np.array(sv)
    N, L, H = A.shape
    print(f"{N} trials, {L} layers, hidden={H}")

    pdf_path = out / "pca_slideshow.pdf"
    tau_levels = sorted(set(tau)); tau_colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
    with PdfPages(pdf_path) as pdf:
        for layer in range(L):
            pc, ev = pca2(A[:, layer, :])
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
            # (1) by tau
            for tv, c in zip(tau_levels, tau_colors):
                m = tau == tv
                axes[0].scatter(pc[m, 0], pc[m, 1], s=16, c=c, alpha=.75, label=f"τ={tv}")
            axes[0].legend(fontsize=8, title="crowd-gap τ"); axes[0].set_title("colored by τ")
            # (2) by private-implied p
            s1 = axes[1].scatter(pc[:, 0], pc[:, 1], s=16, c=pv, cmap="viridis")
            fig.colorbar(s1, ax=axes[1], shrink=.85, label="private-implied p")
            axes[1].set_title("colored by private estimate p")
            # (3) by social-implied s
            s2 = axes[2].scatter(pc[:, 0], pc[:, 1], s=16, c=sv, cmap="magma")
            fig.colorbar(s2, ax=axes[2], shrink=.85, label="social-implied s")
            axes[2].set_title("colored by social estimate s")
            for ax in axes:
                ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.grid(lw=.3, alpha=.4)
            fig.suptitle(f"Layer {layer}/{L-1}  —  PCA of marker activations   "
                         f"(PC1 {ev[0]*100:.1f}%  /  PC2 {ev[1]*100:.1f}%  variance)",
                         fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig); plt.close(fig)
    print(f"wrote {pdf_path} ({L} pages)")

    # ---- probe R^2 table (from analysis.json) ----
    an = json.loads((out / "analysis.json").read_text())
    pr = an["layer_probes"]
    print("\nPer-layer probe R^2 (out-of-sample):")
    print(f"{'layer':>5} | {'private p':>10} | {'social s':>10} | {'E_reward':>10}")
    print("-" * 46)
    for L_ in range(len(pr["private_implied"])):
        print(f"{L_:>5} | {pr['private_implied'][L_]:>10.3f} | "
              f"{pr['social_implied'][L_]:>10.3f} | {pr['E_reward'][L_]:>10.3f}")
    # save csv
    import csv
    with open(out / "probe_r2.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["layer", "private_implied", "social_implied", "E_reward"])
        for L_ in range(len(pr["private_implied"])):
            w.writerow([L_, pr["private_implied"][L_], pr["social_implied"][L_], pr["E_reward"][L_]])
    print(f"\nsaved {out}/probe_r2.csv")


if __name__ == "__main__":
    main()
