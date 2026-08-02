"""Cluster the emotion vectors and render diagonal-block (clustered) cosine
heatmaps. For each layer we hierarchically cluster the 27 emotion vectors,
reorder rows/cols by the dendrogram leaf order so similar emotions sit adjacent
(blocks along the diagonal), and draw cluster dividers.

Outputs (under <run>/emotion_vectors/):
  emotion_cos_clustered.pdf     per-layer clustered heatmaps (slideshow)
  emotion_dendrogram_L<rep>.png dendrogram at a representative layer
  emotion_clustered_L<rep>.png  the clustered heatmap at that layer (standalone)

Usage:
  python cluster_emotion_vectors.py results/all_full            # clean vectors
  python cluster_emotion_vectors.py results/all_full --vectors raw --rep 24 --k 6
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402


def cos_sim(X):
    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
    return Xn @ Xn.T


def layer_linkage(C):
    """Average-linkage clustering on (1 - cosine) distance. Returns (Z, order)."""
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0                      # enforce symmetry for squareform
    Z = linkage(squareform(D, checks=False), method="average")
    order = dendrogram(Z, no_plot=True)["leaves"]
    return Z, order


def plot_clustered(ax, C, names, order, k):
    Cr = C[np.ix_(order, order)]
    nm = [names[i] for i in order]
    im = ax.imshow(Cr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(nm))); ax.set_yticks(range(len(nm)))
    ax.set_xticklabels(nm, rotation=90, fontsize=6)
    ax.set_yticklabels(nm, fontsize=6)
    # draw cluster block dividers along the diagonal
    Z, _ = layer_linkage(C)
    flat = fcluster(Z, t=k, criterion="maxclust")[order]
    bounds = np.where(np.diff(flat) != 0)[0] + 0.5
    for b in bounds:
        ax.axhline(b, color="k", lw=1.0); ax.axvline(b, color="k", lw=1.0)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--vectors", default="clean", choices=["clean", "raw"])
    ap.add_argument("--rep", type=int, default=24, help="representative layer")
    ap.add_argument("--k", type=int, default=6, help="# flat clusters for dividers")
    args = ap.parse_args()

    ev = Path(args.run_dir) / "emotion_vectors"
    names = json.loads((ev / "meta.json").read_text())["emotion_names"]
    vecs = np.load(ev / f"emotion_vectors_{args.vectors}.npy")   # (E, L, H)
    E, L, H = vecs.shape
    sfx = "" if args.vectors == "clean" else "_raw"

    # per-layer clustered heatmap slideshow
    with PdfPages(ev / f"emotion_cos_clustered{sfx}.pdf") as pdf:
        for l in range(L):
            C = cos_sim(vecs[:, l, :])
            _, order = layer_linkage(C)
            fig, ax = plt.subplots(figsize=(8, 7), dpi=110)
            im = plot_clustered(ax, C, names, order, args.k)
            ax.set_title(f"clustered emotion cosine sim — layer {l}/{L-1} "
                         f"({args.vectors})")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # representative layer: standalone clustered heatmap + dendrogram
    rep = args.rep
    C = cos_sim(vecs[:, rep, :])
    Z, order = layer_linkage(C)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=130)
    im = plot_clustered(ax, C, names, order, args.k)
    ax.set_title(f"clustered emotion cosine sim — layer {rep} ({args.vectors})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(ev / f"emotion_clustered{sfx}_L{rep}.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=130)
    dendrogram(Z, labels=names, ax=ax, color_threshold=None)
    ax.set_title(f"emotion-vector dendrogram — layer {rep} ({args.vectors}, "
                 f"avg-linkage, 1-cos)")
    ax.tick_params(axis="x", labelsize=6, rotation=90)
    fig.tight_layout()
    fig.savefig(ev / f"emotion_dendrogram{sfx}_L{rep}.png"); plt.close(fig)

    print(f"[done] clustered heatmaps + dendrogram in {ev}")
    # print the rep-layer clusters for quick reading
    flat = fcluster(Z, t=args.k, criterion="maxclust")
    print(f"[clusters] layer {rep}, k={args.k}:")
    for c in range(1, args.k + 1):
        members = [names[i] for i in range(E) if flat[i] == c]
        print(f"  cluster {c}: {', '.join(members)}")


if __name__ == "__main__":
    main()
