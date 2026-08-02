"""Multipage PDF slideshow: the activation ring's PCA plane per context bin.

One page per context bin: the 7 weekday node-means projected onto that bin's
own top-2 PCA plane. For visual continuity across pages, each bin's 2D
coordinates are aligned to the FINAL bin's via orthogonal Procrustes (rotation/
reflection only), so apparent motion is real geometry change, not basis flips.

Edges drawn on each page:
  solid  = in-context ring adjacency (node i -- i+1; walk transitions)
  dashed = semantic weekday adjacency (Monday -- Tuesday -- ...)

First page: the pretrained weekday ring baseline (neutral templates, no walk).
Last page : all bins projected onto the FINAL bin's plane — each day's
            trajectory through the (fixed) learned plane over context length.

Usage: python make_pca_slides.py [--layer 26] [--out runs]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nglib
from nglib import DAYS_PERMUTED, cm_models, semantic_day_cycle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BINS = [10, 20, 30, 50, 75, 100, 150, 200, 300, 450, 600, 800]
DAY_COLORS = plt.cm.hsv(np.linspace(0, 1, 8)[:7])   # color by SEMANTIC day order


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--bin-width", type=float, default=0.25)
    return ap.parse_args()


def procrustes_align(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Rotate/reflect centered 2D coords X to best match centered ref."""
    Xc, Rc = X - X.mean(0), ref - ref.mean(0)
    U, _, Vt = np.linalg.svd(Rc.T @ Xc)
    return Xc @ (U @ Vt).T


def draw_ring_page(ax, coords, title, sem_cycle):
    """Scatter the 7 days with both edge sets."""
    for i in range(7):                                   # in-context edges
        j = (i + 1) % 7
        ax.plot(*zip(coords[i], coords[j]), color="0.35", lw=1.8, zorder=1)
    for a, b in zip(sem_cycle, sem_cycle[1:] + [sem_cycle[0]]):   # semantic edges
        ax.plot(*zip(coords[a], coords[b]), color="crimson", lw=1.0,
                ls="--", zorder=1, alpha=0.7)
    # color nodes by semantic day index so rotation is easy to track by eye
    day_sem_idx = [(3 * i) % 7 for i in range(7)]
    for i in range(7):
        ax.scatter(*coords[i], s=160, color=DAY_COLORS[day_sem_idx[i]],
                   edgecolor="k", zorder=2)
        ax.annotate(DAYS_PERMUTED[i][:3], coords[i], textcoords="offset points",
                    xytext=(8, 8), fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)


def main():
    args = parse_args()
    out = Path(args.out)
    cap = cm_models.load_capture(str(out / "walk_capture.npz"))
    layer = args.layer if args.layer in cap.acts else sorted(cap.acts)[-1]
    acts, nodes, ctx = cap.acts[layer], cap.meta["node"], cap.meta["context_length"]

    pre = np.load(out / "pretrained_ring.npz")
    pre_key = f"layer_{layer}" if f"layer_{layer}" in pre.files else \
        [k for k in pre.files if k.startswith("layer_")][-1]
    pre_pts = np.asarray(pre[pre_key], dtype=np.float64)
    sem_cycle = semantic_day_cycle()

    metrics = {}
    mpath = out / "shift_metrics.json"
    if mpath.exists():
        metrics = {r["ctx"]: r for r in json.loads(mpath.read_text())["records"]}

    # per-bin node means + own-plane projections
    bin_means, bin_proj = {}, {}
    for c in BINS:
        lo, hi = c * (1 - args.bin_width), c * (1 + args.bin_width)
        m = nglib.node_means(acts, nodes, (ctx >= lo) & (ctx <= hi), 7)
        if np.isnan(m).any():
            continue
        ring = nglib.fit_ring(m)
        bin_means[c] = m
        bin_proj[c] = (m - ring.center) @ ring.plane
    final_c = max(bin_proj)
    final_ring = nglib.fit_ring(bin_means[final_c])

    pdf_path = out / f"pca_slides_layer{layer}.pdf"
    with PdfPages(pdf_path) as pdf:
        # page 1: pretrained baseline
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        pre_ring = nglib.fit_ring(pre_pts)
        draw_ring_page(ax, (pre_pts - pre_ring.center) @ pre_ring.plane,
                       "PRETRAINED weekday ring (neutral templates, no walk)\n"
                       f"layer {layer} — solid: in-context edges, dashed: semantic edges",
                       sem_cycle)
        pdf.savefig(fig); plt.close(fig)

        # one page per bin, aligned to the final bin's frame
        ref = bin_proj[final_c]
        for c in sorted(bin_proj):
            coords = procrustes_align(bin_proj[c], ref)
            r = metrics.get(c, {})
            extra = (f"  angle→pre {r['angle_to_pretrained_deg']:.0f}°, "
                     f"angle→final {r['angle_to_final_deg']:.0f}°, "
                     f"circ {r['circularity']:.2f}"
                     if r else "")
            fig, ax = plt.subplots(figsize=(6.5, 6.5))
            draw_ring_page(ax, coords,
                           f"context ≈ {c} words (own PCA plane){extra}",
                           sem_cycle)
            pdf.savefig(fig); plt.close(fig)

        # last page: trajectories on the FIXED final plane
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        day_sem_idx = [(3 * i) % 7 for i in range(7)]
        for i in range(7):
            path = np.array([
                (bin_means[c][i] - final_ring.center) @ final_ring.plane
                for c in sorted(bin_proj)])
            ax.plot(path[:, 0], path[:, 1], "-", color=DAY_COLORS[day_sem_idx[i]],
                    lw=1.2, alpha=0.8)
            ax.scatter(*path[0], s=40, color=DAY_COLORS[day_sem_idx[i]],
                       marker="s", edgecolor="k", zorder=3)      # start: square
            ax.scatter(*path[-1], s=160, color=DAY_COLORS[day_sem_idx[i]],
                       edgecolor="k", zorder=3)                  # end: circle
            ax.annotate(DAYS_PERMUTED[i][:3], path[-1],
                        textcoords="offset points", xytext=(8, 8), fontsize=11)
        for i in range(7):
            j = (i + 1) % 7
            a = (bin_means[final_c][i] - final_ring.center) @ final_ring.plane
            b = (bin_means[final_c][j] - final_ring.center) @ final_ring.plane
            ax.plot(*zip(a, b), color="0.35", lw=1.5, zorder=1)
        ax.set_title(f"node-mean trajectories on the FINAL plane "
                     f"(ctx {min(bin_proj)}→{final_c}; square=start, circle=end)",
                     fontsize=11)
        ax.set_aspect("equal")
        pdf.savefig(fig); plt.close(fig)

    print(f"[slides] {len(bin_proj) + 2} pages -> {pdf_path}")


if __name__ == "__main__":
    main()
