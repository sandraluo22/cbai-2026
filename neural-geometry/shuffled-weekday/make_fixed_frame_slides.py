"""Fixed-frame PDF slideshow: the ring's motion through TWO FIXED planes.

Companion to make_pca_slides.py, which projects each context bin onto its own
PCA plane and therefore hides the ambient rotation (the within-plane picture is
nearly invariant). Here every page uses the SAME two frames:

  left panel : projection onto the PRETRAINED weekday plane (fixed)
  right panel: projection onto the FINAL learned plane (fixed)

so across pages you watch the ring physically DRAIN out of the pretrained
plane (collapsing toward its centroid) while it GROWS into the learned plane.
Axes are held to a common scale across all pages to keep sizes comparable.
Last page: variance-fraction and ring-radius curves vs context length.

Usage: python make_fixed_frame_slides.py [--layer 26] [--out runs]
"""

from __future__ import annotations

import argparse
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
DAY_COLORS = plt.cm.hsv(np.linspace(0, 1, 8)[:7])


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--bin-width", type=float, default=0.25)
    return ap.parse_args()


def draw(ax, coords, title, sem_cycle, lim):
    for i in range(7):
        j = (i + 1) % 7
        ax.plot(*zip(coords[i], coords[j]), color="0.35", lw=1.6, zorder=1)
    for a, b in zip(sem_cycle, sem_cycle[1:] + [sem_cycle[0]]):
        ax.plot(*zip(coords[a], coords[b]), color="crimson", lw=0.9, ls="--",
                zorder=1, alpha=0.7)
    day_sem_idx = [(3 * i) % 7 for i in range(7)]
    for i in range(7):
        ax.scatter(*coords[i], s=140, color=DAY_COLORS[day_sem_idx[i]],
                   edgecolor="k", zorder=2)
        ax.annotate(DAYS_PERMUTED[i][:3], coords[i], textcoords="offset points",
                    xytext=(7, 7), fontsize=10)
    ax.set(title=title, xlim=(-lim, lim), ylim=(-lim, lim))
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    ax.axhline(0, color="0.9", lw=0.5, zorder=0)
    ax.axvline(0, color="0.9", lw=0.5, zorder=0)


def main():
    args = parse_args()
    out = Path(args.out)
    cap = cm_models.load_capture(str(out / "walk_capture.npz"))
    layer = args.layer if args.layer in cap.acts else sorted(cap.acts)[-1]
    acts, nodes, ctx = cap.acts[layer], cap.meta["node"], cap.meta["context_length"]

    pre = np.load(out / "pretrained_ring.npz")
    pre_key = f"layer_{layer}" if f"layer_{layer}" in pre.files else \
        [k for k in pre.files if k.startswith("layer_")][-1]
    pre_ring = nglib.fit_ring(np.asarray(pre[pre_key], dtype=np.float64))
    sem_cycle = semantic_day_cycle()

    bin_means = {}
    for c in BINS:
        lo, hi = c * (1 - args.bin_width), c * (1 + args.bin_width)
        m = nglib.node_means(acts, nodes, (ctx >= lo) & (ctx <= hi), 7)
        if not np.isnan(m).any():
            bin_means[c] = m
    final_c = max(bin_means)
    fin_ring = nglib.fit_ring(bin_means[final_c])

    # per-bin projections onto the two FIXED planes (centered per bin so the
    # panels show shape, not global drift of the centroid)
    stats = []
    proj = {}
    for c, m in bin_means.items():
        mc = m - m.mean(0)
        tot = float((mc ** 2).sum())
        p_pre, p_fin = mc @ pre_ring.plane, mc @ fin_ring.plane
        proj[c] = (p_pre, p_fin)
        stats.append({
            "ctx": c,
            "var_pre": float((p_pre ** 2).sum() / tot),
            "var_fin": float((p_fin ** 2).sum() / tot),
            "radius": float(nglib.fit_ring(m).radii.mean()),
        })
    lim = 1.05 * max(np.abs(np.concatenate([np.concatenate(v) for v in proj.values()])).max(),
                     1e-9)

    pdf_path = out / f"fixed_frame_slides_layer{layer}.pdf"
    with PdfPages(pdf_path) as pdf:
        for c in sorted(proj):
            s = next(x for x in stats if x["ctx"] == c)
            p_pre, p_fin = proj[c]
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            draw(axes[0], p_pre,
                 f"PRETRAINED plane (fixed) — {100*s['var_pre']:.1f}% of variance",
                 sem_cycle, lim)
            draw(axes[1], p_fin,
                 f"FINAL learned plane (fixed) — {100*s['var_fin']:.1f}% of variance",
                 sem_cycle, lim)
            fig.suptitle(f"context ≈ {c} words   (layer {layer}; common scale; "
                         f"solid: in-context edges, dashed: semantic)", fontsize=12)
            pdf.savefig(fig); plt.close(fig)

        # summary curves
        cs = [s["ctx"] for s in stats]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(cs, [100 * s["var_pre"] for s in stats], "o-",
                     label="in pretrained plane")
        axes[0].plot(cs, [100 * s["var_fin"] for s in stats], "s-",
                     label="in final learned plane")
        axes[0].set(xscale="log", xlabel="context length (words)",
                    ylabel="% of node-mean variance",
                    title="the ring drains from one plane into the other")
        axes[0].legend()
        axes[1].plot(cs, [s["radius"] for s in stats], "o-", color="0.3")
        axes[1].set(xscale="log", xlabel="context length (words)",
                    ylabel="mean ring radius (own plane)",
                    title="ring inflation during learning")
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

    print(f"[slides] {len(proj) + 1} pages -> {pdf_path}")


if __name__ == "__main__":
    main()
