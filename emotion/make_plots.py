"""Visualizations from saved emotion activations.

1. PCA slideshows (multipage PDF + per-layer PNG + GIF) of Q, A1 and A2
   activations at every layer, points color-coded by emotion label.
2. Cosine-similarity distributions between MATCHING A1 and A2 activations:
   one histogram per layer (slideshow) plus a mean-cos-vs-layer summary.

Usage:
  python make_plots.py results/train_200            # ekman coloring (default)
  python make_plots.py results/train_200 --fine     # 28-class coloring
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
from sklearn.decomposition import PCA  # noqa: E402

from goemotions_utils import GOEMOTIONS_LABELS, EKMAN_NAMES


def load_run(run_dir: Path):
    meta = json.loads((run_dir / "meta.json").read_text())
    N, L, H = meta["N"], meta["L"], meta["H"]
    mm = lambda name: np.memmap(run_dir / name, dtype="float16", mode="r", shape=(N, L, H))
    acts = {"Q": mm("q_acts.dat"), "A1": mm("a1_acts.dat"), "A2": mm("a2_acts.dat")}
    primary = np.load(run_dir / "labels_primary.npy")
    ekman = np.load(run_dir / "labels_ekman.npy")
    return meta, acts, primary, ekman


def pca_slideshow(run_dir: Path, name: str, mm, labels, label_names, meta,
                  keep=None, suffix=""):
    L = meta["L"]
    colors = labels.astype(int)
    if keep is not None:
        colors = colors[keep]
    cmap = plt.get_cmap("tab20" if len(label_names) > 10 else "tab10")
    present = sorted(set(colors))

    out = run_dir / "plots" / f"pca_{name}{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    pdf_path = run_dir / "plots" / f"pca_{name}{suffix}_slideshow.pdf"
    png_paths = []
    with PdfPages(pdf_path) as pdf:
        for li in range(L):
            X = np.asarray(mm[:, li, :], dtype=np.float32)
            if keep is not None:
                X = X[keep]
            X = X - X.mean(0, keepdims=True)
            XY = PCA(n_components=2, random_state=0).fit_transform(X)
            fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=130)
            for c in present:
                m = colors == c
                ax.scatter(XY[m, 0], XY[m, 1], s=10, alpha=0.6,
                           color=cmap(c % cmap.N), label=label_names[c])
            ax.set_title(f"PCA of {name} activations — layer {li}/{L - 1}")
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
            ax.legend(fontsize=6, markerscale=1.5, ncol=2,
                      loc="center left", bbox_to_anchor=(1.0, 0.5))
            fig.tight_layout()
            p = out / f"layer_{li:02d}.png"
            fig.savefig(p); pdf.savefig(fig); plt.close(fig)
            png_paths.append(p)
    _maybe_gif(png_paths, run_dir / "plots" / f"pca_{name}{suffix}.gif")
    print(f"[pca] {name}{suffix}: {pdf_path}")


def cos_sim_per_layer(a1, a2):
    """Cosine sim between matching A1/A2 per example, per layer -> (N, L)."""
    a1 = np.asarray(a1, dtype=np.float32)
    a2 = np.asarray(a2, dtype=np.float32)
    num = (a1 * a2).sum(-1)
    den = np.linalg.norm(a1, axis=-1) * np.linalg.norm(a2, axis=-1)
    return num / np.clip(den, 1e-12, None)


def cos_slideshow(run_dir: Path, acts, meta, keep=None, suffix=""):
    L = meta["L"]
    cos = cos_sim_per_layer(acts["A1"], acts["A2"])      # (N, L)
    if keep is not None:
        cos = cos[keep]
    np.save(run_dir / "plots" / f"cos_a1_a2{suffix}.npy", cos)

    out = run_dir / "plots" / f"cos_a1_a2_hist{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    lo = float(np.floor(cos.min() * 20) / 20)
    bins = np.linspace(min(lo, 0.0), 1.0, 61)
    pdf_path = run_dir / "plots" / f"cos_a1_a2_hist{suffix}_slideshow.pdf"
    png_paths = []
    with PdfPages(pdf_path) as pdf:
        for li in range(L):
            fig, ax = plt.subplots(figsize=(6.0, 4.4), dpi=130)
            ax.hist(cos[:, li], bins=bins, color="#3b6ea5", edgecolor="white")
            ax.axvline(cos[:, li].mean(), color="crimson", ls="--",
                       label=f"mean={cos[:, li].mean():.3f}")
            ax.set_title(f"cos(A1, A2) — layer {li}/{L - 1}")
            ax.set_xlabel("cosine similarity"); ax.set_ylabel("frequency")
            ax.legend(fontsize=8)
            fig.tight_layout()
            p = out / f"layer_{li:02d}.png"
            fig.savefig(p); pdf.savefig(fig); plt.close(fig)
            png_paths.append(p)
    _maybe_gif(png_paths, run_dir / "plots" / f"cos_a1_a2_hist{suffix}.gif")

    # summary: mean +/- std cos by layer
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=130)
    mean, std = cos.mean(0), cos.std(0)
    ax.plot(range(L), mean, "-o", ms=3, color="#3b6ea5")
    ax.fill_between(range(L), mean - std, mean + std, alpha=0.2, color="#3b6ea5")
    ax.set_title("Mean cos(A1, A2) by layer"); ax.set_xlabel("layer")
    ax.set_ylabel("cosine similarity"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "plots" / f"cos_a1_a2_mean_by_layer{suffix}.png")
    plt.close(fig)
    print(f"[cos]{suffix}: {pdf_path}")


def _maybe_gif(png_paths, gif_path):
    try:
        from PIL import Image
        frames = [Image.open(p) for p in png_paths]
        frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                       duration=500, loop=0)
    except Exception as e:  # PIL optional
        print(f"[gif] skipped ({e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--fine", action="store_true",
                    help="color PCA by 28 fine emotions instead of 7 Ekman groups")
    ap.add_argument("--no-neutral", action="store_true",
                    help="exclude neutral-labelled examples; outputs get a "
                         "'_noneutral' suffix so they sit beside the full plots")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    (run_dir / "plots").mkdir(exist_ok=True)

    meta, acts, primary, ekman = load_run(run_dir)
    if args.fine:
        labels, names = primary, GOEMOTIONS_LABELS
    else:
        labels, names = ekman, EKMAN_NAMES

    keep, suffix = None, ""
    if args.no_neutral:
        neutral_id = GOEMOTIONS_LABELS.index("neutral")   # primary fine-label id
        keep = primary != neutral_id
        suffix = "_noneutral"
        print(f"[filter] dropping neutral: {int((~keep).sum())} removed, "
              f"{int(keep.sum())}/{len(primary)} kept")

    for name in ("Q", "A1", "A2"):
        pca_slideshow(run_dir, name, acts[name], labels, names, meta,
                      keep=keep, suffix=suffix)
    cos_slideshow(run_dir, acts, meta, keep=keep, suffix=suffix)
    print(f"[done] plots in {run_dir / 'plots'}")


if __name__ == "__main__":
    main()
