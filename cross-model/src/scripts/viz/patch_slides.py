"""Build the cross-model patching results into a multi-page slideshow PDF
(matching the project's other slideshows). Self-contained: reads the patch JSON
(no models / GPU), so it can be (re)run locally on pulled results.

ONE SLIDE PER GRID (square_grid / ring / hex). Each slide has two panels:
  - left : cross-model RSA  vs  patched neighbour accuracy
  - right: cross-model RSA  vs  KL(clean || patched)
In both, the REAL ridge map is blue and the SHUFFLED-PAIRING control is red, and
every point is labelled with the ridge map's held-out R^2 (real points -> r2_test,
control points -> r2_test_shuf).

Usage:
  PYTHONPATH=src python src/scripts/viz/patch_slides.py [all_patch.json] [out_dir]
Defaults: runs/patch/all_patch.json  ->  runs/patch/slides/patch_slides.pdf
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages

GRAPH_ORDER = ["square_grid", "ring", "hex"]


def sp(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def graphs_in(cells):
    present = {c["graph"] for c in cells}
    return [g for g in GRAPH_ORDER if g in present] + sorted(present - set(GRAPH_ORDER))


def _scatter_panel(ax, gc, ykey_real, ykey_ctrl, ylabel):
    """RSA (x) vs a metric (y); real (blue) + control (red); label each point R^2.
    Points are connected by a faint line (sorted by RSA) to make the trend legible."""
    xr = np.array([c["rsa"] for c in gc])
    yr = np.array([c[ykey_real] for c in gc])
    yc = np.array([c[ykey_ctrl] for c in gc])
    o = np.argsort(xr)
    ax.plot(xr[o], yr[o], "-", c="tab:blue", lw=1.1, alpha=.55, zorder=2)
    ax.plot(xr[o], yc[o], "-", c="tab:red", lw=1.1, alpha=.4, zorder=2)
    ax.scatter(xr, yr, c="tab:blue", s=42, zorder=3)
    ax.scatter(xr, yc, c="tab:red", s=42, alpha=.75, zorder=3)
    for c in gc:
        ax.annotate(f"{c['r2_test']:.2f}", (c["rsa"], c[ykey_real]), fontsize=5.5,
                    xytext=(2, 3), textcoords="offset points", color="tab:blue")
        ax.annotate(f"{c['r2_test_shuf']:.2f}", (c["rsa"], c[ykey_ctrl]), fontsize=5.5,
                    xytext=(2, 3), textcoords="offset points", color="tab:red")
    ax.set_xlabel("cross-model RSA (node geometry)")
    ax.set_ylabel(ylabel)
    rho_r = sp(xr, [c[ykey_real] for c in gc])
    rho_c = sp(xr, [c[ykey_ctrl] for c in gc])
    return rho_r, rho_c


def page_graph(pdf, cells, gname, src, tgt, tag=""):
    gc = [c for c in cells if c["graph"] == gname and np.isfinite(c["rsa"])]
    if not gc:
        return
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    ra, ca = _scatter_panel(ax[0], gc, "acc_real", "acc_ctrl", "patched neighbour accuracy")
    ax[0].axhline(float(np.mean([c["acc_clean"] for c in gc])), ls="--", c="k", lw=1)
    ax[0].set_title(f"accuracy vs RSA   (Spearman: real={ra:+.2f}, ctrl={ca:+.2f})", fontsize=10)

    rk, ck = _scatter_panel(ax[1], gc, "kl_real", "kl_ctrl", "KL(clean || patched)   (lower = faithful)")
    ax[1].set_title(f"KL vs RSA   (Spearman: real={rk:+.2f}, ctrl={ck:+.2f})", fontsize=10)

    handles = [Line2D([], [], marker="o", ls="", c="tab:blue", label="real map"),
               Line2D([], [], marker="o", ls="", c="tab:red", label="shuffled control"),
               Line2D([], [], ls="--", c="k", label="clean accuracy")]
    ax[0].legend(handles=handles, fontsize=8, loc="best")
    modetxt = f"  [reconstruction: {tag}]" if tag else ""
    fig.suptitle(f"{gname}   [{src} -> {tgt}]{modetxt}   cross-model activation patching: efficacy vs RSA\n"
                 f"point labels = map held-out R$^2$  (blue = real, red = shuffled-pairing control)",
                 fontsize=12)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def build_slides(cells, out_pdf, src="Qwen", tgt="Llama", tag=""):
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    gs = graphs_in(cells)
    with PdfPages(out_pdf) as pdf:
        for g in gs:
            page_graph(pdf, cells, g, src, tgt, tag)
    print(f"wrote {out_pdf}  ({len(gs)} slides: {', '.join(gs)})", flush=True)


def main():
    jpath = sys.argv[1] if len(sys.argv) > 1 else "runs/patch/all_patch.json"
    odir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(jpath), "slides")
    cells = json.load(open(jpath))
    build_slides(cells, os.path.join(odir, "patch_slides.pdf"))


if __name__ == "__main__":
    main()
