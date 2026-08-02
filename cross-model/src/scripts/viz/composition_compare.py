"""Honest comparison of composition types. For each 2-axis space we show the UNSUPERVISED grid RSA
(rep distances vs 2D-coordinate distances -- no fitting, cannot overfit) and each axis's own RSA. The
grid signature is BALANCE: both axes contribute (graded compositions) vs one axis dominating (binary
products = entangled). We deliberately drop the supervised best-2D / full-dim R^2 numbers -- with n=12
points in ~4096 dims those overfit (best-2D hit 0.97 even for emotion, whose grid RSA is 0.12).
Reads color_grid_<name>_<model>.json (graded) + geometry_fit_<model>.json (binary products).
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

DIR = os.environ.get("DIR", "runs/axes/6_geometry"); MODEL = os.environ.get("MODEL", "Llama")
graded = [("size_hue", "size × hue\n(graded)"), ("color", "lightness × hue\n(graded)"),
          ("emotion", "intensity × valence\n(graded)")]
rows, grid, a1, a2, kinds = [], [], [], [], []
for key, lab in graded:
    d = json.load(open(f"{DIR}/color_grid_{key}_{MODEL}.json"))["best"]
    rows.append(lab); grid.append(d["grid_2d"]); a1.append(d["lightness"]); a2.append(d["hue"]); kinds.append("graded")
fit = json.load(open(f"{DIR}/geometry_fit_{MODEL}.json"))["spaces"]
for key, lab in [("size_color", "size × color\n(binary)"), ("nat_prof", "nat. × profession\n(binary)")]:
    r = fit[key]["rsa"]; rows.append(lab); grid.append(r["product"]); a1.append(r["factorA_only"]); a2.append(r["factorB_only"]); kinds.append("binary")

y = np.arange(len(rows)); w = 0.26
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.barh(y + w, grid, w, color="#111827", label="grid RSA (both axes, unsupervised)")
ax.barh(y, a1, w, color="#93C5FD", label="axis 1 alone")
ax.barh(y - w, a2, w, color="#1D4ED8", label="axis 2 alone")
for yi, (p, q) in zip(y, zip(a1, a2)):                              # label by ACTUAL axis balance, not category
    lo, hi = min(p, q), max(p, q)
    tag_, col = ("grid", "#059669") if lo > 0.3 else (("entangled", "#DC2626") if hi > 0.5 else ("weak", "#9CA3AF"))
    ax.text(-0.28, yi, f"▪ {tag_}", va="center", fontsize=8, color=col)
ax.axhline(len(graded) - 0.5, color="k", ls=":", lw=.7)
ax.set_yticks(y); ax.set_yticklabels(rows, fontsize=8.5); ax.invert_yaxis()
ax.set_xlabel("RSA (unsupervised — no fitting, can't overfit)"); ax.axvline(0, color="k", lw=.5)
ax.set_xlim(-0.3, 1.0)
ax.set_title("Composition geometry (honest, unsupervised): graded axes → balanced 2D grid; "
             "binary axes → one dominates (entangled)", fontsize=9.5)
ax.legend(fontsize=8, frameon=False, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"When does composition invoke a grid? ({MODEL}) — balance of the two axes is the signature", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
for ext in ("pdf",):
    out = f"{DIR}/composition_compare_{MODEL}.{ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
