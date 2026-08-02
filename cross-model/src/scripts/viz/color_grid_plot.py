"""Plot the lightness x hue colour-grid test. Left: the best-2D projection of the composed colour-term
representations, drawn as a lattice (lines connect same-lightness and same-hue neighbours) -- if
composition is grid-like the points form a regular grid. Right: the fit metrics -- overall grid RSA,
supervised best-2D RSA, and each axis alone (both high => genuine 2D grid; one high => entangled).
Reads color_grid_<model>.json.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

DIR = os.environ.get("DIR", "runs/axes/6_geometry"); MODEL = os.environ.get("MODEL", "Llama")
d = json.load(open(f"{DIR}/color_grid_{MODEL}.json"))
light, hue = d["light"], d["hue"]; nL, nH = len(light), len(hue)
coords = np.array(d["coords"], int); emb = np.array(d["embedding"]); items = d["items"]
# orient embedding so lightness ~ x for readability (sign/rotation is arbitrary in best-2D)
HUECOL = ["#DC2626", "#EA580C", "#EAB308", "#16A34A", "#2563EB", "#7C3AED"]

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
idx = {(int(r), int(c)): i for i, (r, c) in enumerate(coords)}
# lattice lines: same lightness (row) and same hue (col)
for li in range(nL):
    pts = [emb[idx[(li, hi)]] for hi in range(nH)]
    ax[0].plot([p[0] for p in pts], [p[1] for p in pts], "-", color="#9CA3AF", lw=1, zorder=1)
for hi in range(nH):
    pts = [emb[idx[(li, hi)]] for li in range(nL)]
    ax[0].plot([p[0] for p in pts], [p[1] for p in pts], "-", color="#D1D5DB", lw=1, ls="--", zorder=1)
for i, (li, hi) in enumerate(coords):
    ax[0].scatter(*emb[i], s=260 - 70 * li, color=HUECOL[hi % len(HUECOL)],
                  edgecolor="k", lw=0.6, zorder=3)
    ax[0].annotate(items[i], emb[i], fontsize=6.5, ha="center", va="center", zorder=4)
ax[0].set_title(f"best-2D projection (L{d['best_layer']}): solid=constant lightness, dashed=constant hue", fontsize=9)
ax[0].set_xticks([]); ax[0].set_yticks([]); ax[0].spines[:].set_visible(False)

b = d["best"]
keys = [("grid_2d", "grid RSA\n(2D coord)"), ("best2d", "best-2D RSA\n(supervised)"),
        ("lightness", "lightness\naxis alone"), ("hue", "hue\naxis alone")]
vals = [b[k] for k, _ in keys]
cols = ["#111827", "#111827", "#6B7280", "#EA580C"]
ax[1].bar(range(len(keys)), vals, color=cols)
for i, v in enumerate(vals): ax[1].text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
ax[1].set_xticks(range(len(keys))); ax[1].set_xticklabels([t for _, t in keys], fontsize=8)
ax[1].set_ylim(0, 1.05); ax[1].set_ylabel("RSA")
ax[1].set_title(f"grid fit  (both axes high ⇒ 2D grid; one high ⇒ entangled)\nlinear decode  R²: light={b['r2_light']:.2f}  hue={b['r2_hue']:.2f}", fontsize=9)
ax[1].axhline(0, color="k", lw=.5); ax[1].spines[["top", "right"]].set_visible(False)

# grid-like if the top variance is planar (best-2D) AND both axes are independently decodable (R^2);
# unequal raw axis-RSA is anisotropy (one axis lower-variance), NOT entanglement.
grid_like = b["best2d"] > 0.6 and min(b["r2_light"], b["r2_hue"]) > 0.6
aniso = abs(b["lightness"] - b["hue"]) > 0.15
verdict = ("grid-like ✓ (both axes independently decodable)" if grid_like else "entangled / one axis dominates")
if grid_like and aniso: verdict += " — anisotropic (axes unequally scaled)"
fig.suptitle(f"Does composing lightness × hue invoke a grid? ({MODEL}) — {verdict}", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
for ext in ("pdf",):
    out = f"{DIR}/color_grid_{MODEL}.{ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
