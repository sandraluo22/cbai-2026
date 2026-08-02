"""Plot the geometry battery FIT stage. Left: for each concept space, the fit to its hypothesised geometry
(RSA for arc/tree/helix/product; equidistance for simplex), coloured by family -- does the model represent
the space with the expected shape? Right: the two informative decompositions -- PRODUCTS (does the joint
rep factorize, or does one axis dominate?) and HELICES (is it a helix, or collapses to pure cycle / pure
line?). Reads geometry_fit_<model>.json.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

DIR = os.environ.get("DIR", "runs/axes/6_geometry"); MODEL = os.environ.get("MODEL", "Llama")
FCOL = {"arc": "#1D4ED8", "simplex": "#059669", "tree": "#7C3AED", "product": "#C2410C", "helix": "#EA580C"}
d = json.load(open(f"{DIR}/geometry_fit_{MODEL}.json"))["spaces"]

order = [n for fam in ["arc", "simplex", "tree", "product", "helix"] for n in d if d[n]["family"] == fam]
fig, ax = plt.subplots(1, 2, figsize=(15, 5.2))

# ---- left: intended-geometry fit per space ----
scores, cols, labs = [], [], []
for n in order:
    r = d[n]
    s = r["equidistance"] if r["family"] == "simplex" else (r["intended_rsa"] or 0)
    scores.append(s); cols.append(FCOL[r["family"]]); labs.append(f"{n}  [{r['best_geom']}]")
y = np.arange(len(order))
ax[0].barh(y, scores, color=cols)
for yi, n in zip(y, order):
    mark = "✓" if d[n]["intended_wins"] else "✗"
    ax[0].text(scores[yi] + 0.01, yi, mark, va="center", fontsize=9,
               color="#059669" if d[n]["intended_wins"] else "#DC2626")
ax[0].set_yticks(y); ax[0].set_yticklabels(labs, fontsize=8); ax[0].invert_yaxis()
ax[0].set_xlabel("fit to hypothesised geometry (RSA, or equidistance for simplex)")
ax[0].set_title("does each concept space match its hypothesised shape?", fontsize=10)
ax[0].set_xlim(0, 1); ax[0].axvline(0, color="k", lw=.5); ax[0].spines[["top", "right"]].set_visible(False)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FCOL.values()]
ax[0].legend(handles, list(FCOL), fontsize=8, frameon=False, loc="lower right")

# ---- right: products (A vs B vs product) and helices (helix vs cycle vs line) ----
prods = [n for n in order if d[n]["family"] == "product"]; helis = [n for n in order if d[n]["family"] == "helix"]
rows = prods + helis; yy = np.arange(len(rows)); w = 0.26
for i, n in enumerate(rows):
    rsa = d[n]["rsa"]
    if d[n]["family"] == "product":
        trip = [("axis A", rsa["factorA_only"], "#93C5FD"), ("axis B", rsa["factorB_only"], "#1D4ED8"), ("product", rsa["product"], "#C2410C")]
    else:
        trip = [("cycle", rsa["cycle"], "#93C5FD"), ("line", rsa["line_prog"], "#1D4ED8"), ("helix", rsa["helix"], "#EA580C")]
    for j, (lab, val, c) in enumerate(trip):
        ax[1].barh(i + (j - 1) * w, val, w, color=c, label=lab if i in (0, len(prods)) else None)
ax[1].set_yticks(yy); ax[1].set_yticklabels(rows, fontsize=8); ax[1].invert_yaxis()
ax[1].axhline(len(prods) - 0.5, color="k", lw=.6, ls=":")
ax[1].text(0.02, len(prods) / 2 - 0.5, "PRODUCTS\n(factorize?)", fontsize=8, va="center")
ax[1].text(0.02, len(prods) + len(helis) / 2 - 0.5, "HELICES\n(cycle+line?)", fontsize=8, va="center")
ax[1].set_xlabel("RSA to candidate structure"); ax[1].axvline(0, color="k", lw=.5)
ax[1].set_title("products: one axis dominates (entangled);  helices: cycle vs line balance", fontsize=10)
ax[1].legend(fontsize=7, frameon=False, loc="lower right"); ax[1].spines[["top", "right"]].set_visible(False)

fig.suptitle(f"Geometry battery — FIT stage ({MODEL}): which shape does the model use for each concept space?", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
for ext in ("pdf",):
    out = f"{DIR}/geometry_fit_{MODEL}.{ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
