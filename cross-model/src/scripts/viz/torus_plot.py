"""Plot the torus random-walk results. Top row: the SWEEP -- RSA to torus vs cylinder vs grid (torus>cyl>grid
everywhere ⇒ the model closes BOTH loops), the wrap-index (torus−grid), and the 4D-torus fit, over all (w,L).
Bottom row: for a representative torus, the PCA embedding coloured by the two circle coordinates (does it form
a torus / two circles?) and the eigenmode (2D-Fourier) power spectrum. Reads torus_walk_<model>.json.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

DIR = os.environ.get("DIR", "runs/axes/7_torus"); MODEL = os.environ.get("MODEL", "Llama")
REPR = os.environ.get("REPR", "w6_L6")
d = json.load(open(f"{DIR}/torus_walk_{MODEL}.json"))["combos"]
keys = list(d.keys())

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3)

# ---- (1) sweep RSA torus/cyl/grid ----
ax = fig.add_subplot(gs[0, 0]); x = np.arange(len(keys)); wd = 0.26
for j, (m, c, lab) in enumerate([("rsa_grid", "#9CA3AF", "grid (no wrap)"), ("rsa_cyl", "#1D4ED8", "cylinder (1 wrap)"), ("rsa_torus", "#C2410C", "torus (both wrap)")]):
    ax.bar(x + (j - 1) * wd, [d[k]["best"][m] for k in keys], wd, color=c, label=lab)
ax.set_xticks(x); ax.set_xticklabels(keys, rotation=90, fontsize=7); ax.set_ylabel("RSA (best layer)")
ax.set_title("does the model wrap? torus > cylinder > grid everywhere", fontsize=9)
ax.legend(fontsize=7.5, frameon=False); ax.spines[["top", "right"]].set_visible(False)

# ---- (2) wrap index + best4d vs n ----
ax = fig.add_subplot(gs[0, 1]); ns = [d[k]["n"] for k in keys]
ax.scatter(ns, [d[k]["wrap_index"] for k in keys], c="#C2410C", s=50, label="wrap index (torus−grid)")
ax.scatter(ns, [d[k]["best"]["best4d_torus"] for k in keys], c="#059669", s=50, marker="s", label="best-4D torus fit")
for k in keys: ax.annotate(k, (d[k]["n"], d[k]["wrap_index"]), fontsize=6, alpha=.7)
ax.axhline(0, color="k", lw=.5); ax.set_xlabel("n nodes (w×L)"); ax.set_ylabel("score")
ax.set_title("wrap-index stays positive; 4D-torus fit ~0.85 across scale", fontsize=9)
ax.legend(fontsize=8, frameon=False); ax.spines[["top", "right"]].set_visible(False)

# ---- (3) eigenmode power spectrum (representative) ----
ax = fig.add_subplot(gs[0, 2]); c = d[REPR]
p = np.array(c["eig_power_by_mode"]); lam = np.array(c["eig_lambdas"]); ks = np.arange(1, len(p))
ax.bar(ks, p[1:], color="#7C3AED")
ax.set_xlabel("eigenmode (by λ, low→high freq)"); ax.set_ylabel("power fraction")
ax.set_title(f"{REPR}: torus eigenmode (2D-Fourier) power\nlow modes = the two circles' fundamentals", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

# ---- (4,5) PCA embedding coloured by length / width ----
emb = np.array(c["embedding"]); coords = np.array(c["coords"]); w, L = c["w"], c["L"]
def nid(i, j): return i * w + j
for col, (which, cmap, ti) in zip([gs[1, 0], gs[1, 1]],
        [(coords[:, 0], plt.cm.twilight, f"length i (major circle, L={L})"),
         (coords[:, 1], plt.cm.hsv, f"width j (minor circle, w={w})")]):
    ax = fig.add_subplot(col)
    # draw torus lattice edges in PC space
    for i in range(L):
        for j in range(w):
            a = nid(i, j)
            for (di, dj) in ((1, 0), (0, 1)):
                b = nid((i + di) % L, (j + dj) % w)
                ax.plot([emb[a, 0], emb[b, 0]], [emb[a, 1], emb[b, 1]], "-", color="#D1D5DB", lw=0.5, zorder=1)
    sctr = ax.scatter(emb[:, 0], emb[:, 1], c=which, cmap=cmap, s=45, zorder=2, edgecolor="k", lw=0.3)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title(f"{REPR} PCA — coloured by {ti}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(sctr, ax=ax, fraction=.046)

# ---- (6) PC2 vs PC3 to expose the second circle ----
ax = fig.add_subplot(gs[1, 2])
sctr = ax.scatter(emb[:, 1], emb[:, 2], c=coords[:, 1], cmap=plt.cm.hsv, s=45, edgecolor="k", lw=0.3)
ax.set_xlabel("PC2"); ax.set_ylabel("PC3"); ax.set_title(f"{REPR} PC2–PC3 — coloured by width j\n(the minor circle in lower-variance PCs)", fontsize=9)
ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(sctr, ax=ax, fraction=.046)

fig.suptitle(f"Random walk on a TORUS ({MODEL}, ctx~{json.load(open(f'{DIR}/torus_walk_{MODEL}.json'))['wlen']}): "
             f"the model represents both cycles (wraps both axes); best-4D torus fit ≈ 0.85 across w∈{{4,5,6}}×L∈{{6,10,16}}", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
for ext in ("pdf",):
    out = f"{DIR}/torus_walk_{MODEL}.{ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
