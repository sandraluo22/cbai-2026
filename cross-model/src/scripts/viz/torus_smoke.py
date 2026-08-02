"""Smoke test: my interpretation of a TORUS random-walk graph before running the model.
A torus = the Cartesian product of two cycles, C_w (minor circle = 'cylinder width' / mini-cycle) and
C_L (major circle = 'cylinder length'). Construction: square grid -> wrap the WIDTH (roll into a cylinder)
-> join the two ends of the cylinder (wrap the LENGTH) -> torus. Both axes are periodic. Node (i,j):
i = 0..L-1 around the major circle, j = 0..w-1 around the minor circle.

Row 1: the flattened w x L lattice drawn as a periodic tiling (faded copies show that both edges wrap).
Row 2: the actual 3D torus embedding, nodes coloured by minor-circle position j.
Row 3: fixed width w=5, varying length L (how big the torus is).

Env: OUTDIR
Out: <OUTDIR>/torus_smoke.pdf   (pure geometry, no model)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

OUTDIR = os.environ.get("OUTDIR", "runs/axes/7_torus")
os.makedirs(OUTDIR, exist_ok=True)
HUE = plt.cm.twilight


def torus_edges(L, w):
    E = []
    for i in range(L):
        for j in range(w):
            E.append(((i, j), ((i + 1) % L, j)))       # along length (major)
            E.append(((i, j), (i, (j + 1) % w)))       # around width (minor)
    return E


def xyz(i, j, L, w, R=3.0, r=1.1):
    th = 2 * np.pi * i / L; ph = 2 * np.pi * j / w
    return ((R + r * np.cos(ph)) * np.cos(th), (R + r * np.cos(ph)) * np.sin(th), r * np.sin(ph))


def flat_periodic(ax, L, w):
    """flattened lattice with faded tiled copies (both axes wrap)."""
    for di, alpha in [(0, 1.0), (-1, 0.18), (1, 0.18)]:
        for dj, al2 in [(0, alpha), (-1, alpha * 0.5), (1, alpha * 0.5)]:
            a = min(alpha, al2) if (di or dj) else 1.0
            ox, oy = di * (L + 0.6), dj * (w + 0.6)
            for i in range(L):
                for j in range(w):
                    if i + 1 < L: ax.plot([i + ox, i + 1 + ox], [j + oy, j + oy], "-", color="#374151", lw=1, alpha=a, zorder=1)
                    if j + 1 < w: ax.plot([i + ox, i + ox], [j + oy, j + 1 + oy], "-", color="#374151", lw=1, alpha=a, zorder=1)
            ax.scatter([i + ox for i in range(L) for j in range(w)], [j + oy for i in range(L) for j in range(w)],
                       c=[HUE(j / w) for i in range(L) for j in range(w)], s=45 if a == 1 else 18, alpha=a, zorder=2, edgecolor="k", lw=0.3)
    # wrap arrows on the central tile
    for j in range(w):
        ax.annotate("", xy=(-0.6, j), xytext=(L - 1, j), arrowprops=dict(arrowstyle="->", color="#C2410C", lw=1, ls="--", alpha=0.7))
    for i in range(L):
        ax.annotate("", xy=(i, -0.6), xytext=(i, w - 1), arrowprops=dict(arrowstyle="->", color="#1D4ED8", lw=1, ls="--", alpha=0.7))
    ax.set_xlim(-1.6, L + 0.6); ax.set_ylim(-1.6, w + 0.6); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"flattened  w={w}, L={L}\n(orange=length wrap, blue=width wrap)", fontsize=8)


def torus3d(ax, L, w):
    for (a, b) in torus_edges(L, w):
        p, q = xyz(*a, L, w), xyz(*b, L, w)
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], "-", color="#9CA3AF", lw=0.8, alpha=0.7)
    P = np.array([xyz(i, j, L, w) for i in range(L) for j in range(w)])
    C = [HUE(j / w) for i in range(L) for j in range(w)]
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=C, s=30, edgecolor="k", lw=0.3, depthshade=True)
    ax.set_box_aspect((1, 1, 0.45)); ax.set_axis_off(); ax.view_init(elev=45, azim=30)
    ax.set_title(f"torus  C_{w} □ C_{L}", fontsize=8)


fig = plt.figure(figsize=(13, 11))
widths = [4, 5, 6]; Lfix = 6
for c, w in enumerate(widths):
    flat_periodic(fig.add_subplot(3, 3, c + 1), Lfix, w)
for c, w in enumerate(widths):
    torus3d(fig.add_subplot(3, 3, 3 + c + 1, projection="3d"), Lfix, w)
for c, Lv in enumerate([4, 7, 10]):
    torus3d(fig.add_subplot(3, 3, 6 + c + 1, projection="3d"), Lv, 5)

fig.suptitle("Torus random-walk graph — smoke test of interpretation\n"
             "torus = C_w (minor circle = cylinder WIDTH / mini-cycle) × C_L (major circle = cylinder LENGTH); both axes wrap.\n"
             "Sweep 1: width w = 4, 5, 6 (rows 1–2, L=6).   Sweep 2: length L = 4, 7, 10 (row 3, w=5).", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = f"{OUTDIR}/torus_smoke.pdf"
fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
