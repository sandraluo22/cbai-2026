"""Render the ground-truth interpolation family G_alpha (torus -> C16(1,3)) exactly as
used by the sweep (edge lists from runs/sweep_spec.json). Two layouts per graph:
circular (node index; circulant edges are chords) and torus grid (r, c) positions.
Edges colored by membership: in BOTH endpoints / torus-only / circ3-only / stray
(created by a degree-preserving swap, in neither endpoint).

Out: figs/galpha_family.png/.pdf + printed stats table
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = json.load(open(os.path.join(HERE, "runs", "sweep_spec.json")))
N = 16
NAMES = [("torus", 0.0), ("interp0.25", 0.25), ("interp0.5", 0.5),
         ("interp0.75", 0.75), ("circ3", 1.0)]


def eset(name):
    return {tuple(e) for e in SPEC["graphs"][name]}


def lam2(name):
    return SPEC["lambda2"][name]


def main():
    E0, E1 = eset("torus"), eset("circ3")
    circ = {i: (np.cos(2 * np.pi * i / N - np.pi / 2),
                np.sin(2 * np.pi * i / N - np.pi / 2)) for i in range(N)}
    tor = {i: (i % 4, -(i // 4)) for i in range(N)}
    CLS = [("both endpoints", "#444444", 2.0),
           ("torus-only", "#2f6f9f", 1.4),
           ("C16(1,3)-only", "#c22f4d", 1.4),
           ("stray (neither)", "#999999", 1.2)]

    fig, axes = plt.subplots(2, 5, figsize=(19, 8))
    print(f"{'graph':12s} {'|E∩torus|':>9s} {'|E∩circ3|':>9s} {'stray':>5s} "
          f"{'edit->torus':>11s} {'edit->circ3':>11s} {'lambda2':>7s}")
    for j, (nm, al) in enumerate(NAMES):
        E = eset(nm)
        n0, n1 = len(E & E0), len(E & E1)
        stray = len(E - E0 - E1)
        print(f"{nm:12s} {n0:9d} {n1:9d} {stray:5d} {len(E ^ E0)//1:11d} "
              f"{len(E ^ E1)//1:11d} {lam2(nm):7.3f}")
        for row, pos, lab in ((0, circ, "circular layout (node index)"),
                              (1, tor, "torus layout (r, c)")):
            ax = axes[row, j]
            for a, b in sorted(E):
                if (a, b) in E0 and (a, b) in E1:
                    col, lw = CLS[0][1], CLS[0][2]
                elif (a, b) in E0:
                    col, lw = CLS[1][1], CLS[1][2]
                elif (a, b) in E1:
                    col, lw = CLS[2][1], CLS[2][2]
                else:
                    col, lw = CLS[3][1], CLS[3][2]
                x = [pos[a][0], pos[b][0]]
                y = [pos[a][1], pos[b][1]]
                if row == 1 and (abs(x[0] - x[1]) > 2 or abs(y[0] - y[1]) > 2):
                    ax.plot(x, y, color=col, lw=lw * 0.7, ls=":", zorder=1)  # torus wrap
                else:
                    ax.plot(x, y, color=col, lw=lw, zorder=1)
            xs = [pos[i][0] for i in range(N)]
            ys = [pos[i][1] for i in range(N)]
            ax.scatter(xs, ys, s=180, c=[plt.cm.hsv(i / N) for i in range(N)],
                       edgecolors="k", zorder=2)
            for i in range(N):
                ax.annotate(str(i), pos[i], ha="center", va="center", fontsize=7,
                            zorder=3)
            if row == 0:
                ax.set_title(f"$\\alpha$={al}  ({nm})\n"
                             f"|E$\\cap$torus|={n0}, |E$\\cap$C16(1,3)|={n1}, "
                             f"stray={stray}, $\\lambda_2$={lam2(nm):.2f}", fontsize=9)
            ax.set_xlabel(lab if row == 1 else "", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect("equal")
    handles = [plt.Line2D([], [], color=c, lw=w, label=l) for l, c, w in CLS]
    handles.append(plt.Line2D([], [], color="0.5", ls=":", label="torus-layout wrap edge"))
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9, frameon=False)
    fig.suptitle("Ground-truth interpolation family $G_\\alpha$: degree-preserving swaps "
                 "from torus C4$\\times$C4 ($\\alpha$=0) to circulant C16($\\pm$1,$\\pm$3) "
                 "($\\alpha$=1); all 16 nodes, 4-regular, 32 edges", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "figs", f"galpha_family.{ext}"), dpi=160)
    print("-> figs/galpha_family.png")


if __name__ == "__main__":
    main()
