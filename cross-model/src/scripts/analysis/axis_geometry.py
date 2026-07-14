"""Geometry of the divider axes: are the x / y / parity READOUT directions near-orthogonal in the
residual stream, and what does the representation look like in the 3-D space they span?

The node-cuts (x, y, parity) are orthogonal by construction IN NODE SPACE. But the residual-space
readout direction that carries each cut, v_c = normalise(Hc^T u_c), lives in R^d and need not be
orthogonal -- that depends on the residual covariance. We report the pairwise |cos| among
{v_x, v_y, v_parity, v_random} at the layer L* that maximises the three cuts' combined power, and
across layers. Then we render the 16 node-means projected onto [v_x, v_y, v_parity] as a 3-D scatter
(coloured by parity): if parity is a real separate axis, the two sublattices split along z.

Reads a node-mean npz (capture_nodemeans.py). CPU-only.
Env: ACTS TAG GRAPH OUTDIR
Out: <OUTDIR>/axis_geometry_<TAG>_<graph>.json + .pdf
"""
from __future__ import annotations
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib.backends.backend_pdf import PdfPages

ACTS = os.environ.get("ACTS", "runs/induction-head/2_probes/divider_basis/nodemeans_Llama_square_grid.npz")
TAG = os.environ.get("TAG", "Llama"); GRAPH = os.environ.get("GRAPH", "square_grid")
OUTDIR = os.environ.get("OUTDIR", "runs/axes/2_geometry/axis_geometry")


def two_colour(A):
    n = A.shape[0]; col = np.zeros(n)
    for s in range(n):
        if col[s] != 0: continue
        col[s] = 1; st = [s]
        while st:
            u = st.pop()
            for v in np.where(A[u] > 0)[0]:
                if col[v] == 0: col[v] = -col[u]; st.append(v)
                elif col[v] == col[u]: return None
    return col.astype(float)


def unit(v):
    v = v - v.mean(); return v / (np.linalg.norm(v) + 1e-9)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    npz = np.load(ACTS, allow_pickle=True)
    A = np.array(npz["adjacency"], float); coords = np.array(npz["coords"], float); n = A.shape[0]
    nL = sum(1 for k in npz.files if k.startswith("layer_"))
    cuts = {"x": unit(coords[:, 0]), "y": unit(coords[:, 1])}
    par = two_colour(A)
    if par is not None: cuts["parity"] = unit(par)
    rng = np.random.default_rng(0); cuts["random"] = unit(rng.standard_normal(n))

    # per-layer captured variance of the named cuts + readout directions
    keys = list(cuts); powers = {k: np.zeros(nL) for k in keys}; dirs_by_L = {}
    for L in range(nL):
        H = npz[f"layer_{L}"].astype(np.float64); Hc = H - H.mean(0)
        tot = (Hc ** 2).sum() + 1e-12; dl = {}
        for k, u in cuts.items():
            v = Hc.T @ u; powers[k][L] = (v ** 2).sum() / tot; dl[k] = v / (np.linalg.norm(v) + 1e-9)
        dirs_by_L[L] = dl
    core = [k for k in keys if k != "random"]
    Lstar = int(np.argmax(sum(powers[k] for k in core)))

    # cos among readout directions at L*
    dl = dirs_by_L[Lstar]
    cos = {a: {b: float(abs(dl[a] @ dl[b])) for b in keys} for a in keys}
    # cos across layers (core pairs)
    pairs = [(core[i], core[j]) for i in range(len(core)) for j in range(i + 1, len(core))]
    cos_by_L = {f"{a}-{b}": [float(abs(dirs_by_L[L][a] @ dirs_by_L[L][b])) for L in range(nL)] for a, b in pairs}

    # 3-D projection of node-means at L* onto [x, y, parity] readouts
    H = npz[f"layer_{Lstar}"].astype(np.float64); Hc = H - H.mean(0)
    zk = "parity" if "parity" in cuts else "random"
    P = np.stack([Hc @ dl["x"], Hc @ dl["y"], Hc @ dl[zk]], 1)

    out = {"tag": TAG, "graph": GRAPH, "n": n, "nL": nL, "Lstar": Lstar,
           "power_at_Lstar": {k: float(powers[k][Lstar]) for k in keys},
           "cos_at_Lstar": cos, "cos_by_layer": cos_by_L, "zaxis": zk}
    json.dump(out, open(f"{OUTDIR}/axis_geometry_{TAG}_{GRAPH}.json", "w"), indent=2)
    make_fig(out, P, par, coords, cos_by_L, keys, cos, f"{OUTDIR}/axis_geometry_{TAG}_{GRAPH}.pdf")
    print(f"[{TAG}/{GRAPH}] L*={Lstar}  |cos| x-y={cos['x']['y']:.2f} x-{zk}={cos['x'][zk]:.2f} "
          f"y-{zk}={cos['y'][zk]:.2f}  (x-random={cos['x']['random']:.2f})", flush=True)


def make_fig(out, P, par, coords, cos_by_L, keys, cos, path):
    zk = out["zaxis"]
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(14, 5))
        # 3-D scatter, two angles
        col = par if par is not None else np.zeros(len(P))
        for a, (az, el) in enumerate([(35, 22), (110, 15)]):
            ax = fig.add_subplot(1, 3, a + 1, projection="3d")
            sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=col, cmap="coolwarm", s=90, edgecolors="k", lw=.4)
            for i in range(len(P)):
                ax.text(P[i, 0], P[i, 1], P[i, 2], str(i), fontsize=6)
            ax.set_xlabel("x-readout"); ax.set_ylabel("y-readout"); ax.set_zlabel(f"{zk}-readout")
            ax.view_init(elev=el, azim=az); ax.set_title(f"{out['tag']} {out['graph']} L{out['Lstar']} (view {a+1})", fontsize=8)
        # cos vs layer
        ax = fig.add_subplot(1, 3, 3)
        for name, series in cos_by_L.items():
            ax.plot(range(out["nL"]), series, label=name, lw=1.2)
        ax.axhline(0, color=".7", lw=.6); ax.set_ylim(0, 1)
        ax.set_title("|cos| between readout dirs vs layer", fontsize=8)
        ax.set_xlabel("layer"); ax.set_ylabel("|cos|"); ax.legend(fontsize=7)
        fig.suptitle(f"{out['tag']} {out['graph']}: node-means in the (x,y,{zk}) readout space; "
                     f"are the axes orthogonal? |cos| at L{out['Lstar']}: "
                     f"x-y={cos['x']['y']:.2f}, x-{zk}={cos['x'][zk]:.2f}, y-{zk}={cos['y'][zk]:.2f}", fontsize=9)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
