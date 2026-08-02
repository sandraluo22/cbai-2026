"""Offline geometry analysis for the dueling-context run (NO RSA).

Inputs: out/nodemeans_dueling.npz, out/gen_log.json (from run_experiment.py).

Per (context in {grid, ring}) x (window in {base, joint_early, joint_mid, joint_late}) x layer:
  Hc = centered per-node mean residuals [16, d]
  1. PCA (SVD): top-PC projections + explained variance -- the paper's visualization.
  2. Coordinate regression: R^2 of predicting Hc from 2D grid coords (x, y) vs from 2D ring
     coords (cos, sin). Both feature sets are 2D + intercept, so R^2_grid vs R^2_ring is a
     fair "which geometry explains the representation" comparison per layer.
  3. Laplacian eigenmode spectra (cross-model gma/family-spectra machinery): energy fraction
     of Hc on each normalized-Laplacian eigenmode of the GRID graph and of the RING graph;
     low-frequency energy = modes 1..2 (the fundamental pair in each basis).

Also plots the behavioral coupling from gen_log.json: probability mass each generator puts on
ring-neighbours vs grid-neighbours of the previous fed node, over joint-generation time.

Out: out/summary.json, out/r2_vs_layer.png, out/pca_grids.png, out/eig_spectra.png,
     out/behavior.png (+ .pdf twins)
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("RUN_OUT", os.path.join(HERE, "runs", "out"))
CTXS = ("grid", "ring")
WINS = ("base", "joint_early", "joint_mid", "joint_late")
WINLABEL = {"base": "pre-interaction (ctx 700-1000)", "joint_early": "joint steps 0-100",
            "joint_mid": "joint steps 100-300", "joint_late": "joint steps 300-600"}


def norm_laplacian_modes(A):
    A = A.astype(float)
    d = A.sum(1)
    di = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    L = np.eye(len(A)) - di[:, None] * A * di[None, :]
    w, U = np.linalg.eigh(L)
    return w, U


def r2_from_coords(Hc, F):
    """R^2 of least-squares predicting centered node-means Hc [n,d] from features F [n,k]
    (columns z-scored, intercept via centering)."""
    Fc = F - F.mean(0)
    Fc = Fc / np.maximum(Fc.std(0), 1e-12)
    B, *_ = np.linalg.lstsq(Fc, Hc, rcond=None)
    resid = Hc - Fc @ B
    return float(1.0 - (resid ** 2).sum() / np.maximum((Hc ** 2).sum(), 1e-12))


def main():
    z = np.load(os.path.join(OUT, "nodemeans_dueling.npz"), allow_pickle=False)
    nL = int(z["n_layers"][0])
    words = [str(w) for w in z["words"]]
    A_grid, A_ring = z["adjacency_grid"], z["adjacency_ring"]
    cg, cr = z["coords_grid"], z["coords_ring"]
    n = len(words)
    wg, Ug = norm_laplacian_modes(A_grid)
    wr, Ur = norm_laplacian_modes(A_ring)

    def H(c, win, L):
        Hm = z[f"{c}_{win}_layer_{L}"].astype(np.float64)
        return Hm - Hm.mean(0)

    res = {c: {win: {"r2_grid": [], "r2_ring": [], "lf_grid": [], "lf_ring": [],
                     "evr2": [], "spec_grid": [], "spec_ring": []}
               for win in WINS} for c in CTXS}
    for c in CTXS:
        for win in WINS:
            for L in range(nL):
                Hc = H(c, win, L)
                r = res[c][win]
                r["r2_grid"].append(r2_from_coords(Hc, cg))
                r["r2_ring"].append(r2_from_coords(Hc, cr))
                tot = (Hc ** 2).sum()
                eg = ((Ug.T @ Hc) ** 2).sum(1) / max(tot, 1e-12)   # energy per grid mode
                er = ((Ur.T @ Hc) ** 2).sum(1) / max(tot, 1e-12)
                r["spec_grid"].append(eg.tolist())
                r["spec_ring"].append(er.tolist())
                r["lf_grid"].append(float(eg[1:3].sum()))          # fundamental pair
                r["lf_ring"].append(float(er[1:3].sum()))
                s = np.linalg.svd(Hc, compute_uv=False)
                r["evr2"].append(float((s[:2] ** 2).sum() / max((s ** 2).sum(), 1e-12)))

    # ---- figure 1: R^2(grid coords) vs R^2(ring coords) across layers ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    colors = {"base": "0.55", "joint_early": "#8ecae6", "joint_mid": "#219ebc",
              "joint_late": "#023047"}
    for ax, c in zip(axes, CTXS):
        for win in WINS:
            ax.plot(res[c][win]["r2_grid"], color=colors[win], ls="-",
                    label=f"grid-coord R$^2$, {win}" if c == "grid" else None)
            ax.plot(res[c][win]["r2_ring"], color=colors[win], ls="--",
                    label=f"ring-coord R$^2$, {win}" if c == "grid" else None)
        ax.set_title(f"{c.upper()}-primed context")
        ax.set_xlabel("layer")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"R$^2$ of node-means from 2D coords")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Which geometry explains the representation? (solid=grid coords, dashed=ring coords)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"r2_vs_layer.{ext}"), dpi=160)
    plt.close(fig)

    # ---- figure 2: PCA maps at a deep layer, base vs joint windows -------
    Lshow = min(26, nL - 1)
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.6))
    ring_order = np.arange(n)                 # ring node i sits at angle 2*pi*i/n
    hue = plt.cm.hsv(ring_order / n)
    for i, c in enumerate(CTXS):
        for j, win in enumerate(WINS):
            ax = axes[i, j]
            Hc = H(c, win, Lshow)
            U, s, Vt = np.linalg.svd(Hc, full_matrices=False)
            P = U[:, :2] * s[:2]
            for a in range(n):                # grid edges (gray) + ring edges (thin red)
                for b in range(a + 1, n):
                    if A_grid[a, b]:
                        ax.plot(P[[a, b], 0], P[[a, b], 1], color="0.75", lw=1, zorder=1)
                    if A_ring[a, b]:
                        ax.plot(P[[a, b], 0], P[[a, b], 1], color="crimson", lw=0.6,
                                alpha=0.55, zorder=2)
            ax.scatter(P[:, 0], P[:, 1], c=hue, s=60, zorder=3, edgecolors="k", lw=0.4)
            for a in range(n):
                ax.annotate(words[a], P[a], fontsize=6, xytext=(2, 2),
                            textcoords="offset points")
            ax.set_title(f"{c}-primed | {WINLABEL[win]}\n"
                         f"EVR(2)={res[c][win]['evr2'][Lshow]:.2f}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"PCA of per-node mean residuals, layer {Lshow} "
                 "(gray = grid edges, red = ring edges, hue = ring position)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"pca_grids.{ext}"), dpi=160)
    plt.close(fig)

    # ---- figure 3: Laplacian eigenmode energy spectra by layer -----------
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey=True)
    for i, c in enumerate(CTXS):
        for j, win in enumerate(WINS):
            ax = axes[i, j]
            Sg = np.array(res[c][win]["spec_grid"])   # [nL, n]
            Sr = np.array(res[c][win]["spec_ring"])
            ax.plot(Sg[:, 1:3].sum(1), color="#023047", label="grid fundamental pair")
            ax.plot(Sr[:, 1:3].sum(1), color="crimson", label="ring fundamental pair")
            ax.plot(Sg[:, -1], color="#023047", ls=":", label="grid parity mode")
            ax.set_title(f"{c}-primed | {win}", fontsize=9)
            ax.grid(alpha=0.3)
            if i == 1: ax.set_xlabel("layer")
            if j == 0: ax.set_ylabel("energy fraction")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Low-frequency Laplacian eigenmode energy of node-means (grid basis vs ring basis)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"eig_spectra.{ext}"), dpi=160)
    plt.close(fig)

    # ---- figure 4: behavioral coupling during joint generation -----------
    log = json.load(open(os.path.join(OUT, "gen_log.json")))
    tg = log["tgen"]
    series = {("ring", "p_ring_nbrs"): [], ("ring", "p_grid_nbrs"): [],
              ("grid", "p_ring_nbrs"): [], ("grid", "p_grid_nbrs"): []}
    ts = {"ring": [], "grid": []}
    for t in range(tg):
        who = "ring" if t % 2 == 0 else "grid"
        vals_r, vals_g = [], []
        for p in range(log["npairs"]):
            step = log["steps"][f"pair{p}"][t]
            vals_r.append(step["p_ring_nbrs"]); vals_g.append(step["p_grid_nbrs"])
        series[(who, "p_ring_nbrs")].append(np.mean(vals_r))
        series[(who, "p_grid_nbrs")].append(np.mean(vals_g))
        ts[who].append(t)

    def smooth(x, k=15):
        x = np.asarray(x, float)
        if len(x) < k: return x
        return np.convolve(x, np.ones(k) / k, mode="valid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, who in zip(axes, ("ring", "grid")):
        for key, col in (("p_ring_nbrs", "crimson"), ("p_grid_nbrs", "#023047")):
            y = smooth(series[(who, key)])
            ax.plot(ts[who][: len(y)], y, color=col,
                    label="mass on RING-neighbours" if key == "p_ring_nbrs"
                    else "mass on GRID-neighbours")
        ax.axhline(2 / 15, color="crimson", ls=":", lw=1, label="ring chance (2 nbrs)")
        ax.axhline(3 / 15, color="#023047", ls=":", lw=1, label="grid chance (mean 3 nbrs)")
        ax.set_title(f"{who.upper()}-primed context's predictions")
        ax.set_xlabel("joint generation step")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("prob. mass on neighbours of previous node")
    axes[0].legend(fontsize=8)
    fig.suptitle("Behavioral coupling: whose graph does each instance follow?")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"behavior.{ext}"), dpi=160)
    plt.close(fig)

    # ---- summary ----------------------------------------------------------
    summary = {"model": log["model"], "npairs": log["npairs"], "ctx": log["ctx"],
               "tgen": tg, "layer_shown": Lshow, "per_layer": {}}
    for c in CTXS:
        summary["per_layer"][c] = {win: {k: res[c][win][k] for k in
                                         ("r2_grid", "r2_ring", "lf_grid", "lf_ring", "evr2")}
                                   for win in WINS}
    # headline numbers at the deep layer
    head = {}
    for c in CTXS:
        head[c] = {win: {"r2_grid": res[c][win]["r2_grid"][Lshow],
                         "r2_ring": res[c][win]["r2_ring"][Lshow],
                         "lf_grid": res[c][win]["lf_grid"][Lshow],
                         "lf_ring": res[c][win]["lf_ring"][Lshow]} for win in WINS}
    summary["headline_layer"] = head
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print(json.dumps(head, indent=2))
    print("DONE ->", OUT)


if __name__ == "__main__":
    main()
