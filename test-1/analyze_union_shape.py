"""IDENTIFY the union shape: Fourier decomposition of converged node-means on Z_16.

The node index order i = 0..15 is simultaneously the ring order AND the row-major grid
order, so the union graph (grid + ring edges) is the circulant graph C16(1,4) minus the
4 wrap column-chords. Circulant graphs are diagonalized by Fourier modes on Z_16, so the
union geometry has a canonical frequency signature:
  f=1 pair  : position along the 16-cycle  (== ring coords; also carries grid ROW, which
              is a staircase over the cycle)
  f=4 pair  : column identity i mod 4      (grid COLUMN fundamental; C16(1,4)'s lowest
              nontrivial eigenvalue block together with f=1)
  f=8       : column parity (harmonic of column)
A pure ring representation = energy at f=1 only. A pure grid representation = energy at
f=1..3 (row staircase) + f=4/8/12 (column). The near-circulant union = f=1 + f=4 blocks.

For joint_late node-means (both primed contexts + fresh control) per layer:
  * energy fraction in each Fourier frequency f = 1..8 (conjugate pairs pooled),
  * R^2 from candidate subspaces: f1 pair (=ring), f4 pair (column), f1+f4 (4D union
    signature), grid coords (2D), union-Laplacian modes 1-2 and 1-4,
  * layer-26 scatter of node-means projected on the f1 plane and the f4 plane.

Also prints the union Laplacian spectrum with each eigenvector's dominant frequency, to
verify the C16(1,4) approximation.

Env: RUN_OUT (default out)
Out: <RUN_OUT>/union_shape.png/.pdf, <RUN_OUT>/union_shape.json
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("RUN_OUT", os.path.join(HERE, "runs", "out"))
n = 16
KEYS = [("grid_joint_late", "GRID-primed late"), ("ring_joint_late", "RING-primed late"),
        ("fresh_late", "FRESH late"), ("grid_base", "GRID-primed base"),
        ("ring_base", "RING-primed base")]


def fourier_pairs():
    """f -> orthonormal basis [16, 1 or 2] of the real Fourier pair at frequency f."""
    i = np.arange(n)
    out = {}
    for f in range(1, 9):
        cols = [np.cos(2 * np.pi * f * i / n)]
        if f < 8:
            cols.append(np.sin(2 * np.pi * f * i / n))
        B = np.stack(cols, 1)
        B = B - B.mean(0)
        Q, _ = np.linalg.qr(B)
        out[f] = Q
    return out


def r2_from_feats(Hc, F):
    Fc = F - F.mean(0)
    Fc = Fc / np.maximum(Fc.std(0), 1e-12)
    B, *_ = np.linalg.lstsq(Fc, Hc, rcond=None)
    resid = Hc - Fc @ B
    return float(1.0 - (resid ** 2).sum() / np.maximum((Hc ** 2).sum(), 1e-12))


def main():
    z = np.load(os.path.join(OUT, "nodemeans_dueling.npz"), allow_pickle=False)
    zf = np.load(os.path.join(OUT, "nodemeans_fresh.npz"), allow_pickle=False)
    nL = int(z["n_layers"][0])
    words = [str(w) for w in z["words"]]
    A_grid = z["adjacency_grid"].astype(float)
    A_ring = z["adjacency_ring"].astype(float)
    A_union = np.maximum(A_grid, A_ring)
    cg = z["coords_grid"]

    def H(key, L):
        Hm = (zf if key.startswith("fresh") else z)[f"{key}_layer_{L}"].astype(np.float64)
        return Hm - Hm.mean(0)

    FP = fourier_pairs()

    # ---- verify near-circulant claim: union Laplacian modes vs Fourier freqs ----
    d = A_union.sum(1)
    di = 1.0 / np.sqrt(d)
    Lap = np.eye(n) - di[:, None] * A_union * di[None, :]
    w, U = np.linalg.eigh(Lap)
    mode_freq = []
    for k in range(1, n):
        pw = {f: float((FP[f].T @ U[:, k] > -np.inf).sum() and
                       np.linalg.norm(FP[f].T @ U[:, k]) ** 2) for f in FP}
        fbest = max(pw, key=pw.get)
        mode_freq.append((k, round(float(w[k]), 3), fbest, round(pw[fbest], 2)))
    print("union Laplacian modes (idx, lambda, dominant Fourier f, purity):")
    print(" ", mode_freq[:6], "...")

    # ---- Fourier energy spectra by layer ----------------------------------
    spec = {key: np.zeros((nL, 8)) for key, _ in KEYS}
    for key, _ in KEYS:
        for L in range(nL):
            Hc = H(key, L)
            tot = (Hc ** 2).sum()
            for f in range(1, 9):
                spec[key][L, f - 1] = ((FP[f].T @ Hc) ** 2).sum() / max(tot, 1e-12)

    # ---- candidate-subspace R^2 -------------------------------------------
    wl, Ul = np.linalg.eigh(Lap)
    CANDS = {
        "f1 pair (= ring coords, 2D)": FP[1],
        "f4 pair (column, 2D)": FP[4],
        "f1 + f4 (union signature, 4D)": np.concatenate([FP[1], FP[4]], 1),
        "grid coords (2D)": cg,
        "union Lap modes 1-2": Ul[:, 1:3],
        "union Lap modes 1-4": Ul[:, 1:5],
    }
    r2 = {key: {c: [r2_from_feats(H(key, L), F) for L in range(nL)]
                for c, F in CANDS.items()} for key, _ in KEYS[:3]}

    # ---- figure -------------------------------------------------------------
    Lshow = min(26, nL - 1)
    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.15])

    # row 1: energy heatmaps for the three late keys + base references
    for j, (key, lab) in enumerate(KEYS[:3]):
        ax = fig.add_subplot(gs[0, j * 2:j * 2 + 2])
        im = ax.imshow(spec[key].T, aspect="auto", origin="lower", cmap="magma",
                       vmin=0, vmax=0.45)
        ax.set_yticks(range(8)); ax.set_yticklabels([f"f={f}" for f in range(1, 9)],
                                                    fontsize=7)
        ax.set_xlabel("layer"); ax.set_title(f"{lab}: Fourier energy", fontsize=9)
        if j == 2: fig.colorbar(im, ax=ax, fraction=0.04)

    # row 2 left: R^2 curves (grid-primed late shown; others in json)
    ax = fig.add_subplot(gs[1, 0:2])
    ccol = {"f1 pair (= ring coords, 2D)": "crimson", "f4 pair (column, 2D)": "#fb8500",
            "f1 + f4 (union signature, 4D)": "#023047", "grid coords (2D)": "0.4",
            "union Lap modes 1-2": "#2a9d8f", "union Lap modes 1-4": "#8ecae6"}
    for cname in CANDS:
        ax.plot(np.mean([r2[k][cname] for k, _ in KEYS[:3]], 0), color=ccol[cname],
                label=cname)
    ax.set_xlabel("layer"); ax.set_ylabel(r"R$^2$ (mean of 3 late contexts)")
    ax.legend(fontsize=6.5); ax.grid(alpha=0.3); ax.set_title("candidate subspaces")

    # row 2 mid/right: layer-26 projections on f1 plane and f4 plane (fresh ctx)
    hue = plt.cm.hsv(np.arange(n) / n)
    for j, (f, plane_lab) in enumerate([(1, "f=1 plane (cycle position)"),
                                        (4, "f=4 plane (column identity)")]):
        for kk, (key, lab) in enumerate(KEYS[:2]):
            ax = fig.add_subplot(gs[1, 2 + j * 2 + kk])
            # restrict the node-means to the frequency-f node-space plane, then take the
            # exact 2D coords of that rank-<=2 configuration via SVD
            Hc = H(key, Lshow)
            Hf = FP[f] @ (FP[f].T @ Hc)                # [16, d], rank <= FP[f].shape[1]
            Uu, ss, _ = np.linalg.svd(Hf, full_matrices=False)
            P2 = Uu[:, :2] * ss[:2]
            ax.scatter(P2[:, 0], P2[:, 1], c=hue, s=55, edgecolors="k", lw=0.4)
            for a in range(n):
                ax.annotate(words[a] if f == 1 else f"{words[a]}({a%4})", P2[a],
                            fontsize=5.5, xytext=(2, 2), textcoords="offset points")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{lab}\n{plane_lab}", fontsize=8)
    fig.suptitle(f"Union-shape identification via Fourier modes on Z16 (layer {Lshow}); "
                 "f=1 == ring/cycle, f=4 == grid column")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"union_shape.{ext}"), dpi=160)
    plt.close(fig)

    summary = {
        "union_lap_modes_dominant_freq": mode_freq,
        "fourier_energy_deep_mean_24_31": {
            key: {f"f{f}": float(spec[key][24:, f - 1].mean()) for f in range(1, 9)}
            for key, _ in KEYS},
        "r2_deep_mean_24_31": {key: {c: float(np.mean(v[24:])) for c, v in r2[key].items()}
                               for key in r2},
        "layer_shown": Lshow,
    }
    json.dump(summary, open(os.path.join(OUT, "union_shape.json"), "w"), indent=2)
    print(json.dumps(summary["fourier_energy_deep_mean_24_31"], indent=2))
    print(json.dumps(summary["r2_deep_mean_24_31"], indent=2))
    print("DONE ->", OUT)


if __name__ == "__main__":
    main()
