"""Turn-RESOLVED manifold of the convergence trajectory, done properly:
  * seed-center (remove which word each game converged to)
  * NORMALIZE turn -> relative progress p in [0,1] per seed (short & long games comparable)
  * turn-conditioned centroids with EQUAL per-seed weight (not per-point)
Then characterize the curve:
  * effective dimensionality (participation ratio) -- is it a low-D arc?
  * per-segment speed and TURNING ANGLE -- where/how much it curves
  * SWING-BY test: project centroids on the start->end axis (along) and the orthogonal
    complement (swing amplitude). Overshoot = along exceeds the endpoint then returns;
    swing = orthogonal amplitude bulges in the middle then collapses.

Usage: python src/qwen32_turn_manifold.py <acts.npz> [LAYER]
"""
from __future__ import annotations
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

KBINS = 6


def seed_center_pool(z):
    A = np.concatenate([z["A1"].astype(np.float32), z["A2"].astype(np.float32)], 0)
    m1, m2 = z["meta1"], z["meta2"]
    rolls = np.concatenate([np.array([r for r, _, _ in m1]),
                            np.array([r for r, _, _ in m2]) + 1000])   # distinct seed ids per player
    turns = np.concatenate([np.array([t for _, t, _ in m1]),
                            np.array([t for _, t, _ in m2])])
    return A, rolls, turns


def centroids(A, rolls, turns, L):
    """Equal-per-seed, normalized-progress turn-conditioned centroids (seed-centered)."""
    X = A[:, L, :].astype(np.float32).copy()
    for s in np.unique(rolls):
        X[rolls == s] -= X[rolls == s].mean(0)            # seed-center: remove word identity
    edges = np.linspace(0, 1, KBINS + 1)
    per_seed_bin = {k: [] for k in range(KBINS)}
    for s in np.unique(rolls):
        idx = np.where(rolls == s)[0]; tt = turns[idx].astype(float)
        if len(idx) < 2:
            continue
        p = (tt - tt.min()) / (tt.max() - tt.min() + 1e-9)  # relative progress in [0,1]
        for k in range(KBINS):
            m = (p >= edges[k]) & (p <= edges[k + 1] if k == KBINS - 1 else p < edges[k + 1])
            if m.any():
                per_seed_bin[k].append(X[idx[m]].mean(0))   # this seed's mean in this bin
    C = np.array([np.mean(per_seed_bin[k], 0) for k in range(KBINS)])
    ncontrib = [len(per_seed_bin[k]) for k in range(KBINS)]
    return C, ncontrib


def analyze(C):
    d = np.diff(C, axis=0)                                # segment vectors
    speed = np.linalg.norm(d, axis=1)
    # turning angle between consecutive segments
    ang = []
    for i in range(len(d) - 1):
        u = d[i] / (np.linalg.norm(d[i]) + 1e-9); v = d[i + 1] / (np.linalg.norm(d[i + 1]) + 1e-9)
        ang.append(np.degrees(np.arccos(np.clip(u @ v, -1, 1))))
    # effective dim of the centroid cloud (participation ratio of its covariance)
    Cc = C - C.mean(0); lam = np.linalg.svd(Cc, compute_uv=False) ** 2
    pr = float((lam.sum() ** 2) / (lam ** 2).sum())
    # swing-by decomposition: along start->end axis vs orthogonal amplitude
    e = C[-1] - C[0]; e /= np.linalg.norm(e) + 1e-9
    along = (C - C[0]) @ e
    orth = np.linalg.norm((C - C[0]) - np.outer(along, e), axis=1)
    return speed, np.array(ang), pr, along, orth


def main():
    npz = sys.argv[1] if len(sys.argv) > 1 else "runs/game-1/qwen32/qwen32_pca_w2v/qwen32_pca_acts.npz"
    Ls = [int(sys.argv[2])] if len(sys.argv) > 2 else [8, 16, 32, 48, 63]
    z = np.load(npz, allow_pickle=True)
    A, rolls, turns = seed_center_pool(z)

    print(f"[manifold] {KBINS} normalized-progress bins, equal per-seed weight, seeds>=2 turns")
    for L in Ls:
        C, nc = centroids(A, rolls, turns, L)
        speed, ang, pr, along, orth = analyze(C)
        end = np.linalg.norm(C[-1] - C[0])
        overshoot = float(along.max() - along[-1])        # >0 => overshoots the endpoint
        swing = float(orth.max())                          # peak orthogonal excursion
        print(f"\n=== layer {L} (bin contributors {nc}) ===")
        print(f"  effective dim (participation ratio): {pr:.2f}   [~1=straight line, 2=planar arc, higher=complex]")
        print(f"  segment speed (early->late): {np.round(speed,1)}")
        print(f"  turning angle between segments (deg): {np.round(ang,0)}   [0=straight, 90=right-angle turn]")
        print(f"  along start->end axis: {np.round(along,1)}  (endpoint={along[-1]:.1f}; overshoot={overshoot:+.1f})")
        print(f"  orthogonal swing amplitude: {np.round(orth,1)}  (peak={swing:.1f} = {100*swing/ (end+1e-9):.0f}% of start-end dist)")

    # figure for the first layer
    L = Ls[0]; C, nc = centroids(A, rolls, turns, L)
    speed, ang, pr, along, orth = analyze(C)
    P = PCA(3).fit_transform(C - C.mean(0))
    fig = plt.figure(figsize=(15, 4.4))
    ax0 = fig.add_subplot(1, 3, 1)
    sc = ax0.scatter(P[:, 0], P[:, 1], c=range(KBINS), cmap="viridis", s=90, zorder=3)
    ax0.plot(P[:, 0], P[:, 1], "-k", alpha=.5)
    for i in range(KBINS):
        ax0.annotate(f"{i}", (P[i, 0], P[i, 1]), fontsize=8)
    fig.colorbar(sc, ax=ax0, label="progress bin"); ax0.set_title(f"centroid trajectory PCA @L{L} (PR={pr:.2f})", fontsize=9)
    ax0.set_xlabel("PC1"); ax0.set_ylabel("PC2")
    ax1 = fig.add_subplot(1, 3, 2)
    ax1.plot(range(KBINS), along, "-o", label="along start→end")
    ax1.plot(range(KBINS), orth, "-o", label="orthogonal swing")
    ax1.axhline(along[-1], color="gray", ls=":", label="endpoint (along)")
    ax1.set_xlabel("progress bin"); ax1.set_title("swing-by decomposition", fontsize=9); ax1.legend(fontsize=8)
    ax2 = fig.add_subplot(1, 3, 3)
    ax2.plot(range(1, KBINS), speed, "-o", color="tab:red", label="speed")
    ax2.plot(range(1, KBINS - 1), ang, "-o", color="tab:purple", label="turn angle (deg)")
    ax2.set_xlabel("segment"); ax2.set_title("speed & curvature", fontsize=9); ax2.legend(fontsize=8)
    fig.suptitle("Qwen3-32B convergence trajectory — turn-resolved manifold (equal per-seed, normalized progress)", fontsize=10)
    fig.tight_layout()
    out = npz.replace("_acts.npz", "_turn_manifold.pdf"); fig.savefig(out)
    print(f"\n[manifold] figure -> {out}")


if __name__ == "__main__":
    main()
