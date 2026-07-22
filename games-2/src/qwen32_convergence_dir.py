"""Find the MAIN CONVERGENCE DIRECTION for each Qwen player, per layer.

Definition: for player p at layer L, take p's activations, SEED-CENTER them (subtract
each seed's own mean over its turns -> removes which word that game converged to),
then take PC1 of the seed-centered activations. That top principal component is the
dominant axis the representation moves along within a game; we orient it so the
projection INCREASES with turn. That signed unit vector d_pL is the convergence
direction. The projection of an activation onto d_pL is its "convergence coordinate"
(should rise from early to late turns).

Reports per layer:
  * frac_var : fraction of within-seed variance PC1 captures (how 1-D the motion is)
  * turn_r   : correlation of the convergence coordinate with turn index (should be +)
  * cos12    : cosine(dir_Qwen1, dir_Qwen2) -- do the two players converge the same way?
Saves the direction vectors (per player, per layer) to an npz.

Usage: python src/qwen32_convergence_dir.py runs/game-1/qwen32/qwen32_pca_w2v/qwen32_pca_acts.npz
"""
from __future__ import annotations
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def seed_center(X, rolls):
    Y = X.astype(np.float32).copy()
    for s in np.unique(rolls):
        m = rolls == s
        Y[m] -= Y[m].mean(0)
    return Y


def convergence_dir(X, rolls, turns):
    """PC1 of seed-centered X, oriented so projection increases with turn.
    Returns (unit_direction, frac_var_PC1, corr(proj,turn))."""
    Y = seed_center(X, rolls)
    Yc = Y - Y.mean(0)
    tot = (Yc ** 2).sum()
    if tot < 1e-8:                               # degenerate (e.g. embedding layer: identical last token)
        return np.zeros(X.shape[1], np.float32), float("nan"), float("nan")
    U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
    d = Vt[0]
    frac = float(S[0] ** 2 / (S ** 2).sum())
    proj = Yc @ d
    r = float(np.corrcoef(proj, turns.astype(float))[0, 1])
    if r < 0:                                   # orient toward increasing turn
        d = -d; proj = -proj; r = -r
    return d / (np.linalg.norm(d) + 1e-9), frac, r


def main():
    npz = sys.argv[1] if len(sys.argv) > 1 else "runs/game-1/qwen32/qwen32_pca_w2v/qwen32_pca_acts.npz"
    z = np.load(npz, allow_pickle=True)
    A1, A2 = z["A1"].astype(np.float32), z["A2"].astype(np.float32)
    m1, m2 = z["meta1"], z["meta2"]
    P1, P2 = [str(p) for p in z["players"]]
    r1 = np.array([r for r, _, _ in m1]); t1 = np.array([t for _, t, _ in m1])
    r2 = np.array([r for r, _, _ in m2]); t2 = np.array([t for _, t, _ in m2])
    nL = A1.shape[1] - 1; H = A1.shape[2]

    D1 = np.zeros((nL + 1, H), np.float32); D2 = np.zeros((nL + 1, H), np.float32)
    fv1, fv2, tr1, tr2, cos12 = [], [], [], [], []
    for L in range(nL + 1):
        d1, f1, rr1 = convergence_dir(A1[:, L, :], r1, t1)
        d2, f2, rr2 = convergence_dir(A2[:, L, :], r2, t2)
        D1[L] = d1; D2[L] = d2
        fv1.append(f1); fv2.append(f2); tr1.append(rr1); tr2.append(rr2)
        cos12.append(float(d1 @ d2))
    out = os.path.join(os.path.dirname(npz), "qwen32_convergence_dir.npz")
    np.savez_compressed(out, dir1=D1, dir2=D2, players=np.array([P1, P2]),
                        frac_var1=fv1, frac_var2=fv2, turn_corr1=tr1, turn_corr2=tr2, cos12=cos12)

    pk = int(np.nanargmax(np.array(fv1) + np.array(fv2)))
    print(f"[dir] {P1}/{P2}: {nL+1} layers. peak-1D layer L{pk}: "
          f"PC1 var {fv1[pk]*100:.0f}%/{fv2[pk]*100:.0f}%, turn-corr {tr1[pk]:+.2f}/{tr2[pk]:+.2f}, "
          f"cos(dir1,dir2)={cos12[pk]:+.2f}")
    print(f"[dir] cos(dir1,dir2): median {np.median(cos12):+.2f}, min {np.min(cos12):+.2f}, max {np.max(cos12):+.2f}")
    print(f"[dir] saved direction vectors -> {out}")

    layers = np.arange(nL + 1)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    ax[0].plot(layers, np.array(fv1) * 100, "-o", ms=3, color="tab:blue", label=P1)
    ax[0].plot(layers, np.array(fv2) * 100, "-o", ms=3, color="tab:orange", label=P2)
    ax[0].set_title("PC1 (convergence dir) % of within-seed variance", fontsize=9)
    ax[0].set_xlabel("layer"); ax[0].set_ylabel("% var"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].plot(layers, tr1, "-o", ms=3, color="tab:blue"); ax[1].plot(layers, tr2, "-o", ms=3, color="tab:orange")
    ax[1].axhline(0, color=".7", lw=.6); ax[1].set_ylim(-1, 1)
    ax[1].set_title("corr(convergence coordinate, turn)", fontsize=9); ax[1].set_xlabel("layer"); ax[1].grid(alpha=.3)
    ax[2].plot(layers, cos12, "-o", ms=3, color="tab:purple"); ax[2].axhline(1, color=".7", ls=":", lw=.8)
    ax[2].set_ylim(-1.05, 1.05); ax[2].set_title("cosine(dir Qwen1, dir Qwen2)", fontsize=9)
    ax[2].set_xlabel("layer"); ax[2].grid(alpha=.3)
    fig.suptitle("Main convergence direction per Qwen (PC1 of seed-centered activations)", fontsize=11)
    fig.tight_layout()
    pdf = out.replace(".npz", ".pdf"); fig.savefig(pdf)
    print(f"[dir] summary plot -> {pdf}")


if __name__ == "__main__":
    main()
