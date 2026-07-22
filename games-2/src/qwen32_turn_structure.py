"""Do the Qwen3-32B self-play games share a common TURN trajectory in representation
space, or is each game idiosyncratic? Two per-layer metrics from the cached acts.

  (1) shared turn DIRECTION: for each (seed, player) trajectory take consecutive
      turn-step vectors act(t+1)-act(t); measure the mean pairwise cosine across all
      trajectories' first steps. Chance floor for H-dim random vectors ~ 1/sqrt(H).
      >> floor => games move in a shared direction; ~floor => idiosyncratic.
  (2) turn DECODABILITY: leave-one-seed-out ridge R2 predicting turn index from the
      seed-centered activation. >0 => a shared linear turn axis exists.

Sanity: mean within-seed cross-player cosine of activations (should be ~1: the two
Qwens are near-identical).

Usage: python src/qwen32_turn_structure.py runs/game-1/qwen32/qwen32_pca/qwen32_pca_acts.npz
"""
from __future__ import annotations
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def unit(v, ax=-1, eps=1e-9):
    n = np.linalg.norm(v, axis=ax, keepdims=True)
    return v / np.clip(n, eps, None)


def first_steps(A, rolls, turns, L):
    """One step vector per seed: act(2nd turn) - act(1st turn) at layer L."""
    steps = []
    for s in np.unique(rolls):
        idx = np.where(rolls == s)[0]
        tt = turns[idx]; order = idx[np.argsort(tt)]
        if len(order) >= 2:
            steps.append(A[order[1], L, :].astype(np.float32) - A[order[0], L, :].astype(np.float32))
    return np.array(steps)


def mean_pairwise_cos(V):
    U = unit(V)
    C = U @ U.T
    iu = np.triu_indices(len(U), 1)
    return float(C[iu].mean()), float(C[iu].std())


def turn_r2_loso(A, rolls, turns, L):
    """Leave-one-seed-out ridge R2: predict turn from seed-centered activation @ L."""
    X = A[:, L, :].astype(np.float32).copy()
    y = turns.astype(np.float32)
    for s in np.unique(rolls):                       # seed-center features AND target
        m = rolls == s
        X[m] -= X[m].mean(0); y = y.copy()
    yc = y - y.mean()
    seeds = np.unique(rolls)
    pred = np.zeros(len(y)); alpha = 1e3
    for s in seeds:
        te = rolls == s; tr = ~te
        if tr.sum() < 3 or te.sum() < 1:
            pred[te] = 0; continue
        Xtr = X[tr]; mu = Xtr.mean(0); Xtr = Xtr - mu
        U, S, Vt = np.linalg.svd(Xtr, full_matrices=False)
        coef = Vt.T @ ((S / (S**2 + alpha)) * (U.T @ yc[tr]))     # 1-D target -> (H,)
        pred[te] = (X[te] - mu) @ coef
    sst = (yc**2).sum()
    return float(1 - ((yc - pred)**2).sum() / sst) if sst > 0 else float("nan")


def main():
    npz = sys.argv[1] if len(sys.argv) > 1 else "runs/game-1/qwen32/qwen32_pca/qwen32_pca_acts.npz"
    z = np.load(npz, allow_pickle=True)
    A1, A2 = z["A1"], z["A2"]; m1, m2 = z["meta1"], z["meta2"]
    r1 = np.array([r for r, _, _ in m1]); t1 = np.array([t for _, t, _ in m1])
    r2 = np.array([r for r, _, _ in m2]); t2 = np.array([t for _, t, _ in m2])
    nL = A1.shape[1] - 1; H = A1.shape[2]
    floor = 1.0 / np.sqrt(H)
    # pool both players as separate trajectories
    A = np.concatenate([A1, A2], 0); rolls = np.concatenate([r1, r2]); turns = np.concatenate([t1, t2])
    # tag seed by (roll, player) so a seed's two players are distinct trajectories for steps
    roll_pl = np.concatenate([r1 * 10, r2 * 10 + 1])

    # sanity: within-seed cross-player cosine at a mid layer
    Lm = nL // 2
    cs = []
    for s in np.unique(r1):
        i1 = np.where(r1 == s)[0]; i2 = np.where(r2 == s)[0]
        for a in i1:
            for b in i2:
                if t1[a] == t2[b]:
                    cs.append(float(unit(A1[a, Lm]) @ unit(A2[b, Lm])))
    print(f"[sanity] within-seed cross-player cosine @L{Lm}: {np.mean(cs):.3f} (near 1 => the two Qwens are ~identical)")

    layers = list(range(nL + 1))
    dir_mean, dir_std, r2 = [], [], []
    for L in layers:
        V = first_steps(A, roll_pl, turns, L)
        mc, sc = mean_pairwise_cos(V) if len(V) >= 2 else (float("nan"), float("nan"))
        dir_mean.append(mc); dir_std.append(sc)
        r2.append(turn_r2_loso(A, rolls, turns, L))
    dm = np.array(dir_mean)
    print(f"[direction] chance floor ~{floor:.3f}; mean step-cosine across layers: "
          f"min {np.nanmin(dm):.3f}, median {np.nanmedian(dm):.3f}, max {np.nanmax(dm):.3f} "
          f"(peak @L{int(np.nanargmax(dm))})")
    print(f"[decode]    turn R2 (LOSO): median {np.nanmedian(r2):.3f}, max {np.nanmax(r2):.3f} @L{int(np.nanargmax(r2))}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].plot(layers, dir_mean, "-o", ms=3, color="tab:purple")
    ax[0].axhline(floor, color="r", ls=":", label=f"chance floor {floor:.3f}")
    ax[0].axhline(0, color=".7", lw=.6)
    ax[0].set_xlabel("layer"); ax[0].set_ylabel("mean pairwise cosine of turn-step vectors")
    ax[0].set_title("Shared turn DIRECTION across seeds", fontsize=10); ax[0].legend(fontsize=8)
    ax[1].plot(layers, r2, "-o", ms=3, color="tab:green")
    ax[1].axhline(0, color=".7", lw=.6)
    ax[1].set_xlabel("layer"); ax[1].set_ylabel("leave-one-seed-out R²")
    ax[1].set_title("Turn DECODABILITY (seed-centered)", fontsize=10)
    fig.suptitle("Do Qwen3-32B self-play games share a turn trajectory? (per layer)", fontsize=11)
    fig.tight_layout()
    out = npz.replace("_acts.npz", "_turn_structure.pdf")
    fig.savefig(out); print(f"[done] {out}")


if __name__ == "__main__":
    main()
