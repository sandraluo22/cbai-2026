"""Project the convergence direction OUT of the (seed-centered) activations, then take
the top PC of the residual -> the main axis ORTHOGONAL to convergence ('what the model
carries/leaves behind independent of converging'). Saves dir1/dir2 in the same format
as the convergence-dir npz so logit_lens_propagate.py can read it.

Usage: python src/qwen32_orthogonal_dir.py <acts.npz> <convergence_dir.npz>
"""
from __future__ import annotations
import os
import sys
import numpy as np


def seed_center(X, rolls):
    Y = X.astype(np.float32).copy()
    for s in np.unique(rolls):
        m = rolls == s; Y[m] -= Y[m].mean(0)
    return Y


def main():
    acts = sys.argv[1]; convn = sys.argv[2]
    z = np.load(acts, allow_pickle=True); c = np.load(convn, allow_pickle=True)
    A1, A2 = z["A1"].astype(np.float32), z["A2"].astype(np.float32)
    m1, m2 = z["meta1"], z["meta2"]; P1, P2 = [str(p) for p in z["players"]]
    r1 = np.array([r for r, _, _ in m1]); r2 = np.array([r for r, _, _ in m2])
    Dc1, Dc2 = c["dir1"], c["dir2"]
    nL = A1.shape[1] - 1; H = A1.shape[2]

    def orth_dir(X, rolls, dconv):
        Y = seed_center(X, rolls); Y = Y - Y.mean(0)
        if (Y ** 2).sum() < 1e-8:
            return np.zeros(H, np.float32), float("nan"), float("nan")
        d = dconv / (np.linalg.norm(dconv) + 1e-9)
        Y = Y - np.outer(Y @ d, d)                       # project OUT the convergence direction
        U, S, Vt = np.linalg.svd(Y, full_matrices=False)
        dorth = Vt[0]; frac = float(S[0] ** 2 / (S ** 2).sum())
        return dorth / (np.linalg.norm(dorth) + 1e-9), frac, float(abs(dorth @ d))

    D1 = np.zeros((nL + 1, H), np.float32); D2 = np.zeros((nL + 1, H), np.float32)
    fv1, orthchk, cos12 = [], [], []
    for L in range(nL + 1):
        d1, f1, o1 = orth_dir(A1[:, L, :], r1, Dc1[L])
        d2, f2, o2 = orth_dir(A2[:, L, :], r2, Dc2[L])
        D1[L] = d1; D2[L] = d2; fv1.append(f1); orthchk.append(o1); cos12.append(float(d1 @ d2))
    out = os.path.join(os.path.dirname(acts), "qwen32_orthogonal_dir.npz")
    np.savez_compressed(out, dir1=D1, dir2=D2, players=np.array([P1, P2]),
                        frac_var1=fv1, cos12=cos12)
    print(f"[orth] saved -> {out}")
    print(f"[orth] PC1-after-projout captures (median) {np.nanmedian(fv1)*100:.0f}% of the remaining variance")
    print(f"[orth] orthogonality check |cos(dorth, dconv)| median = {np.nanmedian(orthchk):.3f} (want ~0)")
    print(f"[orth] cos(dorth Qwen1, dorth Qwen2) median = {np.nanmedian(cos12):+.2f}")


if __name__ == "__main__":
    main()
