"""Render Gemma-layer x Qwen-layer CKA heatmaps from the saved all-layer subsample.

Produces a side-by-side figure:
  (left)  high-context CKA  -- reuses the pod-computed cka_heatmap.npy
  (right) pooled CKA        -- computed here over a context-balanced subsample
Each model's grid-recovery peak layer (Gemma L23, Qwen L12) is marked.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = "runs/v1/square_grid"
GEMMA_PEAK, QWEN_PEAK = 23, 12          # from grid_per_layer.json


def load_layers(npz_path):
    z = np.load(npz_path, allow_pickle=False)
    layers = [int(l) for l in z["_layers"]]
    return z, layers


def cka(X, Y):
    X = X.astype(np.float32); Y = Y.astype(np.float32)
    Xc = X - X.mean(0); Yc = Y - Y.mean(0)
    hsic = np.linalg.norm(Xc.T @ Yc) ** 2
    den = np.linalg.norm(Xc.T @ Xc) * np.linalg.norm(Yc.T @ Yc)
    return float(hsic / den) if den > 0 else np.nan


def pooled_heatmap(n_sub=3000, seed=0):
    zg, gl = load_layers(f"{RUN}/acts_sub_gemma.npz")
    zq, ql = load_layers(f"{RUN}/acts_sub_qwen.npz")
    n = zg["meta_walk_id"].shape[0]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, min(n_sub, n), replace=False))
    # subsample + cast once per layer to keep memory small
    Xg = {L: zg[f"layer_{L}"][idx].astype(np.float32) for L in gl}
    Yq = {L: zq[f"layer_{L}"][idx].astype(np.float32) for L in ql}
    H = np.array([[cka(Xg[g], Yq[q]) for q in ql] for g in gl])
    return H, gl, ql


def draw(ax, H, gl, ql, title):
    im = ax.imshow(H, origin="lower", aspect="auto", cmap="viridis",
                   vmin=0, vmax=1, extent=[ql[0] - .5, ql[-1] + .5,
                                           gl[0] - .5, gl[-1] + .5])
    ax.axhline(GEMMA_PEAK, color="white", lw=.8, ls="--", alpha=.7)
    ax.axvline(QWEN_PEAK, color="white", lw=.8, ls="--", alpha=.7)
    bi, bj = np.unravel_index(int(np.nanargmax(H)), H.shape)
    ax.plot(ql[bj], gl[bi], "r*", ms=12)
    ax.set_xlabel("Qwen layer")
    ax.set_ylabel("Gemma layer")
    ax.set_title(f"{title}\nmax {H[bi,bj]:.2f} @ G{gl[bi]}/Q{ql[bj]}  "
                 f"(grid peaks: G{GEMMA_PEAK},Q{QWEN_PEAK})", fontsize=9)
    return im


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # left: high-context (pod-computed)
    hi = np.load(f"{RUN}/cka_heatmap.npy")
    gl_full, ql_full = list(range(hi.shape[0])), list(range(hi.shape[1]))
    im0 = draw(axes[0], hi, gl_full, ql_full, "linear CKA  (high-context occ.)")

    # right: pooled (computed locally)
    H, gl, ql = pooled_heatmap()
    im1 = draw(axes[1], H, gl, ql, "linear CKA  (pooled over all context)")
    np.save(f"{RUN}/cka_heatmap_pooled.npy", H)

    fig.colorbar(im1, ax=axes, label="CKA", fraction=.025, pad=.02)
    out = f"{RUN}/cka_heatmaps.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    print(f"high-ctx max  {hi.max():.3f} @ G{np.unravel_index(hi.argmax(),hi.shape)}")
    print(f"pooled   max  {np.nanmax(H):.3f} @ G{np.unravel_index(np.nanargmax(H),H.shape)}")
    # quick look at the grid-peak cell specifically
    print(f"grid-peak cell  G{GEMMA_PEAK}/Q{QWEN_PEAK}:  "
          f"high-ctx={hi[GEMMA_PEAK, QWEN_PEAK]:.3f}  pooled={H[GEMMA_PEAK, QWEN_PEAK]:.3f}")


if __name__ == "__main__":
    main()
