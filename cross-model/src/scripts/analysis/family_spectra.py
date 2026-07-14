"""(1) variance vs eigenmode for every family, and (2) can a linear map from the model's represented
eigenmode powers reconstruct the base transition/adjacency matrix?

Reconstruction idea: the graph's transition matrix is built from its Laplacian eigenvectors V with
eigenvalues w (T = I - D^-1 L). The model represents each eigenmode k with power p_k (from node-means).
Form the "represented affinity" R = sum_k p_k v_k v_k^T. If the model concentrates power on the coarse
(low-frequency) modes that make up the graph's connectivity, R's off-diagonal should track the
adjacency A / transition T -- i.e. T is linearly recoverable from the represented eigenmodes. A
structureless graph (flat p_k) gives R ~ I, no adjacency info. We report corr(R_offdiag, A_offdiag),
edge-AUC, and the R^2 of a least-squares fit T ~ alpha*R + beta.

Reads nodemeans_<TAG>_<fam>.npz (markov_families). CPU-only. Env: MFDIR TAG(Llama) OUTDIR
"""
from __future__ import annotations
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

MFDIR = os.environ.get("MFDIR", "runs/axes/1_decomposition/markov_families")
TAG = os.environ.get("TAG", "Llama")
OUTDIR = os.environ.get("OUTDIR", "runs/axes/1_decomposition")
FAMS = os.environ.get("FAMS", "grid,ring,tree,smallworld,sbm4,er_random,sbm2").split(",")


def spectrum(fam):
    z = np.load(f"{MFDIR}/nodemeans_{TAG}_{fam}.npz", allow_pickle=True)
    A = np.array(z["adjacency"], float); n = A.shape[0]; nL = sum(1 for k in z.files if k.startswith("layer_"))
    L = np.diag(A.sum(1)) - A; w, V = np.linalg.eigh(L)
    best = -1; P = None; Hc_star = None
    for l in range(nL):
        H = z[f"layer_{l}"].astype(float); Hc = H - H.mean(0); c = V.T @ Hc; p = (c ** 2).sum(1); p[0] = 0; p /= p.sum() + 1e-12
        lb = p[(np.round(w, 3) == np.round(w[1], 3))].sum()
        if lb > best: best = lb; P = p; Hc_star = Hc
    return A, w, V, P, Hc_star


def auc(scores, labels):
    order = np.argsort(scores); labels = labels[order]
    pos = labels.sum(); neg = len(labels) - pos
    if pos == 0 or neg == 0: return float("nan")
    ranks = np.arange(1, len(labels) + 1)
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def reconstruct(A, w, V, P, Hc):
    n = A.shape[0]
    R = (V[:, 1:] * P[1:]) @ V[:, 1:].T                 # semi-trivial: uses the graph's OWN eigenvectors
    G = Hc @ Hc.T                                       # NON-trivial: representation Gram, no graph info
    G = G / (np.abs(G).max() + 1e-9)
    deg = A.sum(1); T = A / np.maximum(deg[:, None], 1)
    iu = np.triu_indices(n, 1); a_off = A[iu]; t_off = T[iu]; lab = (a_off > 0).astype(int)
    def fit_r2(x):
        X = np.stack([x, np.ones_like(x)], 1); coef, *_ = np.linalg.lstsq(X, t_off, rcond=None); pred = X @ coef
        ss = ((t_off - t_off.mean()) ** 2).sum(); return 1 - ((t_off - pred) ** 2).sum() / ss if ss > 0 else float("nan")
    return dict(aucA_R=auc(R[iu], lab), aucA_G=auc(G[iu], lab),
                r2_R=float(fit_r2(R[iu])), r2_G=float(fit_r2(G[iu])),
                R=R, G=G, T=T, A=A)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    data = {f: spectrum(f) for f in FAMS}
    rec = {f: reconstruct(*data[f]) for f in FAMS}

    # ---- (1) variance vs eigenmode, small multiples ----
    with PdfPages(f"{OUTDIR}/family_spectra_{TAG}.pdf") as pdf:
        ncol = 4; nrow = int(np.ceil(len(FAMS) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3 * nrow))
        for ax, f in zip(np.array(axes).flat, FAMS):
            A, w, V, P, Hc = data[f]; k = np.arange(1, len(w))
            ax.bar(k, P[1:], color="tab:purple")
            ax.set_title(f"{f}", fontsize=9); ax.set_xlabel("eigenmode (freq→)"); ax.set_ylim(0, 0.32)
            ax.set_ylabel("variance frac", fontsize=7)
        for ax in np.array(axes).flat[len(FAMS):]: ax.axis("off")
        fig.suptitle(f"{TAG}: node-mean variance vs Laplacian eigenmode, per family", fontsize=11)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ---- (2) transition-matrix reconstruction: semi-trivial (R, uses graph eigvecs) vs non-trivial (G, representation only) ----
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.6)); x = np.arange(len(FAMS)); w = 0.38
        ax[0].bar(x - w / 2, [rec[f]["aucA_R"] for f in FAMS], w, color=".6", label="R (uses graph eigvecs — semi-trivial)")
        ax[0].bar(x + w / 2, [rec[f]["aucA_G"] for f in FAMS], w, color="tab:orange", label="G = HcHcᵀ (representation only — non-trivial)")
        ax[0].axhline(0.5, color=".4", ls="--", lw=.8); ax[0].set_ylim(0.3, 1.0)
        ax[0].set_xticks(x); ax[0].set_xticklabels(FAMS, rotation=35, ha="right", fontsize=8)
        ax[0].set_ylabel("edge-recovery AUC"); ax[0].legend(fontsize=7)
        ax[0].set_title("recover adjacency from the representation", fontsize=9)
        ax[1].bar(x - w / 2, [rec[f]["r2_R"] for f in FAMS], w, color=".6"); ax[1].bar(x + w / 2, [rec[f]["r2_G"] for f in FAMS], w, color="tab:orange")
        ax[1].set_xticks(x); ax[1].set_xticklabels(FAMS, rotation=35, ha="right", fontsize=8)
        ax[1].set_ylabel("R² (T ≈ α·· + β)"); ax[1].set_title("linear fit of transition matrix", fontsize=9)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ---- (3) A / G / T heatmaps for grid + a random family ----
        for f in ["grid", "er_random"]:
            if f not in rec: continue
            fig, ax = plt.subplots(1, 3, figsize=(13, 4))
            for a, (M, ti) in zip(ax, [(rec[f]["A"], "adjacency A"), (rec[f]["G"], "representation Gram G=HcHcᵀ"), (rec[f]["T"], "transition T")]):
                im = a.imshow(M, cmap="magma"); a.set_title(f"{f}: {ti}", fontsize=9); a.set_xticks([]); a.set_yticks([])
                fig.colorbar(im, ax=a, fraction=.046)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    out = {f: {k: rec[f][k] for k in ("aucA_R", "aucA_G", "r2_R", "r2_G")} for f in FAMS}
    json.dump(out, open(f"{OUTDIR}/family_spectra_{TAG}.json", "w"), indent=2)
    print(f"{'family':11}{'AUC_R(triv)':>12}{'AUC_G(repr)':>12}{'R2_R':>7}{'R2_G':>7}")
    for f in FAMS:
        print(f"{f:11}{rec[f]['aucA_R']:>12.2f}{rec[f]['aucA_G']:>12.2f}{rec[f]['r2_R']:>7.2f}{rec[f]['r2_G']:>7.2f}")
    print(f"DONE -> {OUTDIR}/family_spectra_{TAG}.pdf")


if __name__ == "__main__":
    main()
