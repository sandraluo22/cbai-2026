"""How aligned is the supervised best-2D *RSA* subspace with the coord-*probe* readout subspace?

Both are 2-D subspaces of a layer's activation space that claim to carry the graph geometry:
  - RSA basis : top-6 node-mean PCs regressed onto the graph layout (as in best_2d / the ablation
                'best2dRSA' column) -> a d x 2 embedding basis.
  - probe basis: leave-one-node-out ridge readout activation->coord (coord_decode) -> d x 2.
We compare them by principal angles between the two column spaces (cos of the 2 principal angles;
1 = identical subspace). Runs LOCALLY off the cached all-layer means (allmeans_<graph>_<model>.npz).

Env: GRAPH(square_grid) MEANDIR(cross_layer_heatmap dir) OUTDIR
Out: <OUTDIR>/basis_cossim_<graph>.json + .pdf
"""
from __future__ import annotations
import os, json, glob
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from config import get_config
import graph as G

GKW = {"square_grid": dict(graph_type="grid", grid_rows=4, grid_cols=4),
       "ring": dict(graph_type="ring", ring_size=16),
       "hex": dict(graph_type="hex", hex_rows=4, hex_cols=4),
       "days": dict(graph_type="ring", ring_size=7, word_set="days")}
GRAPH = os.environ.get("GRAPH", "square_grid")
MEANDIR = os.environ.get("MEANDIR", "runs/induction-head/2_probes/cross_layer_heatmap")
OUTDIR = os.environ.get("OUTDIR", "runs/induction-head/2_probes/basis_cossim")
ALPHAS = [1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6]


def _r2(y, yh):
    tot = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ((y - yh) ** 2).sum() / tot) if tot > 0 else float("nan")


def rsa_basis(H, coords):
    """Supervised best-2D embedding basis: top-6 PCs regressed onto layout coords (d x 2)."""
    Hc = H - H.mean(0)
    U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    k = min(6, Vt.shape[0]); Z = U[:, :k] * S[:k]
    W = np.linalg.lstsq(Z, coords - coords.mean(0), rcond=None)[0]
    return Vt[:k].T @ W                                              # d x 2


def probe_basis(H, coords):
    """LOO-ridge coord-probe readout directions in raw activation space (d x 2) + mean LOO R²."""
    n, d = H.shape
    mu = H.mean(0); sd = H.std(0) + 1e-6; Xs = (H - mu) / sd; Yc = coords - coords.mean(0)
    folds = []
    for kf in range(n):
        idx = [i for i in range(n) if i != kf]
        U, S, Vt = np.linalg.svd(Xs[idx], full_matrices=False)
        folds.append((np.array(idx), (Xs[kf] @ Vt.T), U.T.copy(), S))
    best = (-9.0, ALPHAS[0])
    for a in ALPHAS:
        pred = np.zeros((n, 2))
        for kf, (idx, proj, UT, S) in enumerate(folds):
            ytr = Yc[idx]; ymu = ytr.mean(0)
            pred[kf] = proj @ ((S / (S ** 2 + a))[:, None] * (UT @ (ytr - ymu))) + ymu
        sc = 0.5 * (_r2(Yc[:, 0], pred[:, 0]) + _r2(Yc[:, 1], pred[:, 1]))
        if sc > best[0]: best = (sc, a)
    a = best[1]
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    coef_std = Vt.T @ ((S / (S ** 2 + a))[:, None] * (U.T @ Yc))
    return coef_std / sd[:, None], float(best[0])                    # d x 2, LOO R²


def principal_cos(A, B):
    """cos of principal angles between column spaces of A(d x2) and B(d x2) -> 2 values in [0,1]."""
    Qa, _ = np.linalg.qr(A); Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.clip(s, 0, 1)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    cfg = replace(get_config("gemma_qwen"), **GKW[GRAPH])
    graph = G.build_graph(cfg); coords = np.array(graph.coords, float)
    pre = f"allmeans_{GRAPH}_"
    files = sorted(glob.glob(f"{MEANDIR}/{pre}*.npz"))
    if not files:
        raise SystemExit(f"no allmeans for {GRAPH} in {MEANDIR}")
    out = {"graph": GRAPH, "models": {}}
    for f in files:
        tag = os.path.basename(f)[len(pre):-4]
        means = np.load(f)["means"].astype(float)                    # [nL, n, d]
        nL = means.shape[0]; cos1 = []; cos2 = []; pr2 = []
        for L in range(nL):
            H = means[L]
            Br = rsa_basis(H, coords); Bp, r2 = probe_basis(H, coords)
            c = principal_cos(Br, Bp)
            cos1.append(float(c[0])); cos2.append(float(c[1])); pr2.append(r2)
        out["models"][tag] = {"n_layers": nL, "cos_principal_1": cos1, "cos_principal_2": cos2,
                              "probe_r2": pr2}
        peak = int(np.nanargmax(pr2))
        print(f"[{tag}/{GRAPH}] at probe-peak L{peak} (R²={pr2[peak]:.2f}): "
              f"cos∠=({cos1[peak]:.3f}, {cos2[peak]:.3f})", flush=True)
    json.dump(out, open(f"{OUTDIR}/basis_cossim_{GRAPH}.json", "w"), indent=2)
    make_fig(out, f"{OUTDIR}/basis_cossim_{GRAPH}.pdf")
    print(f"DONE -> {OUTDIR}/basis_cossim_{GRAPH}.pdf", flush=True)


def make_fig(out, path):
    order = ["Llama", "Gemma", "Qwen"]; models = [m for m in order if m in out["models"]] + [m for m in out["models"] if m not in order]
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.6), squeeze=False)
        for j, m in enumerate(models):
            r = out["models"][m]; L = list(range(r["n_layers"]))
            ax[0, j].plot(L, r["cos_principal_1"], "-o", ms=3, color="tab:green", label="cos ∠₁ (subspace)")
            ax[0, j].plot(L, r["cos_principal_2"], "-o", ms=3, color="tab:olive", label="cos ∠₂")
            ax2 = ax[0, j].twinx(); ax2.plot(L, r["probe_r2"], "--", color="tab:gray", lw=1, label="probe LOO R²")
            ax2.set_ylim(-0.6, 1.0); ax2.set_ylabel("probe R²", color="gray")
            ax[0, j].set_ylim(0, 1.02); ax[0, j].set_xlabel("layer"); ax[0, j].set_ylabel("cos(principal angle)")
            ax[0, j].set_title(m, fontsize=9); ax[0, j].legend(fontsize=7, loc="lower center")
        fig.suptitle(f"[{out['graph']}] RSA best-2D subspace vs coord-probe readout subspace — cos of principal angles "
                     "(1 = identical 2-D subspace)", fontsize=10)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
