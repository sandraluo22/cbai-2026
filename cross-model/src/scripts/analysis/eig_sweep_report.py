"""Report for head_eig_sweep: correlation matrix (page 1) + all eigenmode damage maps (page 2) + an
INTERPRETATION page (page 3) that names each head-circuit — which heads build 'position', 'parity',
'community', etc. — by (a) clustering eigenmodes into circuits (shared-head correlation) and
(b) labelling each mode by correlating its eigenvector with known structure (coords / 2-colouring /
community membership).

Reads head_eig_sweep_<TAG>_<fam>.json + adjacency from the markov_families node-means npz. CPU-only.
Env: TAG(Llama) FAM(grid) HE MFDIR
"""
from __future__ import annotations
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

TAG = os.environ.get("TAG", "Llama"); FAM = os.environ.get("FAM", "grid")
HE = os.environ.get("HE", "runs/axes/4_circuits/head_eig_sweep")
MFDIR = os.environ.get("MFDIR", "runs/axes/1_decomposition/markov_families")


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


def canon_coords(fam, n):
    if fam == "grid": return np.array([[i // 4, i % 4] for i in range(n)], float)
    if fam == "ring": return np.array([[np.cos(2 * np.pi * i / n), np.sin(2 * np.pi * i / n)] for i in range(n)], float)
    return None


def label_mode(vk, wk, fam, A, coords, wmin, wmax):
    labs = []
    if coords is not None:
        for ci, nm in enumerate(["x-position", "y-position"]):
            c = coords[:, ci] - coords[:, ci].mean(); c /= np.linalg.norm(c) + 1e-9
            if abs(vk @ c) > 0.6: labs.append(nm)
    par = two_colour(A)
    if par is not None:
        p = par - par.mean(); p /= np.linalg.norm(p) + 1e-9
        if abs(vk @ p) > 0.6: labs.append("parity/checkerboard")
    if fam.startswith("sbm"):
        k = 2 if fam == "sbm2" else 4; block = np.repeat(np.arange(k), len(vk) // k).astype(float)
        b = block - block.mean(); b /= np.linalg.norm(b) + 1e-9
        if abs(vk @ b) > 0.6: labs.append("community split")
    if not labs:
        band = "low-freq" if wk < wmin + (wmax - wmin) / 3 else ("high-freq" if wk > wmax - (wmax - wmin) / 3 else "mid-freq")
        labs.append(f"{band} (no simple name)")
    return " + ".join(labs)


def cluster(C, strong, thr=0.5):
    """greedy: group strong modes whose damage-map correlation > thr."""
    groups = []; seen = set()
    for k in strong:
        if k in seen: continue
        g = [k]; seen.add(k)
        for j in strong:
            if j not in seen and C[k, j] > thr: g.append(j); seen.add(j)
        groups.append(g)
    return groups


def main():
    d = json.load(open(f"{HE}/head_eig_sweep_{TAG}_{FAM}.json"))
    w = np.array(d["eigenvalues"]); cp = np.array(d["clean_power"]); C = np.array(d["corr"])
    D = np.array(d["damage"]); nE = D.shape[0]
    z = np.load(f"{MFDIR}/nodemeans_{TAG}_{FAM}.npz", allow_pickle=True); A = np.array(z["adjacency"], float); n = A.shape[0]
    L = np.diag(A.sum(1)) - A; ww, V = np.linalg.eigh(L); Vn = V[:, 1:]      # eigenvectors of non-trivial modes
    coords = canon_coords(FAM, n)
    strong = [k for k in range(nE) if cp[k] > 0.05]
    groups = cluster(C, strong)

    with PdfPages(f"{HE}/head_eig_sweep_{TAG}_{FAM}_report.pdf") as pdf:
        # page 1: correlation matrix
        fig, ax = plt.subplots(figsize=(7, 6)); im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"{TAG} {FAM}: eigenmode circuit-overlap\n(corr of per-mode single-head damage; 1=same heads)", fontsize=9)
        ax.set_xlabel("eigenmode (freq→)"); ax.set_ylabel("eigenmode (freq→)"); fig.colorbar(im, ax=ax, fraction=.046)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        # page 2: all damage maps
        ncol = 5; nrow = int(np.ceil(nE / ncol)); fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.3 * nrow))
        for k, ax in enumerate(np.array(axes).flat):
            if k >= nE: ax.axis("off"); continue
            Dk = D[k]; lim = np.abs(Dk).max() + 1e-9; ax.imshow(Dk, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.set_title(f"m{k+1} λ{w[k]:.1f} p{cp[k]:.2f}", fontsize=7); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{TAG} {FAM}: single-head damage per eigenmode (red=head builds it)", fontsize=10)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        # page 3: interpretation summary
        fig = plt.figure(figsize=(11, 8.5)); ax = fig.add_subplot(111); ax.axis("off")
        lines = [f"{TAG} — {FAM}:  which heads build which structure", ""]
        wmin, wmax = w.min(), w.max()
        for gi, g in enumerate(sorted(groups, key=lambda gg: -cp[gg].sum()), 1):
            g = sorted(g, key=lambda k: -cp[k])
            names = {label_mode(Vn[:, k], w[k], FAM, A, coords, wmin, wmax) for k in g}
            heads = []
            for k in g[:2]:
                heads += [(l, h) for l, h, _ in d["top_heads"][str(k)][:3]]
            heads = list(dict.fromkeys(heads))[:5]
            pw = cp[g].sum()
            lines.append(f"CIRCUIT {gi}  —  {' / '.join(sorted(names))}")
            lines.append(f"    modes {[k+1 for k in g]} (λ {', '.join(f'{w[k]:.1f}' for k in g)}),  total power {pw:.2f}")
            lines.append(f"    top heads: " + ", ".join(f"L{l}H{h}" for l, h in heads))
            lines.append("")
        # note anti-correlations between circuits
        gl = sorted(groups, key=lambda gg: -cp[gg].sum())
        if len(gl) >= 2:
            cc = np.mean([C[a, b] for a in gl[0] for b in gl[1]])
            lines.append(f"Circuit 1 vs Circuit 2 cross-correlation: {cc:+.2f}  "
                         + ("(anti-correlated → compete for variance)" if cc < -0.2 else "(≈independent)"))
        ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)
        pdf.savefig(fig); plt.close(fig)
    print(f"[{TAG}/{FAM}] report -> {HE}/head_eig_sweep_{TAG}_{FAM}_report.pdf  ({len(groups)} circuits)", flush=True)
    for g in sorted(groups, key=lambda gg: -cp[gg].sum()):
        names = {label_mode(Vn[:, k], w[k], FAM, A, coords, w.min(), w.max()) for k in g}
        print(f"   circuit modes {[k+1 for k in g]}: {' / '.join(sorted(names))}")


if __name__ == "__main__":
    main()
