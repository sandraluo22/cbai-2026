"""Experiment 3 -- eigenmode purpose index. For an R x C grid, the graph Laplacian
eigenvectors are outer (Kronecker) products of the two 1-D path-graph eigenmodes,
each a cosine of rising frequency: const, grad (coordinate ramp), fold (U-shape),
alt (parity alternation). This builds the |cos-similarity| matrix between every grid
eigenmode m0..m{n-1} and every Kronecker product (row-mode x col-mode), so each grid
mode's "purpose" (e.g. alt x fold, grad x const = coordinate) is read off directly.

Modes are DEGENERATE by eigenvalue -> within a degenerate group eigh returns an
arbitrary rotation, so only the SUBSPACE has a clean label; the heatmap shows this as
a mode matching several products at ~0.7 rather than one at ~1.0. We annotate each grid
mode with its eigenvalue and the auto-label (parity / coord / product / other).

Env: ROWS(4) COLS(4) OUTDIR(runs/axes/1_decomposition/eigmode_index)
Pure graph math -- no model, no GPU.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

ROWS = int(os.environ.get("ROWS", "4")); COLS = int(os.environ.get("COLS", "4"))
OUTDIR = os.environ.get("OUTDIR", "runs/axes/1_decomposition/eigmode_index")
BANDS = ["const", "grad", "fold", "alt"]                      # 1-D path modes, ascending frequency


def path_modes(m):
    d = np.ones(m); d[1:-1] = 2
    L = np.diag(d) - (np.abs(np.subtract.outer(range(m), range(m))) == 1).astype(float)
    w, U = np.linalg.eigh(L)                                  # ascending: const, grad, fold, alt, ...
    return w, U


def grid_laplacian_modes(rows, cols):
    n = rows * cols
    coords = np.array([[i // cols, i % cols] for i in range(n)], float)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if abs(coords[i, 0] - coords[j, 0]) + abs(coords[i, 1] - coords[j, 1]) == 1:
                A[i, j] = 1.0
    d = A.sum(1); di = 1 / np.sqrt(d); L = np.eye(n) - di[:, None] * A * di[None, :]
    return np.linalg.eigh(L)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    n = ROWS * COLS
    w, U = grid_laplacian_modes(ROWS, COLS)
    wr, Ur = path_modes(ROWS); wc, Uc = path_modes(COLS)

    # candidate Kronecker products, labelled band(row) x band(col)
    cand_vecs, cand_names = [], []
    for a in range(ROWS):
        for b in range(COLS):
            cand_vecs.append(np.outer(Ur[:, a], Uc[:, b]).ravel())
            cand_names.append(f"{BANDS[a]}×{BANDS[b]}")   # e.g. alt×fold
    cand = np.array(cand_vecs)

    def unit(x): x = x - x.mean(); nx = np.linalg.norm(x); return x / (nx + 1e-12)
    Un = np.array([unit(U[:, k]) for k in range(n)])
    Cn = np.array([unit(c) for c in cand])
    S = np.abs(Un @ Cn.T)                                    # |similarity| grid-mode x product

    # auto-label each grid mode from the coordinate (grad×const/const×grad) and parity (alt×alt) refs
    def semantic(bands):                                     # {const,grad,fold,alt} pair -> human tag
        s = set(bands)
        if bands == ("const", "const"): return "trivial"
        if s == {"alt"}:                return "parity"                       # alt×alt = checkerboard
        if s <= {"const", "grad"}:      return "coord"                        # ramp along one/both axes
        if "alt" in s and "grad" in s:  return "parity×coord"
        if "alt" in s and "fold" in s:  return "parity×fold"
        if s <= {"const", "fold"}:      return "fold"
        return "×".join(bands)
    labels = []
    for k in range(n):
        if w[k] < 1e-6:                                      # DC mode (removed by centering everywhere)
            labels.append("trivial"); continue
        a, b = cand_names[int(S[k].argmax())].split("×")
        labels.append(semantic((a, b)))

    # ---- heatmap ----
    fig, ax = plt.subplots(figsize=(1.0 + 0.42 * len(cand_names), 0.9 + 0.34 * n))
    im = ax.imshow(S, cmap="magma", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cand_names))); ax.set_xticklabels(cand_names, rotation=90, fontsize=6)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"m{k}  λ={w[k]:.2f}  [{labels[k]}]" for k in range(n)], fontsize=6.5)
    for k in range(n):                                       # star the best match per mode
        ax.text(S[k].argmax(), k, "★", ha="center", va="center", color="#39FF14", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.025, label="|cosine similarity|")
    ax.set_title(f"Eigenmode purpose index -- {ROWS}×{COLS} grid (Laplacian mode × Kronecker product)", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf",):
        out = f"{OUTDIR}/eigmode_index_{ROWS}x{COLS}.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)

    # ---- JSON index ----
    idx = {"rows": ROWS, "cols": COLS, "products": cand_names,
           "modes": [{"mode": k, "eigenvalue": float(w[k]), "label": labels[k],
                      "best_match": cand_names[int(S[k].argmax())],
                      "best_sim": float(S[k].max()),
                      "top3": sorted([(round(float(S[k, j]), 3), cand_names[j]) for j in range(len(cand_names))],
                                     reverse=True)[:3]} for k in range(n)]}
    json.dump(idx, open(f"{OUTDIR}/eigmode_index_{ROWS}x{COLS}.json", "w"), indent=2)
    print(f"DONE -> {OUTDIR}/eigmode_index_{ROWS}x{COLS}.json")


if __name__ == "__main__":
    main()
