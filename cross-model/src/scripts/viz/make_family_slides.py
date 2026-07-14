"""Per-family slide deck: page 1 = a clear picture of the Markov chain (force-directed graph, node
colour = index, arrows implied by the uniform transition) + its transition matrix; then one page per
LAYER (finest common depth grid) of node-mean 2-D PCA, Llama|Gemma|Qwen side by side (1x3), edges
overlaid so you can see where each model recovers the layout.

Reads nodemeans_<TAG>_<fam>.npz from the markov_families dir (all 3 models). CPU-only.
Env: FAM MFDIR OUTDIR
Out: <OUTDIR>/family_<fam>.pdf
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

FAM = os.environ.get("FAM", "grid")
MFDIR = os.environ.get("MFDIR", "runs/axes/1_decomposition/markov_families")
OUTDIR = os.environ.get("OUTDIR", "runs/axes/1_decomposition/family_slides")
MODELS = ["Llama", "Gemma", "Qwen"]


def load(tag):
    p = f"{MFDIR}/nodemeans_{tag}_{FAM}.npz"
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None


def spring_layout(A, iters=400):
    """Fruchterman-Reingold force-directed layout, spectral-initialised (deterministic)."""
    n = A.shape[0]; L = np.diag(A.sum(1)) - A; w, V = np.linalg.eigh(L)
    pos = V[:, 1:3].copy(); pos = pos / (np.abs(pos).max() + 1e-9)
    k = 1.3 / np.sqrt(n)
    for it in range(iters):
        d = pos[:, None, :] - pos[None, :, :]; dist = np.linalg.norm(d, axis=2) + 1e-9
        rep = (k * k / (dist * dist))[:, :, None] * d                       # repel all
        att = (A * dist / k)[:, :, None] * (-d) / dist[:, :, None]          # attract along edges
        disp = rep.sum(1) + att.sum(1)
        ln = np.linalg.norm(disp, axis=1, keepdims=True) + 1e-9
        temp = 0.1 * (1 - it / iters) + 0.005
        pos = pos + disp / ln * np.minimum(ln, temp)
    pos -= pos.mean(0); pos /= np.abs(pos).max() + 1e-9
    return pos


def pca2(H):
    Hc = H - H.mean(0); U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    return Hc @ Vt[:2].T, (S[:2] ** 2 / (S ** 2).sum())


def draw_graph(ax, A, pos, edges, title):
    n = A.shape[0]; deg = A.sum(1)
    label = len(edges) <= 40                                   # skip labels on dense graphs (clutter)
    for i, j in edges:
        pi, pj = 1.0 / deg[i], 1.0 / deg[j]
        ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], "-", color=".6",
                lw=0.6 + 3.0 * (pi + pj) / 2, zorder=1)         # width ∝ transition probability
        if label:
            mx, my = (pos[i, 0] + pos[j, 0]) / 2, (pos[i, 1] + pos[j, 1]) / 2
            txt = f"{pi:.2f}" if abs(pi - pj) < 1e-3 else f"{pi:.2f}|{pj:.2f}"
            ax.text(mx, my, txt, fontsize=5.5, ha="center", va="center", zorder=2,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=.75))
    ax.scatter(pos[:, 0], pos[:, 1], c=np.arange(n), cmap="turbo", s=340, edgecolors="k", lw=.8, zorder=3)
    for i in range(n): ax.text(pos[i, 0], pos[i, 1], str(i), fontsize=8, ha="center", va="center", zorder=4, fontweight="bold")
    ax.set_title(title, fontsize=11); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax.margins(0.12)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    data = {m: load(m) for m in MODELS}
    ref = next((d for d in data.values() if d is not None), None)
    if ref is None:
        print(f"[{FAM}] no node-means found"); return
    A = np.array(ref["adjacency"], float); n = A.shape[0]
    deg = A.sum(1); T = A / np.maximum(deg[:, None], 1)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j]]
    pos = spring_layout(A)
    nLs = {m: (sum(1 for k in data[m].files if k.startswith("layer_")) if data[m] is not None else 0) for m in MODELS}
    maxnL = max(nLs.values())

    with PdfPages(f"{OUTDIR}/family_{FAM}.pdf") as pdf:
        # page 1: the chain (clear) + transition matrix
        fig, ax = plt.subplots(1, 2, figsize=(13, 5.8))
        draw_graph(ax[0], A, pos, edges, f"{FAM}: the Markov chain\n{n} states, {len(edges)} transitions, mean degree {deg.mean():.1f}")
        im = ax[1].imshow(T, cmap="magma", vmin=0)
        ax[1].set_title("transition matrix\nP(next=j | current=i) = 1/deg(i)", fontsize=11)
        ax[1].set_xlabel("next state j"); ax[1].set_ylabel("current state i")
        fig.colorbar(im, ax=ax[1], fraction=.046)
        fig.suptitle(f"Markov family: {FAM}   (uniform random walks; edge width/label = transition probability 1/deg)", fontsize=12)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # one page per layer (finest common depth grid = maxnL steps)
        for step in range(maxnL):
            frac = step / max(maxnL - 1, 1)
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            for k, m in enumerate(MODELS):
                dm = data[m]
                if dm is None:
                    ax[k].text(.5, .5, f"{m}: not captured", ha="center", va="center"); ax[k].axis("off"); continue
                nL = nLs[m]; L = int(round(frac * (nL - 1)))
                P, evr = pca2(dm[f"layer_{L}"].astype(float))
                for i, j in edges: ax[k].plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], "-", color=".82", lw=.6, zorder=1)
                ax[k].scatter(P[:, 0], P[:, 1], c=np.arange(n), cmap="turbo", s=120, edgecolors="k", lw=.4, zorder=2)
                for i in range(n): ax[k].text(P[i, 0], P[i, 1], str(i), fontsize=6, ha="center", va="center", zorder=3)
                ax[k].set_title(f"{m}  layer {L}/{nL-1}  (PC var {evr[0]*100:.0f}·{evr[1]*100:.0f}%)", fontsize=9)
                ax[k].set_xticks([]); ax[k].set_yticks([])
            fig.suptitle(f"{FAM} — node-mean PCA, depth {frac:.0%}   (edges = graph adjacency)", fontsize=11)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(f"[{FAM}] DONE ({maxnL} layer pages) -> {OUTDIR}/family_{FAM}.pdf", flush=True)


if __name__ == "__main__":
    main()
