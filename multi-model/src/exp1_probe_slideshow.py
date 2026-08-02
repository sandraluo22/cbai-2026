"""Per-layer probe slideshow for Exp1's transfer condition.

For EVERY Qwen layer, one page with two scatter subplots of the 16 node-means
that Qwen builds while reading **Llama's generated walk**, each decoded into
predicted grid (row, col):

  LEFT  — "old" / reference probe: a coord probe fit on Qwen's REAL-WALK
          node-means (the grid Qwen learns from genuine walks), then applied to
          the llama_gen node-means. No leakage (fit and eval are different
          conditions). Tests: does the probe learned from real walks read the
          grid out of the *transferred* representation?
  RIGHT — "newly fitted" probe: a fresh LEAVE-ONE-NODE-OUT probe on the
          llama_gen node-means themselves (each node predicted from the other
          15). The honest in-condition recovery.

Points are placed at their PREDICTED (col, row); true grid edges are drawn
between them, so a recovered grid looks like a clean 4x4 lattice and a failed
one looks tangled. Panel titles show the decode R².

Capture (needs the GPU + both models) runs once and caches the node-means to
`<RUN_DIR>/probe_slideshow_nodemeans.npz`; re-plotting is CPU-only from that
cache (delete it to recapture). Mirrors Exp1's knobs exactly so the walks match.

Env: PRESET GRAPH NSEED XCTX GSTEPS NWALKS_REAL WLEN_REAL CTXLO TEMP RUN_DIR DEVICE
Out: <RUN_DIR>/exp1_probe_slideshow.pdf  (+ probe_slideshow_nodemeans.npz)
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import common as C  # noqa: E402
import graph as G   # noqa: E402

GRAPH = os.environ.get("GRAPH", "square_grid")
NSEED = int(os.environ.get("NSEED", "6" if C.PRESET != "smoke" else "3"))
XCTX = int(os.environ.get("XCTX", "80" if C.PRESET != "smoke" else "15"))
GSTEPS = int(os.environ.get("GSTEPS", "220" if C.PRESET != "smoke" else "40"))
NWALKS_REAL = int(os.environ.get("NWALKS_REAL", "12" if C.PRESET != "smoke" else "4"))
WLEN_REAL = int(os.environ.get("WLEN_REAL", "300" if C.PRESET != "smoke" else "40"))
CTXLO = int(os.environ.get("CTXLO", "100" if C.PRESET != "smoke" else "5"))
TEMP = float(os.environ.get("TEMP", "1.0"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/main")
NPZ = os.path.join(RUN_DIR, "probe_slideshow_nodemeans.npz")
ALPHAS = C.ALPHAS


# ---- ridge probe (full fit, and LOO) on 16 node-means -----------------------
def _standardize(X):
    mu = X.mean(0); sd = X.std(0) + 1e-6
    return (X - mu) / sd, mu, sd


def fit_full(X, Y, alpha):
    Xs, mu, sd = _standardize(X)
    ymu = Y.mean(0); Yc = Y - ymu
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    coef = Vt.T @ ((S / (S ** 2 + alpha))[:, None] * (U.T @ Yc))
    return {"coef": coef, "mu": mu, "sd": sd, "ymu": ymu}


def predict(P, X):
    return ((X - P["mu"]) / P["sd"]) @ P["coef"] + P["ymu"]


def best_alpha_loo(X, Y):
    """Pick alpha maximizing mean LOO R^2 on (X,Y); return alpha and LOO preds."""
    folds = C._prep_folds(X); n = len(folds)
    best = (-9.0, None, None)
    for a in ALPHAS:
        pred = np.zeros((n, 2))
        for k, (idx, proj, UT, S) in enumerate(folds):
            ytr = Y[idx]; ymu = ytr.mean(0)
            pred[k] = proj @ ((S / (S ** 2 + a))[:, None] * (UT @ (ytr - ymu))) + ymu
        score = C._r2(Y[:, 0], pred[:, 0]) + C._r2(Y[:, 1], pred[:, 1])
        if score > best[0]:
            best = (score, a, pred.copy())
    return best[1], best[2]


def r2_pair(Y, P):
    return C._r2(Y[:, 0], P[:, 0]), C._r2(Y[:, 1], P[:, 1])


# ---- capture (GPU) ----------------------------------------------------------
def capture():
    dev = C.default_device()
    cfg = C.make_cfg(GRAPH, n_walks=max(NSEED, NWALKS_REAL, 8),
                     walk_length=max(XCTX, WLEN_REAL), device=dev)
    graph, n, coords = C.build_grid(cfg)
    real_cfg = C.make_cfg(GRAPH, n_walks=NWALKS_REAL, walk_length=WLEN_REAL, device=dev)
    real_walks = G.generate_walks(graph, real_cfg)
    seeds = G.generate_walks(graph, cfg)[:NSEED]

    print("[slideshow] loading Llama, regenerating walks (matches Exp1 seeds)", flush=True)
    llama, ltok = C.load_model("Llama", cfg)
    cand = C.candidate_token_ids(ltok, graph, dev)
    gen_walks = []
    for si, seed in enumerate(seeds):
        nodes, _ = C.generate_walk(llama, ltok, graph, cand, dev, seed.nodes[:XCTX], GSTEPS,
                                   temp=TEMP, rng=np.random.default_rng(1000 + si))
        gen_walks.append(C.mkwalk(nodes, graph))
    C.free(llama, ltok)

    print("[slideshow] loading Qwen, capturing node-means (all layers)", flush=True)
    qwen, qtok = C.load_model("Qwen", cfg)
    gen_nm, _ = C.node_means_all_layers(qwen, qtok, graph, gen_walks, dev, n, ctxlo=CTXLO)
    real_nm, _ = C.node_means_all_layers(qwen, qtok, graph, real_walks, dev, n, ctxlo=CTXLO)
    C.free(qwen, qtok)

    nL = len(gen_nm)
    gen_arr = np.stack([gen_nm[L] for L in range(nL)])      # (nL,16,H)
    real_arr = np.stack([real_nm[L] for L in range(nL)])
    os.makedirs(RUN_DIR, exist_ok=True)
    np.savez_compressed(NPZ, gen=gen_arr.astype(np.float32), real=real_arr.astype(np.float32),
                        coords=coords, adjacency=np.array(_pad_adj(graph), dtype=object),
                        edges=np.array(_edge_list(graph)))
    print(f"[slideshow] cached node-means -> {NPZ}  (nL={nL})", flush=True)


def _edge_list(graph):
    es = []
    for u in range(graph.n_nodes):
        for v in graph.adjacency[u]:
            if u < v:
                es.append((u, v))
    return es


def _pad_adj(graph):
    return [list(a) for a in graph.adjacency]


# ---- plotting (CPU, from cache) ---------------------------------------------
def draw_panel(ax, pred, coords, edges, title, words):
    # edges between predicted positions of true neighbours; plot as (col,row)
    for (u, v) in edges:
        ax.plot([pred[u, 1], pred[v, 1]], [pred[u, 0], pred[v, 0]],
                "-", color="0.75", lw=0.8, zorder=1)
    sc = ax.scatter(pred[:, 1], pred[:, 0], c=coords[:, 0], cmap="viridis",
                    s=90, zorder=2, edgecolors="k", linewidths=0.4)
    for i, w in enumerate(words):
        ax.annotate(w, (pred[i, 1], pred[i, 0]), fontsize=6, ha="center",
                    va="center", xytext=(0, 8), textcoords="offset points")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("predicted col"); ax.set_ylabel("predicted row")
    ax.invert_yaxis()
    return sc


def plot():
    z = np.load(NPZ, allow_pickle=True)
    gen = z["gen"]; real = z["real"]; coords = z["coords"]; edges = [tuple(e) for e in z["edges"]]
    nL = gen.shape[0]
    # words: cross-model WORDS order for a 4x4 grid
    from config import WORDS
    words = WORDS[:gen.shape[1]]

    old_r2, new_r2 = [], []
    pages = []
    for L in range(nL):
        Xg, Xr = gen[L], real[L]
        # old/reference probe: fit on real-walk node-means, apply to llama_gen
        a_ref, _ = best_alpha_loo(Xr, coords)
        Pref = fit_full(Xr, coords, a_ref)
        pred_old = predict(Pref, Xg)
        r_old = r2_pair(coords, pred_old)
        # new probe: LOO on llama_gen node-means
        _, pred_new = best_alpha_loo(Xg, coords)
        r_new = r2_pair(coords, pred_new)
        old_r2.append(np.mean(r_old)); new_r2.append(np.mean(r_new))
        pages.append((L, pred_old, r_old, pred_new, r_new))

    peak = int(np.nanargmax(new_r2))
    with PdfPages(os.path.join(RUN_DIR, "exp1_probe_slideshow.pdf")) as pdf:
        # overview page
        fig, ax = plt.subplots(figsize=(8, 4.6), dpi=120)
        ax.plot(range(nL), old_r2, "-o", ms=3, color="tab:purple",
                label="old/reference probe (fit on real walk)")
        ax.plot(range(nL), new_r2, "-o", ms=3, color="tab:red",
                label="newly fitted probe (LOO on llama_gen)")
        ax.axhline(0, color=".7", lw=.6); ax.axvline(peak, color="tab:red", ls=":", lw=1)
        ax.set_ylim(-0.6, 1.0); ax.set_xlabel("Qwen layer"); ax.set_ylabel("mean grid R²")
        ax.set_title("Qwen grid recovery from Llama-generated walk: reference vs fresh probe", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        # per-layer pages
        for (L, pred_old, r_old, pred_new, r_new) in pages:
            fig, ax = plt.subplots(1, 2, figsize=(11, 5.2), dpi=120)
            draw_panel(ax[0], pred_old, coords, edges,
                       f"OLD probe (fit on real walk → applied to llama_gen)\n"
                       f"row R²={r_old[0]:+.2f}  col R²={r_old[1]:+.2f}", words)
            draw_panel(ax[1], pred_new, coords, edges,
                       f"NEW probe (LOO on llama_gen)\n"
                       f"row R²={r_new[0]:+.2f}  col R²={r_new[1]:+.2f}", words)
            fig.suptitle(f"Qwen LAYER {L}/{gen.shape[0]-1} — 16 node-means from Llama's generated walk, "
                         f"decoded into grid coords", fontsize=10)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(f"[slideshow] wrote {RUN_DIR}/exp1_probe_slideshow.pdf "
          f"({nL+1} pages; peak new-probe L{peak} R²={new_r2[peak]:.2f})", flush=True)


def main():
    if not os.path.exists(NPZ):
        capture()
    plot()


if __name__ == "__main__":
    main()
