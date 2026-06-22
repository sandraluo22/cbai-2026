"""Two controls, done properly, offline.

(1) CLEAN matched-vs-mismatched: fix the trivial version. Instead of scoring
    A@C predictions against random *unpaired* B@C' vectors (which fails just
    because the items are unrelated), we map A@C through the ridge map and ask
    which Qwen CONTEXT-centroid the prediction is nearest to. A diagonal-heavy
    confusion matrix = the map preserves context; off-diagonal = it doesn't.
    Centroids are well-defined across contexts, so no unpaired artifact.

(2) PAPER's control: the p_seen memorization baseline (Fig 5). Overlay the
    models' rule-following accuracy (neighbour mass) and the strong in-context
    counter against p_seen1/p_seen2 to see what beats memorization.
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import replace

from config import get_config
import graph as G
import align as A

CFG = get_config("gemma_qwen")
CHK = list(CFG.context_checkpoints)            # 10,30,100,300,1000
N = 16


# ---------- (1) clean matched vs mismatched ----------
def clean_match_mismatch():
    za = np.load("runs/square_grid/gemma_qwen/acts_model_a.npz", allow_pickle=False)
    zb = np.load("runs/square_grid/gemma_qwen/acts_model_b.npz", allow_pickle=False)
    meta = {k[5:]: za[k] for k in za.files if k.startswith("meta_")}
    rng = np.random.default_rng(0)
    sub = np.sort(rng.choice(meta["walk_id"].shape[0], 60000, replace=False))
    XA = za["layer_32"][sub].astype(np.float64)
    XB = zb["layer_28"][sub].astype(np.float64)
    meta = {k: v[sub] for k, v in meta.items()}

    train, test = A.split_by_walk(meta, CFG.test_frac, CFG.seed)
    rm = A.fit_ridge(XA[train], XB[train], CFG.ridge_alpha)

    bm = lambda C: test & A.context_bin_mask(meta, C, CFG.checkpoint_window)
    qcent = np.stack([XB[bm(C)].mean(0) for C in CHK])      # Qwen centroid per ctx

    conf = np.zeros((len(CHK), len(CHK)))
    for i, C in enumerate(CHK):
        pred = rm.predict(XA[bm(C)])                        # map Gemma@C -> Qwen-space
        d = ((pred[:, None, :] - qcent[None, :, :]) ** 2).sum(-1)
        nn = d.argmin(1)                                    # nearest Qwen-ctx centroid
        for j in range(len(CHK)):
            conf[i, j] = (nn == j).mean()
    return conf


# ---------- (2) paper's p_seen control + counter ----------
def pseen_control():
    acc = json.load(open("runs/square_grid/accuracy/accuracy.json"))
    ctxs = [r["ctx"] for r in acc["gemma"]["by_context"]]
    gm = [r["neighbor_mass"] for r in acc["gemma"]["by_context"]]
    qm = [r["neighbor_mass"] for r in acc["qwen"]["by_context"]]
    ps1 = [1 - ((N - 1) / N) ** l for l in ctxs]
    ps2 = [ps1[k] - ctxs[k] * (1 / N) * ((N - 1) / N) ** (ctxs[k] - 1)
           for k in range(len(ctxs))]

    # strong in-context counter (neighbour mass)
    cfg40 = replace(CFG, n_walks=40)
    graph = G.build_grid_graph(cfg40)
    walks = G.generate_walks(graph, cfg40)
    bins = [(1, 15), (15, 50), (50, 150), (150, 500), (500, 1001)]
    recs = []
    for wk in walks:
        counts = np.zeros((16, 16))
        for s in range(len(wk.nodes) - 1):
            cur, nxt = wk.nodes[s], wk.nodes[s + 1]
            c = counts[cur]
            mass = 1.0 if c.sum() > 0 else len(graph.neighbors(cur)) / 16
            recs.append((s + 1, mass)); counts[cur, nxt] += 1
    recs = np.array(recs)
    cm = [recs[(recs[:, 0] >= lo) & (recs[:, 0] < hi)][:, 1].mean() for lo, hi in bins]
    return ctxs, gm, qm, cm, ps1, ps2


def main():
    conf = clean_match_mismatch()
    ctxs, gm, qm, cm, ps1, ps2 = pseen_control()

    print("=== (1) clean matched-vs-mismatched: confusion matrix ===")
    print("rows = Gemma context, cols = nearest Qwen context-centroid of prediction")
    print("        " + "  ".join(f"Q{c:>4}" for c in CHK))
    for i, C in enumerate(CHK):
        print(f"G{C:>4}   " + "  ".join(f"{conf[i,j]:.2f}" for j in range(len(CHK))))
    print(f"matched (diagonal) mean = {np.diag(conf).mean():.3f}  "
          f"(chance = {1/len(CHK):.2f})")

    print("\n=== (2) paper p_seen control (rule-following = neighbour mass) ===")
    print(f"{'ctx':>5} {'Gemma':>7} {'Qwen':>7} {'counter':>8} {'p_seen1':>8} {'p_seen2':>8}")
    for k, l in enumerate(ctxs):
        print(f"{l:>5} {gm[k]:>7.3f} {qm[k]:>7.3f} {cm[k]:>8.3f} {ps1[k]:>8.3f} {ps2[k]:>8.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    im = ax[0].imshow(conf, vmin=0, vmax=1, cmap="viridis", origin="lower")
    ax[0].set_xticks(range(len(CHK))); ax[0].set_xticklabels(CHK)
    ax[0].set_yticks(range(len(CHK))); ax[0].set_yticklabels(CHK)
    ax[0].set_xlabel("nearest Qwen context-centroid"); ax[0].set_ylabel("Gemma context")
    ax[0].set_title("(1) Clean matched-vs-mismatched\n(diagonal = map preserves context)")
    for i in range(len(CHK)):
        for j in range(len(CHK)):
            ax[0].text(j, i, f"{conf[i,j]:.2f}", ha="center", va="center",
                       color="w" if conf[i, j] < 0.5 else "k", fontsize=8)
    fig.colorbar(im, ax=ax[0], fraction=.046)

    ax[1].plot(ctxs, gm, "-o", label="Gemma (model)")
    ax[1].plot(ctxs, qm, "-o", label="Qwen (model)")
    ax[1].plot(ctxs, cm, "-s", label="in-context counter")
    ax[1].plot(ctxs, ps1, "--", label="p_seen1 (paper)")
    ax[1].plot(ctxs, ps2, "--", label="p_seen2 (paper)")
    ax[1].set_xscale("log"); ax[1].set_xlabel("context length"); ax[1].set_ylabel("neighbour mass")
    ax[1].set_title("(2) Paper p_seen memorization control"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig("runs/square_grid/gemma_qwen/controls_redo.png", dpi=140)
    print("\nwrote runs/square_grid/gemma_qwen/controls_redo.png")


if __name__ == "__main__":
    main()
