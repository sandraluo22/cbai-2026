"""Visualizations for the days-of-week (semantic-prior) condition.

Fig 1 (dual structure): per model at a deep layer, PC1-2 with the NATURAL
weekday cycle drawn (green) and PC3-4 with the IN-CONTEXT ring drawn (purple),
so you can see the pretrained order owns the top PCs and the context structure
lives below it.

Fig 2 (per-layer sweep): how PC1-2 weekday-RSA and PC3-4 context-RSA vary by
layer for each model.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import replace

from config import get_config, DAYS
import graph as G

cfg = replace(get_config("gemma_qwen"), graph_type="ring", ring_size=7, word_set="days")
GRAPH = G.build_graph(cfg)
WORDS = GRAPH.words
N = 7
IU = np.triu_indices(N, 1)
CTX_D = GRAPH.distance_matrix()[IU]
nat = [DAYS.index(w) for w in WORDS]
SEM_D = np.array([[min(abs(nat[i] - nat[j]), 7 - abs(nat[i] - nat[j]))
                   for j in range(N)] for i in range(N)])[IU]
SEM_ADJ = [[j for j in range(N) if (abs(nat[i] - nat[j]) % 7) in (1, 6)] for i in range(N)]
MODELS = ["Llama", "Gemma", "Qwen"]


def spearman(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def node_means(z, L, node, mask):
    X = z[f"layer_{L}"].astype(np.float32)
    H = np.full((N, X.shape[1]), np.nan, np.float32)
    for k in range(N):
        m = mask & (node == k)
        if m.any():
            H[k] = X[m].mean(0)
    return H


def scores_of(H):
    Hc = H - H.mean(0)
    _, _, Vt = np.linalg.svd(Hc, full_matrices=False)
    return Hc @ Vt.T


def pc_rsa(scores, pcs, target):
    s = scores[:, pcs]
    d = np.linalg.norm(s[:, None, :] - s[None, :, :], axis=2)[IU]
    return spearman(d, target)


def load(m):
    z = np.load(f"runs/days/{m}_acts_sub.npz", allow_pickle=False)
    layers = [int(l) for l in z["_layers"]]
    return z, layers, z["meta_node"], z["meta_context_length"] >= 300


# ---------------- Fig 1: dual structure ----------------
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for col, m in enumerate(MODELS):
    z, layers, node, mask = load(m)
    deep = min(layers, key=lambda L: abs(L - 0.8 * max(layers)))
    sc = scores_of(node_means(z, deep, node, mask))
    # top: PC1-2 with weekday-cycle edges (green)
    ax = axes[0, col]; s = sc[:, [0, 1]]
    for i in range(N):
        for j in SEM_ADJ[i]:
            if j > i:
                ax.plot([s[i, 0], s[j, 0]], [s[i, 1], s[j, 1]], color="green", lw=1.4, alpha=.6, zorder=1)
    for i in range(N):
        ax.scatter(*s[i], color="k", zorder=2); ax.annotate(WORDS[i], s[i], fontsize=8)
    ax.set_title(f"{m} L{deep}  PC1-2  (weekday-RSA={pc_rsa(sc,[0,1],SEM_D):+.2f})", fontsize=9)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_xticks([]); ax.set_yticks([])
    # bottom: PC3-4 with in-context ring edges (purple)
    ax = axes[1, col]; s = sc[:, [2, 3]]
    for i in range(N):
        for j in GRAPH.neighbors(i):
            if j > i:
                ax.plot([s[i, 0], s[j, 0]], [s[i, 1], s[j, 1]], color="purple", lw=1.4, alpha=.6, zorder=1)
    for i in range(N):
        ax.scatter(*s[i], color="k", zorder=2); ax.annotate(WORDS[i], s[i], fontsize=8)
    ax.set_title(f"{m} L{deep}  PC3-4  (context-RSA={pc_rsa(sc,[2,3],CTX_D):+.2f})", fontsize=9)
    ax.set_xlabel("PC3"); ax.set_ylabel("PC4"); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Days-of-week (semantic prior):  green = natural weekday cycle (PC1-2),  "
             "purple = in-context ring (PC3-4)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("runs/days/days_dual_structure.png", dpi=140)

# ---------------- Fig 2: per-layer sweep ----------------
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
for col, m in enumerate(MODELS):
    z, layers, node, mask = load(m)
    sem12, ctx34 = [], []
    for L in layers:
        sc = scores_of(node_means(z, L, node, mask))
        sem12.append(pc_rsa(sc, [0, 1], SEM_D))
        ctx34.append(pc_rsa(sc, [2, 3], CTX_D))
    ax = axes2[col]
    ax.plot(layers, sem12, "-o", ms=3, color="green", label="PC1-2 vs weekday semantics")
    ax.plot(layers, ctx34, "-o", ms=3, color="purple", label="PC3-4 vs in-context ring")
    ax.axhline(0, color=".85", lw=.6); ax.set_title(m); ax.set_xlabel("layer")
    ax.legend(fontsize=8)
axes2[0].set_ylabel("RSA")
fig2.suptitle("Days-of-week: weekday semantics (PC1-2) vs in-context ring (PC3-4) by layer")
fig2.tight_layout()
fig2.savefig("runs/days/days_pc_by_layer.png", dpi=140)
print("wrote runs/days/days_dual_structure.png + days_pc_by_layer.png")
