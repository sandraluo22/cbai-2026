"""Days-of-week, PC-resolved (the paper's specific claim): at a deep layer, do
the NATURAL weekday semantics dominate PC1-2 while the conflicting IN-CONTEXT
ring appears in higher PCs (PC3-4)? Per-node means (7 days), high context.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import replace

from config import get_config, DAYS
import graph as G

cfg = replace(get_config("gemma_qwen"), graph_type="ring", ring_size=7, word_set="days")
GRAPH = G.build_graph(cfg)
WORDS = GRAPH.words                          # permuted days (context order)
N = 7
IU = np.triu_indices(N, 1)
CTX_D = GRAPH.distance_matrix()[IU]          # in-context ring (BFS)
nat = [DAYS.index(w) for w in WORDS]
SEM_D = np.array([[min(abs(nat[i] - nat[j]), 7 - abs(nat[i] - nat[j]))
                   for j in range(N)] for i in range(N)])[IU]   # natural weekday cycle


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


def pc_dist(scores, pcs):
    s = scores[:, pcs]
    return np.linalg.norm(s[:, None, :] - s[None, :, :], axis=2)[IU]


MODELS = ["Llama", "Gemma", "Qwen"]
results = {}
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for col, m in enumerate(MODELS):
    z = np.load(f"runs/days/{m}_acts_sub.npz", allow_pickle=False)
    layers = [int(l) for l in z["_layers"]]
    deep = min(layers, key=lambda L: abs(L - 0.8 * max(layers)))
    node = z["meta_node"]; mask = z["meta_context_length"] >= 300
    H = node_means(z, deep, node, mask)
    Hc = H - H.mean(0)
    _, _, Vt = np.linalg.svd(Hc, full_matrices=False)
    scores = Hc @ Vt.T                       # 7 x rank
    res = {}
    for pcs, lab in [([0, 1], "PC1-2"), ([2, 3], "PC3-4")]:
        d = pc_dist(scores, pcs)
        res[lab] = {"semantic_rsa": spearman(d, SEM_D), "context_rsa": spearman(d, CTX_D)}
    results[m] = {"deep_layer": deep, **res}

    for row, pcs, lab in [(0, [0, 1], "PC1-2"), (1, [2, 3], "PC3-4")]:
        ax = axes[row, col]; s = scores[:, pcs]
        for i in range(N):                   # draw the IN-CONTEXT ring edges
            for j in GRAPH.neighbors(i):
                if j > i:
                    ax.plot([s[i, 0], s[j, 0]], [s[i, 1], s[j, 1]], color="0.8", zorder=1)
        for i in range(N):
            ax.scatter(*s[i], zorder=2); ax.annotate(WORDS[i], s[i], fontsize=8)
        r = results[m][lab]
        ax.set_title(f"{m} {lab}: sem-RSA={r['semantic_rsa']:+.2f}, "
                     f"ctx-RSA={r['context_rsa']:+.2f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

fig.suptitle("Days-of-week, PC-resolved: top PCs = weekday semantics?  "
             "higher PCs = in-context ring? (edges = the in-context ring)")
fig.tight_layout()
fig.savefig("runs/days/days_pc_analysis.png", dpi=140)
json.dump(results, open("runs/days/days_pc_analysis.json", "w"), indent=2)
for m in MODELS:
    r = results[m]
    print(f"{m:6} (L{r['deep_layer']}): "
          f"PC1-2 sem={r['PC1-2']['semantic_rsa']:+.2f}/ctx={r['PC1-2']['context_rsa']:+.2f}  | "
          f"PC3-4 sem={r['PC3-4']['semantic_rsa']:+.2f}/ctx={r['PC3-4']['context_rsa']:+.2f}")
print("wrote runs/days/days_pc_analysis.png + .json")
