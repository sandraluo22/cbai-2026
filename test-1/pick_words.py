"""Choose a 16-word subset + node assignment that minimizes Llama's pretrained bigram
prior along UNION-graph edges of a graph pair -- the fix for the 'paper tiger' attractor.

Score of ordered pair (a -> b): restricted softmax over the 36-candidate next words of
the [BOS, " a"] context. Edge score: max of both directions. Objective (lexicographic-ish):
minimize  max_edge + 0.05 * mean_edge  over union edges, by random-restart local search
(position swaps + pool swaps).

Env: PAIR ("grid,ring") -- any two of grid|ring|hex (16 nodes each)
Out: prints the assignment; writes out/words16_<a>_<b>.json
"""
from __future__ import annotations
import json, os, sys
from dataclasses import replace
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs", "out")
sys.path.insert(0, os.path.join(HERE, "..", "cross-model", "src"))
from config import get_config
import graph as G

PAIR = os.environ.get("PAIR", "grid,ring").split(",")
GKW = {"grid": dict(graph_type="grid", grid_rows=4, grid_cols=4),
       "ring": dict(graph_type="ring", ring_size=16),
       "hex": dict(graph_type="hex", hex_rows=4, hex_cols=4),
       "prism": dict(graph_type="prism", prism_k=8),
       "antiprism": dict(graph_type="antiprism", prism_k=8),
       "ring3": dict(graph_type="ring", ring_size=16)}
rng = np.random.default_rng(0)


def adjacency_of(gname):
    g = G.build_graph(replace(get_config("gemma_qwen"), **GKW[gname]))
    adjacency = g.adjacency
    if gname == "ring3":
        adjacency = [sorted([(i - 3) % 16, (i + 3) % 16]) for i in range(16)]
    return adjacency

z = np.load(os.path.join(OUT, "bigram_prior.npz"), allow_pickle=False)
words = [str(w) for w in z["words"]]
L = z["logits"].astype(np.float64)
P = np.exp(L - L.max(1, keepdims=True))
P = P / P.sum(1, keepdims=True)                       # restricted softmax over the pool
S = np.maximum(P, P.T)                                # symmetric pair score
np.fill_diagonal(S, 0.0)
n_pool, n = len(words), 16

# union edges on node indices across the pair's graphs
edges = set()
for gname in PAIR:
    adjacency = adjacency_of(gname)
    assert len(adjacency) == n
    for a in range(n):
        for b in adjacency[a]:
            edges.add(tuple(sorted((a, b))))
edges = sorted(edges)
print(f"pair={PAIR} union edges={len(edges)}")

def cost(assign):
    e = np.array([S[assign[a], assign[b]] for a, b in edges])
    return e.max() + 0.05 * e.mean()

def report(assign, label):
    e = np.array([S[assign[a], assign[b]] for a, b in edges])
    k = int(e.argmax()); a, b = edges[k]
    print(f"{label}: max_edge={e.max():.4f} ({words[assign[a]]}-{words[assign[b]]}) "
          f"mean_edge={e.mean():.4f}")

# baseline = current assignment (WORDS[:16] in order)
base = list(range(16))
report(base, "current WORDS[:16]")
print("worst pool pairs:", [(words[i], words[j], round(S[i, j], 3))
      for i, j in zip(*np.unravel_index(np.argsort(S, axis=None)[::-1][:12:2], S.shape))])

best, best_c = None, np.inf
for restart in range(30):
    perm = list(rng.permutation(n_pool)[:n])
    c = cost(perm)
    for it in range(20000):
        q = perm.copy()
        if rng.random() < 0.5:                        # swap two positions
            i, j = rng.choice(n, 2, replace=False)
            q[i], q[j] = q[j], q[i]
        else:                                         # swap one in from the pool
            outside = [w for w in range(n_pool) if w not in q]
            q[rng.integers(n)] = outside[rng.integers(len(outside))]
        cq = cost(q)
        if cq <= c:
            perm, c = q, cq
    if c < best_c:
        best, best_c = perm, c
report(best, "optimized")
chosen = [words[i] for i in best]
print("assignment (node i -> word):", chosen)
json.dump({"pair": PAIR, "words16": chosen,
           "max_edge_prior": float(max(S[best[a], best[b]] for a, b in edges)),
           "baseline_max_edge_prior": float(max(S[a, b] for a, b in edges))},
          open(os.path.join(OUT, f"words16_{PAIR[0]}_{PAIR[1]}.json"), "w"), indent=2)
