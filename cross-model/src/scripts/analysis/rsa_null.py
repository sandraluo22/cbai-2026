"""Random-points (permutation/random-representation) NULL for RSA -- the correct
baseline for the representational measure. Tells you whether an observed grid
RSA is above chance, which depends strongly on node count n.
"""
import numpy as np
from dataclasses import replace
from config import get_config
import graph as G


def spearman(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def null_rsa(graph, ntrial=20000, d=64, seed=0):
    """RSA of RANDOM node representations against the true graph distance."""
    rng = np.random.default_rng(seed)
    n = graph.n_nodes; iu = np.triu_indices(n, 1)
    GD = graph.distance_matrix()[iu]
    vals = np.empty(ntrial)
    for t in range(ntrial):
        H = rng.standard_normal((n, d))
        R = np.linalg.norm(H[:, None, :] - H[None, :, :], axis=2)[iu]
        vals[t] = spearman(R, GD)
    return vals


if __name__ == "__main__":
    for name, kw in [("n=16 (grid/ring/hex)", dict(graph_type="grid", grid_rows=4, grid_cols=4)),
                     ("n=7 (days)", dict(graph_type="ring", ring_size=7, word_set="days"))]:
        g = G.build_graph(replace(get_config("gemma_qwen"), **kw))
        v = null_rsa(g)
        print(f"{name:22}: 95%={np.percentile(v,95):.3f}  99%={np.percentile(v,99):.3f}")
