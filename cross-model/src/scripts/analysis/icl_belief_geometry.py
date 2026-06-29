"""In-context-learning belief-state geometry, treating the graph as a HIDDEN
process the model must infer in-context (no training; uses cached activations).

The model is never told the graph; early in a walk it is uncertain about the
adjacency and infers it as tokens arrive. The Bayes-optimal observer that does
NOT know the graph models the walk as a Markov chain with a Dirichlet prior on
each row of the transition matrix; its belief at step t is the posterior
predictive over the next node:

    b_t[j] = (N_t(cur -> j) + alpha) / (sum_k N_t(cur -> k) + n*alpha)

where N_t counts transitions seen in the prefix and cur = node at step t. Early
(few counts) -> near-uniform (simplex centre); late -> the true neighbour
distribution (toward the simplex faces). The set of these beliefs across walks
and positions is the in-context mixed-state geometry.

We then fit a linear map  activations -> b_t  (ridge, split by walk) and ask:
does the residual stream linearly encode this belief? We visualise the 16-dim
belief simplex (PCA to 2-D), ground truth vs activation-recovered, coloured by
context length (condensation) and current node.

-> runs/<version>/<graph>/slides/icl_belief_geometry.pdf
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from dataclasses import replace
from config import get_config
import graph as G
import paths as P

GRAPHS = [("square_grid", dict(graph_type="grid", grid_rows=4, grid_cols=4)),
          ("ring", dict(graph_type="ring", ring_size=16)),
          ("hex", dict(graph_type="hex", hex_rows=4, hex_cols=4))]
ALPHA = 1.0          # Dirichlet prior strength
RIDGE = 1e3
NWALKS, WLEN = 100, 1000


def belief_table(gr, walks, alpha=ALPHA):
    """beliefs[wid][t] = posterior-predictive over next node (n-dim)."""
    n = gr.n_nodes
    B = {}
    for w in walks:
        nodes = w.nodes; N = np.zeros((n, n)); bt = np.empty((len(nodes), n))
        for t, cur in enumerate(nodes):
            row = N[cur] + alpha; bt[t] = row / row.sum()
            if t + 1 < len(nodes):
                N[cur, nodes[t + 1]] += 1
        B[w.walk_id] = bt
    return B


def ridge_fit(X, Y, alpha):
    Xc = X - X.mean(0); Yc = Y - Y.mean(0)
    W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(Xc.shape[1]), Xc.T @ Yc)
    return W, X.mean(0), Y.mean(0)


def r2(Y, Yhat):
    return float(1 - ((Y - Yhat) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def main():
    for gname, kw in GRAPHS:
        if not all(__import__("os").path.exists(P.acts_path(gname, m)) for m in P.MODELS):
            print(f"skip {gname}: no acts for {P.VERSION}", flush=True); continue
        gr = G.build_graph(replace(get_config("gemma_qwen"), **kw)); n = gr.n_nodes
        # true limiting beliefs (neighbour distributions) for reference
        attractor = np.zeros((n, n))
        for i in range(n):
            nb = gr.neighbors(i); attractor[i, nb] = 1.0 / len(nb)

        fig, ax = plt.subplots(len(P.MODELS), 3, figsize=(15, 4.6 * len(P.MODELS)))
        for r, m in enumerate(P.MODELS):
            L = P.peak_layer(gname, m)
            z = np.load(P.acts_path(gname, m))
            wid = z["meta_walk_id"]; step = z["meta_step"]; node = z["meta_node"]
            walks = G.generate_walks(gr, replace(get_config("gemma_qwen"),
                                     n_walks=int(wid.max()) + 1, walk_length=int(step.max()) + 1, **kw))
            B = belief_table(gr, walks)
            # validate reproduced walks against the cached metadata
            ok = all(walks[w].nodes[s] == nd for w, s, nd in
                     zip(wid[:300], step[:300], node[:300]))
            assert ok, f"walk mismatch {gname}/{m}"
            X = z[f"layer_{L}"].astype(np.float64)
            Bocc = np.stack([B[w][s] for w, s in zip(wid, step)])      # ground-truth belief per occ
            ctx = step + 1

            te = (wid % 4 == 0)                                        # hold out 1/4 of walks
            tr = ~te
            W, mx, my = ridge_fit(X[tr], Bocc[tr], RIDGE)
            Bhat = (X - mx) @ W + my
            Bhat = np.clip(Bhat, 0, None); Bhat = Bhat / Bhat.sum(1, keepdims=True)
            R2 = r2(Bocc[te], (X[te] - mx) @ W + my)

            # PCA basis from ground-truth beliefs (train), project everything
            mu = Bocc[tr].mean(0); U, S, Vt = np.linalg.svd(Bocc[tr] - mu, full_matrices=False)
            proj = lambda M: (M - mu) @ Vt[:2].T
            At = proj(attractor)
            for c, (M, ttl) in enumerate([(Bocc[te], "ground-truth belief"),
                                          (Bhat[te], f"recovered from acts (R^2={R2:.2f})")]):
                a = ax[r, c]; Q = proj(M)
                sc = a.scatter(Q[:, 0], Q[:, 1], c=np.log10(ctx[te]), s=4, alpha=0.3,
                               cmap="viridis", linewidths=0, rasterized=True)
                a.scatter(At[:, 0], At[:, 1], c="red", marker="*", s=80, zorder=5,
                          edgecolor="k", label="true neighbour dist.")
                a.set_title(f"{m} L{L}: {ttl}", fontsize=9); a.set_xticks([]); a.set_yticks([])
            fig.colorbar(sc, ax=ax[r, 1], fraction=0.046, label="log10 context len")
            # third panel: recovered, coloured by current node
            a = ax[r, 2]; Q = proj(Bhat[te])
            a.scatter(Q[:, 0], Q[:, 1], c=plt.cm.tab20(node[te]), s=4, alpha=0.4, linewidths=0, rasterized=True)
            a.scatter(At[:, 0], At[:, 1], c="k", marker="*", s=60, zorder=5)
            a.set_title(f"{m}: recovered, by current node", fontsize=9); a.set_xticks([]); a.set_yticks([])
            print(f"{gname} {m} L{L}: belief R^2={R2:.3f}", flush=True)

        fig.suptitle(f"{gname} [{P.VERSION}] in-context belief-state geometry "
                     f"(Dirichlet posterior-predictive over next node)\n"
                     f"centre = uncertain (early context) -> red stars = true neighbour distributions (converged)")
        fig.tight_layout()
        import os; os.makedirs(f"{P.gdir(gname)}/slides", exist_ok=True)
        out = f"{P.gdir(gname)}/slides/icl_belief_geometry.pdf"
        fig.savefig(out, dpi=140); plt.close(fig)
        print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
