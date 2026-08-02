"""Bayesian surrogate for mutual in-context learning between differently-primed instances.

MODEL. Each instance m is a Dirichlet-Markov sequence learner with recency discounting:
its predictive for the next node given current node i is

    p_m(j | i, t)  propto  alpha + C_m[i, j; t],
    C_m[i, j; t] = sum over past transitions (i->j) in m's OWN context, each weighted
                   gamma^(age in word-steps)

so at coupling time (t=0) the instances differ ONLY in their prior evidence C_m(0) --
the discounted bigram counts of their own prefix walks (grid vs ring). During the joint
phase every instance observes the SAME generated stream, so their posteriors are pulled
toward a common posterior at a rate set by gamma: with discounting, the effective
evidence is an exponential window of ~1/(1-gamma) recent transitions, and the prefix
influence decays like gamma^t -- predicting (i) convergence of the two predictive
models, (ii) hysteresis ~ gamma^t, (iii) self-reinforcing oscillation traps under top-k
truncation (a repeated edge dominates the recent window, top-k locks it in).

GENERATION exactly mirrors run_experiment.py: alternating generators (ring first),
top-k truncation of the 16-way predictive, sampled token appended to both contexts.

FIT. Grid-search (gamma, alpha) against the LLM run's behavioral curves (probability
mass each generator's PRE-truncation predictive puts on ring- and grid-neighbours of
the previous node, binned over joint time). Then compare, out of fit: occupancy skew,
top oscillating-pair mass, and the trajectory of representational convergence
(Procrustes similarity between the two learners' posterior-graph spectral embeddings
at the base/early/mid/late windows) vs the measured node-mean Procrustes.

Env: RUN(out_k2_fix) -- which LLM run to fit; expects its gen_log.json + summary files.
Out: <RUN>/bayes_fit.png, <RUN>/bayes_summary.json
"""
from __future__ import annotations
import json, os, sys
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cross-model", "src"))
from config import get_config
import graph as G

RUN = os.environ.get("RUN", "runs/out_k2_fix")
OUT = os.path.join(HERE, RUN)
N = 16


def adjacency(g):
    A = np.zeros((N, N), bool)
    for a in range(N):
        for b in g.adjacency[a]:
            A[a, b] = True
    return A


def spectral_embed(C, alpha, k=4):
    """Posterior-mean transition graph -> symmetrized weights -> normalized-Laplacian
    embedding (top-k nontrivial modes, 1/sqrt(lambda) scaled)."""
    P = C + alpha
    W = P + P.T
    np.fill_diagonal(W, 0)
    d = W.sum(1)
    di = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    L = np.eye(N) - di[:, None] * W * di[None, :]
    lam, U = np.linalg.eigh(L)
    E = U[:, 1:k + 1] / np.sqrt(np.maximum(lam[1:k + 1], 1e-6))
    return E


def procrustes_sim(A, B):
    A = A - A.mean(0); A /= max(np.linalg.norm(A), 1e-12)
    B = B - B.mean(0); B /= max(np.linalg.norm(B), 1e-12)
    return float(np.linalg.svd(A.T @ B, compute_uv=False).sum())


def simulate(gamma, alpha, topk, walks_a, walks_b, A_ring, A_grid, T, rng):
    """One coupled generation per pair; returns behavioral series + final counts.
    Learner a = grid-primed, b = ring-primed (b generates at even steps, as in the LLM
    runs). Counts are decayed by gamma each word-step."""
    P = len(walks_a)
    series = {("b", "ring"): np.zeros(T), ("b", "grid"): np.zeros(T),
              ("a", "ring"): np.zeros(T), ("a", "grid"): np.zeros(T)}
    counts = {("b", "ring"): np.zeros(T), ("b", "grid"): np.zeros(T),
              ("a", "ring"): np.zeros(T), ("a", "grid"): np.zeros(T)}
    Cs_t = {}                       # (m, p) -> count matrix, decayed
    prev_own = {}
    snap = {p: {} for p in range(P)}
    occ = np.zeros(N)
    pair_cnt = {}
    for p in range(P):
        for m, wk in (("a", walks_a[p]), ("b", walks_b[p])):
            C = np.zeros((N, N))
            nodes = wk.nodes
            for s in range(1, len(nodes)):
                C *= gamma
                C[nodes[s - 1], nodes[s]] += 1.0
            Cs_t[(m, p)] = C
            prev_own[(m, p)] = nodes[-1]
        snap[p]["base"] = {m: Cs_t[(m, p)].copy() for m in "ab"}
    joint_prev = {}
    for t in range(T):
        who = "b" if t % 2 == 0 else "a"
        for p in range(P):
            pv = prev_own[(who, p)] if t == 0 else joint_prev[p]
            C = Cs_t[(who, p)]
            pred = alpha + C[pv]
            pred = pred / pred.sum()
            series[(who, "ring")][t] += pred[A_ring[pv]].sum()
            series[(who, "grid")][t] += pred[A_grid[pv]].sum()
            counts[(who, "ring")][t] += 1
            counts[(who, "grid")][t] += 1
            pp = pred.copy()
            if topk > 0:
                pp[np.argsort(pp)[:-topk]] = 0.0
            node = int(rng.choice(N, p=pp / pp.sum()))
            # both learners of pair p observe the transition (their own prev -> node)
            for m in "ab":
                pv_m = prev_own[(m, p)] if t == 0 else joint_prev[p]
                Cs_t[(m, p)] *= gamma
                Cs_t[(m, p)][pv_m, node] += 1.0
            joint_prev[p] = node
            if t >= 300:
                occ[node] += 1
                k = tuple(sorted((pv if t > 0 else pv, node)))
                pair_cnt[k] = pair_cnt.get(k, 0) + 1
            if t + 1 in (100, 300, T):
                nm = {100: "early", 300: "mid", T: "late"}[t + 1]
                snap[p][nm] = {m: Cs_t[(m, p)].copy() for m in "ab"}
    for key in series:
        series[key] = series[key] / np.maximum(counts[key], 1)
    return series, occ, pair_cnt, snap


def llm_series(log):
    P, T = log["npairs"], log["tgen"]
    out = {("b", "ring"): np.zeros(T), ("b", "grid"): np.zeros(T),
           ("a", "ring"): np.zeros(T), ("a", "grid"): np.zeros(T)}
    for t in range(T):
        who = "b" if t % 2 == 0 else "a"
        r = np.mean([log["steps"][f"pair{p}"][t]["p_ring_nbrs"] for p in range(P)])
        g = np.mean([log["steps"][f"pair{p}"][t]["p_grid_nbrs"] for p in range(P)])
        out[(who, "ring")][t] = r
        out[(who, "grid")][t] = g
    return out


def binned(x, t_idx, nb=12):
    edges = np.linspace(0, len(x), nb + 1).astype(int)
    return np.array([x[t_idx[(t_idx >= a) & (t_idx < b)]].mean()
                     for a, b in zip(edges[:-1], edges[1:])])


def main():
    log = json.load(open(os.path.join(OUT, "gen_log.json")))
    T = log["tgen"]
    topk = log.get("topk", 2)
    cfg = replace(get_config("gemma_qwen"), n_walks=log["npairs"],
                  walk_length=log["ctx"], seed=log["seed"])
    grid = G.build_graph(replace(cfg, graph_type="grid", grid_rows=4, grid_cols=4))
    ring = G.build_graph(replace(cfg, graph_type="ring", ring_size=16))
    A_grid, A_ring = adjacency(grid), adjacency(ring)
    walks_a = G.generate_walks(grid, replace(cfg, graph_type="grid"))
    walks_b = G.generate_walks(ring, replace(cfg, graph_type="ring"))

    L = llm_series(log)
    tb = np.arange(0, T, 2)          # steps where b (ring) generated
    ta = np.arange(1, T, 2)
    targ = {k: binned(L[k], tb if k[0] == "b" else ta) for k in L}

    GAMMAS = [1.0, 0.999, 0.997, 0.995, 0.99, 0.98, 0.96, 0.93, 0.90, 0.85, 0.75]
    ALPHAS = [0.02, 0.05, 0.15, 0.5, 1.5]
    best = None
    for gam in GAMMAS:
        for al in ALPHAS:
            rng = np.random.default_rng(7)
            S, *_ = simulate(gam, al, topk, walks_a, walks_b, A_ring, A_grid, T, rng)
            loss = sum(((binned(S[k], tb if k[0] == "b" else ta) - targ[k]) ** 2).mean()
                       for k in S)
            if best is None or loss < best[0]:
                best = (loss, gam, al)
    loss, gam, al = best
    print(f"BEST FIT gamma={gam} alpha={al} (mse={loss:.4f})", flush=True)

    rng = np.random.default_rng(7)
    S, occ, pair_cnt, snap = simulate(gam, al, topk, walks_a, walks_b,
                                      A_ring, A_grid, T, rng)

    # representational-convergence prediction: Procrustes between the two learners'
    # posterior-graph spectral embeddings, pooled over pairs, at the 4 windows
    conv_model = {}
    for wname in ("base", "early", "mid", "late"):
        sims = [procrustes_sim(spectral_embed(snap[p][wname]["a"], al),
                               spectral_embed(snap[p][wname]["b"], al))
                for p in snap]
        conv_model[wname] = float(np.mean(sims))
    # measured convergence from the run's node-means (deep layers)
    z = np.load(os.path.join(OUT, "nodemeans_dueling.npz"))
    nL = int(z["n_layers"][0])
    def H(c, w, Lr):
        Hm = z[f"{c}_{w}_layer_{Lr}"].astype(np.float64)
        return Hm - Hm.mean(0)
    conv_llm = {}
    for wname, wkey in (("base", "base"), ("early", "joint_early"),
                        ("mid", "joint_mid"), ("late", "joint_late")):
        conv_llm[wname] = float(np.mean([procrustes_sim(H("grid", wkey, Lr),
                                                        H("ring", wkey, Lr))
                                         for Lr in range(24, nL)]))

    # ---- figure ------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    tt = {"b": tb, "a": ta}
    for ax, m, title in ((axes[0], "b", "RING-primed generator"),
                         (axes[1], "a", "GRID-primed generator")):
        for gname, col in (("ring", "crimson"), ("grid", "#023047")):
            k = 15
            y = np.convolve(L[(m, gname)][tt[m]], np.ones(k) / k, "valid")
            ax.plot(tt[m][:len(y)], y, color=col, lw=1.8,
                    label=f"LLM: mass on {gname}-nbrs")
            ym = np.convolve(S[(m, gname)][tt[m]], np.ones(k) / k, "valid")
            ax.plot(tt[m][:len(ym)], ym, color=col, lw=1.4, ls="--",
                    label=f"Dirichlet-Markov (γ={gam}, α={al})")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("joint step"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("predictive mass on neighbours of prev node")
    axes[0].legend(fontsize=7)
    wnames = ["base", "early", "mid", "late"]
    axes[2].plot(wnames, [conv_llm[w] for w in wnames], "o-", color="#0e7c86",
                 label="LLM node-means (deep layers)")
    axes[2].plot(wnames, [conv_model[w] for w in wnames], "s--", color="#fb8500",
                 label="model posterior-graph embeddings")
    axes[2].set_title("grid-instance vs ring-instance shape similarity", fontsize=10)
    axes[2].set_ylabel("Procrustes similarity"); axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8)
    fig.suptitle(f"Mutual in-context learning as coupled Dirichlet-Markov learners "
                 f"({RUN}, top-k={topk})")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"bayes_fit.{ext}"), dpi=160)

    # trap + occupancy comparison
    tot = sum(pair_cnt.values())
    top_m = sorted(pair_cnt.items(), key=lambda kv: -kv[1])[:3]
    words = log["words"]
    summary = {
        "fit": {"gamma": gam, "alpha": al, "mse": loss, "topk": topk,
                "effective_window_words": (None if gam >= 1.0 else round(1 / (1 - gam)))},
        "convergence_llm": conv_llm, "convergence_model": conv_model,
        "model_top_pairs": [(words[a], words[b], round(c / tot, 3))
                            for (a, b), c in top_m],
        "model_occupancy_maxmin": [int(occ.max()), int(occ.min())],
    }
    json.dump(summary, open(os.path.join(OUT, "bayes_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
