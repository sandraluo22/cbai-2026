"""Run the coupled Dirichlet-Markov surrogate over the SAME cells as the LLM sweep and
compute the SAME outcomes (JS trajectory, T_consensus, win margin, representational
distance), so each experimental knob can be identified with a Bayesian hyperparameter.

Surrogate (per learner m):
  evidence   C_m <- gamma * C_m ; C_m[i,j] += 1 on each observed transition in m's
             OWN context (prefix walk, then the appended joint tokens)
  predictive p_m(j|i) propto (alpha0 + C_m[i,j])^(1/temp)     [temperature = likelihood
             sharpening; alpha0 = symmetric Dirichlet concentration]
  sampling   top-k truncation, renormalize, sample
  coupling   generator's token appended to own context; delivered to partner w.p. q,
             else partner appends its own sample (exactly run_sweep.py's protocol)
  geometry   spectral embedding (modes 1-4, 1/sqrt(lambda)) of the symmetrized
             posterior-mean graph alpha0 + C_m; rep distance = 1 - Procrustes

Fitted Llama constants: gamma=0.96, alpha0=0.05 (from bayes_model.py).
Cells: the alpha / prior-strength / evidence-quality axes of runs/sweep_spec.json.

Out: runs/bayes_sweep.json + figs/bayes_sweep.png (surrogate vs LLM per axis)
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = json.load(open(os.path.join(HERE, "runs", "sweep_spec.json")))
LLM = json.load(open(os.path.join(HERE, "runs", "sweep_summary.json")))
N = 16
GAMMA, ALPHA0 = 0.96, 0.05
NSEEDS = 4


def adj_of(name):
    A = np.zeros((N, N), bool)
    for a, b in SPEC["graphs"][name]:
        A[a, b] = A[b, a] = True
    return A


def walk(A, length, rng):
    nbrs = [np.where(A[i])[0] for i in range(N)]
    x = [int(rng.integers(N))]
    for _ in range(length - 1):
        x.append(int(rng.choice(nbrs[x[-1]])))
    return x


def spec_embed(C):
    W = C + C.T + ALPHA0
    np.fill_diagonal(W, 0)
    d = W.sum(1)
    L = np.eye(N) - (d ** -0.5)[:, None] * W * (d ** -0.5)[None, :]
    lam, U = np.linalg.eigh(L)
    return U[:, 1:5] / np.sqrt(np.maximum(lam[1:5], 1e-6))


def psim(A, B):
    A = A - A.mean(0); A /= max(np.linalg.norm(A), 1e-12)
    B = B - B.mean(0); B /= max(np.linalg.norm(B), 1e-12)
    return float(np.linalg.svd(A.T @ B, compute_uv=False).sum())


def js(p, q):
    m = 0.5 * (p + q)
    def kl(a, b):
        mk = a > 0
        return float((a[mk] * np.log(a[mk] / np.maximum(b[mk], 1e-12))).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def run_cell(cell, seed):
    rng = np.random.default_rng(seed)
    T = cell["tgen"]
    P = cell["npairs"]
    q = cell.get("qcomm", 1.0)
    topk, temp = cell["topk"], cell["temp"]
    AA, AB = adj_of(cell["ga"]), adj_of(cell["gb"])
    a_only, b_only = AA & ~AB, AB & ~AA
    jsr = np.zeros(T)
    margins, rd_base, rd_late = [], [], []
    for p in range(P):
        C = {}
        prev = {}
        for m, A, ctx in (("A", AA, cell["ctx_a"]), ("B", AB, cell["ctx_b"])):
            w = walk(A, ctx, rng)
            Cm = np.zeros((N, N))
            for a, b in zip(w, w[1:]):
                Cm *= GAMMA
                Cm[a, b] += 1
            C[m] = Cm
            prev[m] = w[-1]
        rd_base.append(1 - psim(spec_embed(C["A"]), spec_embed(C["B"])))

        def pred(m):
            row = (ALPHA0 + C[m][prev[m]]) ** (1.0 / temp)
            return row / row.sum()

        def draw(m):
            pp = pred(m).copy()
            if topk > 0:
                pp[np.argsort(pp)[:-topk]] = 0.0
            return int(rng.choice(N, p=pp / pp.sum()))

        gen = np.zeros(T, np.int32)
        for t in range(T):
            who = "B" if t % 2 == 0 else "A"
            oth = "A" if who == "B" else "B"
            jsr[t] += js(pred("A"), pred("B")) / P
            g = draw(who)
            gen[t] = g
            tok = {who: g,
                   oth: (g if rng.random() < q else draw(oth))}
            for m in ("A", "B"):
                C[m] *= GAMMA
                C[m][prev[m], tok[m]] += 1
                prev[m] = tok[m]
        am = bm = tot = 0
        for a, b in zip(gen[T // 2:-1], gen[T // 2 + 1:]):
            tot += 1
            am += a_only[a, b]
            bm += b_only[a, b]
        margins.append((am - bm) / max(tot, 1))
        rd_late.append(1 - psim(spec_embed(C["A"]), spec_embed(C["B"])))
    roll = np.convolve(jsr, np.ones(21) / 21, "valid")
    below = np.where(roll < 0.05)[0]
    return {"js_final": float(jsr[-50:].mean()),
            "T_consensus": int(below[0]) if len(below) else T,
            "win_margin": float(np.mean(margins)),
            "rep_dist_base": float(np.mean(rd_base)),
            "rep_dist_late": float(np.mean(rd_late))}


def main():
    names = [c["name"] for c in SPEC["cells"]
             if c["name"].startswith(("alpha", "prior_", "ev_"))]
    cells = {c["name"]: c for c in SPEC["cells"]}
    res = {}
    for nm in names:
        runs = [run_cell(cells[nm], 100 + s) for s in range(NSEEDS)]
        res[nm] = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
        print(nm, {k: round(v, 3) for k, v in res[nm].items()})
    json.dump(res, open(os.path.join(HERE, "runs", "bayes_sweep.json"), "w"), indent=1)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    an = [f"alpha{a}" for a in alphas]
    ax = axes[0]
    ax.plot(alphas, [LLM[n]["rep_dist_late"] for n in an], "o-", color="#0e7c86",
            label="LLM rep. dist late")
    ax.plot(alphas, [res[n]["rep_dist_late"] for n in an], "o--", color="#0e7c86",
            alpha=0.6, label="surrogate rep. dist late")
    ax.plot(alphas, [LLM[n]["js_final"] for n in an], "s-", color="#fb8500",
            label="LLM JS final")
    ax.plot(alphas, [res[n]["js_final"] for n in an], "s--", color="#fb8500",
            alpha=0.6, label="surrogate JS final")
    ax.set_xlabel(r"prior disagreement $\alpha$")
    ax.set_ylabel("distance / divergence")
    ax.set_title("disagreement axis"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax = axes[1]
    sym = [("prior_100v100", 100), ("prior_300v300", 300), ("alpha1.0", 600),
           ("prior_1000v1000", 1000)]
    asym = [("prior_1000v100", 100), ("prior_1000v300", 300), ("prior_1000v1000", 1000)]
    ax.plot([x for _, x in sym], [LLM[n]["T_consensus"] for n, _ in sym], "o-",
            color="crimson", label="LLM T_consensus (sym)")
    ax.plot([x for _, x in sym], [res[n]["T_consensus"] for n, _ in sym], "o--",
            color="crimson", alpha=0.6, label="surrogate T_consensus (sym)")
    ax.plot([x for _, x in asym], [LLM[n]["win_margin"] * 1000 for n, _ in asym], "s-",
            color="#023047", label="LLM win margin x1000 (A=1000)")
    ax.plot([x for _, x in asym], [res[n]["win_margin"] * 1000 for n, _ in asym], "s--",
            color="#023047", alpha=0.6, label="surrogate win margin x1000")
    ax.set_xlabel("context tokens (both sides for sym; B side for asym)")
    ax.set_ylabel("steps  /  margin x 1000")
    ax.set_title("prior-strength axis"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax = axes[2]
    ev = [("ev_temp0.7", "T=0.7"), ("alpha1.0", "T=1"), ("ev_temp1.3", "T=1.3"),
          ("ev_free", "free"), ("ev_q0.5", "q=.5"), ("ev_q0.25", "q=.25")]
    xs = np.arange(len(ev))
    ax.bar(xs - 0.2, [LLM[n]["js_final"] for n, _ in ev], 0.4, color="#fb8500",
           label="LLM JS final")
    ax.bar(xs + 0.2, [res[n]["js_final"] for n, _ in ev], 0.4, color="#fb8500",
           alpha=0.5, label="surrogate JS final")
    ax.set_xticks(xs); ax.set_xticklabels([l for _, l in ev], fontsize=8)
    ax.set_xlabel("sampling / communication condition")
    ax.set_ylabel("final JS divergence")
    ax.set_title("evidence-quality axis"); ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)
    fig.suptitle("LLM sweep vs coupled Dirichlet-Markov surrogate "
                 f"(gamma={GAMMA}, alpha0={ALPHA0}; no refitting per cell)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "figs", f"bayes_sweep.{ext}"), dpi=160)
    print("DONE -> figs/bayes_sweep.png")


if __name__ == "__main__":
    main()
