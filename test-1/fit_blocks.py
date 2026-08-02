"""Fit update models to the teacher-forced block-dynamics probe.

Candidates, replayed on the identical streams (single-token prediction, no generation):
  exp-gamma : Dirichlet-Markov, predictive prop-to (alpha0 + sum gamma^age counts);
              gamma AND alpha0 fitted on THIS data (best case for the exponential).
  kernel    : measured-kernel model, ZERO refitting: logit_j = sum over past a->j
              transitions of w_row(age) + sum over past c->j (c != a) of w_col(age),
              with w_row/w_col taken from runs/out_probe/update_kernel.json;
              predictive = softmax. Every parameter measured independently.

Comparison: per-step MSE against the LLM's grid-mass/ring-mass series (both contexts),
plus the ratchet statistic: the context's own-graph mass at the end of each of its own
blocks across cycles (does recovery fatigue?).

Out: figs/fig_block_fit.png + printed table
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
N = 16
BINS = [(1, 5), (6, 10), (11, 20), (21, 40), (41, 80), (81, 160), (161, 320), (321, 10 ** 9)]


def binof(age):
    for bi, (lo, hi) in enumerate(BINS):
        if lo <= age <= hi:
            return bi
    return len(BINS) - 1


def adjacency_grid():
    A = np.zeros((N, N), bool)
    for r in range(4):
        for c in range(4):
            i = 4 * r + c
            if c < 3: A[i, i + 1] = A[i + 1, i] = True
            if r < 3: A[i, i + 4] = A[i + 4, i] = True
    return A


def adjacency_ring():
    A = np.zeros((N, N), bool)
    for i in range(N):
        A[i, (i + 1) % N] = A[(i + 1) % N, i] = True
    return A


def series_for(model_fn, pb, ctxname):
    """model_fn(history, prev) -> 16-prob; history = list of (a, b) transitions in
    order (prefix + continuation so far)."""
    P, CTX = pb["P"], pb["ctx"]
    T = pb["bl"] * pb["nb"]
    A_g, A_r = adjacency_grid(), adjacency_ring()
    mg = np.zeros(T); mr = np.zeros(T)
    for p in range(P):
        pref = pb["prefixes"][ctxname][p]
        seq = pb["streams"][p]
        full = pref + seq
        trans = list(zip(full, full[1:]))
        for t in range(T):
            hist = trans[:CTX - 1 + t]              # all transitions before token t
            pv = full[CTX + t - 1]
            pr = model_fn(hist, pv)
            mg[t] += pr[A_g[pv]].sum() / P
            mr[t] += pr[A_r[pv]].sum() / P
    return mg, mr


def make_expgamma(gamma, a0):
    def fn(hist, pv):
        row = np.full(N, a0)
        n = len(hist)
        for i, (a, b) in enumerate(hist):
            if a == pv:
                row[b] += gamma ** (n - i)
        return row / row.sum()
    return fn


def make_kernel(w_row, w_col):
    def fn(hist, pv):
        logit = np.zeros(N)
        n = len(hist)
        for i, (a, b) in enumerate(hist):
            wgt = w_row[binof(n - i)] if a == pv else w_col[binof(n - i)]
            logit[b] += wgt
        e = np.exp(logit - logit.max())
        return e / e.sum()
    return fn


def main():
    pb = json.load(open(os.path.join(HERE, "runs", "out_blockprobe", "blockprobe.json")))
    k = json.load(open(os.path.join(HERE, "runs", "out_probe", "update_kernel.json")))
    w_row, w_col = np.array(k["w_row"]), np.array(k["w_col"])
    T = pb["bl"] * pb["nb"]

    llm = {c: (np.array(pb["series"][c]["grid_mass"]),
               np.array(pb["series"][c]["ring_mass"])) for c in ("grid", "ring")}

    # fit exp-gamma on this data (coarse grid; alpha too)
    best = None
    for gam in (0.999, 0.997, 0.99, 0.98, 0.96, 0.93, 0.90):
        for a0 in (0.02, 0.05, 0.15, 0.5):
            fn = make_expgamma(gam, a0)
            mse = 0
            for c in ("grid", "ring"):
                mg, mr = series_for(fn, pb, c)
                mse += np.mean((mg - llm[c][0]) ** 2) + np.mean((mr - llm[c][1]) ** 2)
            if best is None or mse < best[0]:
                best = (mse, gam, a0)
    _, gam, a0 = best
    print(f"exp-gamma best fit on block data: gamma={gam} alpha={a0} (mse={best[0]/4:.4f})")

    models = {"exp-gamma (fitted here)": make_expgamma(gam, a0),
              "measured kernel (zero refit)": make_kernel(w_row, w_col)}
    out = {}
    for nm, fn in models.items():
        out[nm] = {c: series_for(fn, pb, c) for c in ("grid", "ring")}
        mse = np.mean([np.mean((out[nm][c][i] - llm[c][i]) ** 2)
                       for c in ("grid", "ring") for i in (0, 1)])
        print(f"{nm}: series MSE = {mse:.4f}")

    # ---- figure ------------------------------------------------------------
    sm = lambda x: np.convolve(x, np.ones(9) / 9, "valid")
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    for ax, c in zip(axes, ("grid", "ring")):
        for b in range(pb["nb"]):
            if b % 2 == 0:                          # ring blocks shaded
                ax.axvspan(b * pb["bl"], (b + 1) * pb["bl"], color="#c22f4d", alpha=0.06)
        own = 0 if c == "grid" else 1
        ax.plot(sm(llm[c][own]), color="#111", lw=1.8, label="LLM (measured)")
        ax.plot(sm(out["exp-gamma (fitted here)"][c][own]), color="#999", lw=1.3,
                ls="--", label=f"exp-γ (γ={gam}, fitted)")
        ax.plot(sm(out["measured kernel (zero refit)"][c][own]), color="#0e7c86",
                lw=1.3, ls="--", label="measured kernel (zero refit)")
        ax.set_ylabel(f"{c}-primed ctx:\nmass on OWN ({c}) nbrs")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("continuation step (shaded = ring blocks, unshaded = grid blocks)")
    fig.suptitle("Teacher-forced block dynamics: LLM vs update models "
                 "(single-token prediction on ground-truth blocked streams)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "figs", f"fig_block_fit.{ext}"), dpi=160)

    # ratchet statistic: own-mass at end of each own block
    print("\nratchet: grid-ctx grid-mass at end of each GRID block (steps ~95-100 of blocks 1,3,5)")
    for nm, series in [("LLM", llm)] + [(n2, out[n2]) for n2 in models]:
        vals = [np.mean(series["grid"][0][b * pb["bl"] + 90:(b + 1) * pb["bl"]])
                for b in (1, 3, 5)]
        print(f"  {nm:30s} " + "  ".join(f"{v:.3f}" for v in vals))
    print("DONE -> figs/fig_block_fit.png")


if __name__ == "__main__":
    main()
