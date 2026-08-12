"""Trajectory analysis for the reciprocal fine-tuning experiment (no GPU).

Reads traj_{cond}.json (written per generation by run_recursion.py) and
eval_set.json, and produces traj_summary.pdf plus a printed table.

Panels:
  1  strategy distributions over generations, per condition (A solid, B dashed)
  2  JSD(A,B) over generations, conditions overlaid
  3  per-context label agreement A(x)==B(x) (homogenization beyond marginals)
  4  minority strategy P(M) over generations (extinction question)
  5  quality: length / distinct-2 / base-model fluency
  6  conditional JSD over problem types vs marginal JSD (reciprocal)
"""
from __future__ import annotations

import collections
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
GROUPS = ["E", "I", "M"]
GNAMES = {"E": "exploratory", "I": "informational", "M": "emotional"}
CONDS = ["reciprocal", "self", "frozen", "static"]
CCOL = {"reciprocal": "tab:red", "self": "tab:blue", "frozen": "tab:green",
        "static": "tab:purple"}
GCOL = {"E": "tab:blue", "I": "tab:orange", "M": "tab:green"}


def jsd(p, q):
    p = np.asarray(p, float) + 1e-12
    q = np.asarray(q, float) + 1e-12
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    kl = lambda a, b: float((a * np.log(a / b)).sum())
    return (kl(p, m) + kl(q, m)) / 2 / math.log(2)


def dist_vec(d):
    return [d[g] for g in GROUPS]


def load():
    trajs, seed_trajs = {}, {}
    for c in CONDS:
        f = os.path.join(HERE, f"traj_{c}.json")
        if os.path.exists(f):
            trajs[c] = json.load(open(f))
        for s in (1, 2, 3, 4):
            fs = os.path.join(HERE, f"traj_{c}_s{s}.json")
            if os.path.exists(fs):
                seed_trajs[(c, s)] = json.load(open(fs))
    ev = json.load(open(os.path.join(HERE, "eval_set.json")))
    return trajs, seed_trajs, ev


def agreement(row):
    la, lb = row["A"].get("labels"), row["B"].get("labels")
    if not la or not lb:
        return None
    return float(np.mean([a == b for a, b in zip(la, lb)]))


def cond_jsd(row, problems, min_n=15):
    """Mean JSD(A,B) within problem type (weighted by count)."""
    la, lb = row["A"].get("labels"), row["B"].get("labels")
    if not la or not lb:
        return None
    by = collections.defaultdict(lambda: ([], []))
    for p, a, b in zip(problems, la, lb):
        by[p][0].append(a)
        by[p][1].append(b)
    tot, wsum = 0.0, 0
    for p, (aa, bb) in by.items():
        if len(aa) < min_n:
            continue
        pa = [aa.count(g) / len(aa) for g in GROUPS]
        pb = [bb.count(g) / len(bb) for g in GROUPS]
        tot += len(aa) * jsd(pa, pb)
        wsum += len(aa)
    return tot / wsum if wsum else None


def main():
    trajs, seed_trajs, ev = load()
    problems = ev.get("eval_problem", [])
    if not trajs:
        raise SystemExit("no traj_*.json found")

    fig, axes = plt.subplots(3, 2, figsize=(12, 13))

    # 1: strategy trajectories, one subplot region per condition -> use 2x2 inset grid
    ax = axes[0, 0]
    ax.axis("off")
    sub = fig.add_gridspec(3, 2)[0, 0].subgridspec(2, 2, hspace=0.45, wspace=0.3)
    for i, c in enumerate(CONDS):
        a = fig.add_subplot(sub[i // 2, i % 2])
        if c in trajs:
            gens = [r["gen"] for r in trajs[c]]
            for g in GROUPS:
                a.plot(gens, [r["A"]["dist"][g] for r in trajs[c]], color=GCOL[g],
                       lw=1.4, label=f"A {GNAMES[g]}" if i == 0 else None)
                a.plot(gens, [r["B"]["dist"][g] for r in trajs[c]], color=GCOL[g],
                       lw=1.4, ls="--", label=f"B {GNAMES[g]}" if i == 0 else None)
        a.set_title(c, fontsize=9)
        a.set_ylim(0, 1)
        a.tick_params(labelsize=7)
        if i >= 2:
            a.set_xlabel("generation", fontsize=8)
        if i % 2 == 0:
            a.set_ylabel("P(strategy)", fontsize=8)
        if i == 0:
            a.legend(fontsize=5.5, ncol=2, loc="upper right")

    # 2: JSD (seed 0 bold, replication seeds thin)
    ax = axes[0, 1]
    for c, tr in trajs.items():
        ax.plot([r["gen"] for r in tr], [r["jsd"] for r in tr], color=CCOL[c],
                marker="o", ms=3, label=c)
    for (c, s), tr in seed_trajs.items():
        ax.plot([r["gen"] for r in tr], [r["jsd"] for r in tr], color=CCOL[c],
                lw=1, alpha=0.5)
    ax.set_xlabel("generation")
    ax.set_ylabel("JSD(A, B) over strategy dist (bits)")
    ax.set_title("behavioral divergence (bold seed 0, thin seeds 1-2)")
    ax.legend(fontsize=8)

    # 3: per-context agreement
    ax = axes[1, 0]
    for c, tr in trajs.items():
        ys = [agreement(r) for r in tr]
        gens = [r["gen"] for r in tr]
        if any(y is not None for y in ys):
            ax.plot(gens, ys, color=CCOL[c], marker="o", ms=3, label=c)
    ax.set_xlabel("generation")
    ax.set_ylabel("P(A label == B label) on same context")
    ax.set_title("per-context strategy agreement")
    ax.legend(fontsize=8)

    # 4: minority strategy
    ax = axes[1, 1]
    for c, tr in trajs.items():
        gens = [r["gen"] for r in tr]
        ax.plot(gens, [r["A"]["dist"]["M"] for r in tr], color=CCOL[c], marker="o",
                ms=3, label=f"{c} A")
        ax.plot(gens, [r["B"]["dist"]["M"] for r in tr], color=CCOL[c], marker="s",
                ms=3, ls="--", label=f"{c} B")
    ax.set_xlabel("generation")
    ax.set_ylabel("P(emotional)  (minority strategy)")
    ax.set_title("minority-strategy trajectory")
    ax.legend(fontsize=6, ncol=2)

    # 5: quality metrics
    ax = axes[2, 0]
    for c, tr in trajs.items():
        gens = [r["gen"] for r in tr]
        d2 = [(r["A"]["distinct2"] + r["B"]["distinct2"]) / 2 for r in tr]
        ax.plot(gens, d2, color=CCOL[c], marker="o", ms=3, label=f"{c}")
    ax.set_xlabel("generation")
    ax.set_ylabel("distinct-2 (A/B mean)")
    ax.set_title("lexical diversity")
    ax.legend(fontsize=8)
    ax2 = ax.twinx()
    for c, tr in trajs.items():
        gens = [r["gen"] for r in tr]
        fl = [(r["A"]["fluency"] + r["B"]["fluency"]) / 2 for r in tr]
        ax2.plot(gens, fl, color=CCOL[c], ls=":", lw=1)
    ax2.set_ylabel("base-model logprob (dotted)")

    # 6: conditional vs marginal JSD (per condition)
    ax = axes[2, 1]
    for c, tr in trajs.items():
        gens = [r["gen"] for r in tr]
        cj = [cond_jsd(r, problems) for r in tr]
        if any(x is not None for x in cj):
            ax.plot(gens, cj, color=CCOL[c], marker="o", ms=3, label=f"{c} conditional")
        ax.plot(gens, [r["jsd"] for r in tr], color=CCOL[c], ls="--", lw=1,
                label=f"{c} marginal" if c == "reciprocal" else None)
    ax.set_xlabel("generation")
    ax.set_ylabel("JSD(A,B) (bits)")
    ax.set_title("conditional (per problem type) vs marginal JSD (dashed)")
    ax.legend(fontsize=7)

    fig.tight_layout()
    out = os.path.join(HERE, "traj_summary.pdf")
    fig.savefig(out)
    print("wrote", out)

    # ---- printed table ----------------------------------------------------
    print(f"\n{'cond':<11} gen  A: E/I/M           B: E/I/M           JSD    agree  d2(A/B)     flu(A/B)")
    for c, tr in trajs.items():
        for r in tr:
            if r["gen"] not in (0, len(tr) - 1) and r["gen"] % 5 != 0:
                continue
            ad, bd = r["A"]["dist"], r["B"]["dist"]
            ag = agreement(r)
            print(f"{c:<11} {r['gen']:>3}  "
                  f"{ad['E']:.2f}/{ad['I']:.2f}/{ad['M']:.2f}     "
                  f"{bd['E']:.2f}/{bd['I']:.2f}/{bd['M']:.2f}     "
                  f"{r['jsd']:.3f}  {ag if ag is None else round(ag,2)}   "
                  f"{r['A']['distinct2']:.2f}/{r['B']['distinct2']:.2f}   "
                  f"{r['A']['fluency']:.2f}/{r['B']['fluency']:.2f}")


if __name__ == "__main__":
    main()
