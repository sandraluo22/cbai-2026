"""Fit an RSA / conditional-logit probabilistic model to the open-clue Codenames
transcripts, turning the measured KLs into interpretable, fitted parameters.

Association code:  A[c,i] = MiniLM cos(clue c, board word i)   (from clue_sims.json)

GUESSER (per guesser model): the belief over available board words is modelled as a
literal-listener softmax
        q_i(beta) = softmax_i( beta * A[c,i] )
fit to the OBSERVED belief distribution (guess_dist_clean). beta = clue-sensitivity
(the listener rationality behind `coupling`). Fit quality = KL(observed || model).

SPEAKER (per condition x spymaster): the clue distribution over the logged top-k
candidate clues is modelled as
        q_c(w) = softmax_c( w . [inform_remaining, inform_all_targets, found_assoc, dead_assoc] )
  inform_remaining = mean_i in REMAINING targets A[c,i]      (listener-conditioned -> adaptivity)
  inform_all       = mean_i in ALL original targets A[c,i]   (listener-agnostic)
  found_assoc      = mean_i in FOUND A[c,i]                  (cluing for found = wasteful)
  dead_assoc       = max_i in ELIMINATED A[c,i]              (pointing at a wrong guess)
fit to the observed clue distribution (clue_dist_real). The weight on inform_remaining
(vs inform_all) IS adaptivity; the (negative) weight on dead_assoc IS eliminated-avoidance.

Usage:  python src/codenames_rsa_fit.py [runs/codenames]
Out:    <root>/codenames_rsa_fit.{json,pdf}  (+ printed report)
"""
from __future__ import annotations

import os
import sys
import json
import glob
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = sys.argv[1] if len(sys.argv) > 1 else "runs/codenames"
DIRS = ["llm_codenames_open", "llm_codenames_open_mem", "spy_remaining", "spy_eliminated", "spy_inferred"]
SIMS = json.load(open(os.path.join(ROOT, "clue_sims.json")))
BOARD = SIMS["board"]; BIDX = {w: i for i, w in enumerate(BOARD)}
SIM = SIMS["sims"]


def simvec(clue):
    d = SIM.get(clue)
    return np.array([d[w] for w in BOARD]) if d else None


def load_rows(d):
    out = []
    for f in glob.glob(os.path.join(ROOT, d, "*_transcript.jsonl")):
        for l in open(f):
            if l.strip():
                r = json.loads(l); r["_dir"] = d; out.append(r)
    return out


# ---- conditional-logit MLE: fit theta so softmax(X theta) matches target dist p ----
def fit_logit(examples, F, nonneg=False):
    def nll(theta):
        tot = 0.0
        for X, p in examples:
            z = X @ theta
            tot -= float(np.sum(p * (z - logsumexp(z))))
        return tot
    x0 = np.zeros(F)
    bnds = [(0, None)] * F if nonneg else None
    res = minimize(nll, x0, method="L-BFGS-B", bounds=bnds)
    return res.x


def mean_kl(examples, theta):
    kls = []
    for X, p in examples:
        z = X @ theta; q = np.exp(z - logsumexp(z))
        m = p > 0
        kls.append(float(np.sum(p[m] * (np.log(p[m]) - np.log(q[m] + 1e-12)))))
    return float(np.mean(kls))


def guesser_fit(rows_by_dir):
    """beta per guesser model, fit to observed belief (guess_dist_clean)."""
    ex = defaultdict(list)
    for d, rows in rows_by_dir.items():
        for r in rows:
            gd = r.get("coupling", {}).get("guess_dist_clean")
            sv = simvec(r.get("clue", ""))
            if gd is None or sv is None:
                continue
            p = np.array([gd[w] for w in BOARD])
            avail = p > 1e-6
            if avail.sum() < 2:
                continue
            X = sv[avail][:, None]                              # (K,1) feature = sim
            pp = p[avail] / p[avail].sum()
            ex[r["guesser"]].append((X, pp))
    res = {}
    for g, exs in ex.items():
        beta = fit_logit(exs, 1, nonneg=True)
        res[g] = {"beta": float(beta[0]), "kl_fit": mean_kl(exs, beta),
                  "kl_null": mean_kl(exs, np.zeros(1)), "n": len(exs)}
    return res


def speaker_features(r):
    """Return (candidates, feature matrix (K,4), observed dist) for one round, or None."""
    ad = r.get("adaptivity", {}); cd = ad.get("clue_dist_real")
    if not cd:
        return None
    targets = [BIDX[w] for w in r["targets"] if w in BIDX]
    found = [BIDX[w] for w in r.get("found_so_far", []) if w in BIDX]
    remaining = [t for t in targets if t not in found]
    dead = [BIDX[w] for w in r.get("_dead", []) if w in BIDX]
    cands, X, p = [], [], []
    for c, pr in cd.items():
        sv = simvec(c)
        if sv is None:
            continue
        inform_rem = float(np.mean(sv[remaining])) if remaining else 0.0
        inform_all = float(np.mean(sv[targets])) if targets else 0.0
        found_a = float(np.mean(sv[found])) if found else 0.0
        dead_a = float(np.max(sv[dead])) if dead else 0.0
        cands.append(c); X.append([inform_rem, inform_all, found_a, dead_a]); p.append(pr)
    if len(cands) < 2 or sum(p) <= 0:
        return None
    return np.array(X), np.array(p) / sum(p)


def add_dead(rows):
    """Reconstruct the eliminated (wrong-guess) set present BEFORE each round, per game."""
    by = defaultdict(list)
    for r in rows:
        by[(r["pair"], r["game"])].append(r)
    for key, rs in by.items():
        rs.sort(key=lambda r: r["round"])
        dead = set()
        for r in rs:
            r["_dead"] = sorted(dead)
            for g, ok in zip(r.get("guesses", []), r.get("correct", [])):
                if not ok:
                    dead.add(g)


def speaker_fit(rows_by_dir):
    res = {}
    for d, rows in rows_by_dir.items():
        add_dead(rows)
        ex = defaultdict(list)
        for r in rows:
            f = speaker_features(r)
            if f:
                ex[r["spymaster"]].append(f)
        for spy, exs in ex.items():
            w = fit_logit(exs, 4)
            res[f"{d} | {spy}"] = {"inform_remaining": float(w[0]), "inform_all": float(w[1]),
                                   "found_assoc": float(w[2]), "dead_assoc": float(w[3]),
                                   "kl_fit": mean_kl(exs, w), "kl_null": mean_kl(exs, np.zeros(4)), "n": len(exs)}
    return res


def main():
    rows_by_dir = {d: load_rows(d) for d in DIRS if os.path.isdir(os.path.join(ROOT, d))}
    g = guesser_fit(rows_by_dir)
    s = speaker_fit(rows_by_dir)

    print("=" * 78)
    print("GUESSER (literal-listener softmax  q_i = softmax(beta * sim(clue,i)))  fit to belief")
    for k, v in sorted(g.items()):
        print(f"  {k:10s}: beta={v['beta']:.2f}   fit KL={v['kl_fit']:.3f}  (null KL={v['kl_null']:.3f}, "
              f"explains {100*(1-v['kl_fit']/v['kl_null']):.0f}%)   n={v['n']}")
    print("\nSPEAKER (softmax over candidate clues) fitted weights")
    print(f"  {'condition | spymaster':34s}{'inform_rem':>11s}{'inform_all':>11s}{'found':>8s}{'dead':>8s}{'n':>5s}")
    for k, v in sorted(s.items()):
        print(f"  {k:34s}{v['inform_remaining']:>11.2f}{v['inform_all']:>11.2f}{v['found_assoc']:>8.2f}{v['dead_assoc']:>8.2f}{v['n']:>5d}")

    json.dump({"guesser": g, "speaker": s}, open(os.path.join(ROOT, "codenames_rsa_fit.json"), "w"), indent=2)

    # ---- figure ----
    with PdfPages(os.path.join(ROOT, "codenames_rsa_fit.pdf")) as pdf:
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
        gs = sorted(g); ax[0].bar(range(len(gs)), [g[k]["beta"] for k in gs], color="tab:blue")
        for i, k in enumerate(gs):
            ax[0].text(i, g[k]["beta"], f"beta={g[k]['beta']:.1f}\n{100*(1-g[k]['kl_fit']/g[k]['kl_null']):.0f}% expl",
                       ha="center", va="bottom", fontsize=8)
        ax[0].set_xticks(range(len(gs))); ax[0].set_xticklabels(gs)
        ax[0].set_title("Fitted GUESSER clue-sensitivity  beta\n(literal-listener softmax on MiniLM sim)", fontsize=9)
        ax[0].set_ylabel("beta"); ax[0].grid(alpha=.3, axis="y")
        # speaker: inform_remaining weight (adaptivity) and dead_assoc (avoidance), by condition
        keys = sorted(s); feats = ["inform_remaining", "dead_assoc"]; cols = ["tab:green", "tab:red"]
        y = np.arange(len(keys)); h = 0.38
        for j, fkey in enumerate(feats):
            ax[1].barh(y + (j - 0.5) * h, [s[k][fkey] for k in keys], h, color=cols[j], label=fkey)
        ax[1].axvline(0, color="k", lw=.7); ax[1].set_yticks(y)
        ax[1].set_yticklabels([k.replace(" | ", "\n") for k in keys], fontsize=6)
        ax[1].set_title("Fitted SPEAKER weights\ninform_remaining = adaptivity; dead_assoc = eliminated-avoidance", fontsize=9)
        ax[1].set_xlabel("fitted weight"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3, axis="x")
        fig.suptitle("RSA / conditional-logit fit to LLM Codenames — measured KLs as fitted parameters", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.94]); pdf.savefig(fig); plt.close(fig)
    print("\nwrote", os.path.join(ROOT, "codenames_rsa_fit.pdf"))


if __name__ == "__main__":
    main()
