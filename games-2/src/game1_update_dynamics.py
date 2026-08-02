"""DYNAMICAL model of Game-1: how does each player's next-word DISTRIBUTION update,
turn by turn — and is it a Bayesian update? (The collision model only says picks are
independent samples; this asks what moves the distributions, including in games that
never converge.) No GPU — works on the replayed per-turn dists in *_crossKL.json.

Bayesian-update signatures tested, per player A (same for B), met vs no-meet games:
  1. SURPRISE-COUPLED UPDATES: step size KL(pA_{t+1} ‖ pA_t) should scale with the
     surprisal -log pA_t(partner's word). Control: surprisal of A's OWN word (sampled
     from pA_t, so a Bayesian partner-tracker shouldn't need it).
  2. ACCOMMODATION: mass A places on the partner's last word should RISE after seeing
     it (posterior moves toward evidence).
  3. CONTRACTION: entropy of pA should fall over the game (posterior concentrates).

Word mass = summed prob of top-15 token keys that PREFIX the word (first-token approx;
surprisal is floored at the top-15 minimum => censored, reported as such).

Env: KL_DIR(runs/game-1/2_restricted_core/qwen32_cap24/kl)
     CONDS(reactive,restrict-city,restrict-fruit) OUT_DIR(runs/game-1/6_analyses/update_dynamics)
Out: update_dynamics.pdf + update_dynamics.json + printed stats
"""
from __future__ import annotations
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

KL_DIR = os.environ.get("KL_DIR", "runs/game-1/2_restricted_core/qwen32_cap24/kl")
CONDS = os.environ.get("CONDS", "reactive,restrict-city,restrict-fruit").split(",")
OUT_DIR = os.environ.get("OUT_DIR", "runs/game-1/6_analyses/update_dynamics")


def wmass(dist, word):
    m = sum(p for k, p in dist.items() if len(k) > 1 and word.lower().startswith(k.lower()))
    return m / sum(dist.values())


def stepkl(d1, d2):
    keys = set(d1) | set(d2)
    eps = 1e-4
    p = np.array([d1.get(k, 0) + eps for k in keys]); p /= p.sum()
    q = np.array([d2.get(k, 0) + eps for k in keys]); q /= q.sum()
    return float((q * np.log(q / p)).sum())          # KL(new ‖ old)


def entropy(d):
    p = np.array(list(d.values())); p = p / p.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def spearman(a, b):
    a, b = np.asarray(a), np.asarray(b)
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float("nan")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for cond in CONDS:
        path = os.path.join(KL_DIR, f"{cond}_crossKL.json")
        if not os.path.exists(path):
            continue
        for g, turns in json.load(open(path)).items():
            met = bool(turns[-1]["agreed"])
            T = len(turns)
            for pl, dk, own_k, other_k in (("A", "topA", "pickA", "pickB"),
                                            ("B", "topB", "pickB", "pickA")):
                for i in range(T - 1):
                    d0, d1 = turns[i][dk], turns[i + 1][dk]
                    pos = [v for v in d0.values() if v > 0]
                    floor = max(min(pos) / sum(d0.values()), 1e-4)
                    mp = wmass(d0, turns[i][other_k])
                    ms = wmass(d0, turns[i][own_k])
                    rows.append({
                        "cond": cond, "game": g, "player": pl, "turn": turns[i]["turn"],
                        "progress": (i + 1) / T, "game_met": met,
                        "update": stepkl(d0, d1),
                        "surp_partner": -np.log(max(mp, floor)),
                        "surp_self": -np.log(max(ms, floor)),
                        "censored": mp <= floor,
                        "accommodation": wmass(d1, turns[i][other_k]) - mp,
                        "entropy": entropy(d0)})
    met = [r for r in rows if r["game_met"]]
    nomet = [r for r in rows if not r["game_met"]]

    stats = {}
    for label, sel in (("met", met), ("no_meet", nomet)):
        stats[label] = {
            "n": len(sel),
            "r_update_partner_surp": spearman([r["surp_partner"] for r in sel],
                                              [r["update"] for r in sel]),
            "r_update_self_surp": spearman([r["surp_self"] for r in sel],
                                           [r["update"] for r in sel]),
            "mean_accommodation": float(np.mean([r["accommodation"] for r in sel])),
            "frac_censored": float(np.mean([r["censored"] for r in sel])),
            "r_entropy_progress": spearman([r["progress"] for r in sel],
                                           [r["entropy"] for r in sel])}
        s = stats[label]
        print(f"[dyn] {label:>7} (n={s['n']}): update~partner-surp r={s['r_update_partner_surp']:+.2f} "
              f"| update~self-surp r={s['r_update_self_surp']:+.2f} "
              f"| accommodation {s['mean_accommodation']:+.4f} "
              f"| entropy~progress r={s['r_entropy_progress']:+.2f} "
              f"| censored {s['frac_censored']:.2f}")

    with PdfPages(os.path.join(OUT_DIR, "update_dynamics.pdf")) as pdf:
        # p1: update vs partner surprisal, binned, met vs no-meet
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for sel, color, label in ((met, "tab:green", "met games"),
                                  (nomet, "tab:red", "no-meet games")):
            s = np.array([r["surp_partner"] for r in sel])
            u = np.array([r["update"] for r in sel])
            qs = np.quantile(s, np.linspace(0, 1, 7))
            xs, ys, es = [], [], []
            for lo, hi in zip(qs[:-1], qs[1:]):
                m = (s >= lo) & (s <= hi)
                if m.sum() > 3:
                    xs.append(s[m].mean()); ys.append(u[m].mean())
                    es.append(u[m].std() / np.sqrt(m.sum()))
            ax.errorbar(xs, ys, yerr=es, fmt="o-", color=color,
                        label=f"{label} (r={stats['met' if sel is met else 'no_meet']['r_update_partner_surp']:+.2f})")
        ax.set_xlabel("surprisal of PARTNER's word under current dist  -log p(w_partner)")
        ax.set_ylabel("next-turn update  KL(p_t+1 ‖ p_t)")
        ax.set_title("Bayesian signature 1: update size vs partner surprisal", fontsize=10)
        ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # p2: accommodation by progress
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for sel, color, label in ((met, "tab:green", "met"), (nomet, "tab:red", "no-meet")):
            pr = np.array([r["progress"] for r in sel])
            ac = np.array([r["accommodation"] for r in sel])
            bins = np.linspace(0, 1, 7)
            xs = [(lo + hi) / 2 for lo, hi in zip(bins[:-1], bins[1:])]
            ys = [ac[(pr > lo) & (pr <= hi)].mean() if ((pr > lo) & (pr <= hi)).sum() else np.nan
                  for lo, hi in zip(bins[:-1], bins[1:])]
            ax.plot(xs, ys, "o-", color=color, label=label)
        ax.axhline(0, color="k", alpha=.4)
        ax.set_xlabel("relative game progress")
        ax.set_ylabel("Δ mass on partner's last word (accommodation)")
        ax.set_title("Bayesian signature 2: does the posterior move toward the evidence?",
                     fontsize=10)
        ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # p3: entropy trajectory
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for sel, color, label in ((met, "tab:green", "met"), (nomet, "tab:red", "no-meet")):
            pr = np.array([r["progress"] for r in sel])
            en = np.array([r["entropy"] for r in sel])
            bins = np.linspace(0, 1, 7)
            xs = [(lo + hi) / 2 for lo, hi in zip(bins[:-1], bins[1:])]
            ys = [en[(pr > lo) & (pr <= hi)].mean() if ((pr > lo) & (pr <= hi)).sum() else np.nan
                  for lo, hi in zip(bins[:-1], bins[1:])]
            ax.plot(xs, ys, "o-", color=color, label=label)
        ax.set_xlabel("relative game progress"); ax.set_ylabel("entropy of top-15 dist")
        ax.set_title("Bayesian signature 3: posterior contraction", fontsize=10)
        ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    json.dump({"stats": stats, "rows": rows},
              open(os.path.join(OUT_DIR, "update_dynamics.json"), "w"))
    print(f"[dyn] wrote update_dynamics.pdf + .json -> {OUT_DIR}")


if __name__ == "__main__":
    main()
