"""Probabilistic (independent-sampling collision) model of Game-1 convergence — no GPU.

Model: at each turn the two players independently sample from their next-word
distributions; P(meet at turn t) = sum_w pA(w)·pB(w) (top-15 first-token dists from the
*_crossKL.json replays, temperature-adjusted). Tests whether the picks add any
coordination beyond the distributions themselves.

Outputs (into <KL_DIR>/):
  collision_model.pdf   p1: calibration (predicted collision vs observed agree rate,
                            with AUC);  p2: overlap trajectories, met vs no-meet games
                            (relative game progress);  p3: per-condition pred-vs-obs.
  collision_model.json  raw per-turn (condition, game, turn, c, agreed)

Known residual: first-token collisions that split at the word level (e.g. 'se' ->
seamless/seamstress) make the top calibration bin overpredict — the model is exact only
at the word level, which top-K first-token dists approximate.

Env: KL_DIR(runs/game-1/2_restricted_core/qwen32_cap24/kl) TEMP(0.7)
     CONDS(reactive,restrict-city,restrict-fruit,nolist-city,nolist-fruit,repeatok-city,repeatok-fruit)
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
OUT_DIR = os.environ.get("OUT_DIR", "runs/game-1/6_analyses/collision_model")
TEMP = float(os.environ.get("TEMP", "0.7"))
CONDS = os.environ.get(
    "CONDS",
    "reactive,restrict-city,restrict-fruit,nolist-city,nolist-fruit,repeatok-city,repeatok-fruit"
).split(",")


def overlap(da, db):
    def prep(d):
        w = np.array(list(d.values()), dtype=float) ** (1 / TEMP)
        return {k: v for k, v in zip(d.keys(), w / w.sum())}
    A, B = prep(da), prep(db)
    return float(sum(p * B.get(k, 0.0) for k, p in A.items()))


def main():
    rows = []
    for cond in CONDS:
        path = os.path.join(KL_DIR, f"{cond}_crossKL.json")
        if not os.path.exists(path):
            print(f"[collision] skip {cond} (no {path})")
            continue
        for g, turns in json.load(open(path)).items():
            met = bool(turns[-1]["agreed"])
            T = len(turns)
            for i, t in enumerate(turns):
                rows.append({"cond": cond, "game": g, "turn": t["turn"],
                             "progress": (i + 1) / T, "game_met": met,
                             "c": overlap(t["topA"], t["topB"]), "agreed": bool(t["agreed"])})
    c = np.array([r["c"] for r in rows])
    ag = np.array([r["agreed"] for r in rows], dtype=bool)

    ranks = np.empty(len(c)); ranks[np.argsort(c)] = np.arange(len(c))
    n1, n0 = ag.sum(), (~ag).sum()
    auc = float((ranks[ag].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))

    os.makedirs(OUT_DIR, exist_ok=True)
    with PdfPages(os.path.join(OUT_DIR, "collision_model.pdf")) as pdf:
        # p1 calibration
        bins = [0, .02, .05, .1, .2, .35, .6, 1.01]
        xs, ys, ns = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (c >= lo) & (c < hi)
            if m.sum():
                xs.append(c[m].mean()); ys.append(ag[m].mean()); ns.append(int(m.sum()))
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot([0, 1], [0, 1], "k--", alpha=.4, label="perfect calibration")
        ax.plot(xs, ys, "o-", color="tab:blue")
        for x, y, n in zip(xs, ys, ns):
            ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(6, -10), fontsize=7)
        ax.set_xlabel("predicted collision prob  Σ pA·pB  (temp-adjusted top-15)")
        ax.set_ylabel("observed agreement rate")
        ax.set_title(f"Independent-sampling collision model — calibration\n"
                     f"AUC={auc:.3f} over {len(c)} turns, {int(n1)} agreements "
                     f"({len(CONDS)} conditions)\n"
                     f"top-bin overprediction = first-token collisions that split at word level",
                     fontsize=9)
        ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # p2 overlap trajectories met vs no-meet
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for met, color, label in ((True, "tab:green", "games that meet"),
                                  (False, "tab:red", "games that never meet")):
            sel = [r for r in rows if r["game_met"] == met]
            pbins = np.linspace(0, 1, 7)
            xs2, ys2 = [], []
            for lo, hi in zip(pbins[:-1], pbins[1:]):
                vals = [r["c"] for r in sel if lo < r["progress"] <= hi]
                if vals:
                    xs2.append((lo + hi) / 2); ys2.append(np.mean(vals))
            ax.plot(xs2, ys2, "o-", color=color, label=f"{label} (n turns={len(sel)})")
        ax.set_xlabel("relative game progress"); ax.set_ylabel("mean distribution overlap c")
        ax.set_title("Overlap trajectory: met games' distributions converge; failed games' never do",
                     fontsize=10)
        ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # p3 per-condition pred vs obs met
        fig, ax = plt.subplots(figsize=(8, 5.5))
        conds, preds, obss = [], [], []
        for cond in CONDS:
            sub = [r for r in rows if r["cond"] == cond]
            if not sub:
                continue
            games = {}
            for r in sub:
                games.setdefault(r["game"], []).append(r)
            pred = np.mean([1 - np.prod([1 - r["c"] for r in g]) for g in games.values()])
            obs = np.mean([g[-1]["agreed"] for g in games.values()])
            conds.append(cond); preds.append(pred); obss.append(obs)
        x = np.arange(len(conds))
        ax.bar(x - .18, preds, .36, label="predicted P(meet) (observed turns only)", color="tab:blue")
        ax.bar(x + .18, obss, .36, label="observed met_frac", color="tab:orange")
        ax.set_xticks(x); ax.set_xticklabels(conds, rotation=30, ha="right", fontsize=8)
        ax.set_title("Per-condition predicted vs observed meeting (survival-biased: met games "
                     "stop early)", fontsize=9)
        ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    json.dump(rows, open(os.path.join(OUT_DIR, "collision_model.json"), "w"))
    print(f"[collision] AUC {auc:.3f}; wrote collision_model.pdf + .json -> {OUT_DIR}")


if __name__ == "__main__":
    main()
