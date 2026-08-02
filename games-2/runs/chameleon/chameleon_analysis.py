"""CHAMELEON Phase 1 analysis (no GPU) — joins stimuli + battery + baselines.

Reports (per the level-0 gates in README.md), with binomial SEs and n everywhere:

  1. vote accuracy by condition x tier x impostor_style x agent role — agent vs
     centroid vs lexical baselines, plus agent–centroid AGREEMENT (where they agree,
     nothing mentalistic is demonstrated) and the dissoc-condition split.
  2. calibration: vote-distribution top-mass/entropy in faithful vs all_random vs
     all_same (scapegoat test: confident votes when no impostor exists).
  3. self-knowledge & concealment: P(self=impostor | role), P(vote self | impostor),
     word recovery (impostor guessing the civilian word) — the behavioural
     dissociation triple.
  4. seat-permutation consistency: does the vote follow the clues across twins?
  5. per-round belief trajectories (if the battery ran PERROUND=1).

Env: STIMULI BATTERY(runs/chameleon/battery/battery_QwenInst32.jsonl)
     BASELINES(runs/chameleon/battery/baselines.jsonl)
     OUT_DIR(runs/chameleon/analysis)
Out: analysis.json + analysis.pdf + printed tables.
"""
from __future__ import annotations
import os
import json
import math
import collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

STIMULI = os.environ.get("STIMULI", "runs/chameleon/stimuli/stimuli.jsonl")
BATTERY = os.environ.get("BATTERY", "runs/chameleon/battery/battery_QwenInst32.jsonl")
BASELINES = os.environ.get("BASELINES", "runs/chameleon/battery/baselines.jsonl")
OUT_DIR = os.environ.get("OUT_DIR", "runs/chameleon/analysis")


def se(p, n):
    return math.sqrt(max(p * (1 - p), 1e-9) / n) if n else float("nan")


def frac(xs):
    xs = [x for x in xs if x is not None]
    n = len(xs)
    p = float(np.mean(xs)) if n else float("nan")
    return {"p": p, "se": se(p, n), "n": n}


def entropy(dist):
    ps = np.array(list(dist.values()), float)
    ps = ps / ps.sum()
    return float(-(ps * np.log(ps + 1e-12)).sum())


def main():
    stims = {s["id"]: s for s in map(json.loads, open(STIMULI))}
    batt = {r["id"]: r for r in map(json.loads, open(BATTERY))}
    base = {}
    if os.path.exists(BASELINES):
        base = {r["id"]: r for r in map(json.loads, open(BASELINES))}
    rows = []
    for sid, s in stims.items():
        if sid not in batt:
            continue
        r, b = batt[sid], base.get(sid, {})
        truth = s["true_impostor_seat"]
        vote_seat = s["players"].index(r["vote_top"])
        word_top = max(r["word_dist"], key=r["word_dist"].get)
        rows.append({
            "id": sid, "cond": s["condition"], "tier": s["tier"], "role": s["true_role"],
            "style": s["impostor_style"], "perm_of": s["permutation_of"],
            "truth_seat": truth, "vote_seat": vote_seat,
            "vote_correct": (vote_seat == truth) if truth is not None else None,
            "vote_self": vote_seat == s["agent_seat"],
            "vote_top_mass": max(r["vote_dist"].values()),
            "vote_entropy": entropy(r["vote_dist"]),
            "centroid_seat": b.get("centroid_seat"),
            "centroid_correct": b.get("centroid_correct"),
            "lexical_correct": b.get("lexical_correct"),
            "agree_centroid": (vote_seat == b.get("centroid_seat")) if b else None,
            "self_p": r["self_p_yes"],
            "word_top": word_top,
            "word_correct": (word_top == s["civilian_word"]) if s["civilian_word"] else None,
            "word_gen_correct": (r.get("word_gen") == s["civilian_word"])
            if s["civilian_word"] and r.get("word_gen") else None,
            "perround": r.get("perround"),
        })

    res = {"n": len(rows)}

    # 1. vote accuracy vs baselines
    acc = {}
    faith = [r for r in rows if r["cond"] in ("faithful", "dissoc") and r["perm_of"] is None]
    for key, sel in [("all_faithful", faith)] + \
                    [(f"tier_{t}", [r for r in faith if r["tier"] == t]) for t in ("near", "mid", "far")] + \
                    [(f"style_{st}", [r for r in faith if r["style"] == st]) for st in ("own", "mimic")] + \
                    [("dissoc", [r for r in faith if r["cond"] == "dissoc"]),
                     ("agent_is_impostor", [r for r in faith if r["role"] == "impostor"])]:
        acc[key] = {"agent": frac([r["vote_correct"] for r in sel]),
                    "centroid": frac([r["centroid_correct"] for r in sel]),
                    "lexical": frac([r["lexical_correct"] for r in sel]),
                    "agent_centroid_agree": frac([r["agree_centroid"] for r in sel])}
    res["vote_accuracy"] = acc

    # 2. calibration / scapegoating
    cal = {}
    for cond in ("faithful", "dissoc", "all_random", "all_same"):
        sel = [r for r in rows if r["cond"] == cond and r["perm_of"] is None]
        if sel:
            cal[cond] = {"top_mass": frac([r["vote_top_mass"] for r in sel]),
                         "entropy": frac([r["vote_entropy"] for r in sel])}
    res["calibration"] = cal

    # 3. self-knowledge / concealment / word recovery
    imp = [r for r in rows if r["role"] == "impostor"]
    civ = [r for r in rows if r["role"] == "civilian"]
    res["self_knowledge"] = {
        "self_p_when_impostor": frac([r["self_p"] for r in imp]),
        "self_p_when_civilian": frac([r["self_p"] for r in civ]),
        "vote_self_when_impostor": frac([r["vote_self"] for r in imp]),
        "word_recovery_when_impostor": frac([r["word_correct"] for r in imp]),
        "word_correct_when_civilian": frac([r["word_correct"] for r in civ]),
        "word_gen_recovery_when_impostor": frac([r["word_gen_correct"] for r in imp]),
        "concealed_knowledge": frac([(r["self_p"] > 0.5 and r["word_correct"] and not r["vote_self"])
                                     for r in imp if r["word_correct"] is not None]),
    }

    # 3b. position bias: vote-seat histogram vs truth-seat histogram (a seat-voting
    # agent shows a skewed vote hist over a flat truth hist; late-seat fraction is
    # the summary stat)
    n_seats = max(len(json.loads(open(BATTERY).readline())["vote_dist"]), 2)
    vote_hist = collections.Counter(r["vote_seat"] for r in rows)
    truth_hist = collections.Counter(r["truth_seat"] for r in rows if r["truth_seat"] is not None)
    res["position_bias"] = {
        "vote_seat_hist": {str(s): vote_hist.get(s, 0) for s in range(n_seats)},
        "truth_seat_hist": {str(s): truth_hist.get(s, 0) for s in range(n_seats)},
        "late_seat_vote_frac": frac([r["vote_seat"] >= n_seats - 2 for r in rows]),
        "late_seat_truth_frac": frac([r["truth_seat"] >= n_seats - 2 for r in rows
                                      if r["truth_seat"] is not None]),
    }

    # 4. seat-permutation consistency
    pairs = []
    for r in rows:
        if r["perm_of"] and r["perm_of"] in {x["id"] for x in rows}:
            orig = next(x for x in rows if x["id"] == r["perm_of"])
            if orig["truth_seat"] is not None:
                pairs.append(orig["vote_correct"] == r["vote_correct"])
    res["perm_consistency"] = frac(pairs)

    # 5. per-round trajectories
    traj = collections.defaultdict(list)
    for r in rows:
        if r["perround"]:
            for pr in r["perround"]:
                traj[(r["role"], pr["round"])].append(pr["self_p_yes"])
    if traj:
        res["self_p_by_round"] = {f"{role}_r{rd}": frac(v) for (role, rd), v in sorted(traj.items())}

    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(res, open(os.path.join(OUT_DIR, "analysis.json"), "w"), indent=1)

    # ---- pdf ----
    with PdfPages(os.path.join(OUT_DIR, "analysis.pdf")) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        ax = axes[0, 0]
        keys = ["tier_near", "tier_mid", "tier_far", "dissoc"]
        x = np.arange(len(keys))
        for i, who in enumerate(("agent", "centroid", "lexical")):
            ys = [acc[k][who]["p"] for k in keys]
            es = [acc[k][who]["se"] for k in keys]
            ax.bar(x + (i - 1) * 0.25, ys, 0.25, yerr=es, label=who, capsize=3)
        ax.axhline(1 / 5, ls="--", c="k", alpha=.4)
        ax.set_xticks(x, keys)
        ax.set_ylabel("vote accuracy")
        ax.legend()
        ax.set_title("impostor detection vs level-0 baselines")

        ax = axes[0, 1]
        conds = list(cal)
        ax.bar(conds, [cal[c]["top_mass"]["p"] for c in conds],
               yerr=[cal[c]["top_mass"]["se"] for c in conds], capsize=3)
        ax.axhline(1 / 5, ls="--", c="k", alpha=.4)
        ax.set_ylabel("vote top mass")
        ax.set_title("confidence when the announced story is false\n(all_random / all_same = scapegoat test)")

        ax = axes[1, 0]
        sk = res["self_knowledge"]
        labels = ["P(self|imp)", "P(self|civ)", "vote-self|imp", "word-rec|imp", "concealed"]
        vals = [sk["self_p_when_impostor"]["p"], sk["self_p_when_civilian"]["p"],
                sk["vote_self_when_impostor"]["p"], sk["word_recovery_when_impostor"]["p"],
                sk["concealed_knowledge"]["p"]]
        ax.bar(labels, vals)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=20)
        ax.set_title("self-knowledge / concealment (behavioural dissociation)")

        ax = axes[1, 1]
        if traj:
            for role in ("impostor", "civilian"):
                pts = sorted((rd, np.mean(v)) for (ro, rd), v in traj.items() if ro == role)
                if pts:
                    ax.plot(*zip(*pts), "o-", label=f"agent={role}")
            ax.set_xlabel("round")
            ax.set_ylabel("P(self = impostor)")
            ax.legend()
            ax.set_title("belief trajectory (forked per-round elicitation)")
        else:
            ax.axis("off")
            ax.text(.5, .5, "run battery with PERROUND=1\nfor belief trajectories", ha="center")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    for k in ("all_faithful", "dissoc"):
        a = acc[k]
        print(f"{k:>14}: agent {a['agent']['p']:.2f}±{a['agent']['se']:.2f} | "
              f"centroid {a['centroid']['p']:.2f} | agree {a['agent_centroid_agree']['p']:.2f} "
              f"(n={a['agent']['n']})")
    print(f"self-knowledge: {json.dumps({k: round(v['p'], 2) for k, v in res['self_knowledge'].items()})}")
    pb = res["position_bias"]
    print(f"position bias: votes {pb['vote_seat_hist']} vs truth {pb['truth_seat_hist']} "
          f"(late-seat vote frac {pb['late_seat_vote_frac']['p']:.2f} vs "
          f"truth {pb['late_seat_truth_frac']['p']:.2f})")
    print(f"[analysis] wrote {OUT_DIR}/analysis.json + analysis.pdf")


if __name__ == "__main__":
    main()
