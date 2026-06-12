"""Drift vs selection — the measurements that actually mean something.

This replaces the (uninformative) single-run similarity heatmap as the headline.
It uses the analytic null model (pure numpy, CPU) to demonstrate the core
research contrast, and emits the SAME figures the LLM runs will produce:

  1. accuracy_overlay.png : P(group answer correct) over rounds, NEUTRAL drift
     (no ground-truth signal) vs SELECTION (persistent bias toward the truth).
     Neutral sits at chance (1/K); selection climbs above it.

  2. final_accuracy_vs_N.png : the drift->selection crossover. Under weak
     selection + the Hard channel, small N locks into a random vertex before the
     signal can act (drift wins); larger N converges slowly enough that selection
     accumulates and the TRUE answer wins more reliably.

  3. consensus_time_vs_N.png : why (1) and (2) happen — consensus time grows with
     N, giving selection more rounds to act.

    python run_selection_study.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from qsg import analysis
from qsg.qsg_reference import consensus_time, run_reference

K = 7
GT = 0                      # ground-truth label index
CHANCE = 1.0 / K
M_HARD = 1.0               # Hard channel: where drift is strongest
ALPHA = 0.3
ROUNDS = 250
NS = np.array([2, 4, 8, 16, 32, 64])
SEEDS = 120
SELECTION = 0.01          # weak persistent bias toward truth (models partial info)


def run_condition(n: int, selection: float, seed: int):
    """One numpy QSG run. Unbiased random init; optional weak selection nudge."""
    res = run_reference(
        n, K, ALPHA, M_HARD, ROUNDS, seed=seed,
        ground_truth=GT, selection_strength=selection,
    )
    return res


def accuracy_curve_over_seeds(n: int, selection: float):
    """Mean / sem of group-correct over rounds, averaged across seeds."""
    accs = []
    for s in range(SEEDS):
        res = run_condition(n, selection, seed=s)
        accs.append(analysis.accuracy_curves(res.beliefs, GT)["group_correct"])
    accs = np.array(accs)                       # (seeds, rounds+1)
    return accs.mean(0), accs.std(0) / np.sqrt(SEEDS)


def final_acc_and_consensus(n: int, selection: float):
    finals, ctimes = [], []
    for s in range(SEEDS):
        res = run_condition(n, selection, seed=s)
        finals.append(float(np.argmax(res.beliefs[-1].mean(0)) == GT))
        ct = consensus_time(res.U, 0.95)
        ctimes.append(ct if ct is not None else ROUNDS)
    finals, ctimes = np.array(finals), np.array(ctimes)
    return (finals.mean(), finals.std() / np.sqrt(SEEDS),
            ctimes.mean(), ctimes.std() / np.sqrt(SEEDS))


def main() -> None:
    out = Path("results/_selection_study")
    out.mkdir(parents=True, exist_ok=True)
    rounds_axis = np.arange(ROUNDS + 1)

    # 1) accuracy over rounds at a representative N, neutral vs selection
    N_rep = 16
    print(f"accuracy overlay at N={N_rep} ({SEEDS} seeds each) ...")
    neu_mean, neu_sem = accuracy_curve_over_seeds(N_rep, 0.0)
    sel_mean, sel_sem = accuracy_curve_over_seeds(N_rep, SELECTION)
    analysis.save_accuracy_overlay(
        {
            f"neutral drift (N={N_rep})": (rounds_axis, neu_mean, neu_sem),
            f"selection (N={N_rep})": (rounds_axis, sel_mean, sel_sem),
        },
        out, chance=CHANCE,
    )
    print(f"  neutral final acc = {neu_mean[-1]:.2f} (chance {CHANCE:.2f}),  "
          f"selection final acc = {sel_mean[-1]:.2f}")

    # 2 & 3) sweep over N
    print(f"sweeping N over {list(NS)} ...")
    sel_acc_m, sel_acc_s, neu_acc_m, neu_acc_s = [], [], [], []
    sel_ct_m, sel_ct_s, neu_ct_m, neu_ct_s = [], [], [], []
    for n in NS:
        am, asem, cm, csem = final_acc_and_consensus(n, SELECTION)
        sel_acc_m.append(am); sel_acc_s.append(asem); sel_ct_m.append(cm); sel_ct_s.append(csem)
        am, asem, cm, csem = final_acc_and_consensus(n, 0.0)
        neu_acc_m.append(am); neu_acc_s.append(asem); neu_ct_m.append(cm); neu_ct_s.append(csem)
        print(f"  N={n:3d}  selection acc={sel_acc_m[-1]:.2f}  "
              f"neutral acc={neu_acc_m[-1]:.2f}  consensus_t(sel)={sel_ct_m[-1]:.0f}")

    analysis.save_vs_N(
        NS,
        {"selection": (np.array(sel_acc_m), np.array(sel_acc_s)),
         "neutral drift": (np.array(neu_acc_m), np.array(neu_acc_s))},
        out, ylabel="P(final group answer = truth)",
        title="Drift → selection crossover: does larger N select the TRUE answer?",
        chance=CHANCE, prefix="final_accuracy_vs_N",
    )
    analysis.save_vs_N(
        NS,
        {"selection": (np.array(sel_ct_m), np.array(sel_ct_s)),
         "neutral drift": (np.array(neu_ct_m), np.array(neu_ct_s))},
        out, ylabel="consensus time (rounds to U≥0.95)",
        title="Consensus time grows with N (gives selection time to act)",
        prefix="consensus_time_vs_N", logx=True,
    )
    print(f"\nDone. Figures -> {out}/")


if __name__ == "__main__":
    main()
