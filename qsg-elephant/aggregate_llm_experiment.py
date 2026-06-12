"""Aggregate LLM runs into the drift-vs-selection overlay (the meaningful figure).

Groups runs by the neutral_ablation flag, averages group-accuracy over seeds, and
emits the SAME accuracy overlay as the numpy study — but from real Llama beliefs:

  * full      (agents see partial clues)  -> ground-truth SELECTION
  * ablation  (clues stripped)            -> neutral DRIFT  (should sit at chance)

    python aggregate_llm_experiment.py results_exp --out results_exp/_summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qsg import analysis


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    exp = Path(args.exp_dir)
    out = Path(args.out) if args.out else exp / "_summary"
    out.mkdir(parents=True, exist_ok=True)

    runs = [d for d in exp.iterdir() if (d / "manifest.json").exists()]
    by_cond: dict[str, list[np.ndarray]] = {"full (selection)": [], "ablation (drift)": []}
    by_cond_frac: dict[str, list[np.ndarray]] = {"full (selection)": [], "ablation (drift)": []}
    gt = K = None
    for d in runs:
        man = json.loads((d / "manifest.json").read_text())
        beliefs = np.load(d / "beliefs.npy")
        gt = man["ground_truth_index"]
        K = len(man["candidates"])
        acc = analysis.accuracy_curves(beliefs, gt)
        cond = "ablation (drift)" if man["neutral_ablation"] else "full (selection)"
        by_cond[cond].append(acc["group_correct"])
        by_cond_frac[cond].append(acc["frac_correct"])
        print(f"  {d.name}: cond={cond} final_group_correct={acc['group_correct'][-1]:.0f} "
              f"final_frac={acc['frac_correct'][-1]:.2f}")

    chance = 1.0 / K
    group_curves, frac_curves = {}, {}
    for cond, arrs in by_cond.items():
        if not arrs:
            continue
        a = np.array(arrs)
        rounds = np.arange(a.shape[1])
        group_curves[cond] = (rounds, a.mean(0), a.std(0) / np.sqrt(len(arrs)))
        f = np.array(by_cond_frac[cond])
        frac_curves[cond] = (rounds, f.mean(0), f.std(0) / np.sqrt(len(arrs)))

    analysis.save_accuracy_overlay(
        group_curves, out, chance=chance,
        title="Llama-3.1-8B: neutral drift (ablation) vs ground-truth selection",
        prefix="llm_accuracy_overlay_group",
    )
    analysis.save_accuracy_overlay(
        frac_curves, out, chance=chance,
        title="Llama-3.1-8B: fraction of agents correct — drift vs selection",
        prefix="llm_accuracy_overlay_frac",
    )
    print(f"\nFigures -> {out}/")


if __name__ == "__main__":
    main()
