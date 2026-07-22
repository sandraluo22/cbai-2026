"""Behavioral test: does the guesser ACT on the spymaster's regime (not just encode it)?

Uses the metadata replay (meta_<guesser>.jsonl, 3 modes with behavioral read-outs).

  (3a) Repeated-clue exploration (mode 0): when the spymaster repeats a clue, does the
       guesser guess FURTHER from that clue's literal meaning (clue<->guess MiniLM sim
       falling with repeat_count)? A falling sim = it stops trusting the exhausted clue
       and explores -- behaviorally responding to the stuck spymaster.
  (3b) Non-adaptive spymaster (mode 2): it clues for already-found words
       (clue_topboard_found high). Does the guesser recognize the wasted clue -- i.e., is
       its guess pulled toward the found region less than the clue would suggest -- and
       how do recovery / entropy compare across modes?

Usage:  python src/codenames_probe_behavior.py [runs/codenames/probe3]
Out:    <dir>/codenames_probe_behavior.{pdf,json}  (+ printed report)
"""
from __future__ import annotations

import os
import sys
import glob
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

DIR = sys.argv[1] if len(sys.argv) > 1 else "runs/codenames/probe3"
MODE_NAME = {0: "memoryless", 1: "memory", 2: "nonadaptive"}


def load():
    rows = []
    for path in glob.glob(os.path.join(DIR, "meta_*.jsonl")):
        g = os.path.basename(path).replace("meta_", "").replace(".jsonl", "")
        for l in open(path):
            if l.strip():
                r = json.loads(l); r["guesser"] = g; rows.append(r)
    return pd.DataFrame(rows)


def main():
    df = load()
    df["mode_name"] = df["mode"].map(MODE_NAME)
    report = {}

    # ---- (3a) repeated-clue exploration (mode 0) ----
    print("=" * 76)
    print("[3a] REPEATED-CLUE EXPLORATION (mode 0): clue<->guess sim vs repeat_count")
    a = {}
    for g, sub in df[df["mode"] == 0].groupby("guesser"):
        by = sub.groupby("repeat_count").agg(sim=("clue_guess_sim", "mean"),
                                             sim_se=("clue_guess_sim", "sem"),
                                             ncorr=("n_correct", "mean"), n=("clue", "size"))
        rc = sub["repeat_count"].values.astype(float); sim = sub["clue_guess_sim"].values
        slope, _, r, p, _ = stats.linregress(rc, sim)
        a[g] = {"slope": float(slope), "r": float(r), "p": float(p),
                "by_repeat": {int(k): {"sim": float(v.sim), "n": int(v.n), "ncorr": float(v.ncorr)}
                              for k, v in by.iterrows()}}
        print(f"  {g}: slope(sim vs repeat_count) = {slope:+.4f}  (r={r:+.2f}, p={p:.3f})")
        for k, v in by.iterrows():
            print(f"      repeat#{int(k)}: clue-guess sim={v.sim:+.3f}  n_correct={v.ncorr:.2f}  (n={int(v.n)})")
    report["repeat_exploration"] = a

    # ---- (3b) non-adaptive spymaster ----
    print("\n[3b] NON-ADAPTIVE SPYMASTER (mode 2) vs others")
    b = {"by_mode": {}, "found_region": {}}
    agg = df.groupby("mode_name").agg(clue_topboard_found=("clue_topboard_found", "mean"),
                                      clue_guess_sim=("clue_guess_sim", "mean"),
                                      target_mass=("target_mass", "mean"),
                                      belief_entropy=("belief_entropy", "mean"),
                                      n_correct=("n_correct", "mean"), n=("clue", "size"))
    print(agg.to_string(float_format=lambda v: f"{v:.3f}"))
    b["by_mode"] = json.loads(agg.to_json(orient="index"))

    # does the guesser guess AWAY from the found region when the clue points there?
    print("\n  clue<->guess sim when the clue's top board word is ALREADY FOUND vs not:")
    for g, sub in df.groupby("guesser"):
        f1 = sub[sub["clue_topboard_found"] == 1]["clue_guess_sim"]
        f0 = sub[sub["clue_topboard_found"] == 0]["clue_guess_sim"]
        if len(f1) > 5 and len(f0) > 5:
            t, p = stats.mannwhitneyu(f1, f0)
            print(f"    {g}: found-region clue sim={f1.mean():+.3f} (n={len(f1)}) vs "
                  f"non-found={f0.mean():+.3f} (n={len(f0)})  Δ={f1.mean()-f0.mean():+.3f} (MWU p={p:.3f})")
            b["found_region"][g] = {"sim_found": float(f1.mean()), "sim_notfound": float(f0.mean()),
                                    "delta": float(f1.mean() - f0.mean()), "p": float(p)}
    report["nonadaptive"] = b

    # ---- figure ----
    with PdfPages(os.path.join(DIR, "codenames_probe_behavior.pdf")) as pdf:
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.7))
        # (3a) sim vs repeat_count
        for g, sub in df[df["mode"] == 0].groupby("guesser"):
            by = sub.groupby("repeat_count").agg(sim=("clue_guess_sim", "mean"), se=("clue_guess_sim", "sem"))
            ax[0].errorbar(by.index, by["sim"], yerr=by["se"], marker="o", capsize=2, label=g)
        ax[0].set_xlabel("clue repeat count"); ax[0].set_ylabel("clue↔guess MiniLM sim")
        ax[0].set_title("(3a) mode 0: does the guess move OFF a repeated clue?", fontsize=9)
        ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
        # (3b) recovery + entropy by mode
        order = ["memoryless", "memory", "nonadaptive"]
        present = [m for m in order if m in agg.index]
        x = np.arange(len(present))
        ax[1].bar(x - 0.2, [agg.loc[m, "target_mass"] for m in present], 0.4, label="recovery", color="tab:blue")
        ax[1].bar(x + 0.2, [agg.loc[m, "n_correct"] for m in present], 0.4, label="n_correct/round", color="tab:green")
        ax[1].set_xticks(x); ax[1].set_xticklabels(present, fontsize=8)
        ax[1].set_title("(3b) guesser success by spymaster mode", fontsize=9); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, axis="y")
        # (3b) clue-topboard-found rate + sim-when-found by mode
        ax[2].bar(x, [agg.loc[m, "clue_topboard_found"] for m in present], 0.5, color="tab:orange")
        ax[2].set_xticks(x); ax[2].set_xticklabels(present, fontsize=8)
        ax[2].set_ylabel("P(clue points at an already-found word)")
        ax[2].set_title("(3b) non-adaptive spymaster clues for found words", fontsize=9); ax[2].grid(alpha=.3, axis="y")
        fig.suptitle("Behavioral: does the guesser act on the spymaster's regime?", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.94]); pdf.savefig(fig); plt.close(fig)

    json.dump(report, open(os.path.join(DIR, "codenames_probe_behavior.json"), "w"), indent=2)
    print("\nwrote", os.path.join(DIR, "codenames_probe_behavior.pdf"))


if __name__ == "__main__":
    main()
