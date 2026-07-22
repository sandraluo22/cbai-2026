"""Extract the probe DIRECTION at the last two turns and interpret it.

For each guesser, at each of its last two turns, we refit the probe (StandardScaler ->
PCA -> RidgeCV) on the best layer, recover the direction in activation space
    w = (ridge.coef_ @ pca.components_) / scaler.scale_
and get an out-of-fold projection score per sample (cross_val_predict). We then ask
WHAT the direction encodes by correlating that score with observable features from the
metadata replay, split into two families:
    repetition/staticness : is_repeat, distinct_ratio(-), n_distinct(-)
    task-progress         : found_before, remaining_before, wrong_before
A disentangling OLS (score ~ distinct_ratio + remaining_before + found_before) shows
which family survives controlling for the other. Exemplars (top/bottom score) print the
clue histories so the direction is concrete.

Usage:  python src/codenames_probe_direction.py [runs/codenames/probe]
Out:    <dir>/codenames_probe_direction.{pdf,json}, probe_directions.npz  (+ printed report)
"""
from __future__ import annotations

import os
import sys
import json
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_predict

DIR = sys.argv[1] if len(sys.argv) > 1 else "runs/codenames/probe"
KPCA = 30
REP_FEATS = ["is_repeat", "distinct_ratio", "n_distinct"]
PROG_FEATS = ["found_before", "remaining_before", "wrong_before"]
FEATS = REP_FEATS + PROG_FEATS


def load(gtag):
    d = np.load(os.path.join(DIR, f"probe_{gtag}.npz"), allow_pickle=True)
    acts, layers = d["acts"].astype(np.float32), d["layers"]
    npz_meta = pd.DataFrame({"mode": d["mode"], "game": d["game"], "round": d["round"],
                             "aidx": np.arange(len(d["mode"]))})
    meta = pd.DataFrame([json.loads(l) for l in open(os.path.join(DIR, f"meta_{gtag}.jsonl")) if l.strip()])
    merged = npz_meta.merge(meta, on=["mode", "game", "round"], how="inner")
    return acts, layers, merged


def make_pipe(n, H):
    k = int(min(KPCA, (n * 4) // 5 - 1, H))
    return Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=k, random_state=0)),
                     ("ridge", RidgeCV(alphas=(1.0, 10.0, 100.0, 1000.0)))]), k


def direction(pipe):
    sc, pca, ridge = pipe["sc"], pipe["pca"], pipe["ridge"]
    return (ridge.coef_ @ pca.components_) / sc.scale_     # w in activation space


def analyze(gtag, best_layer):
    acts, layers, merged = load(gtag)
    pos = int(np.where(layers == best_layer)[0][0])
    turns = sorted(merged["round"].unique())
    last2 = turns[-2:]
    out = {"guesser": gtag, "best_layer": int(best_layer), "last_two_turns": [int(t) for t in last2], "turns": {}}
    dirs = {}
    for t in last2:
        sub = merged[merged["round"] == t].reset_index(drop=True)
        X = acts[sub["aidx"].values, pos, :]
        y = sub["mode"].values.astype(float)
        n = len(y)
        pipe, k = make_pipe(n, X.shape[1])
        cv = KFold(5, shuffle=True, random_state=0)
        score = cross_val_predict(pipe, X, y, cv=cv)       # out-of-fold projection
        pipe.fit(X, y)
        dirs[t] = direction(pipe)
        sub = sub.assign(score=score)
        # correlations of the probe score with each observable feature
        cors = {}
        for f in FEATS:
            if sub[f].nunique() > 1:
                r, p = stats.pearsonr(sub["score"], sub[f])
                cors[f] = {"r": float(r), "p": float(p)}
            else:
                cors[f] = {"r": 0.0, "p": 1.0}
        # disentangling OLS (standardized predictors)
        z = sub.copy()
        for f in ["distinct_ratio", "remaining_before", "found_before"]:
            s = z[f].std()
            z[f + "_z"] = (z[f] - z[f].mean()) / (s if s > 1e-9 else 1.0)
        ols = smf.ols("score ~ distinct_ratio_z + remaining_before_z + found_before_z", data=z).fit()
        out["turns"][int(t)] = {
            "n": int(n), "k_pca": int(k),
            "corr": cors,
            "auc_like_r_with_mode": float(stats.pearsonr(sub["score"], sub["mode"])[0]),
            "ols_coef": {k2: float(v) for k2, v in ols.params.items()},
            "ols_p": {k2: float(v) for k2, v in ols.pvalues.items()},
        }
        # exemplars: top/bottom by score with clue histories
        clue_hist = (merged[merged["round"] <= t].sort_values(["mode", "game", "round"])
                     .groupby(["mode", "game"])["clue"].apply(list))
        ex = []
        s2 = sub.sort_values("score")
        for tag, row in [("LOW (memoryless-like)", s2.iloc[0]), ("LOW", s2.iloc[1]),
                         ("HIGH (memory-like)", s2.iloc[-1]), ("HIGH", s2.iloc[-2])]:
            hist = clue_hist.get((row["mode"], row["game"]), [])
            ex.append({"tag": tag, "mode": int(row["mode"]), "game": int(row["game"]),
                       "score": float(row["score"]), "clues": hist})
        out["turns"][int(t)]["exemplars"] = ex
    return out, dirs, (acts, layers, merged, pos)


def main():
    prob = json.load(open(os.path.join(DIR, "codenames_probe.json")))
    results, all_dirs = {}, {}
    for gtag, s in prob.items():
        res, dirs, _ = analyze(gtag, s["best_layer"])
        results[gtag] = res
        for t, w in dirs.items():
            all_dirs[f"{gtag}_turn{t}"] = w

    # ---- report ----
    for gtag, res in results.items():
        print("=" * 76)
        print(f"GUESSER {gtag}  (best layer {res['best_layer']})  last two turns {res['last_two_turns']}")
        for t, td in res["turns"].items():
            print(f"\n  turn {t}  (n={td['n']}, PCA k={td['k_pca']}, corr(score,mode)={td['auc_like_r_with_mode']:+.2f})")
            print("    correlation of probe score with observable features:")
            for f in FEATS:
                c = td["corr"][f]
                fam = "REP " if f in REP_FEATS else "PROG"
                print(f"      [{fam}] {f:18s} r={c['r']:+.3f}  (p={c['p']:.3f})")
            print("    disentangling OLS  score ~ distinct_ratio + remaining + found (standardized):")
            for k2 in ["distinct_ratio_z", "remaining_before_z", "found_before_z"]:
                print(f"      {k2:20s} coef={td['ols_coef'].get(k2,0):+.3f}  p={td['ols_p'].get(k2,1):.3f}")
            print("    exemplars (clue history up to this turn):")
            for e in td["exemplars"]:
                print(f"      {e['tag']:22s} score={e['score']:+.2f}  mode={'mem' if e['mode'] else 'nomem'}  clues={e['clues']}")

    # ---- figure ----
    with PdfPages(os.path.join(DIR, "codenames_probe_direction.pdf")) as pdf:
        for gtag, res in results.items():
            acts, layers, merged, pos = None, None, None, None
            fig, axes = plt.subplots(1, len(res["turns"]), figsize=(7.0 * len(res["turns"]), 4.8), squeeze=False)
            for ax, (t, td) in zip(axes[0], res["turns"].items()):
                rs = [td["corr"][f]["r"] for f in FEATS]
                colors = ["tab:red"] * len(REP_FEATS) + ["tab:blue"] * len(PROG_FEATS)
                ax.barh(range(len(FEATS)), rs, color=colors)
                ax.set_yticks(range(len(FEATS))); ax.set_yticklabels(FEATS, fontsize=8)
                ax.axvline(0, color="k", lw=.7); ax.invert_yaxis()
                ax.set_xlabel("Pearson r  (probe score vs feature)")
                ax.set_title(f"{gtag}  turn {t}  (n={td['n']})", fontsize=10)
                ax.grid(alpha=.3, axis="x")
            from matplotlib.patches import Patch
            fig.legend([Patch(color="tab:red"), Patch(color="tab:blue")],
                       ["repetition / staticness", "task-progress"], loc="lower center", ncol=2, fontsize=9, frameon=False)
            fig.suptitle(f"What the probe direction encodes — guesser {gtag} (last two turns)", fontsize=11)
            fig.tight_layout(rect=[0, 0.06, 1, 0.95]); pdf.savefig(fig); plt.close(fig)
    np.savez(os.path.join(DIR, "probe_directions.npz"), **all_dirs)
    json.dump(results, open(os.path.join(DIR, "codenames_probe_direction.json"), "w"), indent=2)
    print("\nwrote", os.path.join(DIR, "codenames_probe_direction.pdf"),
          "and probe_directions.npz")


if __name__ == "__main__":
    main()
