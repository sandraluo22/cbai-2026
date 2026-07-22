"""Four linear probes on the GUESSER (B)'s residual stream, per layer:

  1. target_multihot -> does B represent A's hidden target set?   (mean AUC over 12 items)
  2. policy          -> H5: does B represent which A-variant it faces?  (3-class accuracy)
  3. belief_entropy  -> is B's own uncertainty explicitly represented?  (regression R^2)
  4. adaptivity      -> "how well A understands me" / level-2         (regression R^2)

All cross-validated with GroupKFold by GAME (an entire game -- all policies, all rounds --
stays in one fold), so a game's target set / clue stream can't leak across train and test.
Pipeline: StandardScaler -> PCA -> {Ridge | LogisticRegression}.

Usage:  python src/codenames_probe4.py [runs/codenames/probe4]
Out:    <dir>/codenames_probe4.{pdf,json}  (+ printed report)
"""
from __future__ import annotations

import os
import sys
import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score
from sklearn.metrics import roc_auc_score

DIR = sys.argv[1] if len(sys.argv) > 1 else "runs/codenames/probe4"
KPCA = 40


def splits(groups, k=5):
    return list(GroupKFold(k).split(np.zeros(len(groups)), groups=groups))


def probe_layer(X, d, groups):
    cv = splits(groups)
    k = int(min(KPCA, min(len(tr) for tr, _ in cv) - 1, X.shape[1]))
    base = [StandardScaler(), PCA(n_components=k, random_state=0)]
    out = {}
    # 1. target set (12-dim) -> mean AUC over items via out-of-fold Ridge scores
    pred = cross_val_predict(make_pipeline(*base, Ridge(alpha=10.0)), X, d["target"], cv=cv)
    aucs = [roc_auc_score(d["target"][:, i], pred[:, i])
            for i in range(d["target"].shape[1]) if len(np.unique(d["target"][:, i])) > 1]
    out["target_auc"] = float(np.mean(aucs))
    # 2. policy -> 3-class accuracy
    out["policy_acc"] = float(cross_val_score(
        make_pipeline(*base, LogisticRegression(max_iter=2000, C=1.0)),
        X, d["policy"], cv=cv, scoring="accuracy").mean())
    # 3, 4 regressions -> R^2
    for key in ("entropy", "adaptivity"):
        out[key + "_r2"] = float(cross_val_score(
            make_pipeline(*base, Ridge(alpha=10.0)), X, d[key], cv=cv, scoring="r2").mean())
    return out


def analyze(path):
    z = np.load(path, allow_pickle=True)
    acts = z["acts"].astype(np.float32)
    d = {"target": z["target"], "policy": z["policy"], "entropy": z["entropy"],
         "adaptivity": z["adaptivity"]}
    groups, layers = z["game"], z["layers"]
    gtag = os.path.basename(path).replace("probe4_", "").replace(".npz", "")
    res = {"guesser": gtag, "layers": [int(x) for x in layers], "n": int(len(groups)),
           "chance": {"policy_acc": float(1 / len(np.unique(d["policy"]))), "target_auc": 0.5}, "by_layer": {}}
    for li in range(acts.shape[1]):
        res["by_layer"][int(layers[li])] = probe_layer(acts[:, li, :], d, groups)
    return res


PANELS = [("target_auc", "1. A's target set  (mean AUC)", 0.5),
          ("policy_acc", "2. A's policy/type  (3-class acc)", None),
          ("entropy_r2", "3. B's belief entropy  (R^2)", 0.0),
          ("adaptivity_r2", "4. 'how well A gets me' = adaptivity  (R^2)", 0.0)]


def main():
    files = sorted(glob.glob(os.path.join(DIR, "probe4_*.npz")))
    results = [analyze(f) for f in files]

    for r in results:
        print("=" * 74)
        print(f"GUESSER {r['guesser']}  (n={r['n']}, layers {r['layers']})")
        for key, ttl, _ in PANELS:
            vals = [r["by_layer"][L][key] for L in sorted(r["by_layer"])]
            best = max(vals) if "acc" in key or "auc" in key else max(vals)
            print(f"  {ttl:42s} best={best:+.3f}  by layer: " +
                  " ".join(f"{v:+.2f}" for v in vals))

    with PdfPages(os.path.join(DIR, "codenames_probe4.pdf")) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        for ax, (key, ttl, chance) in zip(axes.ravel(), PANELS):
            for r in results:
                Ls = sorted(r["by_layer"])
                ax.plot(Ls, [r["by_layer"][L][key] for L in Ls], "-o", label=r["guesser"])
            ch = chance if chance is not None else (results[0]["chance"]["policy_acc"] if results else 1/3)
            ax.axhline(ch, color="0.5", ls="--", lw=1, label="chance/null")
            ax.set_title(ttl, fontsize=10); ax.set_xlabel("layer"); ax.grid(alpha=.3); ax.legend(fontsize=8)
        fig.suptitle("Linear probes on the guesser (B)'s residual stream, per layer\n"
                     "what B represents about the spymaster (A) and about itself", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.94]); pdf.savefig(fig); plt.close(fig)

    json.dump(results, open(os.path.join(DIR, "codenames_probe4.json"), "w"), indent=2)
    print("\nwrote", os.path.join(DIR, "codenames_probe4.pdf"))


if __name__ == "__main__":
    main()
