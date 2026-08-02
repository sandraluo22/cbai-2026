"""STUCK-PRIOR probe (local, no GPU): can a linear probe on the UNRESTRICTED player's
residual stream classify Qwen32 restricted games as STUCK (never meet + morphological
perseveration) vs FAST-MET (meet within 24)?

Labels (cap-100 outcomes): stuck = never met AND perseveration > 0.4; fast_met = met
by turn 23; late-met excluded. Both classes are cleanly bimodal (pers 0.58-0.98 vs ~0).

METHODOLOGY NOTES (learned the hard way):
  * Per-turn pooled accuracy is CONTAMINATED by survival bias: fast games end by ~t8,
    so late bins contain only stuck games and a constant-feature probe scores 0.99
    (verified: layer-0 activations are literally constant yet "decode" 0.81 overall).
    Late-turn decodability is therefore trivial/reflective, not evidence.
  * The meaningful test is GAME-LEVEL EARLY prediction: mean activation over turns 1-3
    (one vote per game, balanced 15v14), leave-one-game-out AND holding out the same
    rollout in the other condition (city/fruit share forced start words).

RESULT (2026-07-24): early game-level decoding is AT/BELOW CHANCE at every layer
(balanced 0.28-0.55, AUC 0.23-0.52; t<=1 and t<=3 windows alike). Together with the
KL baseline (game AUC 0.71 overall, 0.55 at t1-3): the stuck regime has NO early
signature in distributions OR activations — stuckness is not a pre-set internal mode
but emerges dynamically through the no-repeat/perseveration feedback loop, exactly as
the urn model predicts.

Env: ACTS_NPZ CAP100_DIR LAYERS OUT_DIR(runs/game-1/6_analyses/stuck_probe)
"""
from __future__ import annotations
import os
import json
import collections
import numpy as np
from sklearn.linear_model import LogisticRegression

ACTS_NPZ = os.environ.get("ACTS_NPZ",
                          "runs/game-1/6_analyses/constraint_probe/qwen32_constraint_probe_acts.npz")
CAP100_DIR = os.environ.get("CAP100_DIR", "runs/game-1/2_restricted_core/qwen32_cap100")
LAYERS = [int(x) for x in os.environ.get("LAYERS", "8,16,24,32,40,48,56,64").split(",")]
OUT_DIR = os.environ.get("OUT_DIR", "runs/game-1/6_analyses/stuck_probe")


def game_labels():
    labels = {}
    for c in ("restrict-city", "restrict-fruit"):
        games = collections.defaultdict(list)
        for line in open(os.path.join(CAP100_DIR, f"game1_yoked_{c}_transcript.jsonl")):
            d = json.loads(line)
            games[d["rollout"]].append(d)
        for g, ts in games.items():
            met = ts[-1]["turn"] if ts[-1]["agreed"] else None
            A = [t["A"] for t in ts]
            pers = float(np.mean([any(w[:4] == p[:4] and len(w) > 3 for p in A[:i])
                                  for i, w in enumerate(A)][1:])) if len(A) > 1 else 0.0
            if met is None and pers > 0.4:
                labels[(c, g)] = 1
            elif met is not None and met <= 23:
                labels[(c, g)] = 0
    return labels


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    z = np.load(ACTS_NPZ, allow_pickle=True)
    acts = z["acts"].astype(np.float32)
    cond = np.array([str(c) for c in z["cond"]])
    game = z["game"].astype(int)
    turn = z["turn"].astype(int)
    labs = game_labels()

    res = {}
    for T_EARLY in (1, 3):
        feats, ys, ids = [], [], []
        for (c, g), lab in sorted(labs.items()):
            m = (cond == c) & (game == g) & (turn <= T_EARLY)
            if m.sum():
                feats.append(acts[m].mean(0)); ys.append(lab); ids.append((c, g))
        F = np.stack(feats); ys = np.array(ys)
        res[f"t<={T_EARLY}"] = {}
        for L in LAYERS:
            X = F[:, L, :]
            Xn = (X - X.mean(0)) / (X.std(0) + 1e-6)
            correct = np.zeros(len(ys)); scores = np.zeros(len(ys))
            for i in range(len(ys)):
                tr = np.ones(len(ys), bool); tr[i] = False
                for j, (c2, g2) in enumerate(ids):     # shared-start pair holdout
                    if g2 == ids[i][1] and j != i:
                        tr[j] = False
                clf = LogisticRegression(max_iter=2000, C=0.05)
                clf.fit(Xn[tr], ys[tr])
                correct[i] = clf.predict(Xn[i:i + 1])[0] == ys[i]
                scores[i] = clf.decision_function(Xn[i:i + 1])[0]
            bal = 0.5 * (correct[ys == 1].mean() + correct[ys == 0].mean())
            ranks = np.empty(len(ys)); ranks[np.argsort(scores)] = np.arange(len(ys))
            n1, n0 = ys.sum(), (1 - ys).sum()
            auc = float((ranks[ys == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))
            res[f"t<={T_EARLY}"][L] = {"balanced_acc": float(bal), "auc": auc}
            print(f"[stuck] t<={T_EARLY} L{L:>2}: balanced {bal:.2f}  AUC {auc:.2f}")
    json.dump({"n_games": len(ys), "n_stuck": int(ys.sum()), "results": res,
               "verdict": "no early signature; late decodability is survival-bias artifact"},
              open(os.path.join(OUT_DIR, "stuck_probe.json"), "w"), indent=1)
    print(f"[stuck] wrote -> {OUT_DIR}")


if __name__ == "__main__":
    main()
