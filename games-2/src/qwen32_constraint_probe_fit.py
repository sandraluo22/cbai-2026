"""LATENT-CONSTRAINT PROBE, fit stage (local, no GPU): from the captured activations
(qwen32_constraint_probe.py), test whether the UNRESTRICTED player A linearly encodes
WHICH constraint its partner is under — and how early.

Probe: logistic regression city-vs-fruit (the partner's secret), leave-one-game-out CV,
per layer. Reported by TURN BIN, against A's behavioural drift (fraction of A's words
in the partner's category — from game1_yoked_baselines.json) to ask the key question:
is the constraint DECODABLE BEFORE the behaviour shifts?
Control probe: restricted-vs-reactive (is *some* constraint present?), 2-way.

Env: ACTS_NPZ(runs/game-1/6_analyses/constraint_probe/qwen32_constraint_probe_acts.npz)
     LAYERS(0,8,16,24,32,40,48,56,64) OUT_DIR(runs/game-1/6_analyses/constraint_probe)
"""
from __future__ import annotations
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression

ACTS_NPZ = os.environ.get("ACTS_NPZ",
                          "runs/game-1/6_analyses/constraint_probe/qwen32_constraint_probe_acts.npz")
LAYERS = [int(x) for x in os.environ.get("LAYERS", "0,8,16,24,32,40,48,56,64").split(",")]
OUT_DIR = os.environ.get("OUT_DIR", "runs/game-1/6_analyses/constraint_probe")
TURN_BINS = [(1, 2), (3, 4), (5, 8), (9, 16), (17, 24)]


def logo_acc(X, y, groups, mask_eval=None):
    """Leave-one-game-out accuracy; mask_eval optionally restricts which eval points count."""
    hits, tot = 0, 0
    per_point = np.full(len(y), np.nan)
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, C=0.05)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        per_point[te] = (pred == y[te]).astype(float)
    sel = ~np.isnan(per_point)
    if mask_eval is not None:
        sel &= mask_eval
    return float(np.nanmean(per_point[sel])), per_point


def main():
    z = np.load(ACTS_NPZ, allow_pickle=True)
    acts = z["acts"].astype(np.float32)
    cond = np.array([str(c) for c in z["cond"]])
    game = z["game"].astype(int)
    turn = z["turn"].astype(int)
    groups = np.array([f"{c}:{g}" for c, g in zip(cond, game)])
    print(f"[probe-fit] {acts.shape[0]} pts, conds: {dict(zip(*np.unique(cond, return_counts=True)))}")

    is_cf = np.isin(cond, ["restrict-city", "restrict-fruit"])
    y_cf = (cond == "restrict-city").astype(int)
    y_rx = is_cf.astype(int)                      # restricted vs reactive

    res = {"city_vs_fruit": {}, "restricted_vs_reactive": {}, "by_turnbin": {}}
    for L in LAYERS:
        X = acts[:, L, :]
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Xn = (X - mu) / sd
        acc_cf, pp_cf = logo_acc(Xn[is_cf], y_cf[is_cf], groups[is_cf])
        acc_rx, _ = logo_acc(Xn, y_rx, groups)
        res["city_vs_fruit"][L] = acc_cf
        res["restricted_vs_reactive"][L] = acc_rx
        binaccs = []
        t_cf = turn[is_cf]
        for lo, hi in TURN_BINS:
            m = (t_cf >= lo) & (t_cf <= hi)
            binaccs.append(float(np.nanmean(pp_cf[m])) if m.sum() else None)
        res["by_turnbin"][L] = binaccs
        print(f"[probe-fit] L{L:>2}: city-vs-fruit {acc_cf:.2f} | restricted-vs-reactive "
              f"{acc_rx:.2f} | by turn-bin {['%.2f' % b if b is not None else ' - ' for b in binaccs]}")

    bestL = max(res["city_vs_fruit"], key=res["city_vs_fruit"].get)
    os.makedirs(OUT_DIR, exist_ok=True)
    with PdfPages(os.path.join(OUT_DIR, "constraint_probe.pdf")) as pdf:
        fig, ax = plt.subplots(figsize=(8, 5))
        Ls = list(res["city_vs_fruit"].keys())
        ax.plot(Ls, [res["city_vs_fruit"][L] for L in Ls], "o-", label="city vs fruit (partner's secret)")
        ax.plot(Ls, [res["restricted_vs_reactive"][L] for L in Ls], "s-",
                label="restricted vs reactive")
        ax.axhline(0.5, color="k", ls="--", alpha=.5, label="chance")
        ax.set_xlabel("layer"); ax.set_ylabel("LOGO accuracy (A's activations only)")
        ax.set_title("Does the UNRESTRICTED player encode its partner's hidden constraint?",
                     fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        labels = [f"t{lo}-{hi}" for lo, hi in TURN_BINS]
        for L in [8, bestL, 56] if bestL not in (8, 56) else [8, 32, 56]:
            if L in res["by_turnbin"]:
                ax.plot(labels, res["by_turnbin"][L], "o-", label=f"layer {L}")
        # behavioural drift reference (met games, city+fruit mean, from baselines json)
        try:
            b = json.load(open("runs/game-1/2_restricted_core/qwen32_cap24/game1_yoked_baselines.json"))
            drift = np.nanmean([[x if x is not None else np.nan
                                 for x in b[c]["met"]["in_category_by_triplet"]]
                                for c in ("restrict-city", "restrict-fruit")], axis=0)
            ax.plot(labels, drift[:len(labels)], "k--", alpha=.6,
                    label="A's behavioural drift (met games)")
        except Exception as e:
            print("[probe-fit] drift overlay skipped:", e)
        ax.axhline(0.5, color="k", ls=":", alpha=.4)
        ax.set_xlabel("turn bin"); ax.set_ylabel("accuracy / in-category fraction")
        ax.set_title("Decoding time-course vs behavioural drift — is the constraint known "
                     "before behaviour moves?", fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    json.dump(res, open(os.path.join(OUT_DIR, "constraint_probe.json"), "w"), indent=1)
    print(f"[probe-fit] best layer {bestL} ({res['city_vs_fruit'][bestL]:.2f}); wrote -> {OUT_DIR}")


if __name__ == "__main__":
    main()
