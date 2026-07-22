"""Is the guesser's 'static-vs-adaptable spymaster' signal just surface clue-repetition,
or an abstract representation of ADAPTIVENESS?

Three spymaster modes were captured: 0 memoryless (repeats), 1 memory (diverse,
adaptive), 2 diverse-but-NON-adaptive.

  Probe A (repetition)   : mode 0 vs 1   -- the trivial axis (repeat vs diverse).
  Probe B (adaptiveness) : mode 1 vs 2   -- BOTH diverse; differ ONLY in whether the
                           spymaster conditions on the guesser. R^2 rising with turn
                           here = the guesser tracks adaptiveness beyond repetition.
  Transfer               : train Probe A (repetition), then project mode-2 samples.
                           If mode 2 lands with mode 1 -> the repetition axis is blind to
                           non-adaptiveness (it only sees diversity). Probe B separating
                           them then shows adaptiveness is a distinct, decodable axis.

Usage:  python src/codenames_probe_transfer.py [runs/codenames/probe3]
Out:    <dir>/codenames_probe_transfer.{pdf,json}  (+ printed report)
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict

DIR = sys.argv[1] if len(sys.argv) > 1 else "runs/codenames/probe3"
MIN_N = 24
KPCA = 30
NPERM = 4
MODE_NAME = {0: "memoryless", 1: "memory", 2: "nonadaptive"}


def pipe_for(n, H):
    k = int(min(KPCA, (n * 4) // 5 - 1, H))
    return Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=k, random_state=0)),
                     ("ridge", RidgeCV(alphas=(1.0, 10.0, 100.0, 1000.0)))]), k


def paired_sel(mode, game, rnd, t, a, b):
    m = rnd == t
    ga = set(game[m & (mode == a)].tolist()); gb = set(game[m & (mode == b)].tolist())
    common = ga & gb
    return m & np.isin(game, list(common)) & np.isin(mode, [a, b])


def r2_by_turn(acts, mode, game, rnd, li, a, b):
    out = {}
    for t in sorted(int(x) for x in np.unique(rnd)):
        sel = paired_sel(mode, game, rnd, t, a, b)
        if sel.sum() < MIN_N:
            continue
        X, y = acts[sel, li, :], (mode[sel] == b).astype(float)
        pipe, k = pipe_for(len(y), X.shape[1])
        cv = KFold(5, shuffle=True, random_state=t)
        r2 = float(cross_val_score(pipe, X, y, cv=cv, scoring="r2").mean())
        rng = np.random.default_rng(t)
        null = float(np.mean([cross_val_score(pipe, X, rng.permutation(y), cv=cv, scoring="r2").mean()
                              for _ in range(NPERM)]))
        out[t] = (r2, null, int(len(y)))
    return out


def best_layer_for(acts, mode, game, rnd, a, b):
    best, val = 0, -1e9
    for li in range(acts.shape[1]):
        bt = r2_by_turn(acts, mode, game, rnd, li, a, b)
        mv = np.mean([v[0] for t, v in bt.items() if t >= 2]) if bt else -1e9
        if mv > val:
            best, val, bestbt = li, mv, bt
    return best, bestbt


def analyze(path):
    d = np.load(path, allow_pickle=True)
    acts = d["acts"].astype(np.float32)
    mode, game, rnd, layers = d["mode"], d["game"], d["round"], d["layers"]
    gtag = os.path.basename(path).replace("probe_", "").replace(".npz", "")
    have = set(int(x) for x in np.unique(mode))
    res = {"guesser": gtag, "modes_present": sorted(have)}

    # Probe A (0v1) and Probe B (1v2)
    for name, (a, b) in {"A_repetition": (0, 1), "B_adaptiveness": (1, 2)}.items():
        if a in have and b in have:
            bl, bt = best_layer_for(acts, mode, game, rnd, a, b)
            res[name] = {"contrast": [int(a), int(b)], "best_layer": int(layers[bl]),
                         "by_turn": {int(t): {"r2": v[0], "null": v[1], "n": v[2]} for t, v in bt.items()}}

    # Transfer: fit Probe A per turn on {0,1}, project mode-2 samples
    if {0, 1, 2} <= have:
        bl = int(np.where(layers == res["A_repetition"]["best_layer"])[0][0])
        trans = {}
        for t in sorted(int(x) for x in np.unique(rnd)):
            selA = paired_sel(mode, game, rnd, t, 0, 1)
            if selA.sum() < MIN_N:
                continue
            X01, y01 = acts[selA, bl, :], (mode[selA] == 1).astype(float)
            pipe, _ = pipe_for(len(y01), X01.shape[1])
            # honest in-distribution scores for 0/1 via CV; mode-2 via a model fit on all 0/1
            s01 = cross_val_predict(pipe, X01, y01, cv=KFold(5, shuffle=True, random_state=t))
            pipe.fit(X01, y01)
            sel2 = (rnd == t) & (mode == 2)
            s2 = pipe.predict(acts[sel2, bl, :]) if sel2.sum() else np.array([])
            m0 = float(np.mean(s01[mode[selA] == 0])); m1 = float(np.mean(s01[mode[selA] == 1]))
            trans[int(t)] = {"score_mode0": m0, "score_mode1": m1,
                             "score_mode2": float(np.mean(s2)) if len(s2) else None,
                             "n2": int(sel2.sum())}
        res["transfer_probeA_on_mode2"] = trans
    return res


def main():
    files = sorted(glob.glob(os.path.join(DIR, "probe_*.npz")))
    results = [analyze(f) for f in files]

    for r in results:
        print("=" * 76)
        print(f"GUESSER {r['guesser']}  modes present {r['modes_present']}")
        for name in ("A_repetition", "B_adaptiveness"):
            if name in r:
                b = r[name]
                ts = sorted(b["by_turn"])
                print(f"\n  Probe {name}  (modes {b['contrast']}, best layer {b['best_layer']})")
                for t in ts:
                    v = b["by_turn"][t]
                    print(f"      turn {t}: R²={v['r2']:+.3f}  (null {v['null']:+.3f}, n={v['n']})")
        if "transfer_probeA_on_mode2" in r:
            print("\n  TRANSFER: Probe-A (repetition) score by mode  [mode2 should reveal which axis it is]")
            for t, v in r["transfer_probeA_on_mode2"].items():
                s2 = f"{v['score_mode2']:+.3f}" if v["score_mode2"] is not None else "n/a"
                print(f"      turn {t}: mode0={v['score_mode0']:+.3f}  mode1={v['score_mode1']:+.3f}  "
                      f"mode2={s2}  (n2={v['n2']})")

    # figure
    with PdfPages(os.path.join(DIR, "codenames_probe_transfer.pdf")) as pdf:
        fig, axes = plt.subplots(1, len(results), figsize=(7.2 * len(results), 5.0), squeeze=False)
        for ax, r in zip(axes[0], results):
            for name, c in (("A_repetition", "tab:gray"), ("B_adaptiveness", "tab:purple")):
                if name in r:
                    b = r[name]; ts = sorted(b["by_turn"])
                    ax.plot(ts, [b["by_turn"][t]["r2"] for t in ts], "-o", color=c,
                            label=f"Probe {name.split('_')[1]} (modes {b['contrast']}), L{b['best_layer']}")
            ax.axhline(0, color="k", lw=.7)
            ax.set_title(f"guesser {r['guesser']}", fontsize=10)
            ax.set_xlabel("turn"); ax.set_ylabel("cross-validated R²"); ax.legend(fontsize=8); ax.grid(alpha=.3)
        fig.suptitle("Repetition axis (trivial) vs ADAPTIVENESS axis (memory vs diverse-non-adaptive)\n"
                     "R² rising for the adaptiveness probe = the guesser represents adaptiveness beyond repetition",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.92]); pdf.savefig(fig); plt.close(fig)

        # transfer panel
        fig, axes = plt.subplots(1, len(results), figsize=(7.2 * len(results), 4.6), squeeze=False)
        for ax, r in zip(axes[0], results):
            tr = r.get("transfer_probeA_on_mode2", {})
            ts = sorted(tr)
            for key, c, lab in (("score_mode0", "tab:red", "mode0 memoryless"),
                                ("score_mode1", "tab:green", "mode1 memory"),
                                ("score_mode2", "tab:orange", "mode2 nonadaptive (projected)")):
                ys = [tr[t][key] for t in ts if tr[t].get(key) is not None]
                xs = [t for t in ts if tr[t].get(key) is not None]
                ax.plot(xs, ys, "-o", color=c, label=lab)
            ax.set_title(f"guesser {r['guesser']}: Probe-A (repetition) score by mode", fontsize=10)
            ax.set_xlabel("turn"); ax.set_ylabel("repetition-probe score"); ax.legend(fontsize=8); ax.grid(alpha=.3)
        fig.suptitle("Transfer: does the repetition axis 'see' non-adaptiveness? "
                     "(mode2 tracking mode1 = repetition axis is blind to adaptiveness)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.93]); pdf.savefig(fig); plt.close(fig)

    json.dump(results, open(os.path.join(DIR, "codenames_probe_transfer.json"), "w"), indent=2)
    print("\nwrote", os.path.join(DIR, "codenames_probe_transfer.pdf"))


if __name__ == "__main__":
    main()
