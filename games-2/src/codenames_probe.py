"""Probe: can the GUESSER distinguish a static (memoryless) spymaster from an
adaptable (memory) one, and does that ability GROW with turns?

For each guesser model and each turn n, fit a cross-validated linear probe
(StandardScaler -> PCA -> RidgeCV) on the guesser's turn-n activations to predict the
mode label (0=memoryless, 1=memory). If the guesser encodes the distinction, the
cross-validated R^2 should rise with n. A permutation null (shuffled labels) gives the
0-floor, and turn 1 is a built-in floor (identical clue => identical activations).

Paired-survival selection: at turn n we use only game seeds that reached round n in
BOTH modes, so class counts are balanced and "surviving to turn n" is not itself the
signal.

Usage:  python src/codenames_probe.py [runs/codenames/probe]
Out:    <dir>/codenames_probe.pdf + codenames_probe.json
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
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score, KFold

DIR = sys.argv[1] if len(sys.argv) > 1 else "runs/codenames/probe"
MIN_N = 24                 # need at least this many samples (both modes) to probe a turn
KPCA = 30
NPERM = 5


def probe(X, y, seed=0):
    n = len(y)
    k = int(min(KPCA, (n * 4) // 5 - 1, X.shape[1]))   # must fit inside the 5-fold train split
    cv = KFold(5, shuffle=True, random_state=seed)
    pipe = make_pipeline(StandardScaler(),
                         PCA(n_components=k, random_state=seed),
                         RidgeCV(alphas=(1.0, 10.0, 100.0, 1000.0)))
    r2 = float(cross_val_score(pipe, X, y, cv=cv, scoring="r2").mean())
    rng = np.random.default_rng(seed)
    nulls = [float(cross_val_score(pipe, X, rng.permutation(y), cv=cv, scoring="r2").mean())
             for _ in range(NPERM)]
    return r2, float(np.mean(nulls)), float(np.std(nulls)), n, k


def analyze_file(path):
    d = np.load(path, allow_pickle=True)
    acts = d["acts"].astype(np.float32)        # (N, L, H)
    mode, game, rnd = d["mode"], d["game"], d["round"]
    layers = d["layers"]
    gtag = os.path.basename(path).replace("probe_", "").replace(".npz", "")
    turns = sorted(int(t) for t in np.unique(rnd))
    res = {}                                   # (layer_idx_pos) -> {turn: (r2, null, nullsd, n, k)}
    for li in range(acts.shape[1]):
        res[li] = {}
        for t in turns:
            m = rnd == t
            g0 = set(game[m & (mode == 0)].tolist())
            g1 = set(game[m & (mode == 1)].tolist())
            common = g0 & g1
            sel = m & np.isin(game, list(common))
            if sel.sum() < MIN_N:
                continue
            X, y = acts[sel, li, :], mode[sel].astype(float)
            res[li][t] = probe(X, y, seed=li * 100 + t)
    return gtag, layers, res, turns


def best_layer(res):
    """layer position with the highest mean R^2 over turns>=2 (excludes the turn-1 floor)."""
    best, bestval = 0, -1e9
    for li, byturn in res.items():
        vals = [v[0] for t, v in byturn.items() if t >= 2]
        mv = np.mean(vals) if vals else -1e9
        if mv > bestval:
            best, bestval = li, mv
    return best


def trend_slope(byturn):
    """WLS slope of R^2 vs turn (weighted by n), a summary of 'increases with turns'."""
    ts = sorted(byturn)
    if len(ts) < 2:
        return float("nan")
    x = np.array(ts, float); yv = np.array([byturn[t][0] for t in ts]); w = np.array([byturn[t][3] for t in ts], float)
    W = np.diag(w); A = np.vstack([x, np.ones_like(x)]).T
    beta = np.linalg.lstsq(W @ A, W @ yv, rcond=None)[0]
    return float(beta[0])


def main():
    files = sorted(glob.glob(os.path.join(DIR, "probe_*.npz")))
    if not files:
        print(f"no probe_*.npz in {DIR}"); return
    summary = {}
    analyses = [analyze_file(f) for f in files]

    with PdfPages(os.path.join(DIR, "codenames_probe.pdf")) as pdf:
        # page 1: R^2 vs turn (best layer bold + all layers faint + null band)
        fig, axes = plt.subplots(1, len(analyses), figsize=(7.2 * len(analyses), 4.9), squeeze=False)
        for ax, (gtag, layers, res, turns) in zip(axes[0], analyses):
            bl = best_layer(res)
            depth = layers.max()
            for li, byturn in res.items():
                ts = sorted(byturn)
                r2s = [byturn[t][0] for t in ts]
                is_best = li == bl
                ax.plot(ts, r2s, "-o" if is_best else "-", lw=2.6 if is_best else 1,
                        alpha=1 if is_best else 0.25, color="tab:green" if is_best else "0.6",
                        label=f"layer {layers[li]}/{depth} (best)" if is_best else None, zorder=3 if is_best else 1)
            # null band from best layer
            bt = res[bl]; ts = sorted(bt)
            nm = np.array([bt[t][1] for t in ts]); ns = np.array([bt[t][2] for t in ts])
            ax.fill_between(ts, nm - ns, nm + ns, color="0.7", alpha=.4, label="permutation null")
            ax.axhline(0, color="k", lw=.7)
            sl = trend_slope(res[bl])
            ax.set_title(f"guesser = {gtag}   best layer {layers[bl]}/{depth}\n"
                         f"R² vs turn slope = {sl:+.3f}", fontsize=10)
            ax.set_xlabel("turn (round)"); ax.set_ylabel("cross-validated R²  (predict mode)")
            ax.legend(fontsize=8); ax.grid(alpha=.3)
            summary[gtag] = {"best_layer": int(layers[bl]), "depth": int(depth), "slope_R2_vs_turn": sl,
                             "by_turn_best": {int(t): {"r2": bt[t][0], "null": bt[t][1], "n": bt[t][3]} for t in ts}}
        fig.suptitle("Can the guesser distinguish a STATIC (memoryless) vs ADAPTABLE (memory) spymaster?\n"
                     "linear probe on guesser activations per turn — R² rising with turn = yes", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.93]); pdf.savefig(fig); plt.close(fig)

        # page 2: layer x turn R^2 heatmaps
        fig, axes = plt.subplots(1, len(analyses), figsize=(7.2 * len(analyses), 4.6), squeeze=False)
        for ax, (gtag, layers, res, turns) in zip(axes[0], analyses):
            ts_all = sorted({t for byturn in res.values() for t in byturn})
            H = np.full((len(res), len(ts_all)), np.nan)
            for i, li in enumerate(sorted(res)):
                for j, t in enumerate(ts_all):
                    if t in res[li]:
                        H[i, j] = res[li][t][0]
            im = ax.imshow(H, aspect="auto", cmap="viridis", vmin=0, vmax=max(0.05, np.nanmax(H)))
            ax.set_xticks(range(len(ts_all))); ax.set_xticklabels(ts_all)
            ax.set_yticks(range(len(res))); ax.set_yticklabels([str(layers[li]) for li in sorted(res)])
            ax.set_xlabel("turn"); ax.set_ylabel("layer"); ax.set_title(f"guesser = {gtag}: R²(layer, turn)", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=.046)
        fig.suptitle("Probe R² by layer and turn", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.93]); pdf.savefig(fig); plt.close(fig)

    json.dump(summary, open(os.path.join(DIR, "codenames_probe.json"), "w"), indent=2)
    print("wrote", os.path.join(DIR, "codenames_probe.pdf"))
    for g, s in summary.items():
        print(f"  {g}: best layer {s['best_layer']}/{s['depth']}, R²-vs-turn slope {s['slope_R2_vs_turn']:+.3f}")
        for t, v in s["by_turn_best"].items():
            print(f"      turn {t}: R²={v['r2']:+.3f} (null {v['null']:+.3f}, n={v['n']})")


if __name__ == "__main__":
    main()
