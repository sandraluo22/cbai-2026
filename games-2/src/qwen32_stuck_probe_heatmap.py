"""Per-(layer, turn) linear probe heatmap for the STUCK-PRIOR regime (local, no GPU).

For every layer L and turn t: ridge-regress the game's FINAL perseveration index
(continuous 0..1, from the cap-100 transcript) from A's activation at (L, t);
leave-one-game-out (+ shared-start pair holdout) cross-validated R^2. Page 2: AUC for
the binary stuck-vs-fast_met label. SURVIVAL BIAS handled by masking: cells are shown
only where >=MIN_GAMES games are still alive and the target has sd >= 0.08 (R^2) /
both classes have >=3 alive (AUC) — fast games end by ~t8, so late cells reflect only
the within-stuck gradient.

Env: ACTS_NPZ CAP100_DIR OUT_DIR(runs/game-1/6_analyses/stuck_probe) MIN_GAMES(8)
Out: stuck_probe_heatmap.pdf + stuck_probe_heatmap.npz
"""
from __future__ import annotations
import os
import json
import collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import Ridge, LogisticRegression

ACTS_NPZ = os.environ.get("ACTS_NPZ",
                          "runs/game-1/6_analyses/constraint_probe/qwen32_constraint_probe_acts.npz")
CAP100_DIR = os.environ.get("CAP100_DIR", "runs/game-1/2_restricted_core/qwen32_cap100")
OUT_DIR = os.environ.get("OUT_DIR", "runs/game-1/6_analyses/stuck_probe")
MIN_GAMES = int(os.environ.get("MIN_GAMES", "8"))


def targets():
    pers, stuck = {}, {}
    for c in ("restrict-city", "restrict-fruit"):
        games = collections.defaultdict(list)
        for line in open(os.path.join(CAP100_DIR, f"game1_yoked_{c}_transcript.jsonl")):
            d = json.loads(line)
            games[d["rollout"]].append(d)
        for g, ts in games.items():
            met = ts[-1]["turn"] if ts[-1]["agreed"] else None
            A = [t["A"] for t in ts]
            p = float(np.mean([any(w[:4] == q[:4] and len(w) > 3 for q in A[:i])
                               for i, w in enumerate(A)][1:])) if len(A) > 1 else 0.0
            pers[(c, g)] = p
            if met is None and p > 0.4:
                stuck[(c, g)] = 1
            elif met is not None and met <= 23:
                stuck[(c, g)] = 0
    return pers, stuck


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    z = np.load(ACTS_NPZ, allow_pickle=True)
    acts = z["acts"].astype(np.float32)
    cond = np.array([str(c) for c in z["cond"]])
    game = z["game"].astype(int)
    turn = z["turn"].astype(int)
    pers, stuck = targets()

    nL = acts.shape[1]
    turns = sorted(set(turn))
    R2 = np.full((nL, len(turns)), np.nan)
    AUC = np.full((nL, len(turns)), np.nan)
    NAL = np.zeros(len(turns), int)

    for ti, t in enumerate(turns):
        m = turn == t
        ids = [(c, g) for c, g in zip(cond[m], game[m])]
        keep = [i for i, k in enumerate(ids) if k in pers]
        if not keep:
            continue
        ids = [ids[i] for i in keep]
        Xt_all = acts[m][keep]
        y = np.array([pers[k] for k in ids])
        yb = np.array([stuck.get(k, -1) for k in ids])
        NAL[ti] = len(ids)
        ok_r2 = len(ids) >= MIN_GAMES and y.std() >= 0.08
        okb = (yb == 1).sum() >= 3 and (yb == 0).sum() >= 3
        if not (ok_r2 or okb):
            continue
        rolls = np.array([g for _, g in ids])
        for L in range(nL):
            X = Xt_all[:, L, :]
            Xn = (X - X.mean(0)) / (X.std(0) + 1e-6)
            preds = np.zeros(len(y)); scoreb = np.zeros(len(y))
            for i in range(len(y)):
                tr = rolls != rolls[i]              # LOGO + shared-start pair holdout
                if ok_r2:
                    rg = Ridge(alpha=1e3, solver="svd")
                    rg.fit(Xn[tr], y[tr])
                    preds[i] = rg.predict(Xn[i:i + 1])[0]
                if okb and yb[i] >= 0 and len(set(yb[tr][yb[tr] >= 0])) == 2:
                    lg = LogisticRegression(max_iter=1000, C=0.05)
                    sel = tr & (yb >= 0)
                    lg.fit(Xn[sel], yb[sel])
                    scoreb[i] = lg.decision_function(Xn[i:i + 1])[0]
            if ok_r2:
                R2[L, ti] = 1 - ((y - preds) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            if okb:
                mb = yb >= 0
                s, lab = scoreb[mb], yb[mb]
                ranks = np.empty(len(s)); ranks[np.argsort(s)] = np.arange(len(s))
                n1, n0 = lab.sum(), (1 - lab).sum()
                AUC[L, ti] = (ranks[lab == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)
        print(f"[hm] turn {t}: n={len(ids)} r2_ok={ok_r2} auc_ok={okb} "
              f"bestR2={np.nanmax(R2[:, ti]) if ok_r2 else float('nan'):.2f}", flush=True)

    np.savez_compressed(os.path.join(OUT_DIR, "stuck_probe_heatmap.npz"),
                        R2=R2, AUC=AUC, turns=np.array(turns), n_alive=NAL)
    with PdfPages(os.path.join(OUT_DIR, "stuck_probe_heatmap.pdf")) as pdf:
        for M, name, vmin, vmax, cmap in ((R2, "LOGO R^2 predicting final perseveration",
                                           -0.5, 1.0, "RdBu_r"),
                                          (AUC, "LOGO AUC stuck vs fast-met",
                                           0.0, 1.0, "RdBu_r")):
            fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 9), height_ratios=[12, 1.6],
                                          sharex=True)
            im = ax.imshow(M, aspect="auto", origin="lower", cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           extent=[turns[0] - .5, turns[-1] + .5, -0.5, nL - 0.5])
            ax.set_ylabel("layer")
            ax.set_title(f"{name}\n(gray = masked: survival bias — too few games / no "
                         f"label variance alive)", fontsize=10)
            fig.colorbar(im, ax=ax, shrink=.8)
            ax2.bar(turns, NAL, color="tab:gray")
            ax2.set_xlabel("turn"); ax2.set_ylabel("games\nalive", fontsize=8)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    best = np.unravel_index(np.nanargmax(R2), R2.shape)
    print(f"[hm] peak R2 {np.nanmax(R2):.2f} at layer {best[0]}, turn {turns[best[1]]}")
    print(f"[hm] wrote -> {OUT_DIR}/stuck_probe_heatmap.pdf")


if __name__ == "__main__":
    main()
