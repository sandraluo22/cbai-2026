"""Quantitative test of the two dynamics hypotheses on the PICK trajectories (local):

DEFFUANT / bounded-confidence: when a player moves, what fraction of the gap toward the
partner's last word does it close, as a function of how FAR the partner is?
  assimilation a_t = proj(s_{t+1} - s_t, unit(o_t - s_t)) / ||o_t - s_t||
  Deffuant predicts a_t ~ mu > 0 below a confidence radius (cos-similar partner) and
  ~ 0 beyond it (threshold shape). Plotted vs cosine distance(s_t, o_t), met vs no-meet.

POLYA-URN / self-reinforcement: does the walk increasingly revisit its OWN past
neighborhood rather than the partner's?
  For each move, nearest-neighbor cosine distance of s_{t+1} to {own past words} vs to
  {partner past words}, over relative progress. Urn-like trapping = own-history distance
  shrinking (and < partner-history distance) in no-meet games.

Embeddings: mean-pooled input-embedding rows (qwen32_word_embed.npz) — crude but
uniform. Conds: reactive + restrict-city/fruit (baseline regime).

Env: SRC_DIR(runs/game-1/2_restricted_core/qwen32_cap24) EMB_NPZ OUT_DIR
Out: <OUT_DIR>/deffuant_urn.pdf + .json
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

SRC_DIR = os.environ.get("SRC_DIR", "runs/game-1/2_restricted_core/qwen32_cap24")
EMB_NPZ = os.environ.get("EMB_NPZ", "runs/game-1/6_analyses/update_dynamics/qwen32_word_embed.npz")
OUT_DIR = os.environ.get("OUT_DIR", "runs/game-1/6_analyses/update_dynamics")
CONDS = os.environ.get("CONDS", "reactive,restrict-city,restrict-fruit").split(",")


def load_games(cond):
    games = collections.defaultdict(list)
    for line in open(os.path.join(SRC_DIR, f"game1_yoked_{cond}_transcript.jsonl")):
        d = json.loads(line)
        games[d["rollout"]].append(d)
    return dict(games)


def main():
    z = np.load(EMB_NPZ, allow_pickle=True)
    V = z["vecs"] / (np.linalg.norm(z["vecs"], axis=1, keepdims=True) + 1e-9)
    emb = {w: V[i] for i, w in enumerate(z["words"])}

    def E(w):
        return emb.get(w.lower())

    dro, dassim, drev = [], [], []
    for cond in CONDS:
        for g, ts in load_games(cond).items():
            met = ts[-1]["agreed"]
            T = len(ts)
            for pl, ok in (("A", "B"), ("B", "A")):
                seq = [t[pl] for t in ts]
                oth = [t[ok] for t in ts]
                for i in range(T - 1):
                    s0, s1, o = E(seq[i]), E(seq[i + 1]), E(oth[i])
                    if s0 is None or s1 is None or o is None:
                        continue
                    gap = o - s0
                    gn = np.linalg.norm(gap)
                    if gn > 1e-6:
                        a = float(np.dot(s1 - s0, gap / gn) / gn)
                        dassim.append({"cond": cond, "met": met, "progress": (i + 1) / T,
                                       "dist": float(1 - np.dot(s0, o)),
                                       "assim": max(min(a, 2.0), -2.0)})
                    own_past = [E(w) for w in seq[:i + 1] if E(w) is not None]
                    oth_past = [E(w) for w in oth[:i + 1] if E(w) is not None]
                    if own_past and oth_past:
                        d_own = min(1 - np.dot(s1, v) for v in own_past)
                        d_oth = min(1 - np.dot(s1, v) for v in oth_past)
                        drev.append({"cond": cond, "met": met, "progress": (i + 1) / T,
                                     "d_own": float(d_own), "d_oth": float(d_oth)})
    print(f"[deffuant] {len(dassim)} assimilation moves, {len(drev)} revisit points")

    def binned(rows, xk, yk, sel, nb=6, xmax=None):
        rs = [r for r in rows if sel(r)]
        x = np.array([r[xk] for r in rs]); y = np.array([r[yk] for r in rs])
        if xmax:
            m = x <= xmax; x, y = x[m], y[m]
        qs = np.quantile(x, np.linspace(0, 1, nb + 1))
        xs, ys, es = [], [], []
        for lo, hi in zip(qs[:-1], qs[1:]):
            m = (x >= lo) & (x <= hi)
            if m.sum() > 4:
                xs.append(x[m].mean()); ys.append(y[m].mean()); es.append(y[m].std() / np.sqrt(m.sum()))
        return xs, ys, es

    stats = {}
    with PdfPages(os.path.join(OUT_DIR, "deffuant_urn.pdf")) as pdf:
        # p1: Deffuant assimilation vs distance
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for met, color, lab in ((True, "tab:green", "met"), (False, "tab:red", "no-meet")):
            xs, ys, es = binned(dassim, "dist", "assim", lambda r, m=met: r["met"] == m)
            ax.errorbar(xs, ys, yerr=es, fmt="o-", color=color, label=f"{lab} games")
            stats[f"assim_curve_{lab}"] = {"dist": xs, "assim": ys}
        ax.axhline(0, color="k", alpha=.4)
        ax.set_xlabel("cosine distance to partner's last word")
        ax.set_ylabel("fraction of gap closed next move (Deffuant mu)")
        ax.set_title("Bounded-confidence test: assimilation vs distance\n"
                     "Deffuant predicts mu>0 near, ~0 far (threshold)", fontsize=10)
        ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # p2: urn revisit — own vs partner neighborhood over progress
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for met, ls in ((True, "-"), (False, "--")):
            for key, color, lab in (("d_own", "tab:blue", "to OWN history"),
                                     ("d_oth", "tab:orange", "to PARTNER history")):
                xs, ys, es = binned(drev, "progress", key, lambda r, m=met: r["met"] == m)
                ax.errorbar(xs, ys, yerr=es, fmt="o" + ls, color=color,
                            label=f"{lab} ({'met' if met else 'no-meet'})", alpha=.8)
        ax.set_xlabel("relative game progress")
        ax.set_ylabel("NN cosine distance of next word")
        ax.set_title("Urn test: does the walk cling to its own neighborhood?\n"
                     "(solid=met, dashed=no-meet)", fontsize=10)
        ax.legend(fontsize=7); ax.grid(alpha=.3); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    json.dump(stats, open(os.path.join(OUT_DIR, "deffuant_urn.json"), "w"), indent=1)
    # headline numbers
    for lab in ("met", "no-meet"):
        c = stats[f"assim_curve_{lab}"]
        print(f"[deffuant] {lab:>7}: assim near={c['assim'][0]:+.3f} (d~{c['dist'][0]:.2f})  "
              f"far={c['assim'][-1]:+.3f} (d~{c['dist'][-1]:.2f})")
    print(f"[deffuant] wrote deffuant_urn.pdf -> {OUT_DIR}")


if __name__ == "__main__":
    main()
