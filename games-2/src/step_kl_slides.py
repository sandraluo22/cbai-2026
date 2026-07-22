"""Unpaired step-to-step KL slideshow: how much each model's OWN pick distribution
moves from one turn to the next (KL(d_t || d_{t-1})) -- no counterfactual pairing.

Whereas the coupling probe forks the partner's last pick and compares two branches
from the same state, this just tracks the natural drift of each model's distribution
turn by turn: large early KL = the model is still forming its choice; KL -> 0 = it
has settled. Read straight from a saved transcript (no models needed).

Usage:  python src/step_kl_slides.py <transcript.jsonl>
Out:    <transcript basename>_stepkl.pdf
"""
from __future__ import annotations

import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import core as K


def load(tpath):
    recs = [json.loads(l) for l in open(tpath)]
    names = list(recs[0]["picks"].keys())
    vocab = list(recs[0][names[0]]["dist"].keys())
    games = {}
    for r in recs:
        games.setdefault(r["game"], []).append(r)
    for g in games:
        games[g].sort(key=lambda r: r["turn"])
    return names, vocab, games


def dist_seq(game_recs, name, vocab):
    return [np.array([r[name]["dist"][w] for w in vocab]) for r in game_recs]


def step_kls(seq):
    """KL(d_t || d_{t-1}) for t=1..T-1."""
    return [K.kl(seq[t], seq[t - 1]) for t in range(1, len(seq))]


def main():
    tpath = sys.argv[1]
    names, vocab, games = load(tpath)
    nA, nB = names
    x = np.arange(len(vocab))

    # aggregate mean step-KL per turn across games (ragged -> pad with nan)
    maxT = max(len(g) for g in games.values())
    aggA = np.full((len(games), maxT), np.nan)
    aggB = np.full((len(games), maxT), np.nan)
    for gi, recs in games.items():
        sA, sB = dist_seq(recs, nA, vocab), dist_seq(recs, nB, vocab)
        for t, kl in enumerate(step_kls(sA), start=1):
            aggA[gi, t] = kl
        for t, kl in enumerate(step_kls(sB), start=1):
            aggB[gi, t] = kl

    ex = games[0]
    exA, exB = dist_seq(ex, nA, vocab), dist_seq(ex, nB, vocab)
    klA, klB = step_kls(exA), step_kls(exB)

    out = os.path.splitext(tpath)[0] + "_stepkl.pdf"
    with PdfPages(out) as pdf:
        # summary
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        turns = np.arange(maxT)
        ax.plot(turns, np.nanmean(aggA, 0), "-o", color="tab:blue", label=f"{nA} (mean over games)")
        ax.plot(turns, np.nanmean(aggB, 0), "-o", color="tab:orange", label=f"{nB} (mean over games)")
        ax.plot(range(1, len(klA) + 1), klA, ":", color="tab:blue", alpha=.5, label=f"{nA} (game 0)")
        ax.plot(range(1, len(klB) + 1), klB, ":", color="tab:orange", alpha=.5, label=f"{nB} (game 0)")
        ax.set_xlabel("turn t"); ax.set_ylabel("KL(d_t || d_{t-1})  (step-to-step drift)")
        ax.set_title(f"Step-to-step distribution drift per model\n{os.path.basename(tpath)}", fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=.3)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # one slide per step transition (exemplar game 0): d_{t-1} vs d_t
        for t in range(1, len(ex)):
            fig = plt.figure(figsize=(13, 5))
            gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.4, 1.4])
            axk = fig.add_subplot(gs[0]); axA = fig.add_subplot(gs[1]); axB = fig.add_subplot(gs[2])
            ka = klA[t - 1] if t - 1 < len(klA) else np.nan
            kb = klB[t - 1] if t - 1 < len(klB) else np.nan
            axk.bar([0, 1], [ka, kb], color=["tab:blue", "tab:orange"])
            axk.set_xticks([0, 1]); axk.set_xticklabels([nA, nB], fontsize=8, rotation=20)
            axk.set_ylabel("KL(d_t || d_{t-1})"); axk.set_title("step drift", fontsize=9)
            for axm, seq, kl, c, name in ((axA, exA, ka, "tab:blue", nA), (axB, exB, kb, "tab:orange", nB)):
                axm.bar(x - 0.2, seq[t - 1], 0.4, color="0.6", alpha=.8, label=f"turn {t} (d_(t-1))")
                axm.bar(x + 0.2, seq[t], 0.4, color=c, alpha=.85, label=f"turn {t+1} (d_t)")
                axm.set_xticks(x); axm.set_xticklabels(vocab, rotation=90, fontsize=6)
                axm.set_ylim(0, 1); axm.set_title(f"{name}: pick {ex[t-1][name]['pick']}→{ex[t][name]['pick']}  KL={kl:.2f}", fontsize=9)
                axm.legend(fontsize=7)
            fig.suptitle(f"[{ex[t]['mode']}] game 0, step {t}→{t+1}  "
                         f"(picks: {ex[t]['picks'][nA]} / {ex[t]['picks'][nB]})", fontsize=11)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(f"wrote {out}  ({len(ex)} turns, {len(games)} games)")


if __name__ == "__main__":
    main()
