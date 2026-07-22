"""Rebuild ALL Game-1 analysis from a saved transcript -- no models, no GPU.

The transcript JSONL already stores, per (game, turn): both models' full pick
distributions, the picks, agreement, and the coupling swap-distributions. So every
metric and figure can be regenerated locally; you only ever run the models ONCE.

Produces, next to the transcript:
  <base>_summary.json          convergence + coupling (recomputed)
  <base>_coupling_perturn.pdf  per-turn coupling KL + clean-vs-swap shift
  <base>_stepkl.pdf            per-turn UNPAIRED step drift KL(d_t || d_{t-1})

Usage:  python src/analyze_transcript.py <transcript.jsonl>
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


def arr(d, vocab):
    return np.array([d[w] for w in vocab])


def summarize(names, vocab, games):
    nA, nB = names
    conv, ttc, cA, cB = [], [], [], []
    for recs in games.values():
        agreed_turn = next((r["turn"] for r in recs if r["agreed"]), None)
        conv.append(agreed_turn is not None)
        if agreed_turn is not None:
            ttc.append(agreed_turn + 1)
        kA = [r[nA]["coupling"]["kl"] for r in recs if "coupling" in r[nA]]
        kB = [r[nB]["coupling"]["kl"] for r in recs if "coupling" in r[nB]]
        cA.append(np.mean(kA) if kA else 0.0); cB.append(np.mean(kB) if kB else 0.0)
    return {"converged_frac": float(np.mean(conv)),
            "turns_to_converge": float(np.mean(ttc)) if ttc else float("nan"),
            f"coupling_{nA}": float(np.mean(cA)), f"coupling_{nB}": float(np.mean(cB)),
            "n_games": len(games)}


def coupling_perturn(names, vocab, games, out):
    nA, nB = names; x = np.arange(len(vocab)); ex = games[0]
    with PdfPages(out) as pdf:
        for r in ex:
            if "coupling" not in r[nA]:
                continue
            fig = plt.figure(figsize=(13, 5))
            gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.4, 1.4])
            axk, axA, axB = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
            klA, klB = r[nA]["coupling"]["kl"], r[nB]["coupling"]["kl"]
            axk.bar([0, 1], [klA, klB], color=["tab:blue", "tab:orange"])
            axk.set_xticks([0, 1]); axk.set_xticklabels([nA, nB], fontsize=8, rotation=20)
            axk.set_ylabel("coupling KL"); axk.set_title("KL this turn", fontsize=9)
            for axm, name, c in ((axA, nA, "tab:blue"), (axB, nB, "tab:orange")):
                clean = arr(r[name]["dist"], vocab); cp = r[name]["coupling"]
                swap = arr(cp["swap_dist"], vocab)
                axm.bar(x - 0.2, clean, 0.4, color=c, alpha=.85, label="clean (real other-pick)")
                axm.bar(x + 0.2, swap, 0.4, color="0.5", alpha=.8,
                        label=f"swap other {cp['swap_other_from']}→{cp['swap_other_to']}")
                axm.set_xticks(x); axm.set_xticklabels(vocab, rotation=90, fontsize=6)
                axm.set_ylim(0, 1); axm.set_title(f"{name}: pick={r[name]['pick']}  KL={cp['kl']:.2f}", fontsize=9)
                axm.legend(fontsize=7)
            agreed = "  ★ AGREED" if r["agreed"] else ""
            fig.suptitle(f"[{r['mode']}] turn {r['turn']+1}: {nA}={r['picks'][nA]}, {nB}={r['picks'][nB]}{agreed}", fontsize=11)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(f"  {out}")


def stepkl(names, vocab, games, out):
    nA, nB = names; x = np.arange(len(vocab))
    seqs = {g: {n: [arr(r[n]["dist"], vocab) for r in recs] for n in names} for g, recs in games.items()}
    kl = lambda s: [K.kl(s[t], s[t - 1]) for t in range(1, len(s))]
    maxT = max(len(recs) for recs in games.values())
    agg = {n: np.full((len(games), maxT), np.nan) for n in names}
    for g in games:
        for n in names:
            for t, v in enumerate(kl(seqs[g][n]), start=1):
                agg[n][g, t] = v
    ex = games[0]; exk = {n: kl(seqs[0][n]) for n in names}
    with PdfPages(out) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 4.8)); turns = np.arange(maxT)
        for n, c in ((nA, "tab:blue"), (nB, "tab:orange")):
            m = np.nanmean(agg[n], 0)
            ax.plot(turns[1:], m[1:], "-o", color=c, label=f"{n} (mean)")
        ax.set_xlabel("turn t"); ax.set_ylabel("KL(d_t || d_{t-1})")
        ax.set_title("Unpaired step-to-step distribution drift", fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=.3)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        for t in range(1, len(ex)):
            fig = plt.figure(figsize=(13, 5)); gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.4, 1.4])
            axk, axA, axB = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
            ka = exk[nA][t - 1] if t - 1 < len(exk[nA]) else np.nan
            kb = exk[nB][t - 1] if t - 1 < len(exk[nB]) else np.nan
            axk.bar([0, 1], [ka, kb], color=["tab:blue", "tab:orange"])
            axk.set_xticks([0, 1]); axk.set_xticklabels([nA, nB], fontsize=8, rotation=20); axk.set_title("step drift", fontsize=9)
            for axm, n, kv, c in ((axA, nA, ka, "tab:blue"), (axB, nB, kb, "tab:orange")):
                s = seqs[0][n]
                axm.bar(x - 0.2, s[t - 1], 0.4, color="0.6", alpha=.8, label=f"turn {t} (d_(t-1))")
                axm.bar(x + 0.2, s[t], 0.4, color=c, alpha=.85, label=f"turn {t+1} (d_t)")
                axm.set_xticks(x); axm.set_xticklabels(vocab, rotation=90, fontsize=6); axm.set_ylim(0, 1)
                axm.set_title(f"{n}: {ex[t-1][n]['pick']}→{ex[t][n]['pick']}  KL={kv:.2f}", fontsize=9); axm.legend(fontsize=7)
            fig.suptitle(f"[{ex[t]['mode']}] game 0 step {t}→{t+1}", fontsize=11)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(f"  {out}")


def main():
    tpath = sys.argv[1]
    base = os.path.splitext(tpath)[0]
    names, vocab, games = load(tpath)
    s = summarize(names, vocab, games)
    json.dump(s, open(base + "_summary.json", "w"), indent=2)
    print(f"summary: {s}")
    print("figures:")
    coupling_perturn(names, vocab, games, base + "_coupling_perturn.pdf")
    stepkl(names, vocab, games, base + "_stepkl.pdf")


if __name__ == "__main__":
    main()
