"""Per-turn KL plot for a Codenames LLM transcript: the GUESSER's KL (coupling =
KL(guess|swapped-clue || guess|real-clue)) and the SPYMASTER's KL (adaptivity =
KL(clue|guesser-found-X || clue|naive)) as a function of round, for each role
ordering. Mean over games (bold) with +/-SE band, and faint per-game traces.

The two KLs live on different scales (coupling is a bounded board-simplex KL;
adaptivity is a top-N full-vocab KL), so each pairing gets twin y-axes:
  left  = guesser coupling   right = spymaster adaptivity.

Usage:  python src/codenames_kl_turns.py <transcript.jsonl>
Out:    <base>_klturns.pdf  (reads the transcript only; no GPU)
"""
from __future__ import annotations

import os
import sys
import json
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

COUP_C, ADAPT_C = "tab:blue", "tab:red"


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    pairs = {}                                             # pair -> dict(coup, adapt): [game][round]
    meta = {}
    for r in rows:
        p = r["pair"]
        d = pairs.setdefault(p, {"coup": defaultdict(dict), "coup2": defaultdict(dict),
                                 "adapt": defaultdict(dict),
                                 "spymaster": r["spymaster"], "guesser": r["guesser"]})
        d["coup"][r["game"]][r["round"]] = r["coupling"]["kl"]
        if "coupling2" in r:
            d["coup2"][r["game"]][r["round"]] = r["coupling2"]["kl"]
        d["adapt"][r["game"]][r["round"]] = r["adaptivity"]["kl"]
    return pairs


def _grid(bygame):
    G = max(bygame) + 1
    R = max(max(rr) for rr in bygame.values()) + 1
    a = np.full((G, R), np.nan)
    for g, rr in bygame.items():
        for r, v in rr.items():
            a[g, r] = v
    return a


def plot(path, out=None):
    pairs = load(path)
    out = out or (os.path.splitext(path)[0].replace("_transcript", "") + "_klturns.pdf")
    names = list(pairs)
    with PdfPages(out) as pdf:
        fig, axes = plt.subplots(1, len(names), figsize=(7.2 * len(names), 4.8), squeeze=False)
        for ax, pname in zip(axes[0], names):
            d = pairs[pname]
            coup, adapt = _grid(d["coup"]), _grid(d["adapt"])
            has2 = any(d["coup2"].values())
            coup2 = _grid(d["coup2"]) if has2 else None
            rounds = np.arange(1, coup.shape[1] + 1)
            axr = ax.twinx()
            # faint per-game traces
            for g in range(coup.shape[0]):
                ax.plot(rounds, coup[g], color=COUP_C, alpha=0.12, lw=1)
                axr.plot(rounds, adapt[g], color=ADAPT_C, alpha=0.12, lw=1)
            # mean +/- SE
            for arr, axis, c in ((coup, ax, COUP_C), (adapt, axr, ADAPT_C)):
                m = np.nanmean(arr, 0); se = np.nanstd(arr, 0) / np.sqrt(arr.shape[0])
                axis.plot(rounds, m, "-o", color=c, lw=2.2)
                axis.fill_between(rounds, m - se, m + se, color=c, alpha=0.18)
            # 2nd-pick coupling (dashed, same left axis as 1st-pick coupling)
            if coup2 is not None:
                m2 = np.nanmean(coup2, 0)
                ax.plot(rounds, m2, "--s", color=COUP_C, lw=2.0, alpha=0.8)
            ax.set_xlabel("round (turn)")
            ax.set_ylabel(f"guesser coupling KL  ({d['guesser']})", color=COUP_C)
            axr.set_ylabel(f"spymaster adaptivity KL  ({d['spymaster']})", color=ADAPT_C)
            ax.tick_params(axis="y", labelcolor=COUP_C); axr.tick_params(axis="y", labelcolor=ADAPT_C)
            ax.set_ylim(bottom=0); axr.set_ylim(bottom=0)
            ax.set_xticks(rounds); ax.grid(alpha=.3)
            ax.set_title(f"{pname}\nspymaster={d['spymaster']}  guesser={d['guesser']}", fontsize=10)
        # shared legend
        from matplotlib.lines import Line2D
        fig.legend([Line2D([0], [0], color=COUP_C, lw=2.2, marker="o"),
                    Line2D([0], [0], color=COUP_C, lw=2.0, ls="--", marker="s"),
                    Line2D([0], [0], color=ADAPT_C, lw=2.2, marker="o")],
                   ["guesser coupling — 1st pick  KL(guess|swap || guess|real)",
                    "guesser coupling — 2nd pick  (autoregressive)",
                    "spymaster adaptivity  KL(clue|B-found || clue|B-naive)"],
                   loc="lower center", ncol=3, fontsize=8, frameon=False)
        fig.suptitle("Per-turn theory-of-mind KL in open-clue Codenames "
                     "(mean over games ± SE; faint = individual games)", fontsize=11)
        fig.tight_layout(rect=[0, 0.06, 1, 0.96]); pdf.savefig(fig); plt.close(fig)
    print("wrote", out)
    return out


if __name__ == "__main__":
    plot(sys.argv[1])
