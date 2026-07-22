"""Compare the GUESSER's KL and the SPYMASTER's KL over turns, for the 1st and 2nd
guess distributions.

  guesser KL   = coupling  = KL(guess-dist | swapped clue || guess-dist | real clue)
                 - 1st-pick distribution (coupling)  and  2nd-pick distribution (coupling2)
  spymaster KL = adaptivity = KL(clue | B-state || clue | naive)

Per role-ordering, two panels (twin y-axes: guesser coupling left/blue, spymaster
adaptivity right/red), mean over games +/- SE:
  LEFT  : 1st-guess coupling  vs  spymaster adaptivity
  RIGHT : 2nd-guess coupling  vs  spymaster adaptivity

Usage:  python src/codenames_kl_compare.py <transcript.jsonl> [more.jsonl ...]
Out:    <base>_klcompare.pdf
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

GUES_C, SPY_C = "tab:blue", "tab:red"


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    pairs = {}
    for r in rows:
        p = r["pair"]
        d = pairs.setdefault(p, {"c1": defaultdict(dict), "c2": defaultdict(dict), "ad": defaultdict(dict),
                                 "spymaster": r["spymaster"], "guesser": r["guesser"]})
        d["c1"][r["game"]][r["round"]] = r["coupling"]["kl"]
        if "coupling2" in r:
            d["c2"][r["game"]][r["round"]] = r["coupling2"]["kl"]
        d["ad"][r["game"]][r["round"]] = r["adaptivity"]["kl"]
    return pairs


def grid(bg, R):
    G = max(bg) + 1
    a = np.full((G, R), np.nan)
    for g, rr in bg.items():
        for r, v in rr.items():
            a[g, r] = v
    return a


def _mse(arr):
    m = np.nanmean(arr, 0)
    denom = np.clip(np.sum(~np.isnan(arr), 0), 1, None)
    return m, np.nanstd(arr, 0) / np.sqrt(denom)


def panel(ax, coup, adapt, guesser, spymaster, title):
    rounds = np.arange(1, coup.shape[1] + 1)
    axr = ax.twinx()
    m, se = _mse(coup)
    ax.plot(rounds, m, "-o", color=GUES_C, lw=2.3, label="guesser coupling")
    ax.fill_between(rounds, m - se, m + se, color=GUES_C, alpha=.18)
    ma, sea = _mse(adapt)
    axr.plot(rounds, ma, "-s", color=SPY_C, lw=2.3, label="spymaster adaptivity")
    axr.fill_between(rounds, ma - sea, ma + sea, color=SPY_C, alpha=.18)
    ax.set_xlabel("turn")
    ax.set_ylabel(f"guesser coupling KL  ({guesser})", color=GUES_C)
    axr.set_ylabel(f"spymaster adaptivity KL  ({spymaster})", color=SPY_C)
    ax.tick_params(axis="y", labelcolor=GUES_C); axr.tick_params(axis="y", labelcolor=SPY_C)
    ax.set_ylim(bottom=0); axr.set_ylim(bottom=0)
    ax.set_xticks(rounds); ax.grid(alpha=.3); ax.set_title(title, fontsize=9)


def plot(path, out=None):
    pairs = load(path)
    out = out or (os.path.splitext(path)[0].replace("_transcript", "") + "_klcompare.pdf")
    names = list(pairs)
    with PdfPages(out) as pdf:
        fig, axes = plt.subplots(len(names), 2, figsize=(14, 4.7 * len(names)), squeeze=False)
        for i, pn in enumerate(names):
            d = pairs[pn]
            R = max(max(rr) for m in ("c1", "c2", "ad") if d[m] for rr in d[m].values()) + 1
            c1, ad = grid(d["c1"], R), grid(d["ad"], R)
            panel(axes[i][0], c1, ad, d["guesser"], d["spymaster"],
                  f"{pn}: 1st-guess coupling  vs  spymaster adaptivity")
            if any(d["c2"].values()):
                panel(axes[i][1], grid(d["c2"], R), ad, d["guesser"], d["spymaster"],
                      f"{pn}: 2nd-guess coupling  vs  spymaster adaptivity")
            else:
                axes[i][1].axis("off")
                axes[i][1].text(.5, .5, "2nd-guess KL not logged\n(re-run game_llm_open.py)",
                                ha="center", va="center", fontsize=10)
        fig.suptitle("Guesser KL vs spymaster KL over turns   "
                     "(blue = guesser coupling, red = spymaster adaptivity; mean over games ± SE)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96]); pdf.savefig(fig); plt.close(fig)
    print("wrote", out)
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        plot(p)
