"""The KL between the spymaster and the guesser, per turn, over the board.

Both are put on the shared 12-word board simplex:
  spymaster p = uniform over its true target set T   (what A means)
  guesser   b = its belief over the board            (what B thinks)
KL(p || b) per round = how far B's belief is from A's targets (0 = perfectly aligned).
Direction is KL(spymaster || guesser) because b can put mass off-target (making the
reverse KL infinite); this one stays finite and shrinks as B recovers T.

Usage:  python src/codenames_kl_agents.py <transcript.jsonl> [more ...]
Out:    <base>_klagents.pdf
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


def kl_pb(targets, b):
    p = np.zeros(len(b))
    for i in targets:
        p[i] = 1.0
    p /= p.sum()
    b = b / b.sum()
    m = p > 0
    return float(np.sum(p[m] * (np.log(p[m]) - np.log(b[m] + 1e-12))))


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    pairs = defaultdict(lambda: defaultdict(dict)); meta = {}
    for r in rows:
        board = list(r["belief"]); bidx = {w: i for i, w in enumerate(board)}
        b = np.array([r["belief"][w] for w in board])
        tgt = [bidx[w] for w in r["targets"] if w in bidx]
        pairs[r["pair"]][r["game"]][r["round"]] = kl_pb(tgt, b)
        meta[r["pair"]] = (r["spymaster"], r["guesser"])
    return pairs, meta


def grid(bg):
    G = max(bg) + 1; R = max(max(rr) for rr in bg.values()) + 1
    a = np.full((G, R), np.nan)
    for g, rr in bg.items():
        for rd, v in rr.items():
            a[g, rd] = v
    return a


def plot(path, out=None):
    pairs, meta = load(path)
    out = out or (os.path.splitext(path)[0].replace("_transcript", "") + "_klagents.pdf")
    with PdfPages(out) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        for pn in pairs:
            a = grid(pairs[pn]); rounds = np.arange(1, a.shape[1] + 1)
            m = np.nanmean(a, 0)
            se = np.nanstd(a, 0) / np.sqrt(np.clip(np.sum(~np.isnan(a), 0), 1, None))
            spy, gue = meta[pn]
            line, = ax.plot(rounds, m, "-o", lw=2.3, label=f"{pn}  (spy={spy}, guesser={gue})")
            ax.fill_between(rounds, m - se, m + se, color=line.get_color(), alpha=.18)
        ax.set_xlabel("turn"); ax.set_ylabel("KL(spymaster targets ‖ guesser belief)  [nats]")
        ax.set_title("KL between spymaster and guesser over turns\n(lower = the two minds more aligned; mean over games ± SE)", fontsize=10)
        ax.set_ylim(bottom=0); ax.grid(alpha=.3); ax.legend(fontsize=8)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print("wrote", out)
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        plot(p)
