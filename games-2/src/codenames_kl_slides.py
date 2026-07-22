"""Per-turn SLIDESHOW for a Codenames LLM transcript: one page per turn, showing the
two distributions behind each side's theory-of-mind KL.

  * LEFT  (guesser / coupling): the guess distribution over the board words under the
    REAL clue vs the SWAPPED clue. KL between them = coupling.
  * RIGHT (spymaster / adaptivity): the clue distribution (top tokens) given the REAL
    guesser-found state vs a NAIVE guesser. top-N KL between them = adaptivity.

Flip through the pages to watch, turn by turn, how much the clue moves the guess and
how much the guesser's revealed state moves the clue.

Usage:  python src/codenames_kl_slides.py <transcript.jsonl>
Out:    <base>_klslides.pdf   (reads the transcript only; no GPU)
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

CLEAN_C, ALT_C = "tab:blue", "0.55"


def _san(s):
    """Latin-1-safe label: non-ASCII glyphs (e.g. Qwen's CJK clue tokens) render as
    boxes in the default matplotlib font, so show their unicode-escape instead."""
    return s if all(ord(c) < 128 for c in s) else s.encode("unicode_escape").decode()


def _bars(ax, da, db, la, lb, order=None, rot=90):
    keys = order if order is not None else list(dict.fromkeys(list(da) + list(db)))
    x = np.arange(len(keys)); w = 0.4
    ax.bar(x - w / 2, [da.get(k, 0) for k in keys], w, color=CLEAN_C, alpha=.85, label=la)
    ax.bar(x + w / 2, [db.get(k, 0) for k in keys], w, color=ALT_C, alpha=.85, label=lb)
    ax.set_xticks(x); ax.set_xticklabels([_san(k) for k in keys], rotation=rot, fontsize=7)
    ax.set_ylim(0, 1); ax.legend(fontsize=7)
    return keys


def _mark_targets(ax, targets):
    for lab in ax.get_xticklabels():
        if lab.get_text() in targets:
            lab.set_color("tab:green"); lab.set_fontweight("bold")


def _annot_guess(ax, board, dist, r, rank):
    """Annotate the actual guess of the given rank on `ax` with its prob under `dist`."""
    if len(r["guesses"]) < rank:
        return
    g, ok = r["guesses"][rank - 1], r["correct"][rank - 1]
    if g in board:
        xi = board.index(g); p = dist.get(g, 0.0)
        ax.annotate(f"guess{rank}{'✓' if ok else '✗'}\np={p:.2g}", xy=(xi, p),
                    xytext=(xi, 0.62), fontsize=6.5, ha="center",
                    color="darkgreen" if ok else "firebrick",
                    arrowprops=dict(arrowstyle="->", lw=.8, color="0.3"))


def plot(path, out=None):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    out = out or (os.path.splitext(path)[0].replace("_transcript", "") + "_klslides.pdf")
    with PdfPages(out) as pdf:
        for r in rows:
            board = list(r["belief"])                      # fixed board-word order
            cp = r["coupling"]; ad = r["adaptivity"]; cp2 = r.get("coupling2")
            has_clue = "clue_dist_real" in ad              # spymaster dists (needs re-logged transcript)
            clue, swapc = _san(r["clue"]), _san(cp["clue_swapped_to"])
            fig, axes = plt.subplots(1, 3, figsize=(20, 5.2))
            # panel 0 -- guesser FIRST-pick distribution (clean vs swap clue)
            _bars(axes[0], cp["guess_dist_clean"], cp["guess_dist_swap"],
                  f"real clue '{clue}'", f"swap '{swapc}'", order=board)
            axes[0].set_title(f"GUESSER {r['guesser']} 1st pick   coupling1 KL={cp['kl']:.2f}", fontsize=9)
            _mark_targets(axes[0], r["targets"]); _annot_guess(axes[0], board, cp["guess_dist_clean"], r, 1)
            # panel 1 -- guesser SECOND-pick distribution (autoregressive; 1st pick held fixed)
            if cp2:
                _bars(axes[1], cp2["guess_dist2_clean"], cp2["guess_dist2_swap"],
                      f"real clue '{clue}'", f"swap '{swapc}'", order=board)
                axes[1].set_title(f"GUESSER {r['guesser']} 2nd pick  |  1st='{_san(cp2['cond_first'])}'   "
                                  f"coupling2 KL={cp2['kl']:.2f}", fontsize=9)
                _mark_targets(axes[1], r["targets"]); _annot_guess(axes[1], board, cp2["guess_dist2_clean"], r, 2)
            else:
                axes[1].axis("off")
                axes[1].text(0.5, 0.5, "2nd-pick distribution not logged\n(re-run game_llm_open.py)",
                             ha="center", va="center", fontsize=10)
            # panel 2 -- spymaster clue distribution (real vs naive guesser-state)
            if has_clue:
                _bars(axes[2], ad["clue_dist_real"], ad["clue_dist_naive"],
                      "real guesser-state", "naive guesser")
                shown = sum(ad["clue_dist_real"].values()); k = len(ad["clue_dist_real"])
                axes[2].set_title(f"SPYMASTER {r['spymaster']}: clue dist  "
                                  f"(top-{k} of vocab, Σ={shown:.2f})   adaptivity KL={ad['kl']:.2f}", fontsize=9)
            else:
                axes[2].axis("off")
                axes[2].text(0.5, 0.5, f"spymaster clue distribution not logged\n(adaptivity KL={ad['kl']:.2f})",
                             ha="center", va="center", fontsize=10)
            tgt = ", ".join(r["targets"])
            gsr = ", ".join(f"{g}{'✓' if ok else '✗'}" for g, ok in zip(r["guesses"], r["correct"]))
            fig.suptitle(f"{r['pair']}   game {r['game']}  round {r['round']+1}\n"
                         f"HINT: '{_san(r['clue'])}' ({r['count']})      GUESSED: {gsr}\n"
                         f"targets: {tgt}      |      found before: {', '.join(r['found_so_far']) or 'nothing'}",
                         fontsize=10)
            fig.tight_layout(rect=[0, 0, 1, 0.90]); pdf.savefig(fig); plt.close(fig)
    print("wrote", out, f"({len(rows)} slides)")
    return out


if __name__ == "__main__":
    plot(sys.argv[1])
