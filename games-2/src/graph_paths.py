"""Plot both players' NAVIGATION PATHS on the grid from a game1_graph transcript.
One panel per game: the ROWSxCOLS word-grid, with Qwen1's and Qwen2's trajectories
(node sequence over rounds) drawn as round-graded paths; start squares and the meeting
node marked. Shows how the two models move toward each other on the hidden graph.

Usage: python src/graph_paths.py <..._transcript.jsonl> [ROWS COLS]
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

from game1_graph import WORDS       # single source of truth for node words (64)


def main():
    tpath = sys.argv[1]
    ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    COLS = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    def coord(w):
        i = WORDS.index(w); return (i // COLS, i % COLS)

    games = {}
    for l in open(tpath):
        r = json.loads(l); games.setdefault(r["game"], []).append(r)
    for g in games:
        games[g].sort(key=lambda r: r["round"])

    out = os.path.splitext(tpath)[0].replace("_transcript", "") + "_paths.pdf"
    ng = len(games); ncol = 5; nrow = int(np.ceil(ng / ncol))
    with PdfPages(out) as pdf:
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.4 * nrow))
        axes = np.array(axes).reshape(-1)
        for gi in sorted(games):
            ax = axes[gi]; rows = games[gi]
            # start positions from the transcript if present, else fall back to corners
            if "startA" in rows[0]:
                a0, b0 = coord(rows[0]["startA"]), coord(rows[0]["startB"])
            else:
                (a0, b0) = ((0, 0), (ROWS - 1, COLS - 1)) if gi % 2 == 0 else ((0, COLS - 1), (ROWS - 1, 0))
            Apath = [a0] + [coord(r["A_word"]) for r in rows]
            Bpath = [b0] + [coord(r["B_word"]) for r in rows]
            # grid word labels
            for rr in range(ROWS):
                for cc in range(COLS):
                    ax.text(cc, rr, WORDS[rr * COLS + cc], ha="center", va="center", fontsize=5, color="0.6")
            def draw(path, cmap, marker, label):
                P = np.array(path)
                ax.plot(P[:, 1], P[:, 0], "-", color="0.4", alpha=.3, lw=1, zorder=1)
                ax.scatter(P[:, 1], P[:, 0], c=range(len(P)), cmap=cmap, s=45, marker=marker,
                           edgecolors="k", linewidths=.4, zorder=3, label=label)
                ax.scatter([P[0, 1]], [P[0, 0]], marker="s", s=130, facecolors="none",
                           edgecolors=("tab:blue" if cmap == "Blues" else "tab:orange"), linewidths=1.8, zorder=2)
            draw(Apath, "Blues", "o", "Qwen1")
            draw(Bpath, "Oranges", "^", "Qwen2")
            met = any(r["met"] for r in rows)
            if met:
                mnode = coord([r for r in rows if r["met"]][0]["A_word"])
                ax.scatter([mnode[1]], [mnode[0]], marker="*", s=320, color="red", zorder=4)
            ax.set_xlim(-.6, COLS - .4); ax.set_ylim(ROWS - .4, -.6)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"game {gi}: {'MET @r%d' % ([r['round'] for r in rows if r['met']][0]+1) if met else 'no-meet (%d r)' % len(rows)}",
                         fontsize=8)
        for k in range(ng, len(axes)):
            axes[k].axis("off")
        axes[0].legend(fontsize=7, loc="upper left")
        fig.suptitle(f"Qwen3-32B grid navigation paths — {os.path.basename(tpath)}\n"
                     "(square=start, ★=meeting node; marker shade = round; blue=Qwen1, orange=Qwen2)", fontsize=10)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(f"wrote {out}  ({ng} games)")


if __name__ == "__main__":
    main()
