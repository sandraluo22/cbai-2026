"""Animated viewer: play a game1_graph game frame-by-frame as a GIF. Each frame is one
round; both models are shown moving on the word-grid with fading trails, the current
round + grid-distance, and a flash when they meet. Open the GIF in any browser / Quick
Look.

Usage: python src/graph_animate.py <..._transcript.jsonl> [ROWS COLS] [GAME=all|<idx>] [FPS=3]
Out:   <base>_game<gi>.gif  for each game (or the chosen one)
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from game1_graph import WORDS

ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
COLS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
GAME = os.environ.get("GAME", "all")
FPS = int(os.environ.get("FPS", "3"))
MAXFRAMES = int(os.environ.get("MAXFRAMES", "160"))


def coord(w):
    i = WORDS.index(w); return (i // COLS, i % COLS)


def animate_game(gi, rows, out):
    startA = rows[0].get("startA"); startB = rows[0].get("startB")
    a0 = coord(startA) if startA else (0, 0)
    b0 = coord(startB) if startB else (ROWS - 1, COLS - 1)
    A = [a0] + [coord(r["A_word"]) for r in rows]
    B = [b0] + [coord(r["B_word"]) for r in rows]
    dists = [abs(A[k][0] - B[k][0]) + abs(A[k][1] - B[k][1]) for k in range(len(A))]
    T = len(A)
    step = max(1, int(np.ceil(T / MAXFRAMES)))
    frames = list(range(0, T, step))
    if frames[-1] != T - 1:
        frames.append(T - 1)

    fig, ax = plt.subplots(figsize=(6.2, 6.2))

    def draw(t):
        ax.clear()
        for rr in range(ROWS):
            for cc in range(COLS):
                ax.text(cc, rr, WORDS[rr * COLS + cc], ha="center", va="center", fontsize=6, color="0.75")
        PA, PB = np.array(A[:t + 1]), np.array(B[:t + 1])
        # fading trails
        ax.plot(PA[:, 1], PA[:, 0], "-", color="tab:blue", alpha=.35, lw=1.5)
        ax.plot(PB[:, 1], PB[:, 0], "-", color="tab:orange", alpha=.35, lw=1.5)
        if t > 0:
            ax.scatter(PA[:-1, 1], PA[:-1, 0], color="tab:blue", alpha=.25, s=30)
            ax.scatter(PB[:-1, 1], PB[:-1, 0], color="tab:orange", alpha=.25, s=30)
        # start squares
        ax.scatter([a0[1]], [a0[0]], marker="s", s=140, facecolors="none", edgecolors="tab:blue", lw=1.5)
        ax.scatter([b0[1]], [b0[0]], marker="s", s=140, facecolors="none", edgecolors="tab:orange", lw=1.5)
        # current positions (big)
        met = A[t] == B[t]
        ax.scatter([A[t][1]], [A[t][0]], color="tab:blue", s=340, marker="o", edgecolors="k", lw=1, zorder=5, label="Qwen1")
        ax.scatter([B[t][1]], [B[t][0]], color="tab:orange", s=240, marker="^", edgecolors="k", lw=1, zorder=5, label="Qwen2")
        if met:
            ax.scatter([A[t][1]], [A[t][0]], marker="*", s=900, color="red", zorder=4)
        ax.set_xlim(-.6, COLS - .4); ax.set_ylim(ROWS - .4, -.6)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"game {gi} — round {t}/{T-1}   grid-distance = {dists[t]}"
                     + ("   ★ MET!" if met else ""), fontsize=11)
        ax.legend(loc="upper left", fontsize=8)

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 // FPS)
    anim.save(out, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"  {out}  ({len(frames)} frames, {T-1} rounds)")


def main():
    tpath = sys.argv[1]
    games = {}
    for l in open(tpath):
        r = json.loads(l); games.setdefault(r["game"], []).append(r)
    for g in games:
        games[g].sort(key=lambda r: r["round"])
    base = os.path.splitext(tpath)[0].replace("_transcript", "")
    targets = sorted(games) if GAME == "all" else [int(GAME)]
    for gi in targets:
        animate_game(gi, games[gi], f"{base}_game{gi}.gif")


if __name__ == "__main__":
    main()
