"""Visualize the update dynamics: SUPPORT RASTERS. One page per game: y = tokens
(ordered by first appearance in either player's top-15), x = turn. Cell color: blue =
mass in A's dist only, orange = B only, purple = both (shared support). The dynamics
claim in one picture: met games grow a purple band (supports merge, then a sample
collides); no-meet games stay two disjoint color blocks (mutual evidence-discounting).

Env: KL_DIR(runs/game-1/2_restricted_core/qwen32_cap24/kl)
     GAMES("reactive:0,reactive:2,restrict-city:0,restrict-city:4,restrict-fruit:3,
            restrict-fruit:1,nolist-city:0,repeatok-fruit:8" — cond:game pairs)
     OUT_PDF(runs/game-1/6_analyses/update_dynamics/support_rasters.pdf) MAXTOK_PAGE(60)
"""
from __future__ import annotations
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

KL_DIR = os.environ.get("KL_DIR", "runs/game-1/2_restricted_core/qwen32_cap24/kl")
GAMES = os.environ.get(
    "GAMES",
    "reactive:0,reactive:2,restrict-city:0,restrict-city:4,restrict-fruit:3,"
    "restrict-fruit:1,nolist-city:0,repeatok-fruit:8").split(",")
OUT_PDF = os.environ.get("OUT_PDF", "runs/game-1/6_analyses/update_dynamics/support_rasters.pdf")
MAXTOK_PAGE = int(os.environ.get("MAXTOK_PAGE", "60"))


def norm(d):
    z = sum(d.values())
    return {k: v / z for k, v in d.items() if len(k) > 1}


def main():
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        for spec in GAMES:
            cond, g = spec.strip().rsplit(":", 1)
            path = os.path.join(KL_DIR, f"{cond}_crossKL.json")
            turns = json.load(open(path)).get(g)
            if turns is None:
                print(f"[raster] skip {spec}")
                continue
            toks = []
            for t in turns:
                for k in list(norm(t["topA"])) + list(norm(t["topB"])):
                    if k not in toks:
                        toks.append(k)
            toks = toks[:MAXTOK_PAGE]
            ix = {k: i for i, k in enumerate(toks)}
            T = len(turns)
            img = np.ones((len(toks), T, 3))
            for x, t in enumerate(turns):
                A, B = norm(t["topA"]), norm(t["topB"])
                for k, i in ix.items():
                    a = min(A.get(k, 0) * 4, 1.0)
                    b = min(B.get(k, 0) * 4, 1.0)
                    # blue for A, orange for B, both -> purple-ish blend on white
                    img[i, x] = (1 - .85 * a - .05 * b,
                                 1 - .45 * a - .55 * b,
                                 1 - .05 * a - .85 * b)
            met_turn = next((t["turn"] for t in turns if t["agreed"]), None)
            fig, ax = plt.subplots(figsize=(max(6, T * .5), max(5, len(toks) * .16)))
            ax.imshow(img, aspect="auto", interpolation="nearest")
            ax.set_yticks(range(len(toks)))
            ax.set_yticklabels(toks, fontsize=5.5)
            ax.set_xticks(range(T))
            ax.set_xticklabels([t["turn"] for t in turns], fontsize=6)
            ax.set_xlabel("turn")
            picks = "  ".join(f"{t['pickA']}/{t['pickB']}" for t in turns[:12])
            ax.set_title(f"{cond} game {g} — support raster (blue=A, orange=B, purple=shared)\n"
                         f"{'MET @' + str(met_turn) if met_turn else 'NO MEET'}   picks: {picks}"
                         + ("…" if T > 12 else ""), fontsize=8)
            if met_turn is not None:
                x = [t["turn"] for t in turns].index(met_turn)
                ax.axvline(x, color="green", lw=2, alpha=.6)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
            print(f"[raster] {spec}: {len(toks)} tokens x {T} turns")
    print(f"[raster] wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
