"""Baselines + behavioural read-outs for the yoked/restricted Game-1 runs (no GPU).

1. YOKED CHANCE FLOOR: pair each yoked game's ghost sequence (A) with the LIVE player
   (B) of every OTHER yoked game; a same-word-same-round collision in a mismatched pair
   is what "meeting" looks like with zero interaction. Also reported for reactive
   sequences (short, so an underestimate over long horizons).
2. RESTRICTED-GAME DRIFT: fraction of the UNRESTRICTED player's (A's) words inside the
   partner's secret category, by turn-triplet, split met vs no-meet. Category vocab =
   union of the restricted player's words across games (+ singular-stripped match).
3. PERSEVERATION: fraction of A's words sharing a 4-char prefix with one of A's own
   earlier words (morphological-loop pathology).
4. MEETING WORD: is it in the restricted category (i.e. did A do all the moving)?

Env: SRC_DIR(runs/game-1/qwen32/qwen32_variations_cap100)
Out: <SRC_DIR>/game1_yoked_baselines.json  (+ printed table)
"""
from __future__ import annotations
import os
import json
import itertools
import collections
import numpy as np

SRC_DIR = os.environ.get("SRC_DIR", "runs/game-1/qwen32/qwen32_variations_cap100")


def load(path):
    games = collections.defaultdict(list)
    for line in open(path):
        d = json.loads(line)
        games[d["rollout"]].append(d)
    return dict(games)


def collision_null(games):
    A = {g: [t["A"] for t in ts] for g, ts in games.items()}
    B = {g: [t["B"] for t in ts] for g, ts in games.items()}
    real = [any(a == b for a, b in zip(A[g], B[g])) for g in games]
    null = []
    for i, j in itertools.permutations(games, 2):
        L = min(len(A[i]), len(B[j]))
        null.append(any(a == b for a, b in zip(A[i][:L], B[j][:L])))
    return float(np.mean(real)), float(np.mean(null)), len(null), \
        float(np.mean([len(ts) for ts in games.values()]))


def drift(games):
    vocab = {w for ts in games.values() for t in ts for w in [t["B"]]}
    in_cat = lambda w: w in vocab or w.rstrip("s") in vocab
    met_g = {g for g, ts in games.items() if ts[-1]["agreed"]}
    out = {}
    for label, sel in (("met", met_g), ("no_meet", set(games) - met_g)):
        buck = collections.defaultdict(list)
        pers = []
        for g in sel:
            A = [t["A"] for t in games[g]]
            for i, w in enumerate(A):
                buck[min(i // 3, 5)].append(in_cat(w))
            if len(A) > 1:
                pers.append(np.mean([any(w[:4] == p[:4] and len(w) > 3 for p in A[:i])
                                     for i, w in enumerate(A)][1:]))
        out[label] = {"n": len(sel),
                      "in_category_by_triplet": [float(np.mean(buck[b])) if buck[b] else None
                                                 for b in range(6)],
                      "perseveration": float(np.mean(pers)) if pers else None}
    out["meet_words"] = {g: games[g][-1]["A"] for g in met_g}
    out["meet_word_in_category"] = float(np.mean([in_cat(games[g][-1]["A"]) for g in met_g])) \
        if met_g else None
    return out


def main():
    res = {"src": SRC_DIR}
    for cond in ("yoked", "reactive"):
        p = os.path.join(SRC_DIR, f"game1_yoked_{cond}_transcript.jsonl")
        if os.path.exists(p):
            real, null, npairs, mlen = collision_null(load(p))
            res[f"{cond}_floor"] = {"real_met": real, "shuffled_pair_met": null,
                                    "n_pairs": npairs, "mean_len": mlen}
            print(f"{cond:>9}: real met {real:.2f} vs shuffled-pair null {null:.3f} "
                  f"({npairs} pairs, mean len {mlen:.1f})")
    for cond in ("restrict-city", "restrict-fruit"):
        p = os.path.join(SRC_DIR, f"game1_yoked_{cond}_transcript.jsonl")
        if os.path.exists(p):
            res[cond] = drift(load(p))
            d = res[cond]
            for lab in ("met", "no_meet"):
                row = " ".join(f"{x:.2f}" if x is not None else " -- "
                               for x in d[lab]["in_category_by_triplet"])
                print(f"{cond} {lab:>7} (n={d[lab]['n']}): A-in-category {row}  "
                      f"perseveration={d[lab]['perseveration']}")
            print(f"{cond} meet-word in category: {d['meet_word_in_category']}")
    out = os.path.join(SRC_DIR, "game1_yoked_baselines.json")
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
