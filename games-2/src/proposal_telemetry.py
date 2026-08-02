"""FIXED proposal-distribution telemetry (2026-08-02): Monte-Carlo profile of what A
actually proposes at every state of the strict-run games — replaces the broken
first-token logit readout (which measured a formatting token).

For each recorded A-state (strict transcripts, turns 1..MAXT): prefill once, replicate
the KV cache K(64) ways, sample K 4-token continuations in parallel, clean each to a
word, classify:
  used        : exactly a previously-used word
  self_family : shares a 4-prefix with one of A's own previous words (and not used)
  category    : in the partner's category wordlist or previously said by B (and novel)
  other       : novel, none of the above
Collapse pressure = used + self_family. Written per (game, turn) alongside the game's
onset turn and the turn's logged resample burden.

Env: MODEL(QwenInst32) SRC_DIR(runs/qwen32_strict) CATS(city,fruit) K(64) MAXT(24)
     TEMP(0.7) OUT(runs/qwen32_strict/proposal_telemetry.jsonl)
"""
from __future__ import annotations
import os
import json
import collections
import numpy as np
import llm_agents as LA
import qwen32_pca as G
from game1_strict import CATWORDS, CONCEPTS

MODEL = os.environ.get("MODEL", "QwenInst32")
SRC_DIR = os.environ.get("SRC_DIR", "runs/qwen32_strict")
CATS = os.environ.get("CATS", "city,fruit").split(",")
K = int(os.environ.get("K", "64"))
MAXT = int(os.environ.get("MAXT", "24"))
TEMP = float(os.environ.get("TEMP", "0.7"))
OUT = os.environ.get("OUT", os.path.join(SRC_DIR, "proposal_telemetry.jsonl"))
START_FILE = os.environ.get("START_FILE", "runs/game-1/qwen32/qwen32_pca_w2v/start_words.txt")


def main():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)

    @torch.no_grad()
    def propose_k(prompt):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        batch = ids.repeat(K, 1)
        out = model.generate(batch, max_new_tokens=4, do_sample=True, temperature=TEMP,
                             top_p=0.95, pad_token_id=tok.eos_token_id)
        return [G.clean_word(tok.decode(out[i, ids.shape[1]:], skip_special_tokens=True))
                for i in range(K)]

    starts = []
    for line in open(START_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split("\t") if "\t" in line else line.split()
            starts.append((p[-2], p[-1]))

    of = open(OUT, "w")
    for cat in CATS:
        rows = [json.loads(l) for l in open(os.path.join(SRC_DIR, f"game1_strict_{cat}_transcript.jsonl"))]
        games = collections.defaultdict(list)
        for r in rows:
            games[r["rollout"]].append(r)
        catset = set(CATWORDS[cat])
        for roll, ts in sorted(games.items()):
            ts.sort(key=lambda r: r["turn"])
            sa, sb = starts[roll]
            histA = [(sb, sa)]
            used = {sa, sb}
            own = [sa]
            bsaid = {sb}
            onset = next((t["turn"] for i, t in enumerate(ts)
                          if len(t["A"]) > 3 and any(t["A"][:4] == p[:4] and len(p) > 3
                                                     for p in [x["A"] for x in ts[:i]] + [sa])), None)
            for t in ts:
                if t["turn"] > MAXT:
                    break
                body = (G.OPEN_PROMPT + " "
                        + " ".join(f"Round {k+1}: the other player said {o}, you said {s}."
                                   for k, (o, s) in enumerate(histA))
                        + " Words already used (do not repeat): " + ", ".join(sorted(used)) + ".")
                words = propose_k(LA._render(tok, body) + "\nMy word:")
                fams = {w[:4] for w in own if len(w) > 3}
                cls = collections.Counter()
                for w in words:
                    if not w:
                        cls["junk"] += 1
                    elif w in used:
                        cls["used"] += 1
                    elif len(w) > 3 and w[:4] in fams:
                        cls["self_family"] += 1
                    elif w in catset or w in bsaid:
                        cls["category"] += 1
                    else:
                        cls["other"] += 1
                of.write(json.dumps({"cat": cat, "rollout": roll, "turn": t["turn"],
                                     "onset": onset, "met": ts[-1]["agreed"],
                                     "resamplesA": t["resamplesA"],
                                     "frac": {k: v / K for k, v in cls.items()}}) + "\n")
                of.flush()
                used |= {t["A"], t["B"]}
                own.append(t["A"])
                bsaid.add(t["B"])
                histA.append((t["B"], t["A"]))
            print(f"[ptel] {cat} roll {roll} done ({min(len(ts), MAXT)} states)", flush=True)
    of.close()
    print(f"[ptel] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
