"""MECH6 (2026-08-03): FIRST-PERSON version of the mech5 role swap.

Same byte-identical third-person history as mech5 ("Round k: Player 1 said A,
Player 2 said B.", planted triple in one player's slots). But instead of asking
the model to ADVISE a player (witness frame), the final sentence ASSIGNS the
model a role and elicits its own move first-person:
  " You are Player {n}. It is time to say your next word." + prefill "\\nMy word:"

Factorial: planted-owner {P1, P2} x assigned-role {P1, P2} x {planted, base}
x streams.

mech5 (witness frame) found partner-matching: near-total family capture when
asked about the NON-owner (0.93-0.99) vs owner (0.31-0.59). If the first-person
frame suppresses partner-matching (the live-game hypothesis), the assigned-role
== non-owner cells should drop far below mech5's 0.93-0.99, while assigned-role
== owner stays near in-game capture (~0.5). If frame does not matter, mech5's
pattern replicates.

Measures: K=64 MC planted-family mass (novel words only), category mass.
Env: MODEL(QwenInst32) SRC_DIR START_FILE N_STREAMS(4) K(64) TEMP(0.7)
     RUN_DIR(runs/mech6_firstperson)
"""
from __future__ import annotations
import os
import json
import collections
import numpy as np
import llm_agents as LA
import qwen32_pca as G
from game1_strict import CATWORDS

MODEL = os.environ.get("MODEL", "QwenInst32")
SRC_DIR = os.environ.get("SRC_DIR", "runs/qwen32_strict")
START_FILE = os.environ.get("START_FILE", "runs/game-1/qwen32/qwen32_pca_w2v/start_words.txt")
N_STREAMS = int(os.environ.get("N_STREAMS", "4"))
K = int(os.environ.get("K", "64"))
TEMP = float(os.environ.get("TEMP", "0.7"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/mech6_firstperson")

PLANT = ["planted", "planting", "plantings"]
FAMP = "plan"
FILL = ["window", "carpet", "stapler"]

OPEN3 = ("Two players are playing a word game. Each round, both players say one word "
         "at the same time, then the two words are revealed. They win the round only "
         "if they both said the same word. Neither player may say any word that has "
         "already been said by either player.")


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)
    catset = set(CATWORDS["city"])

    rows_t = [json.loads(l) for l in open(os.path.join(SRC_DIR, "game1_strict_city_transcript.jsonl"))]
    games = collections.defaultdict(list)
    for r in rows_t:
        games[r["rollout"]].append(r)
    streams = sorted(games.items(), key=lambda kv: -len(kv[1]))[:N_STREAMS]
    starts = []
    for line in open(START_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split("\t") if "\t" in line else line.split()
            starts.append((p[-2], p[-1]))

    results = []
    for si, (roll, ts) in enumerate(streams):
        Bseq = [t["B"] for t in ts]
        sa, sb = starts[roll]
        for owner in ("P1", "P2"):
            for content in ("planted", "base"):
                aw = PLANT if content == "planted" else FILL
                rounds = []
                fi = iter(["ribbon", "saddle", "pencil"])
                for k in range(3):
                    if owner == "P1":
                        p1w, p2w = aw[k], Bseq[k]
                    else:
                        p1w, p2w = (next(fi), aw[k]) if content == "planted" else (Bseq[k], aw[k])
                    rounds.append((p1w, p2w))
                hist = " " + " ".join(f"Round {k+1}: Player 1 said {w1}, Player 2 said {w2}."
                                      for k, (w1, w2) in enumerate(rounds))
                used = sorted({sa, sb} | {w for w1, w2 in rounds for w in (w1, w2)})
                base_body = (OPEN3 + hist
                             + " Words already used (may not be repeated): " + ", ".join(used) + ".")
                ctx = {w for w1, w2 in rounds for w in (w1, w2)} | set(PLANT)
                for role in ("P1", "P2"):
                    n = role[1]
                    body = base_body + f" You are Player {n}. It is time to say your next word."
                    prompt = LA._render(tok, body) + "\nMy word:"
                    enc = tok(prompt, return_tensors="pt").to(dev)
                    with torch.no_grad():
                        out = model.generate(enc.input_ids.repeat(K, 1), max_new_tokens=4, do_sample=True,
                                             temperature=TEMP, top_p=0.95, pad_token_id=tok.eos_token_id)
                    ws = [G.clean_word(tok.decode(out[i, enc.input_ids.shape[1]:], skip_special_tokens=True))
                          for i in range(K)]
                    fam = float(np.mean([1 if (w and w not in ctx and len(w) > 3 and w[:4] == FAMP)
                                         else 0 for w in ws]))
                    cat = float(np.mean([1 if (w and w not in ctx and w in catset) else 0 for w in ws]))
                    results.append({"owner": owner, "role": role, "content": content,
                                    "stream": roll, "fam_mass": fam, "cat_mass": cat})
                    json.dump({"per_cell": results}, open(os.path.join(RUN_DIR, "mech6.json"), "w"))
        print(f"[mech6] stream {roll} done", flush=True)

    summary = {}
    for owner in ("P1", "P2"):
        for role in ("P1", "P2"):
            p = float(np.mean([r["fam_mass"] for r in results
                               if r["owner"] == owner and r["role"] == role and r["content"] == "planted"]))
            b = float(np.mean([r["fam_mass"] for r in results
                               if r["owner"] == owner and r["role"] == role and r["content"] == "base"]))
            summary[f"owner{owner}_role{role}"] = {"planted": p, "base": b, "capture": p - b}
            print(f"[mech6] === owner {owner} role {role}: capture {p - b:.3f}", flush=True)
    json.dump({"per_cell": results, "summary": summary},
              open(os.path.join(RUN_DIR, "mech6.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
