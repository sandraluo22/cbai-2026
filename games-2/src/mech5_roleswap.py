"""MECH5 (2026-08-03): DIRECT TEST OF H2b (role-bound continuation).

The history is BYTE-IDENTICAL across the request conditions: a third-person
transcript "Round k: Player 1 said A, Player 2 said B." with the planted triple in
one player's slots. Only the FINAL REQUEST varies: ask for Player 1's vs Player 2's
next word ("What word do you think Player N should say next?" -> "Player N should
say:"). Full factorial: planted-owner {P1, P2} x asked-player {P1, P2} x
{planted, base} x streams.

H2b predicts capture ONLY when asked-player == planted-owner (interaction).
Surface-structural accounts (H2a) predict equal capture for both requests, since
the prompt differs only in the final sentence's player index.

If capture needs a consistent textual series (surface-structural H2a), scrambling
the per-round format should destroy it. If capture is role-bound (H2b — bound to
"these are the answering player's words" however rendered), it should survive.

Per-round formats (all express the same round content):
  f0  "Round k: the other player said B, you said A."
  f1  "Round k: you said A, the other player said B."
  f2  "Them: B / You: A" (line pair)
  f3  "Round k - the other player; B. you; A."
  f4  "You: A | Them: B"
  f5  "then they said B and you said A;"

Conditions (x {planted, base} x N_STREAMS, planted = planted/planting/plantings
in A's slots, rounds 1-3, gap 0):
  fixed_f0    all rounds f0 (mech4 reference format)
  fixed_f4    all rounds f4 (a compact format, consistency control)
  alt_f0f1    alternate f0/f1 by round
  all_diff    round k uses format f_k (3 different formats in 3 rounds)
  scramble_a  per-round formats [f2, f5, f3]
  scramble_b  per-round formats [f4, f0, f2]

Measures: K=64 MC planted-family mass (novel), category mass, raw family prob.
Env: MODEL(QwenInst32) SRC_DIR START_FILE N_STREAMS(4) K(64) TEMP(0.7)
     RUN_DIR(runs/mech5_roleswap)
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
RUN_DIR = os.environ.get("RUN_DIR", "runs/mech5_roleswap")

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

    def fid(w):
        return tok(" " + w, add_special_tokens=False)["input_ids"][0]
    fam_ids = torch.tensor(sorted({fid(w) for w in PLANT + [FAMP, "plants", "plant", "planter"]}))

    @torch.no_grad()
    def measure(body, ctx_words):  # unused in mech5
        prompt = LA._render(tok, body) + "\nMy word:"
        enc = tok(prompt, return_tensors="pt").to(dev)
        logits = model(enc.input_ids).logits[0, -1].float()
        pfam = float(torch.softmax(logits, 0)[fam_ids].sum())
        out = model.generate(enc.input_ids.repeat(K, 1), max_new_tokens=4, do_sample=True,
                             temperature=TEMP, top_p=0.95, pad_token_id=tok.eos_token_id)
        ws = [G.clean_word(tok.decode(out[i, enc.input_ids.shape[1]:], skip_special_tokens=True))
              for i in range(K)]
        fam = float(np.mean([1 if (w and w not in ctx_words and len(w) > 3 and w[:4] == FAMP)
                             else 0 for w in ws]))
        cat = float(np.mean([1 if (w and w not in ctx_words and w in catset) else 0 for w in ws]))
        return pfam, fam, cat

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
                base_body = ("You are witnessing a word game between two players. " + OPEN3 + hist
                             + " Words already used (may not be repeated): " + ", ".join(used) + ".")
                ctx = {w for w1, w2 in rounds for w in (w1, w2)} | set(PLANT)
                for asked in ("P1", "P2"):
                    n = asked[1]
                    body = base_body + f" What word do you think Player {n} should say next?"
                    prompt_prefill = f"\nPlayer {n} should say:"
                    prompt = LA._render(tok, body) + prompt_prefill
                    import torch
                    enc = tok(prompt, return_tensors="pt").to(dev)
                    logits = model(enc.input_ids).logits[0, -1].float()
                    pfam = float(torch.softmax(logits, 0)[fam_ids].sum())
                    out = model.generate(enc.input_ids.repeat(K, 1), max_new_tokens=4, do_sample=True,
                                         temperature=TEMP, top_p=0.95, pad_token_id=tok.eos_token_id)
                    ws = [G.clean_word(tok.decode(out[i, enc.input_ids.shape[1]:], skip_special_tokens=True))
                          for i in range(K)]
                    fam = float(np.mean([1 if (w and w not in ctx and len(w) > 3 and w[:4] == FAMP)
                                         else 0 for w in ws]))
                    cat = float(np.mean([1 if (w and w not in ctx and w in catset) else 0 for w in ws]))
                    results.append({"owner": owner, "asked": asked, "content": content,
                                    "stream": roll, "pfam": pfam, "fam_mass": fam, "cat_mass": cat})
                    json.dump({"per_cell": results}, open(os.path.join(RUN_DIR, "mech5.json"), "w"))
        print(f"[mech5] stream {roll} done", flush=True)

    summary = {}
    for owner in ("P1", "P2"):
        for asked in ("P1", "P2"):
            p = float(np.mean([r["fam_mass"] for r in results
                               if r["owner"] == owner and r["asked"] == asked and r["content"] == "planted"]))
            b = float(np.mean([r["fam_mass"] for r in results
                               if r["owner"] == owner and r["asked"] == asked and r["content"] == "base"]))
            summary[f"owner{owner}_ask{asked}"] = {"planted": p, "base": b, "capture": p - b}
            print(f"[mech5] === owner {owner} ask {asked}: capture {p - b:.3f}", flush=True)
    json.dump({"per_cell": results, "summary": summary},
              open(os.path.join(RUN_DIR, "mech5.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
