"""DOSE-8 PLACEMENT (2026-08-07): is HIGH-dose capture lexical (position- and
slot-independent) or still bound to the answering player's column?

Shuffle-v2 showed order-invariance WITHIN the self column at high dose; mech4
showed slot-dependence at dose 3 (self 0.50 / partner 0.11 / list 0.12). The
missing cell: dose 8 in OTHER placements. Cells (8 plant-family words each,
replayed city B streams, MC K=64 family mass at the branch):
  self8     8 family words as A's own words
  partner8  8 family words as B's words (A neutral fillers)
  list8     history neutral, 8 family words in an appended "unrelated word list"
  fill      neutral control
Env: MODEL(QwenInst32) SRC_DIR START_FILE N_STREAMS(6) K(64) TEMP(0.7)
     RUN_DIR(runs/dose8_placement)
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
N_STREAMS = int(os.environ.get("N_STREAMS", "6"))
K = int(os.environ.get("K", "64"))
TEMP = float(os.environ.get("TEMP", "0.7"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/dose8_placement")

PLANT8 = ["planted", "planting", "plantings", "replant",
          "replanted", "planter", "planters", "plantation"]
FILL8 = ["window", "carpet", "stapler", "napkin", "candle", "basket", "ribbon", "saddle"]
FAMP = "plan"


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)
    catset = set(CATWORDS["city"])

    def body_of(hist, used, extra=""):
        s = G.OPEN_PROMPT + extra + " " + " ".join(
            f"Round {k+1}: the other player said {o}, you said {s_}."
            for k, (o, s_) in enumerate(hist))
        return s + " Words already used (do not repeat): " + ", ".join(sorted(used)) + "."

    @torch.no_grad()
    def propose_k(body):
        ids = tok(LA._render(tok, body) + "\nMy word:", return_tensors="pt").input_ids.to(dev)
        out = model.generate(ids.repeat(K, 1), max_new_tokens=4, do_sample=True,
                             temperature=TEMP, top_p=0.95, pad_token_id=tok.eos_token_id)
        return [G.clean_word(tok.decode(out[i, ids.shape[1]:], skip_special_tokens=True))
                for i in range(K)]

    games = collections.defaultdict(list)
    for line in open(os.path.join(SRC_DIR, "game1_strict_city_transcript.jsonl")):
        d = json.loads(line)
        games[d["rollout"]].append(d)
    streams = [sorted(ts, key=lambda r: r["turn"]) for _, ts in
               sorted(games.items(), key=lambda kv: -len(kv[1]))[:N_STREAMS]]
    starts = []
    for line in open(START_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split("\t") if "\t" in line else line.split()
            starts.append((p[-2], p[-1]))

    results = []
    for si, ts in enumerate(streams):
        roll = ts[0]["rollout"]
        Bseq = [t["B"] for t in ts]
        if len(Bseq) < 8:
            Bseq = Bseq + Bseq
        sa, sb = starts[roll]
        for cell in ("self8", "partner8", "list8", "fill"):
            hist = [(sb, sa)]
            used = {sa, sb}
            extra = ""
            for i in range(8):
                if cell == "self8":
                    b, a = Bseq[i], PLANT8[i]
                elif cell == "partner8":
                    b, a = PLANT8[i], FILL8[i]
                else:
                    b, a = Bseq[i], FILL8[i]
                hist.append((b, a))
                used |= {a, b}
            if cell == "list8":
                extra = " Unrelated word list: " + ", ".join(PLANT8) + "."
                used |= set(PLANT8)
            props = propose_k(body_of(hist, used, extra))
            fam = np.mean([1 if (w and w not in used and len(w) > 3 and w[:4] == FAMP)
                           else 0 for w in props])
            cat = np.mean([1 if (w and w not in used and w in catset) else 0 for w in props])
            results.append({"cell": cell, "stream": roll, "fam_mass": float(fam),
                            "cat_mass": float(cat)})
            json.dump({"per_state": results}, open(os.path.join(RUN_DIR, "dose8.json"), "w"))
        print(f"[d8p] stream {roll} done", flush=True)

    out = {"per_state": results, "cells": {}}
    for cell in ("self8", "partner8", "list8", "fill"):
        sel = [r for r in results if r["cell"] == cell]
        out["cells"][cell] = {"fam_mass": float(np.mean([r["fam_mass"] for r in sel])),
                              "cat_mass": float(np.mean([r["cat_mass"] for r in sel]))}
        print(f"[d8p] === {cell}: fam {out['cells'][cell]['fam_mass']:.3f} "
              f"cat {out['cells'][cell]['cat_mass']:.3f}", flush=True)
    json.dump(out, open(os.path.join(RUN_DIR, "dose8.json"), "w"), indent=1)
    print("[d8p] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
