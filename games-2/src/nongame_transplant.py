"""NON-GAME TRANSPLANT (2026-08-05): does family capture need the game at all?

The planted triple is transplanted into ordinary continuation frames with NO game
rules — no win condition, no no-repeat constraint, no players-as-agents framing.
If family mass survives, the no-repeat constraint / game objective is not the
mechanism; if it dies, capture is game-bound.

Frames (each x {planted, base}, interleaver words from real strict-city B streams):
  list    "Here is a list of words from someone's notebook: w1, w2, ... ."
          cue "Next word:"
  prose   "While cleaning out a desk, Maya found a scrap of paper on which
          someone had written the words w1, w2, ... . Underneath, the same
          handwriting continued with one more word." cue "That word was:"
  dialog  two named speakers alternate bare words (no game framing):
          "Sam: tokyo / Alex: planted / ..." cue "Alex:"

Measures per cell (K=64 samples): novel family mass (family word not shown in
the frame), any-family mass (incl. verbatim repeats of the planted words —
meaningful here because nothing forbids repetition), repeat mass (any shown
word), category mass.

Env: MODEL(QwenInst32) SRC_DIR(runs/qwen32_strict) START_FILE N_STREAMS(6) K(64)
     TEMP(0.7) RUN_DIR(runs/nongame_transplant)
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
RUN_DIR = os.environ.get("RUN_DIR", "runs/nongame_transplant")

PLANT = ["planted", "planting", "plantings"]
FAMP = "plan"
FILL = ["window", "carpet", "stapler"]


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)
    catset = set(CATWORDS["city"])

    @torch.no_grad()
    def sample_k(body, cue):
        ids = tok(LA._render(tok, body) + "\n" + cue, return_tensors="pt").input_ids.to(dev)
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

    def interleave(Bwords, awords):
        seq = []
        for i in range(3):
            seq += [Bwords[i], awords[i]]
        return seq

    def frame_body(frame, seq):
        if frame == "list":
            return ("Here is a list of words from someone's notebook: "
                    + ", ".join(seq) + "."), "Next word:"
        if frame == "prose":
            return ("While cleaning out a desk, Maya found a scrap of paper on which "
                    "someone had written the words " + ", ".join(seq)
                    + ". Underneath, the same handwriting continued with one more "
                    "word."), "That word was:"
        lines = []
        for i, w in enumerate(seq):
            lines.append(("Sam: " if i % 2 == 0 else "Alex: ") + w)
        return "\n".join(lines), "Alex:"

    results = []
    for si, ts in enumerate(streams):
        Bwords = [t["B"] for t in ts][:3]
        for frame in ("list", "prose", "dialog"):
            for content in ("planted", "base"):
                aw = PLANT if content == "planted" else FILL
                seq = interleave(Bwords, aw)
                shown = set(seq)
                body, cue = frame_body(frame, seq)
                ws = sample_k(body, cue)
                famn = np.mean([1 if (w and w not in shown and len(w) > 3 and w[:4] == FAMP)
                                else 0 for w in ws])
                fama = np.mean([1 if (w and len(w) > 3 and w[:4] == FAMP) else 0 for w in ws])
                rep = np.mean([1 if (w and w in shown) else 0 for w in ws])
                cat = np.mean([1 if (w and w not in shown and w in catset) else 0 for w in ws])
                results.append({"frame": frame, "content": content, "stream": si,
                                "fam_novel": float(famn), "fam_any": float(fama),
                                "repeat": float(rep), "cat_mass": float(cat),
                                "top_words": sorted(collections.Counter(ws).items(),
                                                    key=lambda kv: -kv[1])[:5]})
                json.dump({"per_cell": results},
                          open(os.path.join(RUN_DIR, "nongame.json"), "w"))
        print(f"[ng] stream {si} done", flush=True)

    out = {"per_cell": results, "cells": {}}
    for frame in ("list", "prose", "dialog"):
        for content in ("planted", "base"):
            sel = [r for r in results if r["frame"] == frame and r["content"] == content]
            out["cells"][f"{frame}_{content}"] = {k: float(np.mean([r[k] for r in sel]))
                for k in ("fam_novel", "fam_any", "repeat", "cat_mass")}
        p = out["cells"][f"{frame}_planted"]; b = out["cells"][f"{frame}_base"]
        print(f"[ng] === {frame}: fam_novel {p['fam_novel']:.3f}/{b['fam_novel']:.3f} "
              f"fam_any {p['fam_any']:.3f}/{b['fam_any']:.3f} repeat {p['repeat']:.3f}",
              flush=True)
    json.dump(out, open(os.path.join(RUN_DIR, "nongame.json"), "w"), indent=1)
    print("[ng] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
