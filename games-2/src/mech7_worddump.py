"""MECH7 (2026-08-03): word-level dump of the mech5/mech6 cells.

Re-runs the exact mech5 (witness) and mech6 (first-person) prompts, but saves
every sampled word (K per prompt) so behavior can be characterized qualitatively:
does the model say cities when simulating the partner? what fills the non-family
mass? Same construction as mech5/mech6, byte-identical histories per frame.

Env: MODEL(QwenInst32) SRC_DIR START_FILE N_STREAMS(4) K(64) TEMP(0.7)
     RUN_DIR(runs/mech7_worddump)
"""
from __future__ import annotations
import os
import json
import collections
import llm_agents as LA
import qwen32_pca as G

MODEL = os.environ.get("MODEL", "QwenInst32")
SRC_DIR = os.environ.get("SRC_DIR", "runs/qwen32_strict")
START_FILE = os.environ.get("START_FILE", "runs/game-1/qwen32/qwen32_pca_w2v/start_words.txt")
N_STREAMS = int(os.environ.get("N_STREAMS", "4"))
K = int(os.environ.get("K", "64"))
TEMP = float(os.environ.get("TEMP", "0.7"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/mech7_worddump")

PLANT = ["planted", "planting", "plantings"]
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
                for frame in ("witness", "firstperson"):
                    for who in ("P1", "P2"):
                        n = who[1]
                        if frame == "witness":
                            body = ("You are witnessing a word game between two players. " + OPEN3
                                    + hist + " Words already used (may not be repeated): "
                                    + ", ".join(used) + "."
                                    + f" What word do you think Player {n} should say next?")
                            prompt = LA._render(tok, body) + f"\nPlayer {n} should say:"
                        else:
                            body = (OPEN3 + hist + " Words already used (may not be repeated): "
                                    + ", ".join(used) + "."
                                    + f" You are Player {n}. It is time to say your next word.")
                            prompt = LA._render(tok, body) + "\nMy word:"
                        enc = tok(prompt, return_tensors="pt").to(dev)
                        with torch.no_grad():
                            out = model.generate(enc.input_ids.repeat(K, 1), max_new_tokens=6,
                                                 do_sample=True, temperature=TEMP, top_p=0.95,
                                                 pad_token_id=tok.eos_token_id)
                        raws = [tok.decode(out[i, enc.input_ids.shape[1]:], skip_special_tokens=True)
                                for i in range(K)]
                        ws = [G.clean_word(r) for r in raws]
                        results.append({"frame": frame, "owner": owner, "who": who,
                                        "content": content, "stream": roll,
                                        "p1_words": [r[0] for r in rounds],
                                        "p2_words": [r[1] for r in rounds],
                                        "used": used, "words": ws, "raw": raws})
                        json.dump(results, open(os.path.join(RUN_DIR, "mech7.json"), "w"))
        print(f"[mech7] stream {roll} done", flush=True)
    json.dump(results, open(os.path.join(RUN_DIR, "mech7.json"), "w"), indent=1)
    print("[mech7] all done", flush=True)


if __name__ == "__main__":
    main()
