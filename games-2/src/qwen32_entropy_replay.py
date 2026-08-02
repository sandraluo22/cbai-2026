"""Replay the recorded self-play games and log each player's OUTPUT ENTROPY at every
answer position -- to test whether the "convergence direction" is just a confidence /
low-entropy axis (deconfound for the steering result).

For each (rollout, turn, player) in the transcript we rebuild the exact prompt from the
recorded history, run one forward pass, and record the entropy of the temp-scaled
next-token distribution plus the top-token probability. Pairs with the activations in
qwen32_pca_acts.npz (same roll/turn/player index order as meta1/meta2).

Env: MODEL(QwenInst32) TRANSCRIPT START_FILE TEMP(0.7) OUT_JSON DEVICE
"""
from __future__ import annotations
import os
import json
import collections
import numpy as np
import llm_agents as LA
import qwen32_pca as G

MODEL = os.environ.get("MODEL", "QwenInst32")
TRANSCRIPT = os.environ.get("TRANSCRIPT",
                            "runs/game1_qwen32_pca_w2v/qwen32_pca_transcript.jsonl")
TEMP = float(os.environ.get("TEMP", "0.7"))
OUT_JSON = os.environ.get("OUT_JSON", "runs/game1_qwen32_pca_w2v/qwen32_entropy.json")


def main():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)

    games = collections.defaultdict(list)
    for line in open(TRANSCRIPT):
        rec = json.loads(line)
        games[rec["rollout"]].append(rec)

    out = []
    with torch.no_grad():
        for roll, turns in sorted(games.items()):
            sa, sb = turns[0]["start"]
            p1, p2 = sorted(turns[0]["picks"].keys())
            histA, histB = [(sb, sa)], [(sa, sb)]
            used = {sa, sb}
            for rec in turns:
                wA, wB = rec["picks"][p1], rec["picks"][p2]
                for player, prompt in ((p1, G.build_prompt(tok, histA, used)),
                                       (p2, G.build_prompt(tok, histB, used))):
                    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
                    logits = model(ids).logits[0, -1].float() / TEMP
                    p = torch.softmax(logits, -1)
                    ent = float(-(p * torch.log(p + 1e-12)).sum())
                    out.append({"rollout": roll, "turn": rec["turn"], "player": player,
                                "entropy": ent, "top_p": float(p.max())})
                used |= {wA, wB}
                histA.append((wB, wA)); histB.append((wA, wB))
            print(f"[entropy] roll {roll} done ({len(turns)} turns)", flush=True)
    json.dump(out, open(OUT_JSON, "w"))
    print(f"[entropy] wrote {len(out)} points -> {OUT_JSON}")


if __name__ == "__main__":
    main()
