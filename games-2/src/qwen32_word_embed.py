"""Export an embedding for every word played in the Game-1 transcripts (pod, ~1 min):
mean-pooled INPUT-embedding rows of the word's tokens ("_word" form). Used locally by
game1_urn_deffuant.py to measure semantic distances / assimilation moves.

Env: MODEL(QwenInst32) SRC_DIR(runs/game-1/qwen32/qwen32_variations)
     OUT_NPZ(runs/qwen32_word_embed.npz)
"""
from __future__ import annotations
import os
import glob
import json
import numpy as np
import llm_agents as LA

MODEL = os.environ.get("MODEL", "QwenInst32")
SRC_DIR = os.environ.get("SRC_DIR", "runs/game-1/qwen32/qwen32_variations")
OUT_NPZ = os.environ.get("OUT_NPZ", "runs/qwen32_word_embed.npz")


def main():
    import torch
    words = set()
    for f in glob.glob(os.path.join(SRC_DIR, "*_transcript.jsonl")):
        for line in open(f):
            d = json.loads(line)
            for k in ("A", "B"):
                if d.get(k):
                    words.add(d[k].lower())
    words = sorted(words)
    model, tok = LA.load(MODEL, "cpu")          # embeddings only; CPU fine
    E = model.get_input_embeddings().weight.detach().float()
    vecs = []
    for w in words:
        ids = tok(" " + w, add_special_tokens=False)["input_ids"]
        vecs.append(E[ids].mean(0).numpy())
    np.savez_compressed(OUT_NPZ, words=np.array(words), vecs=np.stack(vecs))
    print(f"[embed] {len(words)} words -> {OUT_NPZ}")


if __name__ == "__main__":
    main()
