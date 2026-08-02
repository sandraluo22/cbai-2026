"""Embeddings for the chameleon vocab (pod, CPU-only, ~1 min for the 7B; the 32B
loads to CPU RAM). Mean-pooled input-embedding rows (" word" form), same recipe as
src/qwen32_word_embed.py. Includes every bank/clue word (stimuli/clue_vocab.txt) plus
any open-vocab clues the live agent generated in battery files.

Env: MODEL(QwenInst32) VOCAB(runs/chameleon/stimuli/clue_vocab.txt)
     BATTERY_GLOB(runs/chameleon/battery/battery_*.jsonl)
     OUT_NPZ(runs/chameleon/battery/chameleon_word_embed.npz)
"""
from __future__ import annotations
import os
import sys
import glob
import json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
import llm_agents as LA  # noqa: E402

MODEL = os.environ.get("MODEL", "QwenInst32")
VOCAB = os.environ.get("VOCAB", "runs/chameleon/stimuli/clue_vocab.txt")
BATTERY_GLOB = os.environ.get("BATTERY_GLOB", "runs/chameleon/battery/battery_*.jsonl")
OUT_NPZ = os.environ.get("OUT_NPZ", "runs/chameleon/battery/chameleon_word_embed.npz")


def main():
    words = {w.strip().lower() for w in open(VOCAB) if w.strip()}
    for f in glob.glob(BATTERY_GLOB):
        for line in open(f):
            words |= {c.lower() for c in json.loads(line).get("agent_clues", [])}
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
