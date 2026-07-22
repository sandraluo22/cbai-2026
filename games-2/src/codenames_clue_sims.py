"""Compute MiniLM cosine similarity between every clue word appearing in the open-clue
Codenames transcripts and the 12 board words -- the association matrix A[c,i] for the
RSA / conditional-logit fit (codenames_rsa_fit.py).

Env: RUN_ROOT(runs/codenames) SIM_DIRS(comma list) OUT
Out: <OUT> json {"board": [...], "sims": {clue: {board_word: cos}}}
"""
from __future__ import annotations

import os
import json
import glob

import numpy as np

import llm_agents as LA

RUN_ROOT = os.environ.get("RUN_ROOT", "runs/codenames")
DIRS = os.environ.get("SIM_DIRS",
                      "llm_codenames_open,llm_codenames_open_mem,spy_remaining,spy_eliminated,spy_inferred").split(",")
OUT = os.environ.get("OUT", "runs/codenames/clue_sims.json")


def main():
    import torch
    from transformers import AutoModel, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    words = LA.OPEN_BOARD
    etok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    emod = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(dev).eval()

    def emb(ws):
        enc = etok(ws, return_tensors="pt", padding=True, truncation=True).to(dev)
        with torch.no_grad():
            h = emod(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(v, dim=1).cpu().numpy()

    board_emb = emb(words)                                     # (12, D)

    clues = set()
    for d in DIRS:
        for f in glob.glob(os.path.join(RUN_ROOT, d, "*_transcript.jsonl")):
            for l in open(f):
                if not l.strip():
                    continue
                r = json.loads(l)
                clues.add(r.get("clue", ""))
                if "coupling" in r:
                    clues.add(r["coupling"].get("clue_swapped_to", ""))
                ad = r.get("adaptivity", {})
                for key in ("clue_dist_real", "clue_dist_naive"):
                    if key in ad:
                        clues.update(ad[key].keys())
    clues = sorted(c for c in clues if c)

    sims = {}
    B = 256
    for i in range(0, len(clues), B):
        chunk = clues[i:i + B]
        s = emb(chunk) @ board_emb.T                          # (chunk, 12)
        for w, row in zip(chunk, s):
            sims[w] = {words[j]: round(float(row[j]), 4) for j in range(len(words))}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"board": words, "sims": sims}, open(OUT, "w"))
    print(f"[sims] wrote {OUT}: {len(sims)} clues x {len(words)} board words", flush=True)


if __name__ == "__main__":
    main()
