"""Metadata + behavioral replay of the probe games (3 spymaster modes), to interpret
the probe direction AND test whether the guesser ACTS on the spymaster's regime.

Re-plays the same games as codenames_probe_capture.py (same seeds, deterministic) and
logs, per (mode, game, round), observable features + behavioral read-outs. Rows join to
the captured activations on (mode, game, round) within each guesser file.

Modes:  0 memoryless (repeats, adaptive) | 1 memory (diverse, adaptive)
        2 diverse-but-NON-adaptive (ignores the guesser's found-set)

Logged: is_repeat, repeat_count, n_distinct, distinct_ratio, found/wrong/remaining,
belief_entropy (guesser uncertainty after the clue), target_mass (recovery), n_correct,
top_guess, and MiniLM sims: clue<->top_guess, and clue<->its most-associated board word
plus whether THAT word is already found (the non-adaptive spymaster clues for found
words -> a behavioral hook).

Env: MODELS(LlamaInst,QwenInst) GAMES(100) ROUNDS(8) M(4) DEVICE RUN_DIR PROBE_MODES(0,1,2)
Out: <RUN_DIR>/meta_<guesser>.jsonl
"""
from __future__ import annotations

import os
import json
import itertools

import numpy as np

import core as K
import llm_agents as LA

MODELS = os.environ.get("MODELS", "LlamaInst,QwenInst").split(",")
GAMES = int(os.environ.get("GAMES", "100"))
CAP = int(os.environ.get("ROUNDS", "8"))
M = int(os.environ.get("M", "4"))
N = len(LA.OPEN_BOARD)
RUN_DIR = os.environ.get("RUN_DIR", "runs/codenames/probe")
MODE_CFG = {0: dict(remember=False, adaptive=True),
            1: dict(remember=True, adaptive=True),
            2: dict(remember=True, adaptive=False)}
MODES = [int(x) for x in os.environ.get("PROBE_MODES", "0,1,2").split(",")]


def main():
    import torch
    from transformers import AutoModel, AutoTokenizer
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    words = LA.OPEN_BOARD
    loaded = {m: LA.load(m, dev) for m in set(MODELS)}

    etok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    emod = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(dev).eval()
    _ecache = {}

    def emb(w):
        if w not in _ecache:
            enc = etok([w or "."], return_tensors="pt", truncation=True).to(dev)
            with torch.no_grad():
                h = emod(**enc).last_hidden_state
            mm = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * mm).sum(1) / mm.sum(1).clamp(min=1e-9)
            _ecache[w] = torch.nn.functional.normalize(v, dim=1)[0].cpu().numpy()
        return _ecache[w]

    board_emb = np.stack([emb(w) for w in words])              # (N, D)

    files = {}
    for mem in MODES:
        for A, B in itertools.permutations(MODELS, 2):         # A=spymaster, B=guesser
            mS, tS = loaded[A]; mG, tG = loaded[B]
            if B not in files:                                 # open once per guesser
                files[B] = open(os.path.join(RUN_DIR, f"meta_{B}.jsonl"), "w")
            f = files[B]
            for gi in range(GAMES):
                rng = np.random.default_rng(gi)
                targets = sorted(rng.choice(N, M, replace=False).tolist())
                S = LA.LLMSpeakerOpen(mS, tS, words, targets, dev, **MODE_CFG[mem])
                G = LA.LLMListenerOpen(mG, tG, words, dev)
                clues_seen = []
                for r in range(CAP):
                    if not S.remaining:
                        break
                    real, swap, count = S.clue(G)
                    repeat_count = clues_seen.count(real)
                    is_repeat = repeat_count > 0
                    clues_seen.append(real)
                    n_distinct = len(set(clues_seen))
                    found_before = len(G.known)
                    ce = emb(real)
                    board_sim = board_emb @ ce                 # (N,)
                    top_board = int(np.argmax(board_sim))
                    G.update(real, count)
                    gd = G.guess_dist()
                    guesses = G.pick_guesses(count)
                    tg = guesses[0]
                    rec = {
                        "mode": mem, "pair": f"{A}->{B}", "game": gi, "round": r + 1,
                        "clue": real, "count": count, "is_repeat": int(is_repeat),
                        "repeat_count": repeat_count, "n_distinct": n_distinct,
                        "distinct_ratio": n_distinct / (r + 1),
                        "found_before": found_before, "wrong_before": len(G.dead),
                        "remaining_before": len(S.remaining),
                        "belief_entropy": float(K.entropy(gd)),
                        "target_mass": float(K.target_mass(G.belief(), targets)),
                        "top_guess": words[tg], "clue_guess_sim": float(ce @ emb(words[tg])),
                        "clue_topboard": words[top_board],
                        "clue_topboard_found": int(top_board in G.known),
                        "clue_topboard_is_target": int(top_board in targets),
                    }
                    results = []
                    for g in guesses:
                        ok = g in S.remaining
                        results.append(int(ok)); G.observe(g, ok); S.observe(g, ok)
                    rec["n_correct"] = int(sum(results))
                    f.write(json.dumps(rec) + "\n")
                    S.note_clue(real)
            print(f"[meta] mode={mem} {A}->{B} logged for guesser {B}", flush=True)
    for f in files.values():
        f.close()
    print(f"[meta] DONE -> {RUN_DIR}/meta_*.jsonl", flush=True)


if __name__ == "__main__":
    main()
