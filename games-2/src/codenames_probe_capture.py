"""Capture the GUESSER's residual-stream activations, turn by turn, in the
memoryless and memory open-clue Codenames runs -- for the probe that asks:

    can the guesser tell whether the spymaster is STATIC (memoryless, repeats
    clues) vs ADAPTABLE (memory, never repeats)?

Games are PAIRED by seed across modes. For each (mode, role-ordering, game, round)
we store the guesser's last-token hidden state at the guess-time forward pass (the
guesser reasoning about the board given all clues so far), across a set of evenly
spaced layers. The probe (codenames_probe.py) then fits a per-turn linear model on
these activations to predict the mode; R^2 rising with turn = yes, it can tell.

NOTE: at round 1 the memory and memoryless spymasters give the IDENTICAL clue
(memory's clue-history is empty), so round-1 activations coincide across modes ->
a built-in R^2 ~ 0 null floor.

Env: MODELS(LlamaInst,QwenInst) GAMES(100) ROUNDS(8) M(4) NLAYERS(8) DEVICE RUN_DIR
Out: <RUN_DIR>/probe_<guesser>.npz  (acts[N,Lkeep,H] fp16 + mode/pair/game/round/layers)
"""
from __future__ import annotations

import os
import itertools

import numpy as np

import llm_agents as LA

MODELS = os.environ.get("MODELS", "LlamaInst,QwenInst").split(",")
GAMES = int(os.environ.get("GAMES", "100"))
CAP = int(os.environ.get("ROUNDS", "8"))
M = int(os.environ.get("M", "4"))
N = len(LA.OPEN_BOARD)
NLAYERS = int(os.environ.get("NLAYERS", "8"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/codenames/probe")

# mode label -> spymaster config.  0=memoryless (repeats, adaptive), 1=memory (diverse,
# adaptive), 2=diverse-but-NON-adaptive (ignores the guesser's found-set).
MODE_CFG = {0: dict(remember=False, adaptive=True),
            1: dict(remember=True, adaptive=True),
            2: dict(remember=True, adaptive=False)}
MODES = [int(x) for x in os.environ.get("PROBE_MODES", "0,1,2").split(",")]


def keep_layers(total, k):
    return sorted(set(np.linspace(0, total - 1, k).round().astype(int).tolist()))


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    words = LA.OPEN_BOARD
    loaded = {m: LA.load(m, dev) for m in set(MODELS)}

    @torch.no_grad()
    def guess_hidden(G):
        ids = G.tok(G._prompt(), return_tensors="pt").input_ids.to(G.dev)
        hs = G.m(ids, output_hidden_states=True).hidden_states     # tuple (L+1) of (1,seq,H)
        return np.stack([h[0, -1].float().cpu().numpy() for h in hs])   # (L+1, H)

    store = {}      # guesser tag -> dict of lists
    for mem in MODES:
        for A, B in itertools.permutations(MODELS, 2):             # A=spymaster, B=guesser
            mS, tS = loaded[A]; mG, tG = loaded[B]
            st = store.setdefault(B, {"acts": [], "mode": [], "pair": [], "game": [],
                                      "round": [], "layers": None})
            for gi in range(GAMES):
                rng = np.random.default_rng(gi)
                targets = sorted(rng.choice(N, M, replace=False).tolist())
                S = LA.LLMSpeakerOpen(mS, tS, words, targets, dev, **MODE_CFG[mem])
                G = LA.LLMListenerOpen(mG, tG, words, dev)
                for r in range(CAP):
                    if not S.remaining:
                        break
                    real, swap, count = S.clue(G)
                    G.update(real, count)
                    h = guess_hidden(G)                            # (L+1, H)
                    if st["layers"] is None:
                        st["layers"] = keep_layers(h.shape[0], NLAYERS)
                    st["acts"].append(h[st["layers"]].astype(np.float16))
                    st["mode"].append(mem); st["pair"].append(f"{A}->{B}")
                    st["game"].append(gi); st["round"].append(r + 1)
                    for g in G.pick_guesses(count):
                        ok = g in S.remaining
                        G.observe(g, ok); S.observe(g, ok)
                    S.note_clue(real)
            print(f"[probe] mem={mem} {A}->{B}: {len(st['acts'])} samples for guesser {B}", flush=True)

    for gtag, st in store.items():
        acts = np.stack(st["acts"])                                # (Ntot, Lkeep, H) fp16
        path = os.path.join(RUN_DIR, f"probe_{gtag}.npz")
        np.savez_compressed(path, acts=acts, mode=np.array(st["mode"], dtype=np.int8),
                            pair=np.array(st["pair"]), game=np.array(st["game"], dtype=np.int16),
                            round=np.array(st["round"], dtype=np.int8),
                            layers=np.array(st["layers"], dtype=np.int16))
        print(f"[probe] wrote {path}  acts={acts.shape} layers={st['layers']}", flush=True)


if __name__ == "__main__":
    main()
