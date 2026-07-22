"""Capture the GUESSER (B)'s residual stream + four decoding targets, to probe what B
represents about the spymaster (A) and itself.

For each (A-policy, role-ordering, game, round) we store B's last-token hidden state at
each of several layers, plus labels:
  target_multihot : A's hidden target set (12-dim 0/1)      -> does B represent what A meant?
  policy          : which A-variant is being played (0/1/2) -> H5: does B represent A's type?
  belief_entropy  : entropy of B's guess distribution       -> is B's own uncertainty explicit?
  adaptivity      : KL(A_clue|B-state || A_clue|naive)       -> "how well A understands me" (level-2)

A-policies: 0 memoryless (repeats, adaptive) | 1 memory (diverse, adaptive) | 2 non-adaptive.

Env: MODELS(LlamaInst,QwenInst) GAMES(80) ROUNDS(8) M(4) NLAYERS(8) TOPN(50) DEVICE RUN_DIR
Out: <RUN_DIR>/probe4_<guesser>.npz
"""
from __future__ import annotations

import os
import itertools

import numpy as np

import core as K
import llm_agents as LA

MODELS = os.environ.get("MODELS", "LlamaInst,QwenInst").split(",")
GAMES = int(os.environ.get("GAMES", "80"))
CAP = int(os.environ.get("ROUNDS", "8"))
M = int(os.environ.get("M", "4"))
N = len(LA.OPEN_BOARD)
NLAYERS = int(os.environ.get("NLAYERS", "8"))
TOPN = int(os.environ.get("TOPN", "50"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/codenames/probe4")
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

    def topn_kl(lp, lq, n=TOPN):
        p = torch.softmax(lp, 0); q = torch.softmax(lq, 0)
        n = min(n, p.shape[-1])
        idx = torch.unique(torch.cat([torch.topk(p, n).indices, torch.topk(q, n).indices]))
        pp, qq = p[idx], q[idx]
        P = torch.cat([pp, (1 - pp.sum()).clamp(min=1e-9).view(1)]); P = P / P.sum()
        Q = torch.cat([qq, (1 - qq.sum()).clamp(min=1e-9).view(1)]); Q = Q / Q.sum()
        return float((P * (P.clamp(min=1e-12).log() - Q.clamp(min=1e-12).log())).sum())

    @torch.no_grad()
    def guess_hidden(G):
        ids = G.tok(G._prompt("\nMy two guesses:\n1)"), return_tensors="pt").input_ids.to(G.dev)
        hs = G.m(ids, output_hidden_states=True).hidden_states
        return np.stack([h[0, -1].float().cpu().numpy() for h in hs])   # (L+1, H)

    store = {}
    for mem in MODES:
        for A, B in itertools.permutations(MODELS, 2):
            mS, tS = loaded[A]; mG, tG = loaded[B]
            st = store.setdefault(B, {"acts": [], "target": [], "policy": [], "entropy": [],
                                      "adaptivity": [], "game": [], "round": [], "layers": None})
            for gi in range(GAMES):
                rng = np.random.default_rng(gi)
                targets = sorted(rng.choice(N, M, replace=False).tolist())
                tvec = np.zeros(N, dtype=np.int8); tvec[targets] = 1
                S = LA.LLMSpeakerOpen(mS, tS, words, targets, dev, **MODE_CFG[mem])
                G = LA.LLMListenerOpen(mG, tG, words, dev)
                for r in range(CAP):
                    if not S.remaining:
                        break
                    naive = LA.LLMListenerOpen(mG, tG, words, dev)
                    adapt = topn_kl(S.clue_logits(G), S.clue_logits(naive))   # A's ToM of B
                    real, swap, count = S.clue(G)
                    G.update(real, count)
                    h = guess_hidden(G)
                    if st["layers"] is None:
                        st["layers"] = keep_layers(h.shape[0], NLAYERS)
                    st["acts"].append(h[st["layers"]].astype(np.float16))
                    st["target"].append(tvec.copy()); st["policy"].append(mem)
                    st["entropy"].append(float(K.entropy(G.guess_dist())))
                    st["adaptivity"].append(adapt); st["game"].append(gi); st["round"].append(r + 1)
                    for g in G.pick_guesses(count):
                        ok = g in S.remaining
                        G.observe(g, ok); S.observe(g, ok)
                    S.note_clue(real)
            print(f"[probe4] policy={mem} {A}->{B}: {len(st['acts'])} samples for guesser {B}", flush=True)

    for gtag, st in store.items():
        acts = np.stack(st["acts"])
        path = os.path.join(RUN_DIR, f"probe4_{gtag}.npz")
        np.savez_compressed(path, acts=acts, target=np.stack(st["target"]),
                            policy=np.array(st["policy"], dtype=np.int8),
                            entropy=np.array(st["entropy"], dtype=np.float32),
                            adaptivity=np.array(st["adaptivity"], dtype=np.float32),
                            game=np.array(st["game"], dtype=np.int16),
                            round=np.array(st["round"], dtype=np.int8),
                            layers=np.array(st["layers"], dtype=np.int16))
        print(f"[probe4] wrote {path}  acts={acts.shape} layers={st['layers']}", flush=True)


if __name__ == "__main__":
    main()
