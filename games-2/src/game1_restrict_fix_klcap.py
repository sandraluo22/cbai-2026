"""Cross-player KL slides for the game1_restrict_fix runs (nolist-*/repeatok-*).

Same output as game1_yoked_klcap.py (per-condition <cond>_crossKL.json +
<cond>_crossKL_curve.pdf + <cond>_crossKL_perturn.pdf in <SRC_DIR>/kl/), but replays
each game with the EXACT prompts of game1_restrict_fix.py: mode 'nolist' keeps the
no-repeat rule but hides the used-list; 'repeatok' drops the rule; B is restricted to
the concept in both. Reuses klcap's plotting. PLOT_ONLY=1 re-renders without GPU.

Env: MODEL(QwenInst32) SRC_DIR(runs/game1_restrict_fix) START_FILE
     CONDS(nolist-city,nolist-fruit,repeatok-city,repeatok-fruit) TOPK(15) PLOT_ONLY(0)
"""
from __future__ import annotations
import os
import json

import game1_yoked_klcap as K
import game1_restrict_fix as F

SRC_DIR = os.environ.get("SRC_DIR", "runs/game1_restrict_fix")
CONDS = os.environ.get("CONDS", "nolist-city,nolist-fruit,repeatok-city,repeatok-fruit").split(",")
PLOT_ONLY = os.environ.get("PLOT_ONLY", "0") == "1"


def load_games(cond):
    path = os.path.join(SRC_DIR, f"game1_restrict_fix_{cond}_transcript.jsonl")
    games = {}
    for line in open(path):
        r = json.loads(line)
        games.setdefault(r["rollout"], []).append(r)
    for g in games:
        games[g].sort(key=lambda r: r["turn"])
    return games


def capture(cond, model, tok, dev, starts):
    import torch

    @torch.no_grad()
    def readout(prompt):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        logits = None
        for _ in range(3):
            logits = model(ids).logits[0, -1].float()
            s = tok.decode([int(logits.argmax())]).strip()
            if s and any(c.isalpha() for c in s):
                break
            ids = torch.cat([ids, logits.argmax()[None, None]], dim=1)
        p = torch.softmax(logits, -1)
        v, i = p.topk(K.TOPK)
        top = {(tok.decode([tid]).strip() or "·"): round(pv, 4) for tid, pv in zip(i.tolist(), v.tolist())}
        return top, p

    def kl(p, q):
        return float((p * (p.clamp_min(1e-9).log() - q.clamp_min(1e-9).log())).sum())

    mode, concept = cond.split("-", 1)
    games = load_games(cond)
    captured = {}
    for roll, recs in games.items():
        sa, sb = starts[roll]
        histA = [(sb, sa)]; histB = [(sa, sb)]
        turns = []
        for r in recs:
            wA, wB = r["A"], r["B"]
            topA, pA = readout(F.build_prompt(tok, histA, mode))
            topB, pB = readout(F.build_prompt(tok, histB, mode, restrict=concept))
            turns.append({"turn": r["turn"], "agreed": bool(r["agreed"]), "pickA": wA, "pickB": wB,
                          "topA": topA, "topB": topB, "kl_ab": kl(pA, pB), "kl_ba": kl(pB, pA)})
            histA.append((wB, wA)); histB.append((wA, wB))
        captured[roll] = turns
        print(f"[fixKL]   {cond} game {roll} ({sa}/{sb}): {len(turns)} turns", flush=True)
    return captured


def main():
    out_dir = os.path.join(SRC_DIR, "kl")
    os.makedirs(out_dir, exist_ok=True)
    if PLOT_ONLY:
        for cond in CONDS:
            captured = {int(k): v for k, v in
                        json.load(open(os.path.join(out_dir, f"{cond}_crossKL.json"))).items()}
            K.plot_condition(cond, captured, out_dir)
        return
    import torch
    import llm_agents as LA
    dev = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    model, tok = LA.load(K.MODEL, dev)
    starts = K.load_starts()
    for cond in CONDS:
        captured = capture(cond, model, tok, dev, starts)
        json.dump(captured, open(os.path.join(out_dir, f"{cond}_crossKL.json"), "w"))
        K.plot_condition(cond, captured, out_dir)
    print(f"[fixKL] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
