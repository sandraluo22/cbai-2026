"""LATENT-CONSTRAINT PROBE, capture stage (GPU): does the UNRESTRICTED player (A)
internally represent the partner's secret constraint ("B only says cities/fruits")
before A's behaviour drifts into the category?

Replays the recorded game1_yoked runs (reactive / restrict-city / restrict-fruit) and
captures A's residual stream at the answer position, every layer, every turn — A's
prompt is identical in form across conditions (A is never told anything), so any
decodable difference is inferred from the partner's words. Probing happens locally
(`qwen32_constraint_probe_fit.py`) on the saved npz.

Env: MODEL(QwenInst32) SRC_DIR(runs/game-1/qwen32/qwen32_variations) START_FILE
     CONDS(reactive,restrict-city,restrict-fruit) MAXTURN(24) OUT_NPZ DEVICE
Out: OUT_NPZ with acts (N, nLayers+1, H) float16 + cond/game/turn arrays
"""
from __future__ import annotations
import os
import json
import numpy as np
import llm_agents as LA
import game1_yoked as Y

MODEL = os.environ.get("MODEL", "QwenInst32")
SRC_DIR = os.environ.get("SRC_DIR", "runs/game-1/qwen32/qwen32_variations")
START_FILE = os.environ.get("START_FILE", Y.START_FILE)
CONDS = os.environ.get("CONDS", "reactive,restrict-city,restrict-fruit").split(",")
MAXTURN = int(os.environ.get("MAXTURN", "24"))
OUT_NPZ = os.environ.get("OUT_NPZ", "runs/qwen32_constraint_probe_acts.npz")


def load_starts():
    pairs = []
    for line in open(START_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split("\t") if "\t" in line else line.split()
            pairs.append((p[-2], p[-1]))
    return pairs


def load_games(cond):
    games = {}
    for line in open(os.path.join(SRC_DIR, f"game1_yoked_{cond}_transcript.jsonl")):
        r = json.loads(line)
        games.setdefault(r["rollout"], []).append(r)
    for g in games:
        games[g].sort(key=lambda r: r["turn"])
    return games


def main():
    import torch
    dev = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    model, tok = LA.load(MODEL, dev)
    starts = load_starts()

    @torch.no_grad()
    def hidden(prompt):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        hs = model(ids, output_hidden_states=True).hidden_states
        return np.stack([h[0, -1].float().cpu().numpy() for h in hs]).astype(np.float16)

    acts, conds, games_ix, turns_ix = [], [], [], []
    for cond in CONDS:
        games = load_games(cond)
        for roll, recs in sorted(games.items()):
            sa, sb = starts[roll]
            histA = [(sb, sa)]
            used = {sa, sb}
            for r in recs[:MAXTURN]:
                # A's prompt: cond-blind (A never sees the restriction; used-list shown as in the run)
                acts.append(hidden(Y.build_prompt(tok, histA, used)))
                conds.append(cond); games_ix.append(roll); turns_ix.append(r["turn"])
                histA.append((r["B"], r["A"]))
                used |= {r["A"], r["B"]}
            print(f"[probe-cap] {cond} game {roll}: {min(len(recs), MAXTURN)} turns", flush=True)
    np.savez_compressed(OUT_NPZ, acts=np.stack(acts), cond=np.array(conds),
                        game=np.array(games_ix), turn=np.array(turns_ix))
    print(f"[probe-cap] wrote {len(acts)} pts -> {OUT_NPZ}", flush=True)


if __name__ == "__main__":
    main()
