"""Counterfactual removal replay: for an existing gossip transcript, re-read each
neutral listener's end-of-round belief with ONE source's memory entries deleted.

Reconstructs every agent's memory from the step log, then at each round's end
(before that round's reveal) reads the belief three ways: full memory, minus P1's
entries, minus P2's entries. The drop in p(correct) (or p(clue)) under removal is
the causal contribution of that source to the listener's belief — measured on the
SAME conversation history, no re-simulation.

usage: python cf_removal.py <transcript.jsonl> [more transcripts...]
env: MODEL (Qwen32)
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import qsg_gossip as G  # noqa: E402


def main():
    model, tok, _ = G.load(os.environ.get("MODEL", "Qwen32"))
    rng = random.Random(0)
    for path in sys.argv[1:]:
        lines = [json.loads(l) for l in open(path)]
        meta = lines[0]
        n = meta["n"]
        starts = {l["round"]: l for l in lines if l["type"] == "round_start"}
        mems = {i: [] for i in range(n)}
        reveals = {}
        out = []
        cur_round = 0
        def flush_round(r):
            st = starts[r]
            labels = st.get("labels", meta["labels"])
            ids = [tok(l, add_special_tokens=False)["input_ids"][0] for l in labels]
            ci = labels.index(st["correct"])
            cm = {int(k) for k in st.get("clue_map", {})}
            rec = dict(round=r, correct=st["correct"])
            for mode, tag in ((None, "full"), (1, "noP1"), (2, "noP2")):
                ps = []
                for i in range(n):
                    if (i + 1) in cm:
                        continue                       # neutrals only
                    mv = [e for e in mems[i] if mode is None or e[1] != mode]
                    p = G.belief(model, tok, G.user_msg(i, labels, mv, reveals, r,
                                                        None, rng, meta.get("names", False)),
                                 ids)
                    ps.append(float(p[ci]))
                rec[tag] = round(float(np.mean(ps)), 4)
            out.append(rec)
            reveals[r] = st["correct"]
        for l in lines:
            if l["type"] == "step":
                if l["round"] != cur_round:
                    if cur_round:
                        flush_round(cur_round)
                    cur_round = l["round"]
                mems[l["L"] - 1].append((l["round"], l["S"], l["s_label"]))
        if cur_round:
            flush_round(cur_round)
        stem = path.replace("_transcript.jsonl", "_cfremoval.json")
        json.dump(out, open(stem, "w"), indent=1)
        f = np.mean([o["full"] for o in out]); a = np.mean([o["noP1"] for o in out])
        b = np.mean([o["noP2"] for o in out])
        print(f"[{os.path.dirname(path).split('/')[-1]}] mean neutral p(correct): "
              f"full {f:.3f} | without P1 {a:.3f} (d={a-f:+.3f}) | "
              f"without P2 {b:.3f} (d={b-f:+.3f})", flush=True)


if __name__ == "__main__":
    main()
