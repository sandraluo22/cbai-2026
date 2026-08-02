"""Source-identity counterfactual replay: hold message and history fixed, swap the
speaker attribution between the two towers (P1 <-> P2), re-read the listener.

For each recorded conversation where the speaker was a tower, rebuild the
listener's exact memory, then read its belief with the new entry attributed to
the OTHER tower. The actual p_after is already recorded, so
    identity_effect = dm_actual - dm_counterfactual
isolates history-conditioned reputation from message content exactly.

usage: python replay_identity.py <transcript.jsonl> [...]   env: MODEL, STRIDE (3)
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
    stride = int(os.environ.get("STRIDE", "3"))
    rng = random.Random(0)
    for path in sys.argv[1:]:
        lines = [json.loads(l) for l in open(path)]
        meta = lines[0]
        starts = {l["round"]: l for l in lines if l["type"] == "round_start"}
        mems = {i: [] for i in range(meta["n"])}
        reveals = {}
        out, k = [], 0
        cur = 0
        for l in lines:
            if l["type"] == "probe":
                reveals[l["round"]] = starts[l["round"]]["correct"]
            if l["type"] != "step":
                continue
            r, S, L = l["round"], l["S"], l["L"]
            if S in (1, 2):
                k += 1
                if k % stride == 0:
                    labels = starts[r].get("labels", meta["labels"])
                    ids = [tok(x, add_special_tokens=False)["input_ids"][0] for x in labels]
                    other = 2 if S == 1 else 1
                    mv = mems[L - 1] + [(r, other, l["s_label"])]
                    pcf = G.belief(model, tok, G.user_msg(L - 1, labels, mv, reveals, r,
                                                          None, rng, meta.get("names", False)),
                                   ids)
                    j = labels.index(l["s_label"])
                    dm_act = l["p_after"][j] - l["p_before"][j]
                    dm_cf = float(pcf[j]) - l["p_before"][j]
                    out.append(dict(round=r, S=S, L=L, label=l["s_label"],
                                    dm_actual=round(dm_act, 4), dm_swapped=round(dm_cf, 4)))
            mems[L - 1].append((r, S, l["s_label"]))
        stem = path.replace("_transcript.jsonl", "_identitycf.json")
        json.dump(out, open(stem, "w"), indent=1)
        for who, tag in ((1, "truth-teller msg -> attributed to liar"),
                        (2, "liar msg -> attributed to truth-teller")):
            v = [(o["dm_actual"], o["dm_swapped"]) for o in out if o["S"] == who]
            if v:
                a, c = np.mean([x[0] for x in v]), np.mean([x[1] for x in v])
                print(f"[{os.path.dirname(path).split('/')[-1]}] {tag}: "
                      f"dm actual {a:+.3f} -> swapped {c:+.3f} (identity effect {a - c:+.3f}, "
                      f"n={len(v)})", flush=True)


if __name__ == "__main__":
    main()
