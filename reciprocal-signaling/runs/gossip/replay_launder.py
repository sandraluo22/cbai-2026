"""Laundering causal proof: same memory CONTENT, intervened provenance structure.

At each round of a misinformed_all transcript, take each neutral listener's
current-round memory and read its belief in the liar's label under:
  actual        the recorded provenance (P1 originals + whatever echoes occurred)
  concentrated  every wrong-label entry re-attributed to P1 (no echo cover)
  laundered     every wrong-label entry re-attributed to neutrals, P1 absent
  tagged        echoes keep provenance: "P3 (repeating P1): <label>"
If belief(laundered) > belief(concentrated), echo-diffusion causally strengthens
the wrong label; the 'tagged' arm tests whether preserved provenance restores
source accountability.

usage: python replay_launder.py <transcript.jsonl> [...]   env: MODEL
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
        maxr = max(starts)
        for r in sorted(starts):
            st = starts[r]
            labels = st.get("labels", meta["labels"])
            ids = [tok(x, add_special_tokens=False)["input_ids"][0] for x in labels]
            wl = st["clue"]                            # liar's wrong label this round
            for l in lines:
                if l["type"] == "step" and l["round"] == r:
                    mems[l["L"] - 1].append((r, l["S"], l["s_label"]))
            wi = labels.index(wl)
            for i in range(1, n):                      # neutrals
                cur = [e for e in mems[i] if e[0] == r]
                past = [e for e in mems[i] if e[0] < r]
                wrongs = [e for e in cur if e[2] == wl]
                if len(wrongs) < 2:
                    continue
                neutrals = [k for k in range(2, n + 1) if k != i + 1]
                variants = dict(
                    actual=cur,
                    concentrated=[(e[0], 1, e[2]) if e[2] == wl else e for e in cur],
                    laundered=[(e[0], neutrals[j % len(neutrals)], e[2]) if e[2] == wl else e
                               for j, e in enumerate(cur)],
                )
                rec = dict(round=r, agent=i + 1, n_wrong=len(wrongs))
                for tag, cv in variants.items():
                    p = G.belief(model, tok, G.user_msg(i, labels, past + cv, reveals, r,
                                                        None, rng, meta.get("names", False)),
                                 ids)
                    rec[tag] = round(float(p[wi]), 4)
                # tagged: echoes keep provenance via a suffixed label string
                cv = [(e[0], e[1], e[2] + " (repeating P1)") if (e[2] == wl and e[1] != 1) else e
                      for e in cur]
                p = G.belief(model, tok, G.user_msg(i, labels, past + cv, reveals, r,
                                                    None, rng, meta.get("names", False)), ids)
                rec["tagged"] = round(float(p[wi]), 4)
                out.append(rec)
            reveals[r] = st["correct"]
        stem = path.replace("_transcript.jsonl", "_launder.json")
        json.dump(out, open(stem, "w"), indent=1)
        if out:
            f = lambda k: np.mean([o[k] for o in out])
            print(f"[{os.path.dirname(path).split('/')[-1]}] belief in liar's label "
                  f"(n={len(out)}): actual {f('actual'):.3f} | concentrated(P1-only) "
                  f"{f('concentrated'):.3f} | laundered(no P1) {f('laundered'):.3f} | "
                  f"tagged(echoes credit P1) {f('tagged'):.3f}", flush=True)


if __name__ == "__main__":
    main()
