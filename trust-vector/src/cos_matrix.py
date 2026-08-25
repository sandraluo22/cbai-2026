"""Cosine matrix among the directions Sandra asked for (2026-08-14):
fitted, optim, story_comb, story_trust, story_trust@acctnb, story_trust@storynb.
CPU only; reads vectors2.npz + fit2.json + newvecs.json through dirs.load_all.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from dirs import load_all  # noqa: E402

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
NAMES = ["FITTED trust", "optim", "story_comb", "story_trust",
         "story_warmth", "story_comp", "story_trust@acctnb", "story_trust@storynb"]

import json  # noqa: E402

dump = {}
for L in (45, 52):
    D = load_all(OUT, L)
    have = [n for n in NAMES if n in D]
    w = max(len(n) for n in have)
    print(f"\n=== L{L} ===")
    print(" " * (w + 2) + "  ".join(f"{n[:12]:>12}" for n in have))
    for a in have:
        row = "  ".join(f"{float(D[a] @ D[b]):>12.3f}" for b in have)
        print(f"{a:<{w}}  {row}")
    dump[f"L{L}"] = {"names": have,
                     "M": [[float(D[a] @ D[b]) for b in have] for a in have]}
json.dump(dump, open(os.path.join(OUT, "cos_matrix.json"), "w"), indent=1)
print("COS_DONE", flush=True)
