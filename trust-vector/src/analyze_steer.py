"""Paired per-prompt differences with standard errors, from a steer_qsg raw dump.

Run 1 reported arm means only, so a +0.209 next to a random-direction -0.025 had no
spread attached and could not be told apart from noise. Every prompt is evaluated
under every arm in the same order, so the arm-vs-base difference is PAIRED: compute
it per prompt, then take mean and standard error over prompts. Paired is what makes
this sensitive -- the prompt-to-prompt variation in the raw margin is far larger than
any steering effect and cancels in the difference.

The number to read is the mean difference against its own standard error, and against
the random-direction arm computed the same way. `t` here is mean/SE, reported as a
rough magnitude only: prompts are not independent (5 games x 2 schedules x 2 styles x
2 orderings are heavily structured), so treat |t| as descriptive, not as a p-value.

  ANCHOR=name2 python src/analyze_steer.py            # reads out/steer4_<anchor>.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))


def paired(cell, base, scheds):
    d = []
    for s in scheds:
        a = cell["raw"].get(s)
        b = base["raw"].get(s)
        if a and b and len(a) == len(b):
            d += [x - y for x, y in zip(a, b)]
    if not d:
        return None
    d = np.array(d)
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    return dict(mean=float(d.mean()), se=float(se), n=len(d),
                t=float(d.mean() / se) if se and se > 0 else float("nan"))


def main():
    f = os.environ.get("FILE")
    if not f:
        anchor = os.environ.get("ANCHOR", "name2")
        f = os.path.join(OUT, f"steer4_{anchor}.json")
    d = json.load(open(f))
    scheds = d["config"]["scheds"]
    layer = os.environ.get("LAYER")
    print(f"[file] {os.path.basename(f)}   anchor={d['config']['anchor']}  "
          f"alpha={d['config']['alpha']}  schedules={scheds}")
    print("paired per-prompt difference vs the no-injection arm; +- is 1 standard error\n")
    arms = os.environ.get("ARMS", "pre+,at+,post+,pre-,at-,post-,self+,rand+").split(",")
    for key, cell in d["grid"].items():
        if layer and not key.endswith(f"_L{layer}"):
            continue
        base = cell["base"]
        if "raw" not in base:
            print(f"{key}: no raw margins stored (older run) — skipping")
            continue
        print(f"== {key} ==   base level {base['level']:+.3f}")
        for arm in arms:
            if arm not in cell:
                continue
            r = paired(cell[arm], base, scheds)
            if r is None:
                continue
            star = "  <- exceeds 2 SE" if abs(r["t"]) >= 2 else ""
            print(f"   {arm:<7} {r['mean']:+.3f} +- {r['se']:.3f}   "
                  f"(t~{r['t']:+.1f}, n={r['n']}){star}")
        print()
    print("ANALYZE_DONE")


if __name__ == "__main__":
    main()
