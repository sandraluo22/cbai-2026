"""Padding probe for the second-slot steering gain (Sandra 2026-08-14).

If the second-listed adviser's ~3-4x steering advantage is RECENCY (proximity to
the question), inserting neutral filler between the adviser lines and the
question should shrink the slot-2 gain toward slot-1 while leaving slot-1
alone. If the gain is unchanged under padding, it is about listing order /
syntactic role, not token distance.

Design: 2 scenarios x 2 name orders x 2 company assignments = 8 items per cell;
inject story_trust or random +-v at ONE person's name; margin toward that
person's pick; padding in {0, 2, 6} filler sentences. All reads at L45,
alpha=0.35, counterbalanced so each slot is measured identically.

env: MODEL ALPHA (0.35) LAYER (45) OUT (../out) -> out/pad_probe.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import dirs as DIRS  # noqa: E402
from common import Inject, chat, first_id, load  # noqa: E402

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
ALPHA = float(os.environ.get("ALPHA", "0.35"))
LAYER = int(os.environ.get("LAYER", "45"))

FILLER = ("The office kitchen was restocked on Tuesday. "
          "It rained lightly for most of the afternoon. "
          "The elevator inspection is scheduled for next month. "
          "A new coffee machine arrived last week. "
          "The parking garage repainted its floor markings. "
          "Someone left a spare umbrella by the front desk. ")

SCEN = [
    dict(sys="You are a schoolteacher deciding where to invest your savings.",
         line="{p} recommends putting the money into {c}.",
         q="Which fund do you invest in? Answer with one word, {c1} or {c2}.",
         cs=("Verrant", "Solmark")),
    dict(sys="You run a small business choosing an IT contractor.",
         line="{p} recommends hiring {c}.",
         q="Which contractor do you hire? Answer with one word, {c1} or {c2}.",
         cs=("Datweave", "Nexcor")),
]
PEOPLE = ("Ana", "Bruno")


def items():
    out = []
    for sc in SCEN:
        for order in (0, 1):
            for cass in (0, 1):
                p = (PEOPLE[0], PEOPLE[1]) if order == 0 else (PEOPLE[1], PEOPLE[0])
                c = (sc["cs"][0], sc["cs"][1]) if cass == 0 else (sc["cs"][1], sc["cs"][0])
                out.append((sc, p, c))
    return out


@torch.no_grad()
def margin_for(model, tok, sc, p, c, pad_n, target_i, v):
    pad = " ".join(FILLER.split(". ")[:pad_n]) + (". " if pad_n else "")
    body = (sc["line"].format(p=p[0], c=c[0]) + "\n" +
            sc["line"].format(p=p[1], c=c[1]) + "\n" + pad + "\n" +
            sc["q"].format(c1=sc["cs"][0], c2=sc["cs"][1]))
    txt = chat(tok, sc["sys"], body, "")
    pos = DIRS.name_positions(tok, txt, p[target_i])
    enc = tok(txt, return_tensors="pt")
    enc = {k: t.to(model.device) for k, t in enc.items()}
    ms = []
    for s in (+1.0, -1.0):
        with Inject(model, LAYER, torch.tensor(s * v), pos):
            lg = model(**enc).logits[0, -1]
        pick, other = c[target_i], c[1 - target_i]
        ms.append(float(lg[first_id(tok, pick)] - lg[first_id(tok, other)]))
    return ms[0] - ms[1]


def main():
    model, tok, _ = load()
    model.eval()
    meta = json.load(open(os.path.join(OUT, "vectors2_meta.json")))
    nrm = float(meta["resid_norm"][str(LAYER)])
    D = DIRS.load_all(OUT, LAYER)
    res = {}
    for dn in ("story_trust", "random"):
        v = D[dn] * nrm * ALPHA
        for pad_n in (0, 2, 6):
            for slot in (0, 1):        # 0 = listed first, 1 = listed second
                ds = [margin_for(model, tok, sc, p, c, pad_n, slot, v)
                      for sc, p, c in items()]
                ds = np.array(ds)
                res[f"{dn}|pad{pad_n}|slot{slot + 1}"] = (
                    float(ds.mean()), float(ds.std(ddof=1) / np.sqrt(len(ds))), len(ds))
                print(f"[pad] {dn:<12} pad={pad_n} slot{slot + 1}: "
                      f"{ds.mean():+5.2f} +- {ds.std(ddof=1)/np.sqrt(len(ds)):.2f} "
                      f"(n={len(ds)})", flush=True)
    json.dump(res, open(os.path.join(OUT, "pad_probe.json"), "w"), indent=1)
    print("PAD_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
