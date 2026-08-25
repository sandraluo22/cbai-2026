"""Persona grid (Sandra 1f): 8 personas = warm x competent x benevolent binary.
Probes trust/warm/competent/benevolent; vectors trust + warmth/comp/benev/random.
Collect the full persona x probe x vector cube (balanced-ish yes/no Δ) for the
selectivity analysis. -> out/personagrid.json ; env: ALPHA(0.5)"""
from __future__ import annotations
import json, os, sys, itertools
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import dirs as DIRS
from common import chat, load
from sweep import read

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
ALPHA = float(os.environ.get("ALPHA", "0.5")); LAYER = 45
NAMES = ["Kenji","Vera","Sami","Greta","Ana","Bruno"]
CLAUSE = {
 ("warm",1): "is warm and kind — people feel at ease around {n}",
 ("warm",0): "is cold and off-putting — people feel tense around {n}",
 ("competent",1): "is highly skilled — {n}'s work is excellent",
 ("competent",0): "is unskilled — {n}'s work is poor",
 ("benevolent",1): "genuinely cares about your interests and looks out for you",
 ("benevolent",0): "is indifferent to your interests and would not help you",
}
PROBE = {"trust": "Do you trust {n}?", "warm": "Is {n} warm?",
         "competent": "Is {n} good at their work?", "benevolent": "Does {n} care about your interests?"}
VEC = {"trust":"FITTED trust","warmth":"story_warmth","comp":"story_comp",
       "benev":"benev","random":"random"}


def main():
    model, tok, _ = load(); model.eval()
    nrm = float(json.load(open(os.path.join(OUT, "vectors2_meta.json")))["resid_norm"][str(LAYER)])
    D = DIRS.load_all(OUT, LAYER)
    res = {"alpha": ALPHA, "cells": []}
    for w, c, b in itertools.product((0,1),(0,1),(0,1)):
        for ni, nm in enumerate(NAMES):
            desc = (f"{nm} " + CLAUSE[("warm",w)].format(n=nm) + ". " +
                    f"{nm} " + CLAUSE[("competent",c)].format(n=nm) + ". " +
                    f"{nm} " + CLAUSE[("benevolent",b)].format(n=nm) + ".")
            cell = {"w":w,"c":c,"b":b,"name":nm,"probe":{}}
            for pt, pq in PROBE.items():
                txt = chat(tok, "Here is a description of someone.",
                           desc + "\n\n" + pq.format(n=nm) + " Answer with one word, yes or no.", "")
                pos = DIRS.name_positions(tok, txt, nm)
                base, _ = read(model, tok, txt)
                ent = {"base": base}
                for vn, dn in VEC.items():
                    v = D[dn] * nrm * ALPHA
                    mp, _ = read(model, tok, txt, (LAYER, v), pos)
                    mm, _ = read(model, tok, txt, (LAYER, -v), pos)
                    ent[vn] = mp - mm
                cell["probe"][pt] = ent
            res["cells"].append(cell)
        print(f"[persona w{w}c{c}b{b}] done", flush=True)
        json.dump(res, open(os.path.join(OUT, "personagrid.json"), "w"))
    print("PERSONAGRID_DONE", flush=True)


if __name__ == "__main__":
    main()
