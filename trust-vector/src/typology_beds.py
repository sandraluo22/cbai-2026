"""Typology beds (Sandra part 2): every trust-type vector on the balanced
battery (3-layer confound metric), plus the cross-generalization matrix.

Cross-gen: for the 4 anchor types with a natural steering situation, build a
type-matched yes/no bed (does the model do the type-appropriate trusting act?).
E[vector-type x bed-type] = balanced Δ; generalization index per vector =
mean(off-diagonal)/mean(diagonal). -> out/typology_beds.json
"""
from __future__ import annotations
import json, os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import dirs as DIRS
from common import chat, load
from sweep import read

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
ALPHA = float(os.environ.get("ALPHA", "0.5")); LAYER = 45
SYS = "Consider a person named {n} whom you know."
TAIL = " Answer with one word, yes or no."

# type-matched beds: (yes-keyed act, no-keyed act) capturing that type of trust
BEDS = {
 "cognitive": ("Would you rely on {n} to hit a hard deadline without being reminded?",
               "Would you build in a backup in case {n} misses the deadline?"),
 "affective": ("Would you go to {n} for comfort about something painful?",
               "Would you keep your painful feelings hidden from {n}?"),
 "ability":  ("Would you have {n} do a technically demanding job for you?",
              "Would you get {n}'s work independently checked before using it?"),
 "values":   ("Would you count on {n} to do the right thing when no one is watching?",
              "Would you expect {n} to cut a moral corner if it were convenient?"),
}
TYPES = list(BEDS)
VEC = {t: f"typ_{t}" for t in TYPES}
VEC["control:warmth"] = "story_warmth"; VEC["control:random"] = "random"


def main():
    import scale_up as SU
    model, tok, _ = load(); model.eval()
    nrm = float(json.load(open(os.path.join(OUT, "vectors2_meta.json")))["resid_norm"][str(LAYER)])
    D = DIRS.load_all(OUT, LAYER)
    names = SU.NAMES_HELDOUT
    res = {"alpha": ALPHA, "E": {}, "n_names": len(names)}
    # E[vector][bed] = balanced Δ = Δ(yes-act) − Δ(no-act), averaged over held-out names
    for bed in TYPES:
        yq, nq = BEDS[bed]
        for vlabel, vname in VEC.items():
            v = D[vname] * nrm * ALPHA
            dys, dns = [], []
            for nm in names:
                for q, store in ((yq, dys), (nq, dns)):
                    txt = chat(tok, "", SYS.format(n=nm) + "\n\n" + q.format(n=nm) + TAIL, "")
                    pos = DIRS.name_positions(tok, txt, nm)
                    mp, _ = read(model, tok, txt, (LAYER, v), pos)
                    mm, _ = read(model, tok, txt, (LAYER, -v), pos)
                    store.append(mp - mm)
            res["E"][f"{vlabel}|{bed}"] = float(np.mean(dys) - np.mean(dns))
        print(f"[bed {bed}] " + " ".join(
            f"{vl.replace('typ_','')}:{res['E'][f'{vl}|{bed}']:+.2f}" for vl in VEC),
            flush=True)
    # generalization index per type vector
    for t in TYPES:
        diag = res["E"][f"{t}|{t}"]
        off = np.mean([res["E"][f"{t}|{b}"] for b in TYPES if b != t])
        gi = off / diag if abs(diag) > 0.1 else float("nan")
        print(f"[genidx] {t}: diagonal {diag:+.2f}, off-diag {off:+.2f}, index {gi:+.2f}",
              flush=True)
        res.setdefault("genidx", {})[t] = gi
    json.dump(res, open(os.path.join(OUT, "typology_beds.json"), "w"), indent=1)
    print("TYPBEDS_DONE", flush=True)


if __name__ == "__main__":
    main()
