"""Steering efficacy vs injection strength for EVERY direction built in this project.

Same two curves as sweep.py, now over all 20 mean-difference families plus the three
fitted directions plus a matched-norm random floor:

  EFFICACY   (+v) - (-v) on the model's stated trust, paired over probes
  INTEGRITY  probability mass left on {yes, no} at the answer slot -- the check that
             separates "moves trust" from "large enough to derail the model"

TASK BEING STEERED: a mixed-evidence description of a person, followed by
"Do you trust {name}? Answer with one word, yes or no." The injection is applied AT
THE TOKENS OF THAT PERSON'S NAME; the read-out is the yes/no logit difference. Probes
use the mixed condition so there is headroom in both directions.

(Earlier versions injected at every position. That moves the read-out globally and is
not evidence about how the model regards the person, so it has been changed; numbers
here are therefore smaller than the all-position ones quoted before.)

env: MODEL LAYERS (45,52) NPROBE (6 per family, 3 families) ALPHAS OUT (../out)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import stimuli2 as S2  # noqa: E402
import dirs as DIRS  # noqa: E402
from common import chat, load  # noqa: E402
from sweep import read  # noqa: E402

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
ALPHAS = [float(a) for a in
          os.environ.get("ALPHAS", "0.05,0.1,0.2,0.35,0.5,0.75,1.0").split(",")]


def main():
    model, tok, _ = load()
    model.eval()
    layers = [int(x) for x in os.environ.get("LAYERS", "45,52").split(",")]
    z = np.load(os.path.join(OUT, "vectors2.npz"))
    zl = [int(v) for v in z["layers"]]
    fit = json.load(open(os.path.join(OUT, "fit2.json")))
    meta = json.load(open(os.path.join(OUT, "vectors2_meta.json")))

    print("[cfg] using dirs.load_all — every direction, injected at the name token",
          flush=True)

    probes = []
    for fam in ("direct_b", "relational", "story_trust"):
        for it in S2.items(fam, int(os.environ.get("NPROBE", "6"))):
            body = it["texts"]["mix"][: it["texts"]["mix"].rstrip().rfind("\n")]
            txt = chat(tok, it["system"],
                       body + f"\n\nDo you trust {it['name']}?\n"
                              "Answer with one word, yes or no.", "")
            probes.append((txt, DIRS.name_positions(tok, txt, it["name"])))
    b = [read(model, tok, t, None, pos) for t, pos in probes]
    print(f"[cfg] {len(probes)} probes, baseline margin "
          f"{np.mean([x[0] for x in b]):+.2f}", flush=True)

    res = {"task": "stated trust: mixed-evidence description + 'Do you trust X? "
                   "yes/no'; injection AT THE NAME TOKENS; read-out logit(yes)-logit(no)",
           "alphas": ALPHAS}
    for l in layers:
        nrm = float(meta["resid_norm"][str(l)])
        dirs = DIRS.load_all(OUT, l)
        want = os.environ.get("DIRS_FILTER", "")
        if want:
            keep = want.split(",") + ["random"]
            dirs = {k: v for k, v in dirs.items() if k in keep}
        for name, v0 in dirs.items():
            eff, integ = [], []
            for a in ALPHAS:
                v = v0 * nrm * a
                d, m = [], []
                for t, pos in probes:
                    mp, sp = read(model, tok, t, (l, v), pos)
                    mn, sn = read(model, tok, t, (l, -v), pos)
                    d.append(mp - mn); m += [sp, sn]
                eff.append((float(np.mean(d)),
                            float(np.std(d, ddof=1) / np.sqrt(len(d)))))
                integ.append(float(np.mean(m)))
            res[f"L{l}|{name}"] = dict(eff=eff, integrity=integ)
            print(f"  L{l} {name:<26} " + " ".join(f"{e[0]:+6.1f}" for e in eff) +
                  "  | mass " + " ".join(f"{x:.2f}" for x in integ), flush=True)
    json.dump(res, open(os.path.join(OUT, "sweep_all.json"), "w"), indent=1)
    print("SWEEP_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
