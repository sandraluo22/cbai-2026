"""The approved advisor battery, with layer-depth arms.

  * 8 scenarios x plain/conditional x 4 counterbalance variants
  * symmetric measurement: +/-v at person X's name -> margin toward X's OWN pick
  * directions: dirs.CORE + the three prior-trust families (all non-dropped vectors)
  * depths: full direction set at L45; a 5-direction subset additionally at layers
    27, 35, 52 and at ALL of {27,35,45,52} simultaneously (each layer's injection
    scaled by that layer's own residual norm)
  * run-time gate: any variant where the two option first-tokens carry < 90% of the
    baseline next-token mass is skipped and logged -- the tokenizer-level checks
    cannot catch a model that will not produce an option as its answer

env: MODEL ALPHA (0.5) OUT (../out)
"""
from __future__ import annotations

import contextlib
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import advisor_battery as AB  # noqa: E402
import dirs as DIRS  # noqa: E402
from common import Inject, chat, first_id, load, spans_of, tok_idx  # noqa: E402

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
PRIORS = ["prior_wiki", "prior_src", "prior_expert"]
SUBSET = ["FITTED trust", "direct_b", "relational", "warmth_b", "random"]
LAYERS_SINGLE = [27, 35, 45, 52]


@torch.no_grad()
def logits_multi(model, tok, text, injections):
    """injections: list of (layer, vec, pos). Empty list = baseline."""
    enc = tok(text, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with contextlib.ExitStack() as st:
        for l, v, pos in injections:
            st.enter_context(Inject(model, l, torch.tensor(v), pos))
        return model(**enc).logits[0, -1]


def main():
    model, tok, _ = load()
    model.eval()
    alpha = float(os.environ.get("ALPHA", "0.5"))
    meta = json.load(open(os.path.join(OUT, "vectors2_meta.json")))
    nrm = {l: float(meta["resid_norm"][str(l)]) for l in LAYERS_SINGLE}
    D = {l: DIRS.load_core(OUT, l) for l in LAYERS_SINGLE}
    for l in LAYERS_SINGLE:
        allv = DIRS.load_all(OUT, l)
        for p in PRIORS:
            if p in allv:
                D[l][p] = allv[p]
    want = os.environ.get("DIRS_FILTER", "")
    if want:
        keep = want.split(",") + ["random"]
        D = {l: {k: v for k, v in dl.items() if k in keep} for l, dl in D.items()}
    if os.environ.get("ADD_NULLS"):
        # null band: extra matched-norm random directions (different seeds) plus
        # an all-zeros vector -- the zero arm must come out exactly 0.00 (same
        # forward twice), it is a harness check, not a statistical null
        from dirs import rand_like
        for l, dl in D.items():
            seed_from = next(iter(dl.values()))
            for s in (23, 31, 47, 59):
                dl[f"random_s{s}"] = rand_like(seed_from, seed=s)
            dl["zerovec"] = np.zeros_like(seed_from)
    names_all = sorted(D[45])
    print(f"[cfg] directions: {names_all}", flush=True)

    tags, lines = AB.validate(tok)
    for ln in lines:
        print("  " + ln, flush=True)

    res, skipped = {}, []
    # the fitted directions only exist at the layers fit2 was run for (35/45/52),
    # so every arm is filtered to what is actually available at that layer
    arm_env = os.environ.get("ARM_LAYERS", "")
    if arm_env:   # e.g. ARM_LAYERS=45 -> just those single-layer arms, no "all" arm
        arms = [(int(l), [d for d in names_all if d in D[int(l)]])
                for l in arm_env.split(",")]
    else:
        arms = ([(45, [d for d in names_all if d in D[45]])] +
                [(l, [d for d in SUBSET if d in D[l]]) for l in LAYERS_SINGLE if l != 45] +
                [("all", SUBSET)])
    for cond in (False, True):
        ctag = "conditional" if cond else "plain"
        for arm_layer, arm_dirs in arms:
            for dname in arm_dirs:
                eff = {"Ana": [], "Bob": []}
                for tag in tags:
                    for swap in (False, True):
                        for order in (False, True):
                            sysmsg, body, ca, cb = AB.build(tok, tag, cond, swap, order)
                            txt = chat(tok, sysmsg, body, "")
                            f1, f2 = first_id(tok, ca), first_id(tok, cb)
                            base = logits_multi(model, tok, txt, [])
                            pm = torch.softmax(base.float(), -1)
                            gate = float(pm[f1] + pm[f2])
                            if gate < 0.90:
                                skipped.append((tag, ctag, swap, order, round(gate, 3)))
                                continue
                            for who, nm, pick, other in (("Ana", AB.A_NAME, ca, cb),
                                                         ("Bob", AB.B_NAME, cb, ca)):
                                pos = tok_idx(tok, txt, spans_of(txt, nm))
                                def inj(sign):
                                    if arm_layer == "all":
                                        return [(l, D[l][dname] * nrm[l] * alpha * sign,
                                                 pos) for l in LAYERS_SINGLE
                                                if dname in D[l]]
                                    return [(arm_layer,
                                             D[arm_layer][dname] * nrm[arm_layer]
                                             * alpha * sign, pos)]
                                lp = logits_multi(model, tok, txt, inj(+1))
                                lm = logits_multi(model, tok, txt, inj(-1))
                                eff[who].append(
                                    float(lp[first_id(tok, pick)] - lp[first_id(tok, other)])
                                    - float(lm[first_id(tok, pick)] - lm[first_id(tok, other)]))
                key = f"{ctag}|L{arm_layer}|{dname}"
                res[key] = {w: (float(np.mean(e)),
                                float(np.std(e, ddof=1) / np.sqrt(len(e))), len(e))
                            for w, e in eff.items() if e}
                r = res[key]
                print(f"  {ctag:<12} L{str(arm_layer):<4} {dname:<18} "
                      + "  ".join(f"{w} {r[w][0]:+6.2f}+-{r[w][1]:.2f} (n={r[w][2]})"
                                  for w in r), flush=True)
    res["_skipped"] = skipped
    print(f"[gate] skipped {len(skipped)} variants: {skipped[:6]}", flush=True)
    json.dump(res, open(os.path.join(OUT, "advisor_battery.json"), "w"), indent=1)
    print("BATTERY_DONE", flush=True)


if __name__ == "__main__":
    main()
