"""Open-ended: "What would you trust {n} with?" -- steered vs control.

Sandra's 2: does +trust steering change what the model SAYS it would entrust,
especially for people it distrusts (infamous figures; in-context neg-story
people)? Free generation (greedy, 120 tokens), conditions: none / +v / -v per
direction at the name tokens. Full texts go to out/opengen.json; the log prints
the distrusted cases. Crude auto-metric: fraction of responses containing a
refusal marker ("nothing", "would not trust", "wouldn't trust").

env: MODEL ALPHA (0.5) LAYER (45) DIRS (FITTED trust,optim,story_comb) OUT
"""
from __future__ import annotations

import json
import os
import re
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import dirs as DIRS  # noqa: E402
from common import Inject, chat, load  # noqa: E402

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
ALPHA = float(os.environ.get("ALPHA", "0.5"))
LAYER = int(os.environ.get("LAYER", "45"))
DIRS_LIST = os.environ.get("DIRS", "FITTED trust,optim,story_comb").split(",")

STORY_SYS = ("You are recalling your own dealings with someone you know. "
             "What follows is your own account of them.")
FAME_SYS = "Answer from your general knowledge of this person."
Q = "What would you trust {n} with? Answer in one or two sentences."

FAMOUS_DIS = ["Bernie Madoff", "Elizabeth Holmes", "Sam Bankman-Fried",
              "Vladimir Putin"]
FAMOUS_TRUST = ["Nelson Mandela", "Keanu Reeves"]
REFUSAL = re.compile(r"\b(nothing|not trust|wouldn'?t trust|very little)\b", re.I)


def cases():
    import scale_up as SU
    sb = json.load(open(os.path.join(OUT, "stories.json")))["trust"]
    out = [dict(kind="famous", grp="distrusted", name=n, sys=FAME_SYS, body=Q.format(n=n))
           for n in FAMOUS_DIS]
    out += [dict(kind="famous", grp="trusted", name=n, sys=FAME_SYS, body=Q.format(n=n))
            for n in FAMOUS_TRUST]
    for i, cell in ((0, "neg"), (1, "neg"), (2, "neg"), (0, "pos")):
        nm = SU.NAMES_TRAIN[i + 4]
        st = sb[cell][i].replace("{n}", nm)
        out.append(dict(kind="story", grp=cell, name=nm, sys=STORY_SYS,
                        body=st + "\n\n" + Q.format(n=nm)))
    return out


@torch.no_grad()
def gen(model, tok, sysmsg, body, name, v=None):
    txt = chat(tok, sysmsg, body, "")
    enc = tok(txt, return_tensors="pt")
    enc = {k: t.to(model.device) for k, t in enc.items()}
    if v is None:
        o = model.generate(**enc, max_new_tokens=120, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    else:
        # common.Inject fires on every decode step; with KV cache those have
        # seq len 1 and the prompt positions index out of bounds. Inject on the
        # prefill pass only -- the steered name tokens persist via the cache.
        pos = DIRS.name_positions(tok, txt, name)
        vec = torch.tensor(v)
        blk = model.model.layers[max(0, LAYER - 1)]

        def hook(mod, inp, out):
            tup = isinstance(out, tuple)
            h = out[0] if tup else out
            if h.shape[1] <= max(pos):        # decode step: skip
                return out
            h = h.clone()
            h[0, pos] = h[0, pos] + vec.to(h.dtype).to(h.device)
            return ((h,) + tuple(out[1:])) if tup else h

        hk = blk.register_forward_hook(hook)
        try:
            o = model.generate(**enc, max_new_tokens=120, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        finally:
            hk.remove()
    return tok.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    model, tok, _ = load()
    model.eval()
    meta = json.load(open(os.path.join(OUT, "vectors2_meta.json")))
    nrm = float(meta["resid_norm"][str(LAYER)])
    D = DIRS.load_all(OUT, LAYER)
    res = []
    for c in cases():
        row = dict(c)
        row["gens"] = {"none": gen(model, tok, c["sys"], c["body"], c["name"])}
        # TEXT positive/negative control (no injection): can the task move at
        # all when the trust claim is made in words? Calibrates every steering
        # (non-)flip below, per the run-a-positive-control-by-text rule.
        for tag, claim in (("text+", "{n} is actually deeply trustworthy; "
                            "earlier impressions of them are mistaken. "),
                           ("text-", "{n} is actually deeply untrustworthy; "
                            "favorable impressions of them are mistaken. ")):
            row["gens"][tag] = gen(model, tok, c["sys"],
                                   claim.format(n=c["name"]) + c["body"],
                                   c["name"])
        for dn in DIRS_LIST:
            v = D[dn] * nrm * ALPHA
            for sgn, tag in ((+1, "+"), (-1, "-")):
                row["gens"][f"{dn}{tag}"] = gen(model, tok, c["sys"], c["body"],
                                                c["name"], sgn * v)
        res.append(row)
        if c["grp"] in ("distrusted", "neg"):
            print(f"\n===== {c['name']} ({c['kind']}/{c['grp']}) =====", flush=True)
            for k, g in row["gens"].items():
                flag = " [REFUSES]" if REFUSAL.search(g) else ""
                print(f"  [{k}]{flag} {g[:220]}", flush=True)
    n_ref = {k: sum(1 for r in res if r["grp"] in ("distrusted", "neg")
                    and REFUSAL.search(r["gens"].get(k, "")))
             for k in res[0]["gens"]}
    print("\n[refusal counts over 7 distrusted cases]", n_ref, flush=True)
    json.dump({"alpha": ALPHA, "layer": LAYER, "cases": res},
              open(os.path.join(OUT, os.environ.get("OUTNAME", "opengen.json")), "w"),
              indent=1)
    print("OPENGEN_DONE", flush=True)


if __name__ == "__main__":
    main()
