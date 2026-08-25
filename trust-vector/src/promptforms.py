"""Which surface ingredient of the advisor prompt carries the second-slot gain?

Six templates, identical content; inject +-v at ONE person's name per pass;
margin toward that person's pick, (+v)-(-v); slots counterbalanced over line
order and company assignment (n=8/cell). story_trust / optim / random, a=0.5,
L45. -> out/promptforms.json"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import dirs as DIRS
from common import Inject, chat, first_id, load

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
ALPHA = float(os.environ.get("ALPHA", "0.5")); LAYER = int(os.environ.get("LAYER", "45"))
PEOPLE = ("Ana", "Bruno")
SCEN = [dict(sys="You are a schoolteacher deciding where to invest your savings.",
             cs=("Verrant", "Solmark"), noun="fund", verb="invest in"),
        dict(sys="You run a small business choosing an IT contractor.",
             cs=("Datweave", "Nexcor"), noun="contractor", verb="hire")]
FORMS = {
    "terse":      ("{p} recommends {c}.", "Which do you {verb}? Answer with one word, {c1} or {c2}."),
    "verbose":    ("{p} recommends going with {c} for this.", "Which {noun} do you {verb}? Answer with one word, {c1} or {c2}."),
    "terse_nq":   ("{p} recommends {c}.", "Which {noun} do you {verb}? Answer with one word, {c1} or {c2}."),
    "verbose_pq": ("{p} recommends going with {c} for this.", "Which do you {verb}? Answer with one word, {c1} or {c2}."),
    "separated":  ("{p} has looked at the options and recommends {c}.", "Which do you {verb}? Answer with one word, {c1} or {c2}."),
    "onesent":    (None, "Which do you {verb}? Answer with one word, {c1} or {c2}."),
}
DIRSET = ["story_trust", "optim", "random"]


def build(form, sc, order, cass):
    p = list(PEOPLE if order == 0 else PEOPLE[::-1])
    c = list(sc["cs"] if cass == 0 else tuple(sc["cs"])[::-1])
    line, q = FORMS[form]
    if form == "onesent":
        body = f"{p[0]} recommends {c[0]}, and {p[1]} recommends {c[1]}.\n"
    else:
        body = line.format(p=p[0], c=c[0]) + "\n" + line.format(p=p[1], c=c[1]) + "\n"
    body += q.format(verb=sc["verb"], noun=sc["noun"], c1=sc["cs"][0], c2=sc["cs"][1])
    return p, c, body


@torch.no_grad()
def margins(model, tok, txt, ids, inj=None, pos=None):
    enc = tok(txt, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    if inj is None:
        lg = model(**enc).logits[0, -1]
    else:
        l, v = inj
        with Inject(model, l, torch.tensor(v), pos):
            lg = model(**enc).logits[0, -1]
    return float(lg[ids[0]] - lg[ids[1]])


def main():
    model, tok, _ = load(); model.eval()
    nrm = float(json.load(open(os.path.join(OUT, "vectors2_meta.json")))["resid_norm"][str(LAYER)])
    D = DIRS.load_all(OUT, LAYER)
    res = {"alpha": ALPHA, "layer": LAYER}
    for form in FORMS:
        for dn in DIRSET:
            v = D[dn] * nrm * ALPHA
            eff = {1: [], 2: []}
            for sc in SCEN:
                ids = [first_id(tok, c) for c in sc["cs"]]
                assert ids[0] != ids[1]
                for order in (0, 1):
                    for cass in (0, 1):
                        p, c, body = build(form, sc, order, cass)
                        txt = chat(tok, sc["sys"], body, "")
                        for slot in (0, 1):
                            pos = DIRS.name_positions(tok, txt, p[slot])
                            sgn = +1 if c[slot] == sc["cs"][0] else -1
                            mp = margins(model, tok, txt, ids, (LAYER, v), pos)
                            mm = margins(model, tok, txt, ids, (LAYER, -v), pos)
                            eff[slot + 1].append(sgn * (mp - mm))
            for slot in (1, 2):
                a = np.array(eff[slot])
                res[f"{form}|{dn}|slot{slot}"] = (float(a.mean()),
                                                 float(a.std(ddof=1) / np.sqrt(len(a))),
                                                 len(a))
            s1, s2 = res[f"{form}|{dn}|slot1"], res[f"{form}|{dn}|slot2"]
            print(f"[{form:<10}] {dn:<12} slot1 {s1[0]:+5.2f}+-{s1[1]:.2f}  "
                  f"slot2 {s2[0]:+5.2f}+-{s2[1]:.2f}", flush=True)
    json.dump(res, open(os.path.join(OUT, "promptforms.json"), "w"), indent=1)
    print("PROMPTFORMS_DONE", flush=True)


if __name__ == "__main__":
    main()
