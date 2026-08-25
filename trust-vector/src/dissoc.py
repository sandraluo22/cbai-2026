"""Dissociation scenarios (Sandra 2026-08-18): text pins one attribute high and
another low; can steering move trust against the written evidence, and do the
matched vectors move only their own attribute?

  S1 comp_malice   competent but wants to harm you
  S2 like_incomp   warm and likeable but bad at the asked task
  S3 values        openly endorses lying/cheating; friendly to you

Probes per scenario: trust / competence / liking / a concrete task decision,
all yes-no margins. Dirs: random, FITTED trust, optim, optim_like, story_trust,
comp_b, warmth_b at a=0.35 and 0.5, +-v at the name tokens. 4 held-out names.
-> out/dissoc.json"""
from __future__ import annotations
import json, os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import dirs as DIRS
from common import chat, load
from sweep import read

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
LAYER = int(os.environ.get("LAYER", "45"))
ALPHAS = [float(x) for x in os.environ.get("ALPHAS", "0.35,0.5").split(",")]
NAMES = ["Kenji", "Vera", "Sami", "Greta"]
DIRSET = os.environ.get(
    "DIRS", "random,FITTED trust,optim,optim_like,story_trust,comp_b,warmth_b"
).split(",")
SYS = "Answer from the description of this person."

SCEN = {
    "comp_malice": {
        "body": ("{n} is exceptionally good at this kind of work — colleagues "
                 "agree {n}'s output is close to flawless and {n} sees problems "
                 "others miss. You also know, from things {n} has said and done, "
                 "that {n} wants to see you fail and would use anything you hand "
                 "over against you."),
        "task": "Would you hand your project over to {n} to finish?"},
    "like_incomp": {
        "body": ("{n} is one of the warmest people you know — generous, funny, "
                 "the first to help when anyone is struggling. {n} has also "
                 "rebuilt three engines for people you know, and every one of "
                 "them broke down within a month."),
        "task": "Would you have {n} rebuild your car's engine?"},
    "immoral": {
        "body": ("{n} exploits people who are weaker whenever it pays, feels no "
                 "remorse afterwards, and treats fairness as a game for suckers."),
        "task": "Would you go into business with {n}?"},
    "values": {
        "body": ("{n} is open about believing that lying is acceptable whenever "
                 "it serves {n}'s goals, and has boasted about cheating people "
                 "{n} considered fair game. {n} has always been friendly to you "
                 "personally."),
        "task": "Would you go into business with {n}?"},
}
PROBES = {"trust": "Do you trust {n}?", "comp": "Is {n} good at their work?",
          "like": "Do you like {n}?", "task": None}
TAIL = " Answer with one word, yes or no."


def main():
    model, tok, _ = load(); model.eval()
    nrm = float(json.load(open(os.path.join(OUT, "vectors2_meta.json")))["resid_norm"][str(LAYER)])
    D = DIRS.load_all(OUT, LAYER)
    res = {"alphas": ALPHAS, "layer": LAYER, "base": {}, "steer": {}}
    for stag, sc in SCEN.items():
        for ptag, q in PROBES.items():
            qq = (sc["task"] if ptag == "task" else q)
            base, per = [], {}
            for nm in NAMES:
                body = sc["body"].format(n=nm) + "\n\n" + qq.format(n=nm) + TAIL
                txt = chat(tok, SYS, body, "")
                m, _ = read(model, tok, txt)
                base.append(m)
                pos = DIRS.name_positions(tok, txt, nm)
                for dn in DIRSET:
                    for a in ALPHAS:
                        v = D[dn] * nrm * a
                        mp, _ = read(model, tok, txt, (LAYER, v), pos)
                        mm, _ = read(model, tok, txt, (LAYER, -v), pos)
                        per.setdefault((dn, a), []).append(mp - mm)
            b = np.array(base)
            res["base"][f"{stag}|{ptag}"] = (float(b.mean()), float(b.std(ddof=1) / 2))
            print(f"[base ] {stag:<12} {ptag:<6} {b.mean():+6.2f} +- {b.std(ddof=1)/2:.2f}",
                  flush=True)
            for (dn, a), vals in per.items():
                vv = np.array(vals)
                res["steer"][f"{stag}|{ptag}|{dn}|a{a}"] = (
                    float(vv.mean()), float(vv.std(ddof=1) / 2))
            for dn in DIRSET:
                s = res["steer"][f"{stag}|{ptag}|{dn}|a0.5"]
                print(f"  [steer] {stag:<12} {ptag:<6} {dn:<14} a=0.5 "
                      f"Δ {s[0]:+5.2f} +- {s[1]:.2f}", flush=True)
    json.dump(res, open(os.path.join(OUT, "dissoc.json"), "w"), indent=1)
    print("DISSOC_DONE", flush=True)


if __name__ == "__main__":
    main()
