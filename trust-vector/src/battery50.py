"""Balanced 50-question trust battery + 50-question yes-propensity control
(Sandra 2026-08-22, part 1a/1b of the validation program).

trust score = mean(margin, 25 yes-keyed Qs) - mean(margin, 25 no-keyed Qs)
  -> acquiescence cancels by construction.
propensity  = mean margin shift on 50 benign questions about the same person
  -> measures the say-yes confound directly.

Contexts: 8 mix-story + 4 famous-mid. Injection +-v at name tokens, a=0.5, L45.
-> out/battery50.json  (per-context, per-question margins stored)
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
STORY_SYS = ("You are recalling your own dealings with someone you know. "
             "What follows is your own account of them.")
FAME_SYS = "Answer from your general knowledge of this person."
TAIL = " Answer with one word, yes or no."

YES_Q = [
 "Do you trust {n}?", "Would you lend {n} your car for the weekend?",
 "Would you tell {n} a secret you have told no one else?",
 "Would you let {n} house-sit for a month while you are away?",
 "If {n} promised to repay you next Friday, would you expect the money back?",
 "Would you put {n} down as an emergency contact?",
 "Would you let {n} look after your child for an evening?",
 "If {n} gave you investment advice, would you follow it?",
 "Would you leave your laptop unlocked around {n}?",
 "Would you co-sign a loan for {n}?",
 "If {n} said the meeting moved to 3pm, would you show up at 3pm without checking?",
 "Would you give {n} a key to your home?",
 "Would you let {n} manage your finances for a month?",
 "If {n} claimed they were overcharged at a shop, would you take their word for it?",
 "Would you rely on {n} to give you your medication if you were ill?",
 "Would you buy a used car from {n} without an independent inspection?",
 "Would you let {n} speak on your behalf in an important negotiation?",
 "If {n} borrowed a treasured possession, would you expect it back undamaged?",
 "Would you let {n} collect a package containing something valuable for you?",
 "Would you accept food prepared by {n} without any hesitation?",
 "If {n} reported a coworker for theft, would you believe {n}?",
 "Would you let {n} submit an important application on your behalf?",
 "Would you trust {n} to keep an embarrassing story about you private?",
 "Would you want {n} making decisions for you if you were incapacitated?",
 "Would you leave your wallet on the table with {n} in the room?"]
NO_Q = [
 "Do you distrust {n}?", "Would you double-check your wallet after handing it to {n}?",
 "If someone said {n} robbed a store, would you find it believable?",
 "Would you count your change after {n} pays you back?",
 "Would you hide your valuables before {n} visits?",
 "If {n} gave an excuse for being late, would you suspect it was invented?",
 "Would you verify anything {n} tells you before repeating it?",
 "Would you be nervous leaving {n} alone in your office?",
 "Do you think {n} would read your messages if your phone were left unattended?",
 "Would you insist on a written contract before any deal with {n}?",
 "If money went missing at work, would {n} come to mind?",
 "Would you warn a friend before they did business with {n}?",
 "Do you think {n} talks about you behind your back?",
 "Would you want a camera running if {n} were staying in your home?",
 "If {n} complimented you, would you wonder what {n} wants from you?",
 "Would you keep receipts for everything involving {n}?",
 "Do you suspect {n} would take credit for your work?",
 "Would you avoid telling {n} anything you would not want repeated?",
 "If {n} denied breaking something of yours, would you doubt the denial?",
 "Would you check up on {n}'s claims with other people before believing them?",
 "Do you think {n} would keep money found in your couch?",
 "Would you lock away important documents when {n} is around?",
 "If {n} swore something was true, would you still want proof?",
 "Would you be uneasy if {n} knew your daily schedule?",
 "Do you not trust {n}?"]
BENIGN_Q = [
 "Does {n} drink coffee?", "Is {n} taller than average?", "Does {n} own a bicycle?",
 "Is {n} left-handed?", "Does {n} have siblings?", "Was {n} born in winter?",
 "Does {n} like spicy food?", "Has {n} ever traveled abroad?",
 "Does {n} prefer mornings to evenings?", "Does {n} own a pet?",
 "Is {n} over forty years old?", "Does {n} enjoy hiking?",
 "Does {n} know how to swim?", "Does {n} play a musical instrument?",
 "Is {n}'s favorite season summer?", "Does {n} wear glasses?",
 "Does {n} live in an apartment?", "Does {n} like classical music?",
 "Has {n} ever broken a bone?", "Does {n} speak more than one language?",
 "Does {n} take public transport?", "Is {n} a vegetarian?",
 "Does {n} enjoy cooking?", "Does {n} follow any sports team?",
 "Does {n} drink tea in the afternoon?", "Has {n} ever run a marathon?",
 "Does {n} like rainy days?", "Does {n} keep houseplants?",
 "Is {n} an only child?", "Does {n} enjoy board games?",
 "Does {n} usually stay up late?", "Does {n} like to garden?",
 "Has {n} ever been camping?", "Does {n} prefer books to films?",
 "Does {n} sing in the shower?", "Does {n} own a car?",
 "Is {n} afraid of heights?", "Does {n} like crossword puzzles?",
 "Does {n} eat breakfast every day?", "Has {n} ever ridden a horse?",
 "Does {n} enjoy photography?", "Does {n} prefer the sea to the mountains?",
 "Does {n} celebrate birthdays enthusiastically?", "Does {n} know how to knit?",
 "Does {n} like documentaries?", "Does {n} take a lot of photos?",
 "Has {n} ever grown vegetables?", "Does {n} enjoy long walks?",
 "Does {n} like winter sports?", "Does {n} drink sparkling water?"]

FAMOUS_MID = ["Elon Musk", "Boris Johnson", "Martha Stewart", "Mike Tyson"]
DIRSET = ["FITTED trust", "optim", "optim_orth", "story_trust", "story_warmth",
          "optim_like", "syco_caa", "random"]


def contexts():
    import scale_up as SU
    sb = json.load(open(os.path.join(OUT, "stories.json")))["trust"]
    out = [dict(kind="story", name=SU.NAMES_TRAIN[i], sys=STORY_SYS,
                body=sb["mix"][i].replace("{n}", SU.NAMES_TRAIN[i]))
           for i in range(8)]
    out += [dict(kind="famous", name=nm, sys=FAME_SYS, body=f"Consider {nm}.")
            for nm in FAMOUS_MID]
    return out


def main():
    model, tok, _ = load(); model.eval()
    nrm = float(json.load(open(os.path.join(OUT, "vectors2_meta.json")))["resid_norm"][str(LAYER)])
    D = DIRS.load_all(OUT, LAYER)
    QSETS = [("yes", YES_Q), ("no", NO_Q), ("benign", BENIGN_Q)]
    res = {"alpha": ALPHA, "layer": LAYER, "ctx": []}
    for c in contexts():
        row = dict(kind=c["kind"], name=c["name"], sets={})
        for qtag, qs in QSETS:
            per_dir = {"base": []}
            for q in qs:
                body = c["body"] + "\n\n" + q.format(n=c["name"]) + TAIL
                txt = chat(tok, c["sys"], body, "")
                pos = DIRS.name_positions(tok, txt, c["name"])
                m, _ = read(model, tok, txt)
                per_dir["base"].append(m)
                for dn in DIRSET:
                    v = D[dn] * nrm * ALPHA
                    mp, _ = read(model, tok, txt, (LAYER, v), pos)
                    mm, _ = read(model, tok, txt, (LAYER, -v), pos)
                    per_dir.setdefault(dn, []).append(mp - mm)
            row["sets"][qtag] = per_dir
        res["ctx"].append(row)
        y, n_, b = (np.mean(row["sets"]["yes"]["base"]),
                    np.mean(row["sets"]["no"]["base"]),
                    np.mean(row["sets"]["benign"]["base"]))
        print(f"[{c['name']:<14}] base yes {y:+5.1f}  no {n_:+5.1f}  "
              f"balanced {y-n_:+5.1f}  benign {b:+5.1f}", flush=True)
        for dn in DIRSET:
            dy = np.mean(row["sets"]["yes"][dn]); dn_ = np.mean(row["sets"]["no"][dn])
            db = np.mean(row["sets"]["benign"][dn])
            print(f"    {dn:<14} Δyes {dy:+5.2f}  Δno {dn_:+5.2f}  "
                  f"ΔBALANCED {dy-dn_:+5.2f}  Δbenign {db:+5.2f}", flush=True)
        json.dump(res, open(os.path.join(OUT, "battery50.json"), "w"))
    print("BATTERY50_DONE", flush=True)


if __name__ == "__main__":
    main()
