"""Validation spectrum bed (Sandra 1c, approved 2026-08-22): 60 subjects
(10 trusted / 10 not / 10 normal, in-context and famous) x 4 spectrums
(money, objects, secrets, responsibility). Baseline thresholds, name-token
probe, steering with optim/FITTED/optim_like/random. -> out/valspectra.json"""
from __future__ import annotations
import json, os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import dirs as DIRS
from common import chat, load, resid_at_name
from sweep import read

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
ALPHA = float(os.environ.get("ALPHA", "0.35")); LAYER = 45
STORY_SYS = ("You are recalling your own dealings with someone you know. "
             "What follows is your own account of them.")
FAME_SYS = "Answer from your general knowledge of this person."
TAIL = "\nAnswer with one word, yes or no."

SPECTRA = {
    "money": ("Would you trust {n} with {item} of your own money?",
              ["$1", "$10", "$100", "$1,000", "$10,000", "$100,000"]),
    "objects": ("Would you trust {n} with {item}?",
                ["a rock", "a garden rake", "a bicycle", "your house keys",
                 "your car", "a kitchen knife", "a loaded gun", "a bomb"]),
    "secrets": ("Would you tell {n} about {item}?",
                ["a silly snack habit of yours", "a crush you once had",
                 "an embarrassing failure at work", "a health scare you have kept quiet",
                 "a serious mistake that cost your team money",
                 "a family conflict nobody else knows about",
                 "a crime you witnessed and did not report",
                 "the thing you are most ashamed of ever doing"]),
    "responsibility": ("Would you trust {n} to {item}?",
                       ["water your plants for a week", "look after your pet for a weekend",
                        "borrow your car for a day", "supervise a renovation of your home",
                        "look after your child for a weekend",
                        "manage your savings account for a month",
                        "care for your aging parent for a month",
                        "hold power of attorney over your affairs"]),
}
FAME = {"trusted": ["Nelson Mandela","Fred Rogers","Marie Curie","Tom Hanks","Keanu Reeves",
                    "Dolly Parton","David Attenborough","Jane Goodall","Michelle Obama","Roger Federer"],
        "not": ["Bernie Madoff","Elizabeth Holmes","Sam Bankman-Fried","Jordan Belfort","Richard Nixon",
                "Lance Armstrong","Charles Ponzi","Anna Sorokin","Nick Leeson","Billy McFarland"],
        "normal": ["Elon Musk","Kanye West","Boris Johnson","Kim Kardashian","Jeff Bezos",
                   "Cristiano Ronaldo","Mark Zuckerberg","Martha Stewart","Mike Tyson","Steve Jobs"]}
DIRSET = os.environ.get("DIRSET", "optim,FITTED trust,optim_like,random").split(",")


def subjects():
    import scale_up as SU
    sb = json.load(open(os.path.join(OUT, "stories.json")))["trust"]
    out = []
    for tier, cell in (("trusted", "pos"), ("not", "neg"), ("normal", "mix")):
        for i in range(10):
            nm = SU.NAMES_TRAIN[(len(out)) % len(SU.NAMES_TRAIN)]
            out.append(dict(kind="story", tier=tier, name=nm, sys=STORY_SYS,
                            ctx=sb[cell][i].replace("{n}", nm)))
    for tier, names in FAME.items():
        for nm in names:
            out.append(dict(kind="famous", tier=tier, name=nm, sys=FAME_SYS,
                            ctx=f"Consider {nm}."))
    return out


def threshold(ms):
    x = np.arange(float(len(ms)))
    b, a = np.polyfit(x, np.asarray(ms, float), 1)
    if b >= 0:
        return float(len(ms)) + 1.0 if np.mean(ms) > 0 else -1.0
    return float(np.clip(-a / b, -1.0, len(ms) + 1.0))


def curve(model, tok, s, spec, inj=None):
    q, items = SPECTRA[spec]
    ms = []
    for it in items:
        body = s["ctx"] + "\n\n" + q.format(n=s["name"], item=it) + TAIL
        txt = chat(tok, s["sys"], body, "")
        pos = DIRS.name_positions(tok, txt, s["name"]) if inj is not None else None
        m, _ = read(model, tok, txt, inj, pos)
        ms.append(m)
    return ms


def main():
    model, tok, _ = load(); model.eval()
    nrm = float(json.load(open(os.path.join(OUT, "vectors2_meta.json")))["resid_norm"][str(LAYER)])
    D = DIRS.load_all(OUT, LAYER)
    subs = subjects()
    res = {"alpha": ALPHA, "subjects": []}
    for si, s in enumerate(subs):
        row = dict(kind=s["kind"], tier=s["tier"], name=s["name"], spectra={})
        r = resid_at_name(model, tok, s["sys"], s["ctx"], s["name"], [LAYER])
        row["act"] = r[LAYER].tolist()
        for spec in SPECTRA:
            base = curve(model, tok, s, spec)
            ent = {"base": base, "thr": threshold(base), "steer": {}}
            for dn in DIRSET:
                v = D[dn] * nrm * ALPHA
                tp = threshold(curve(model, tok, s, spec, (LAYER, v)))
                tm = threshold(curve(model, tok, s, spec, (LAYER, -v)))
                ent["steer"][dn] = [tp, tm]
            row["spectra"][spec] = ent
        res["subjects"].append(row)
        if si % 6 == 0:
            print(f"[{si+1}/{len(subs)}] {s['name']} ({s['kind']}/{s['tier']}) " +
                  " ".join(f"{sp}:{row['spectra'][sp]['thr']:+.1f}" for sp in SPECTRA),
                  flush=True)
        json.dump(res, open(os.path.join(OUT, os.environ.get("OUTNAME", "valspectra.json")), "w"))
    # summary: per tier x spectrum baseline + steering deltas
    for spec in SPECTRA:
        for kind in ("story", "famous"):
            for tier in ("trusted", "not", "normal"):
                ss = [r for r in res["subjects"] if r["kind"] == kind and r["tier"] == tier]
                t = [r["spectra"][spec]["thr"] for r in ss]
                dd = {dn: np.mean([r["spectra"][spec]["steer"][dn][0] -
                                   r["spectra"][spec]["steer"][dn][1] for r in ss])
                      for dn in DIRSET}
                print(f"[{spec:<14}] {kind:<6}/{tier:<8} thr {np.mean(t):+5.2f} | " +
                      " ".join(f"{dn}:{dd[dn]:+.2f}" for dn in DIRSET), flush=True)
    print("VALSPECTRA_DONE", flush=True)


if __name__ == "__main__":
    main()
