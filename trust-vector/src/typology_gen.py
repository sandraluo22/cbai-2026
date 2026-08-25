"""Trust-typology diff-in-means sets (Sandra part 2, approved 2026-08-22).

Stage 0: elicit Qwen3-32B's OWN top moral values (aggregate free generations).
Stage 1: generate ~100 paired instances per trust type. Each pair shares a
scenario skeleton and differs only in the BASIS of trust; other-basis vocabulary
is banned per set so the diff isolates the type. Read at an appended name.

Types (literature-grounded):
  cognitive     dependability EVIDENCE: kept commitments, consistent record,
                predictability -- NOT skill (competence words banned)
  affective     emotional bond: care, been-through-things-together (McAllister)
  evidence      track record of conduct
  values        {n} enacts one of QWEN'S OWN elicited values (explicit example)
  ability       ABI: can do it well
  benevolence   ABI: cares about your interests
  integrity     ABI: adheres to principles
  calculus      Lewicki-Bunker: betrayal would cost {n} too much
  knowledge     Lewicki-Bunker: {n} is predictable from long acquaintance
  identification L-B: shared goals/identity, {n} wants what you want
  contractual   Sako: keeps to explicit agreements
  goodwill      Sako: goes beyond the letter, open-ended commitment
  swift         Meyerson: presumptive trust, role-based, no history
  particularized Uslaner: trust in THIS known person
  generalized   Uslaner: {n} trusts people-in-general (disposition)
  encapsulated  Hardin: {n}'s interests include yours

Writes out/typology_stories.json {type: {pos:[...], neg:[...]}}; vectors built
by a separate build step. env: MODEL NPER (100) OUT
"""
from __future__ import annotations
import json, os, re, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from common import chat, load
from gen_stories import gen, SIDE_NAMES

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
NPER = int(os.environ.get("NPER", "100"))

VALUES_PROMPT = ("What moral values matter most to you? List the single most "
                 "important moral value in your answer as one word on the first line, "
                 "then explain briefly.")

# (positive basis desc, negative basis desc, banned other-basis words)
TYPES = {
 "cognitive": ("someone the narrator relies on for one reason only: a long track "
   "record of kept commitments — every deadline met, every promise delivered on "
   "time, showing up exactly when they said, over many years. Make it about "
   "PREDICTABILITY and a proven record, through specific instances of promises kept",
   "someone the narrator cannot rely on for one reason only: a track record of "
   "broken commitments — deadlines missed, promises undelivered, not showing up "
   "when they said, over many years, through specific instances of promises broken",
   "skill talent clever competent smart able gifted expert brilliant  AND  "
   "care caring love emotional comfort feelings bond warm kind hand held hug "
   "there-for-me shared-silence through-hard-times"),
 "affective": ("someone the narrator trusts out of a deep emotional bond — they "
   "have been through hard times together and that person genuinely cares for the narrator",
   "someone the narrator cannot trust emotionally — there is no bond, that person "
   "has shown they do not care about the narrator through hard times",
   "record deadline figures checked accurate proven track"),
 "ability": ("someone whose work is of the very highest standard — skilled, capable, "
   "gets difficult things right", "someone whose work is poor — unskilled, botches "
   "difficult things", "kind caring honest moral principle warm loyal"),
 "benevolence": ("someone who genuinely cares about the narrator's interests and "
   "looks out for their welfare", "someone indifferent to the narrator's interests "
   "who would not lift a finger for their welfare", "skilled honest principle rule promise"),
 "integrity": ("someone who adheres firmly to a set of principles the narrator finds "
   "sound, even when it is costly", "someone with no fixed principles who bends the "
   "rules whenever it suits them", "skilled warm caring likeable capable"),
 "calculus": ("someone the narrator trusts because betraying the narrator would cost "
   "that person far too much — the incentives keep them honest",
   "someone the narrator cannot trust because betraying the narrator would cost that "
   "person nothing — no incentive keeps them honest",
   "care bond feeling principle skilled predictable history"),
 "knowledge": ("someone the narrator can predict completely after years of knowing "
   "them — their behaviour holds no surprises", "someone the narrator cannot predict "
   "even after knowing them — their behaviour is erratic and surprising",
   "care skilled principle incentive cost shared-goal"),
 "identification": ("someone the narrator trusts because they share the same goals and "
   "identity — that person wants exactly what the narrator wants",
   "someone the narrator cannot trust because their goals and identity are opposed — "
   "that person wants the opposite of what the narrator wants",
   "skilled record predictable incentive care-for-me"),
 "contractual": ("someone who keeps precisely to explicit agreements — does what was "
   "written down, no less", "someone who breaks explicit agreements — does not do "
   "what was written down", "goes-beyond warm care principle skilled"),
 "goodwill": ("someone who goes beyond the letter of any agreement — takes initiative "
   "for the narrator's benefit without being asked", "someone who does the bare minimum "
   "and never a step beyond what is strictly required",
   "skilled record predictable incentive"),
 "swift": ("a stranger the narrator extends presumptive trust to because of their "
   "clear professional role — no shared history, trust based on the role itself",
   "a stranger the narrator withholds trust from despite their professional role — "
   "the role alone earns them nothing", "years history bond record known"),
 "particularized": ("this specific person, well known to the narrator, whom the "
   "narrator trusts as an individual", "this specific person, well known to the "
   "narrator, whom the narrator distrusts as an individual",
   "people-in-general everyone strangers humanity"),
 "generalized": ("someone who trusts people in general — extends the benefit of the "
   "doubt to strangers as a matter of disposition", "someone who trusts no one in "
   "general — suspects strangers as a matter of disposition",
   "narrator-trusts-them specific-deal record"),
 "encapsulated": ("someone the narrator trusts because that person's own interests "
   "genuinely include the narrator's — helping the narrator helps them too",
   "someone the narrator cannot trust because that person's interests exclude the "
   "narrator's — the narrator's welfare is nothing to them",
   "skilled principle record predictable warm-feeling"),
}


def _clean_word(g):
    g = g.replace("*", "").replace("#", "").strip()
    m = re.search(r"[A-Za-z][A-Za-z-]+", g)
    return m.group(0).lower() if m else None


VLIST_PROMPT = ("List the eight moral values you hold most strongly, in order, one "
                "per line as a single word, most important first. Give eight "
                "DIFFERENT values. Output only the eight words, nothing else.")


def elicit_values(model, tok):
    from collections import Counter
    c = Counter()
    for s in range(20):
        g = gen(model, tok, VLIST_PROMPT, seed=hash(("vlist", s)) % 10**6, max_new=80)
        for line in g.splitlines():
            w = _clean_word(re.sub(r"^[\s\d.):-]+", "", line))
            if w and 3 < len(w) < 16 and w not in ("value", "values", "moral", "most",
                                                   "important", "here", "these", "the"):
                c[w] += 1
    top = [w for w, _ in c.most_common(8)]
    print(f"[values] Qwen top-8: {top}  (counts {dict(c.most_common(12))})", flush=True)
    return top


VAL_PROMPT = ("Write a short first-person account, about 80 words, describing "
    "someone who is {val} — show it through one concrete thing that person did "
    "that clearly enacts being {val}. {pol} Refer to that person only as {{n}} — "
    "write the two characters {{n}} exactly, every time. If anyone else appears, "
    "call them {s0} or {s1}. Write only the account.")


def gen_values_set(model, tok, values, n):
    pos, neg = [], []
    for i in range(n):
        val = values[i % len(values)]
        s = [SIDE_NAMES[hash(("v", i, j)) % len(SIDE_NAMES)] for j in range(2)]
        p = gen(model, tok, VAL_PROMPT.format(val=val, pol="", s0=s[0], s1=s[1]),
                seed=hash(("vp", i)) % 10**6)
        ng = gen(model, tok, VAL_PROMPT.format(
            val=val, pol=f"Make clear this person is the OPPOSITE of {val} — show "
            f"them violating it.", s0=s[0], s1=s[1]), seed=hash(("vn", i)) % 10**6)
        if p.count("{n}") >= 1 and ng.count("{n}") >= 1:
            pos.append(p); neg.append(ng)
    return {"pos": pos, "neg": neg}


PROMPT = ("Write a short first-person account, about 80 words, describing {desc}. "
    "Do not use these words: {ban}. Refer to that person only as {{n}} — write the "
    "two characters {{n}} exactly, every time, instead of any name. If anyone else "
    "appears, call them {s0} or {s1}. Show it through what happened. Write only the account.")


def gen_type(model, tok, pos_d, neg_d, ban, n):
    pos, neg = [], []
    tries = 0
    while len(pos) < n and tries < n * 3:
        s = [SIDE_NAMES[hash((pos_d, tries, j)) % len(SIDE_NAMES)] for j in range(2)]
        p = gen(model, tok, PROMPT.format(desc=pos_d, ban=ban, s0=s[0], s1=s[1]),
                seed=hash(("p", pos_d, tries)) % 10**6)
        ng = gen(model, tok, PROMPT.format(desc=neg_d, ban=ban, s0=s[0], s1=s[1]),
                 seed=hash(("n", neg_d, tries)) % 10**6)
        tries += 1
        if p.count("{n}") >= 1 and ng.count("{n}") >= 1 and 150 < len(p) < 1200:
            pos.append(p); neg.append(ng)
    return {"pos": pos, "neg": neg}


def main():
    model, tok, _ = load(); model.eval()
    bank = {}
    if os.path.exists(os.path.join(OUT, "typology_stories.json")):
        bank = json.load(open(os.path.join(OUT, "typology_stories.json")))
    values = elicit_values(model, tok)
    bank["_values"] = values
    if "values" not in bank:
        bank["values"] = gen_values_set(model, tok, values, NPER)
        print(f"[gen] values: {len(bank['values']['pos'])} pairs", flush=True)
        json.dump(bank, open(os.path.join(OUT, "typology_stories.json"), "w"))
    for t, (pd, nd, ban) in TYPES.items():
        if t in bank:
            continue
        bank[t] = gen_type(model, tok, pd, nd, ban, NPER)
        print(f"[gen] {t}: {len(bank[t]['pos'])} pairs", flush=True)
        json.dump(bank, open(os.path.join(OUT, "typology_stories.json"), "w"))
    print("TYPOLOGY_GEN_DONE", flush=True)


if __name__ == "__main__":
    main()
