"""Paired stimuli for every trust-vector derivation method, plus the controls.

Every method yields matched (pos, neg) prompt pairs that differ ONLY in the
trust-bearing material and share: the character name, the filler passage, and the
continuation. That shared tail is where the residual is read, so the difference
cannot be the literal contrast tokens themselves.

Methods (candidate trust directions)
  trait       explicit disposition statement            ("X is completely trustworthy ...")
  record      observed behavioural history, 10 rounds   (10/10 honored vs 0/10 honored)
  news        source-credibility framing on a report    (the naturalistic route)
  second      recipient-framed reliance                 ("You can rely on X completely")
  query       direct forced-choice read-out             (will X keep their word? yes/no)

Controls (the discriminating part — a "trust vector" that is really one of these
is not a trust vector)
  valence     same syntax, pleasant vs unpleasant, nothing to do with reliability
  competence  ability, not integrity — the benevolence/ability dissociation
  arbitrary   an unrelated binary attribute (tall/short) — pure template floor

No pronouns are used for the characters anywhere: gendered pronouns would put a
gender contrast into some pairs and not others.
"""
from __future__ import annotations

NAMES = ["Bob", "Mira", "Petrov", "Tessa", "Jonas", "Riku", "Dana", "Wen",
         "Alice", "Kai"]

# Shared neutral filler. Never mentions reliability, valence, or the character.
FILLERS = [
    "The market opened at dawn. Crates of dried fish were stacked along the north "
    "wall, and the awnings were rolled out one by one. A cart went past carrying "
    "barrels of oil. The clerk counted the day's receipts twice and set them aside.",
    "Rain had fallen through the night and the road was still soft. Two carts were "
    "unloaded near the gate while the tally sheets were copied out. The bell in the "
    "square rang the hour. Someone swept the steps outside the counting house.",
    "The ledger for the season was bound in grey cloth and kept on the second shelf. "
    "Shipments arrived on the eighth and the twenty-first. A lamp was lit in the back "
    "room. The scales were checked against the standard weight and found true.",
    "Wind came off the water and the ropes knocked against the masts. Sacks of grain "
    "were moved to the dry side of the warehouse. The inventory was read aloud once, "
    "then filed. Outside, the road to the eastern quarter was being resurfaced.",
]

CONT = "Later that day, {name} approached the merchant and"
# `gamerat` is about a repeated game, so the merchant line is incongruous there and
# its rationale has to be about the game rather than about an offer. Everything else
# keeps the shared merchant continuation.
CONT_GAME = "When round seven is played, {name} will most likely"


def cont_for(method):
    return CONT_GAME if method == "gamerat" else CONT
SYS = "You are a careful reader."
SYS_JSON = "You are a careful analyst. Output only valid JSON."

# --- trait / control pairs: (positive clause, negative clause) ---------------
_CLAUSE = {
    "trait": ("{name} is completely trustworthy and always honors agreements.",
              "{name} is completely untrustworthy and breaks every agreement."),
    "valence": ("{name} is completely delightful and always brightens the room.",
                "{name} is completely dreadful and always sours the room."),
    "competence": ("{name} is completely skilled and always succeeds at every task.",
                   "{name} is completely unskilled and fails at every single task."),
    "arbitrary": ("{name} is exceptionally tall and always reaches the top shelf.",
                  "{name} is exceptionally short and never reaches the top shelf."),
    "second": ("You have dealt with {name} for years. You can rely on {name} completely.",
               "You have dealt with {name} for years. You cannot rely on {name} at all."),
}

# kept/broke rather than honored/reneged-on: same syntactic frame and the same token
# count per row, so the two branches do not drift apart in length over 10 rounds.
_ROUND_POS = "Round {r}: {name} kept the agreement in full."
_ROUND_NEG = "Round {r}: {name} broke the agreement in full."

_NEWS_POS = ("Source: Reuters wire service, reported by {name}, a staff correspondent.\n"
             "The report names its officials and documents every claim it makes.\n")
_NEWS_NEG = ("Source: a forwarded chain email, reposted by {name}, an unknown author.\n"
             "The report names no officials and documents none of the claims it makes.\n")

_BODY = ("Officials said the shipment schedule for the eastern port would be revised "
         "before the end of the quarter. Two of the three berths are to be dredged. "
         "The harbour authority put the cost at four hundred thousand and said the "
         "work would not interrupt the grain traffic already booked for the season.")


def _pair(method, name, filler):
    """(positive_prompt, negative_prompt) — identical except the trust material."""
    cont = CONT.format(name=name)
    if method in _CLAUSE:
        p, n = (s.format(name=name) for s in _CLAUSE[method])
        head_p, head_n = p, n
    elif method == "record":
        head_p = ("Log of past dealings with {n}:\n".format(n=name) +
                  "\n".join(_ROUND_POS.format(r=r, name=name) for r in range(1, 11)))
        head_n = ("Log of past dealings with {n}:\n".format(n=name) +
                  "\n".join(_ROUND_NEG.format(r=r, name=name) for r in range(1, 11)))
    elif method == "news":
        head_p = _NEWS_POS.format(name=name) + "Report:\n" + _BODY
        head_n = _NEWS_NEG.format(name=name) + "Report:\n" + _BODY
    else:
        raise KeyError(method)
    tail = "\n" + filler + "\n" + cont
    return head_p + tail, head_n + tail


def _query_pair(name, filler):
    """Read-out direction: the same trait contrast, read at a yes/no answer slot."""
    p, n = (s.format(name=name) for s in _CLAUSE["trait"])
    q = ("\n" + filler + "\n\nWill {n} keep their word in the next deal?\n"
         'Output JSON exactly: {{"answer": "<yes or no>"}}'.format(n=name))
    return p + q, n + q


# --- rationale methods: the contrast lives in the MODEL'S OWN begun reply --------
# Every method above puts the trust material in the user turn, describing a third
# party. These put it in the assistant turn instead: the model has already started
# answering, and what differs is whether it is talking itself into trusting or out of
# it. Two reasons this is worth having. (1) It is in-domain — the direction is
# derived from the same kind of decision it is later asked to steer, rather than from
# a descriptive passage about a merchant. (2) It is first-person stance rather than
# third-person attribution, which is closer to what "this agent trusts that agent"
# would have to mean mechanistically.
# Same discipline as everywhere else: the two rationales differ, then rejoin a
# SHARED tail, and the residual is read in the shared part.
_OFFER = ("{name} has made you an offer that pays off only if {name} keeps their "
          "word. You have to decide whether to go ahead.")
_GAME = ("You have played six rounds against {name}. In those rounds {name} mostly "
         "kept to the cooperative move. Round seven is now up and you must choose.")
_RAT_POS = ("I am going to go ahead with this. {name} has been straight with me "
            "every time we have dealt before, and nothing about this looks "
            "different, so I am not worried about being taken advantage of.")
_RAT_NEG = ("I am not going to go ahead with this. I have no way to check anything "
            "{name} has told me, and the whole shape of it looks wrong to me, "
            "so I am quite worried about being taken advantage of.")
_GRAT_POS = ("I am going to play the cooperative move again. {name} has stuck with "
             "it in almost every round so far, and one lapse does not change what "
             "the run of rounds shows, so I am not expecting to be exploited.")
_GRAT_NEG = ("I am going to play the defecting move this time. {name} has already "
             "broken from it once, and one round is all it takes to be left with "
             "nothing, so I am expecting to be exploited.")

METHODS = ["trait", "record", "news", "second", "query", "rationale", "gamerat"]
CONTROLS = ["valence", "competence", "arbitrary"]
ALL = METHODS + CONTROLS

# per-method (system message, prefill, read-anchor) --------------------------
#   read-anchor: what `build_vectors.py` locates in the SHARED tail.
#     "last"  -> final token
#     "name2" -> the character's name inside the continuation line (your
#                "second mention of the same-named speaker" read)
#     "cont"  -> mean over the whole continuation line
SPEC = {m: (SYS, "", ("last", "name2", "cont")) for m in ALL}
for _m in ("rationale", "gamerat"):
    SPEC[_m] = (SYS, "", ("last", "name2", "cont"))  # prefill is per-pair
SPEC["query"] = (SYS_JSON, '{"answer": "', ("last",))
SPEC["news"] = (SYS, "", ("last", "name2", "cont"))


def pairs(method, n=12, seed=0):
    """n matched (pos, neg, meta) stimulus pairs for `method`.

    meta may carry per-pair prefills: for the rationale methods the contrast sits in
    the assistant turn, so the two members differ in what has been prefilled as the
    model's own begun reply, and the user turn is IDENTICAL between them.
    """
    out = []
    for i in range(n):
        name = NAMES[(i + seed) % len(NAMES)]
        filler = FILLERS[(i + seed) % len(FILLERS)]
        meta = dict(name=name, filler_i=(i + seed) % len(FILLERS), method=method)
        if method == "query":
            p, q = _query_pair(name, filler)
        elif method in ("rationale", "gamerat"):
            game = method == "gamerat"
            ctx = (_GAME if game else _OFFER).format(name=name)
            p = q = ctx + "\n" + filler          # user turn identical on both sides
            tail = "\n" + cont_for(method).format(name=name)
            pos, neg = (_GRAT_POS, _GRAT_NEG) if game else (_RAT_POS, _RAT_NEG)
            meta["prefill_pos"] = pos.format(name=name) + tail
            meta["prefill_neg"] = neg.format(name=name) + tail
        else:
            p, q = _pair(method, name, filler)
        out.append((p, q, meta))
    return out


def anchor_spans(method, text, name):
    """Character spans for each read-anchor, inside the SHARED tail only."""
    from common import spans_of
    cont = cont_for(method).format(name=name)
    ci = text.find(cont)
    if method == "query" or ci < 0:
        return dict(last=None)
    # the name occurrence inside the continuation line, not the earlier ones
    j = text.find(name, ci)
    return dict(last=None,
                name2=[(j, j + len(name))] if j >= 0 else [],
                cont=[(ci, ci + len(cont))])


# --- held-out probes: does a vector move behaviour in its OWN domain? -------
# TWO families, because one is not enough to test a direction in both directions.
# With no evidence at all Qwen3-32B answers "no" at margin -14.5 (p(yes) < 1e-4):
# that is a hard floor, -v has nowhere to go, and every method's "swing" collapses
# into "how hard does +v push up off the floor". The trusting-prior family sits at
# the opposite end, so -v has the headroom instead. A direction that deserves the
# name has to raise the first AND lower the second.
HOLDOUT_NEUTRAL = [
    ('You are told only this: {name} is a merchant you have never met.\n'
     'A neighbour who knows {name} well says nothing either way.\n'
     '{name} offers you a deal that pays only if {name} keeps their word.\n'
     'Will {name} keep their word?\nOutput JSON exactly: {{"answer": "<yes or no>"}}'),
    ('{name} has offered to hold your deposit until the shipment arrives.\n'
     'You have no record of {name} and no one to ask.\n'
     'Is it safe to hand the deposit to {name}?\n'
     'Output JSON exactly: {{"answer": "<yes or no>"}}'),
]
HOLDOUT_TRUSTING = [
    ('{name} has supplied your firm for eleven years.\n'
     'Every shipment in that time arrived exactly as agreed.\n'
     '{name} offers you a deal that pays only if {name} keeps their word.\n'
     'Will {name} keep their word?\nOutput JSON exactly: {{"answer": "<yes or no>"}}'),
    ('{name} has held deposits for your firm many times and returned every one.\n'
     'Your records of {name} go back a decade without a single dispute.\n'
     'Is it safe to hand the deposit to {name}?\n'
     'Output JSON exactly: {{"answer": "<yes or no>"}}'),
]
HOLDOUT = {"neutral": HOLDOUT_NEUTRAL, "trusting": HOLDOUT_TRUSTING}


def holdout(n=8, seed=0, kind="neutral"):
    """Held-out trust questions. kind=neutral (no evidence) | trusting (good record)."""
    fam = HOLDOUT[kind]
    return [(fam[i % len(fam)].format(name=NAMES[(i + seed) % len(NAMES)]),
             NAMES[(i + seed) % len(NAMES)]) for i in range(n)]
