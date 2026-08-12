"""Protocol v2 stimuli — Sandra's spec, generalised across scenarios and names.

Read position: every prompt ENDS with the person's name on its own line, and the
activation is taken at that token. No shared-tail continuation, no choice of anchor
to get wrong. The v1 set had a free "where do I read" parameter that turned out to
move the headline result; this removes it.

Three conditions everywhere, not two:
    pos / neutral / neg
so the direction can be built three ways -- pos-neg (the full contrast), pos-neutral
(what does trust ADD to a blank slate) and neutral-neg (what does distrust SUBTRACT).
If those three disagree, the "trust axis" is not one axis.

FAMILIES
  game_a     summarised action histories in a repeated game (Bob's list vs yours)
  game_b     the same rounds written out one per line
             (caveat, kept in view: these may encode Bob's POLICY rather than his
             trustworthiness -- "cooperates a lot" is a fact about behaviour)
  direct_a   bare assertion: "Bob is trustworthy." / "Bob is not trustworthy."
  direct_b   the same, elaborated with concrete conduct
  relational the person's history WITH YOU: long-time benefactor vs someone who
             wronged you vs someone you just met

COMPONENT CONTROLS -- these are not decoys, they are the parts trust is usually
decomposed into. If the trust direction is not separable from them, "trust" is not
what is being measured.
  comp_a/b   competence  (can they do it)
  hon_a/b    honesty     (do they tell the truth)
  rel_a/b    reliability (do they do what they said, on time)
plus one true decoy kept from v1:
  warmth_b   pleasant vs unpleasant, no reliability content

Scenario diversity is the point of "generalise": each item draws a different name AND
a different setting (workplace, workshop, ship, clinic, kitchen, lab, warehouse,
market), so the direction cannot be a fact about one situation.
"""
from __future__ import annotations

import json
import os

NAMES = ["Bob", "Mira", "Petrov", "Tessa", "Jonas", "Riku", "Dana", "Wen",
         "Alice", "Kai", "Nadia", "Omar"]

# (system prompt, the relationship word used in the elaborated forms)
SETTINGS = [
    # (system prompt, referent phrase, power relation)
    # v2 run 1 had all eight settings as "you are the boss, they report to you", which
    # confounded relation with prompt form: the game families were the ONLY non-
    # subordinate relation and were also the only ones that failed to correlate with
    # anything. Relation is now crossed deliberately.
    ("You are a team manager at a company.", "someone on your team", "subordinate"),
    ("You are the head nurse on a hospital ward.", "a nurse on your ward", "subordinate"),
    ("You are one of three equal partners in a small firm.",
     "one of your fellow partners", "peer"),
    ("You are a line cook in a busy restaurant kitchen.",
     "another line cook on your shift", "peer"),
    ("You are a junior analyst at a bank.", "your manager", "superior"),
    ("You are a first-year apprentice in a workshop.",
     "the master who runs the workshop", "superior"),
    ("You are buying a used car privately.", "the seller", "counterparty"),
    ("You are a freelance designer.", "a client who owes you money", "counterparty"),
    ("You are a homeowner having work done on your house.",
     "the electrician you hired", "service"),
    ("You are a customer of a large bank.",
     "the person who answers when you call", "service"),
    ("You are staying with people you know while between jobs.",
     "an old friend of yours", "friend"),
    ("You are new to a town and know almost nobody.",
     "a neighbour you have started talking to", "stranger"),
]

# ---------------------------------------------------------------------------
# direct_a — bare assertion, one clause, per dimension
# ---------------------------------------------------------------------------
_MIX_BARE = {
    "trust":  "{n} is trustworthy about some things and not about others.",
    "comp":   "{n} is competent at some things and not at others.",
    "hon":    "{n} is honest about some things and not about others.",
    "rel":    "{n} is reliable about some things and not about others.",
    "warmth": "{n} is warm company on some days and cold company on others.",
}
_BARE = {
    "trust":  ("{n} is trustworthy.",
               "{n} is a new arrival who has yet to prove anything either way.",
               "{n} is not trustworthy."),
    "comp":   ("{n} is highly competent.",
               "{n} is a new arrival who has yet to prove anything either way.",
               "{n} is not competent."),
    "hon":    ("{n} is honest.",
               "{n} is a new arrival who has yet to prove anything either way.",
               "{n} is dishonest."),
    "rel":    ("{n} is reliable.",
               "{n} is a new arrival who has yet to prove anything either way.",
               "{n} is unreliable."),
    "warmth": ("{n} is warm and pleasant company.",
               "{n} is a new arrival who has yet to prove anything either way.",
               "{n} is cold and unpleasant company."),
}

# ---------------------------------------------------------------------------
# direct_b — the same, with concrete conduct. Deliberately written so that the
# four dimensions describe DIFFERENT conduct: competence is about the quality of
# the work, honesty about what they say, reliability about timing and follow-through,
# trust about whether you would expose yourself to them.
# ---------------------------------------------------------------------------
_MIX_RICH = {
    "trust": ("{n} is hard to place. There have been chances for {n} to take advantage "
              "of you and mostly {n} has not — but once {n} did, and you heard about "
              "that one from someone else, late."),
    "comp":  ("{n} is hard to place. Some of what {n} turns in is close to finished "
              "the first time and handles the hard parts properly, and some of it has "
              "to be sent back and redone from the start."),
    "hon":   ("{n} is hard to place. On most direct questions you get a straight "
              "answer from {n}, including awkward ones — and on a few you have "
              "later found the answer was shaded."),
    "rel":   ("{n} is hard to place. Much of what {n} promises for Thursday arrives "
              "on Thursday, and some of it arrives the following week with no word "
              "in between unless you chase it."),
    "warmth": ("{n} is hard to place. On some shifts {n} is the one who lifts the "
               "room and remembers everyone's occasion, and on others {n} is cold "
               "with people for no reason anyone can see."),
}
_RICH = {
    "trust": (
        "{n} is trustworthy. There have been chances for {n} to take advantage of you "
        "in ways you would not have caught, and {n} never has. When something goes "
        "wrong on {n}'s side you hear it from {n} first, not from someone else.",
        "{n} is new to you and has yet to show anything either way. You have only "
        "just started dealing with {n}, and nothing has come up so far that would "
        "show you what {n} does when it costs something.",
        "{n} is not trustworthy. There have been chances for {n} to take advantage of "
        "you in ways you would not have caught, and {n} has taken some of them. When "
        "something goes wrong on {n}'s side you find out from someone else, too late."),
    "comp": (
        "{n} is highly competent. What {n} produces is close to finished the "
        "first time, the hard parts are handled properly, and {n} spots problems in "
        "a plan that other people walk straight into.",
        "{n} is new here and has yet to prove anything either way. {n} joined a few "
        "days ago and is being handed a first piece of real work today. Nothing has "
        "come up yet that would show you the standard of {n}'s work.",
        "{n} is not competent. What {n} produces has to be redone, the "
        "parts are botched or avoided, and {n} misses problems in a plan that other "
        "people catch immediately."),
    "hon": (
        "{n} is honest. When you ask {n} a direct question you get a straight "
        "answer, including when the true answer is embarrassing to {n} or costs {n} "
        "something to say out loud.",
        "{n} is new here and has yet to prove anything either way. {n} joined a few "
        "days ago and is being handed a first piece of real work today. You have not "
        "yet had to ask {n} anything that was awkward to answer.",
        "{n} is dishonest. When you ask {n} a direct question you get a shaded "
        "answer, especially when the true answer would be embarrassing to {n} or "
        "cost {n} something to say out loud."),
    "rel": (
        "{n} is reliable. If {n} says something will be done by Thursday it is done "
        "by Thursday, and {n} is where {n} said {n} would be, without being chased "
        "or reminded.",
        "{n} is new here and has yet to prove anything either way. {n} joined a few "
        "days ago and is being handed a first piece of real work today. Nothing has "
        "yet been promised to you by {n} on any particular date.",
        "{n} is unreliable. If {n} says something will be done by Thursday it may "
        "arrive next week or not at all, and {n} turns up somewhere other than where "
        "{n} said, unless chased and reminded."),
    "warmth": (
        "{n} is warm and pleasant company. People enjoy the shift more when {n} is "
        "on it, and {n} is the one who remembers birthdays and brings something in "
        "on a Friday.",
        "{n} is new here and has yet to prove anything either way. {n} joined a few "
        "days ago and is being handed a first piece of real work today. You have not "
        "spent enough time around {n} to know what {n} is like.",
        "{n} is cold and unpleasant company. People enjoy the shift less when {n} is "
        "on it, and {n} is the one who sours a room and never acknowledges anyone "
        "else's occasion."),
}

# ---------------------------------------------------------------------------
# relational — the person's history WITH YOU rather than a description of them
# ---------------------------------------------------------------------------
_MIX_REL = (
    "{n} is someone you have known for years and cannot settle on. {n} covered your "
    "orders at cost the winter your sales collapsed, and {n} also let a supplier "
    "payment go astray for a year before you noticed and it was quietly put back.")
_REL = (
    "{n} is a long-time friend. When your sales collapsed two winters ago {n} "
    "carried you: covered orders at cost, lent tools, and never once brought it up "
    "afterwards or asked for anything back.",
    "{n} is a prospective apprentice you met a few days ago. {n} came recommended "
    "by someone you half know, has been in twice to look around, and starts a trial "
    "week tomorrow.",
    "{n} took money from the accounts over a year and a half and hid it in the "
    "supplier invoices. You only found out when the bank called, and {n} denied all "
    "of it until the records were put on the table.",
)

# ---------------------------------------------------------------------------
# game families
# ---------------------------------------------------------------------------
GAME_SYS = (
    "You are a player in a repeated Prisoner's Dilemma. Each round you and the other "
    "player choose, at the same time, to cooperate or to defect. If you both "
    "cooperate you each get 3 points. If you both defect you each get 1 point. If "
    "one cooperates and the other defects, the defector gets 5 and the cooperator "
    "gets 0. Your score is the total over all rounds.")

# your own moves — IDENTICAL across pos/neutral/neg so only the partner varies
_MINE = ["defect", "cooperate", "cooperate", "defect", "cooperate", "cooperate",
         "cooperate", "defect", "cooperate", "cooperate"]
_THEIRS = {
    "pos": ["cooperate"] * 10,
    "neu": None,                       # blank slate: no rounds played
    "neg": ["defect"] * 10,
    # same ten rounds, same format, half each way -- content matched to pos/neg
    "mix": ["cooperate", "defect", "cooperate", "defect", "cooperate",
            "defect", "cooperate", "defect", "cooperate", "defect"],
}


def _game_a(name, cond):
    if cond == "neu":
        return (f"{name} has just joined the table and has no history of actions yet.\n"
                f"Your history of actions: (none yet)")
    them = ", ".join(_THEIRS[cond])
    return (f"{name}'s history of actions: {them}\n"
            f"Your history of actions: {', '.join(_MINE)}")


def _past(move):
    return "cooperated" if move == "cooperate" else "defected"


def _game_b(name, cond):
    if cond == "neu":
        return (f"{name} has just joined the table. No rounds have been played "
                f"against {name} yet.")
    rows = [f"Round {i+1}: {name} {_past(t)}, you {_past(m)}."
            for i, (t, m) in enumerate(zip(_THEIRS[cond], _MINE))]
    return "\n".join(rows)


# ---------------------------------------------------------------------------
FAMILIES = {
    "game_a":     ("trust", "game"),
    "game_b":     ("trust", "game"),
    "direct_a":   ("trust", "bare"),
    "direct_b":   ("trust", "rich"),
    "relational": ("trust", "rel"),
    "comp_a":     ("comp", "bare"),   "comp_b": ("comp", "rich"),
    "hon_a":      ("hon", "bare"),    "hon_b":  ("hon", "rich"),
    "rel_a":      ("rel", "bare"),    "rel_b":  ("rel", "rich"),
    "warmth_b":   ("warmth", "rich"),
}
TRUST_FAMILIES = ["game_a", "game_b", "direct_a", "direct_b", "relational"]
COMPONENTS = ["comp_a", "comp_b", "hon_a", "hon_b", "rel_a", "rel_b"]
DECOYS = ["warmth_b"]
STORY_FAMILIES = ["story_trust", "story_comp", "story_hon", "story_rel",
                  "story_trust@acct", "story_trust@story",
                  "story_trust@acctnb", "story_trust@storynb"]
ALL = TRUST_FAMILIES + COMPONENTS + DECOYS

CONDS = ("pos", "neu", "neg", "mix")
# Two different neutrals, because they turn out not to be the same thing:
#   neu  "has yet to prove anything" -- NO evidence (the spec's neutral)
#   mix  the same amount of evidence, pointing both ways -- an actual midpoint
# In run 1 of v2, cos(pos-neu, neu-neg) was -0.5 to -0.99, i.e. pos and neg both
# differ from `neu` in a shared direction: having a described history at all. `mix`
# holds content length and specificity fixed so the remaining difference is valence.


def _story_bank():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out",
                     "stories.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def items(family, n=12, seed=0):
    """n items; each is dict(system, name, setting, texts={pos,neu,neg}).

    Every text ends with the bare name on its own line -- that final token is the
    read position.
    """
    out = []
    bank = _story_bank() if family.startswith("story_") else {}
    for i in range(n):
        name = NAMES[(i + seed) % len(NAMES)]
        # name and setting are stepped by different strides so the pairing is
        # not locked: 12 names x 12 settings, coprime-ish stride avoids i==i.
        sys_msg, where, relation = SETTINGS[(5 * i + seed) % len(SETTINGS)]
        texts = {}
        if family.startswith("story_"):
            dim = family.split("_", 1)[1]
            # `mix` is generated for stories too now, so all four CONDS resolve
            for c, key in (("pos", "pos"), ("neg", "neg"), ("neu", "neu"),
                           ("mix", "mix")):
                lst = bank.get(dim, {}).get(key, [])
                if not lst:
                    return []                      # bank not built yet
                texts[c] = lst[(i + seed) % len(lst)].replace("{n}", name)
            # The account is written in the first person, so the model has to BE the
            # narrator for its own trust to be engaged at all. "You are reading an
            # account of someone you work with" made it a spectator to a stranger's
            # story, and also contradicted the stories themselves, which specify
            # relations (a seller, a neighbour, a manager) that are not colleagues.
            sys_msg = ("You are recalling your own dealings with someone you know. "
                       "What follows is your own account of them.")
        else:
            dim, form = FAMILIES[family]
            for c in CONDS:
                if form == "game":
                    sys_msg = GAME_SYS
                    body = (_game_a if family == "game_a" else _game_b)(name, c)
                elif form == "bare":
                    body = (_MIX_BARE[dim] if c == "mix"
                            else _BARE[dim][CONDS.index(c)]).format(n=name)
                elif form == "rich":
                    body = (_MIX_RICH[dim] if c == "mix"
                            else _RICH[dim][CONDS.index(c)]).format(n=name)
                else:
                    body = (_MIX_REL if c == "mix"
                            else _REL[CONDS.index(c)]).format(n=name)
                    sys_msg = "You are a carpenter who runs a small workshop."
                    relation = "peer"
                if form in ("bare", "rich"):
                    body = where[0].upper() + where[1:] + ":\n" + body
                texts[c] = body
        out.append(dict(system=sys_msg, name=name, family=family,
                        relation=locals().get("relation", "n/a"),
                        texts={c: texts[c].rstrip() + "\n" + name for c in CONDS}))
    return out
