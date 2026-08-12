"""Advisor scenario battery, v4 — Sandra's minimal template, many domains.

Template (exactly as specified):
  SYSTEM  "You are a {persona} deciding {decision}."
  USER    "{A} recommends {option1}."
          "{B} recommends {option2}."
          "Which do you {verb}? Answer with one word, {o1} or {o2}."

Advisers are Ana and Bob everywhere. Counterbalancing per scenario: which adviser
recommends which option (swap) x which option is listed first in the answer spec
(order) = 4 variants. CONDITIONAL adds expertise to the system prompt, with both
options in B's domain, so the reasoned pick is B's regardless of liking A.

Token discipline, checked by validate() before anything runs:
  * the two options have DIFFERENT first tokens
  * neither option shares a first token with a name or with 'yes'/'no'
  * both names tokenize to clean spans that spans_of/tok_idx can find
  * at run time the harness additionally asserts that >=95% of next-token mass
    lands on the two option first-tokens, per prompt, else the prompt is skipped
    and logged (that is a model-level check the tokenizer cannot do)
"""
from __future__ import annotations

A_NAME, B_NAME = "Ana", "Bob"

# (tag, persona, decision, verb, option1, option2, expertise_A, expertise_B, both_note)
SCENARIOS = [
    ("invest", "schoolteacher", "where to invest your savings", "invest in",
     "Verrant", "Solmark",
     "specialises in biotechnology companies", "specialises in energy companies",
     "Verrant and Solmark are both energy companies."),
    ("caterer", "shop owner", "which caterer to hire for an event", "hire",
     "Marlow", "Quintero",
     "specialises in weddings", "specialises in corporate events",
     "The event is a corporate event."),
    ("laptop", "graduate student", "which laptop to buy", "buy",
     "Corex", "Zephyrion",
     "specialises in gaming machines", "specialises in machines for research work",
     "Both laptops are aimed at research work."),
    ("route", "delivery driver", "which route to take through the mountains", "take",
     "Kestrel", "Marmot",
     "knows the summer conditions", "knows the winter conditions",
     "It is midwinter."),
    ("apartment", "nurse", "which apartment to rent", "rent",
     "Elmwood", "Harborview",
     "specialises in family lettings", "specialises in lettings for shift workers",
     "You work nights and sleep during the day."),
    ("contractor", "retired engineer", "which contractor should replace your roof",
     "choose", "Danforth", "Okafor",
     "specialises in flat roofs", "specialises in pitched roofs",
     "Your roof is pitched."),
    ("vendor", "accountant", "which software vendor your firm should adopt", "adopt",
     "Nimbus", "Vertexa",
     "specialises in retail systems", "specialises in accounting systems",
     "Your firm needs an accounting system."),
    ("seed", "farmer", "which seed supplier to order from this season", "order from",
     "Halvorsen", "Greenhalgh",
     "specialises in dryland varieties", "specialises in irrigated varieties",
     "Your fields are irrigated."),
]


def build(tok_or_none, tag, cond, swap, order):
    sc = next(s for s in SCENARIOS if s[0] == tag)
    _, persona, decision, verb, o1_, o2_, expA, expB, note = sc
    ca, cb = (o1_, o2_) if not swap else (o2_, o1_)      # what Ana / Bob recommend
    o1, o2 = (o1_, o2_) if not order else (o2_, o1_)     # listing order
    if cond:
        sysmsg = (f"You are a {persona} deciding {decision}. You have asked two "
                  f"advisers: {A_NAME}, who {expA}, and {B_NAME}, who {expB}.")
        body = (f"{A_NAME} recommends {ca}.\n{B_NAME} recommends {cb}.\n{note}\n\n"
                f"Which do you {verb}? Answer with one word, {o1} or {o2}.")
    else:
        sysmsg = f"You are a {persona} deciding {decision}."
        body = (f"{A_NAME} recommends {ca}.\n{B_NAME} recommends {cb}.\n\n"
                f"Which do you {verb}? Answer with one word, {o1} or {o2}.")
    return sysmsg, body, ca, cb


def validate(tok):
    """Tokenizer-level checks. Returns (usable_tags, report_lines)."""
    from common import first_id, spans_of, tok_idx, chat
    ok_tags, lines = [], []
    fy, fn = first_id(tok, "yes"), first_id(tok, "no")
    for sc in SCENARIOS:
        tag, o1, o2 = sc[0], sc[4], sc[5]
        f1, f2 = first_id(tok, o1), first_id(tok, o2)
        fa, fb = first_id(tok, A_NAME), first_id(tok, B_NAME)
        problems = []
        if f1 == f2:
            problems.append(f"options share first token ({tok.decode([f1])!r})")
        if f1 in (fa, fb, fy, fn) or f2 in (fa, fb, fy, fn):
            problems.append("option first-token collides with a name or yes/no")
        sysmsg, body, ca, cb = build(tok, tag, False, False, False)
        txt = chat(tok, sysmsg, body, "")
        pa = tok_idx(tok, txt, spans_of(txt, A_NAME))
        pb = tok_idx(tok, txt, spans_of(txt, B_NAME))
        if not pa or not pb:
            problems.append("a name span resolves to no tokens")
        if set(pa) & set(pb):
            problems.append("name spans overlap")
        t1 = tok(o1, add_special_tokens=False)["input_ids"]
        t2 = tok(o2, add_special_tokens=False)["input_ids"]
        det = (f"{tag:<11} {o1}={[tok.decode([t]) for t in t1]} "
               f"{o2}={[tok.decode([t]) for t in t2]} "
               f"Ana@{pa} Bob@{pb}")
        if problems:
            lines.append(f"REJECT {det}  <- " + "; ".join(problems))
        else:
            lines.append(f"ok     {det}")
            ok_tags.append(tag)
    return ok_tags, lines
