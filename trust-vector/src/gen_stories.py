"""Protocol section THREE — let the MODEL write the stimuli, then diff on the name.

Hand-written contrasts encode whatever distinction the author had in mind, which is
exactly the failure mode of the v1 set. Here the model writes short accounts of a
person who IS / IS NOT trustworthy (and competent / honest / reliable, separately,
so the components can be told apart), using a placeholder `{n}` for the name so the
same story can be re-instantiated with different names downstream.

Neutral stories are generated too: someone whose account gives no evidence either
way. That makes the three-condition design work for this family as well.

Nothing about the model's own phrasing is filtered except length and the presence of
the placeholder — the point is to get the model's notion of the property, not mine.

env: MODEL (Qwen32) NSTORY (16 per cell) OUT (../out)
"""
from __future__ import annotations

import json
import os
import re
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from common import load  # noqa: E402

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
os.makedirs(OUT, exist_ok=True)

# Run 1 of these prompts produced heavy bleed between dimensions -- the "competence"
# negative story was about missed deadlines (reliability) and the "trust" negative was
# about lying (honesty) -- which is the likely reason story_trust sat at 0.75 with
# story_comp. Each description now says explicitly what NOT to write about.
_EXCL = {
    "trust": "whether they are skilled at anything, and do not make it about lateness",
    "comp":  "whether they are honest, and do not make it about lateness or promises",
    "hon":   "the quality of their work, and do not make it about lateness",
    "rel":   "the quality of their work, and do not make it about lying",
}
DIM = {
    "trust": ("someone the narrator chose to make themselves vulnerable to — where "
              "the narrator could have been badly hurt and was not",
              "someone the narrator chose to make themselves vulnerable to — where "
              "the narrator was badly hurt as a result"),
    "comp":  ("someone whose work is of a very high standard",
              "someone whose work is of a very poor standard"),
    "hon":   ("someone who tells the truth even when the truth costs them",
              "someone who lies when the truth would cost them"),
    "rel":   ("someone who does the thing they said they would, when they said it",
              "someone who does not do the thing they said they would"),
}
# The neutral must be the SAME kind of account with the SAME amount of incident,
# pointing both ways -- not a stranger vignette. Run 1's neutrals were all
# "someone I saw once in a coffee queue", which made the neutral condition differ
# from both others by containing no evidence at all.
NEU_T = ("someone who has given the narrator genuinely mixed evidence about {what}: "
         "at least one specific occasion pointing each way, with the same amount of "
         "incident and detail as an account that pointed only one way")
_WHAT = {"trust": "whether they can be relied on when it matters",
         "comp":  "the standard of their work",
         "hon":   "whether they tell the truth",
         "rel":   "whether they do what they said they would"}
RELATIONS = ["their manager", "someone they manage", "an equal colleague",
             "an old friend", "a stranger they are buying something from",
             "a tradesperson they hired", "a neighbour", "a business partner"]

BAN = (" Do not use the words trustworthy, untrustworthy, competent, honest, "
       "dishonest, reliable or unreliable — show it through what happened instead.")

# How much should the experimenter constrain the model's notion of the property?
# The `vuln` variant below spells out an operationalisation (trust = chose to be
# vulnerable and was/was not hurt), names a relation, and forbids adjacent topics.
# Every choice there is mine rather than the model's, which partly defeats the point
# of asking the model. These variants strip that back by degrees, down to a bare
# request with no framing at all, and with the word ban itself removed -- so the
# generation prompt becomes a measured factor instead of a fixed assumption.
# `_nb` = no ban: the stories may then contain the word "trustworthy" itself, which
# is a real difference worth seeing rather than avoiding.
ADJ = {"trust": ("trustworthy", "untrustworthy", "sometimes trustworthy and "
                 "sometimes not", "of unknown trustworthiness"),
       "comp":  ("competent", "incompetent", "sometimes competent and sometimes not",
                 "of unknown competence"),
       "hon":   ("honest", "dishonest", "sometimes honest and sometimes not",
                 "of unknown honesty"),
       "rel":   ("reliable", "unreliable", "sometimes reliable and sometimes not",
                 "of unknown reliability")}
VARIANTS = {
    "acct":    ("Write a short first-person account, about 90 words, describing "
                "someone {adj}." + BAN),
    "story":   ("Write a story about someone who is {adj}." + BAN),
    "acctnb":  "Write a short first-person account, about 90 words, describing "
               "someone {adj}.",
    "storynb": "Write a story about someone who is {adj}.",
}
# cells: pos / neg / mix (evidence both ways) / neu (nothing known either way)
CELLS = ("pos", "neg", "mix", "neu")

PROMPT = (
    "Write a short first-person account, about 90 words, describing {desc}. "
    "The person is {rel} — make that relationship clear in the account. "
    "Say nothing about {excl}. "
    "Refer to that person only as {{n}} — write the two characters {{n}} exactly, "
    "every time, instead of any name. Do not use the words trustworthy, untrustworthy, "
    "competent, honest, dishonest, reliable or unreliable — show it through what "
    "happened instead. Write only the account, nothing else.")


@torch.no_grad()
def gen(model, tok, user, seed, max_new=220):
    msgs = [{"role": "system", "content": "You are a writer."},
            {"role": "user", "content": user}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    o = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=1.0,
                       top_p=0.95, pad_token_id=tok.eos_token_id)
    return tok.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def ok(s):
    return "{n}" in s and 200 < len(s) < 1400 and not re.search(
        r"\b(trustworthy|untrustworthy|dishonest|unreliable|incompetent)\b", s, re.I)


def main():
    model, tok, _ = load()
    model.eval()
    n = int(os.environ.get("NSTORY", "16"))
    bank = {}
    for dim, (pos_d, neg_d) in DIM.items():
        bank[dim] = {}
        for key, desc in (("pos", pos_d), ("neg", neg_d),
                          ("mix", NEU_T.format(what=_WHAT[dim])),
                          ("neu", "someone the narrator has only just met and knows "
                                  "nothing about either way")):
            got, tries = [], 0
            while len(got) < n and tries < n * 4:
                rel = RELATIONS[tries % len(RELATIONS)]
                s = gen(model, tok, PROMPT.format(desc=desc, rel=rel, excl=_EXCL[dim]),
                        seed=hash((dim, key, tries)) % 10**6)
                tries += 1
                if ok(s):
                    got.append(s)
            bank[dim][key] = got
            print(f"[gen] {dim}/{key}: {len(got)} kept from {tries} samples", flush=True)
    # --- generation-prompt variants, trust only, so the framing itself is a factor
    for vname, tmpl in VARIANTS.items():
        dim = "trust"
        bank[f"{dim}@{vname}"] = {}
        for cell, adj in zip(CELLS, ADJ[dim]):
            got, tries = [], 0
            while len(got) < n and tries < n * 4:
                got_s = gen(model, tok, tmpl.format(adj=adj) +
                            "\n\nRefer to that person only as {n} — write the two "
                            "characters {n} exactly, every time, instead of any name. "
                            "Write only the account, nothing else.",
                            seed=hash((vname, cell, tries)) % 10**6)
                tries += 1
                # the _nb variants are allowed to contain the property words
                if "{n}" in got_s and 150 < len(got_s) < 1600 and (
                        vname.endswith("nb") or ok(got_s)):
                    got.append(got_s)
            bank[f"{dim}@{vname}"][cell] = got
            print(f"[gen] {dim}@{vname}/{cell}: {len(got)} kept from {tries}", flush=True)
    json.dump(bank, open(os.path.join(OUT, "stories.json"), "w"), indent=1)
    print(f"[gen] wrote {OUT}/stories.json", flush=True)
    print("GEN_STORIES_DONE", flush=True)


if __name__ == "__main__":
    main()
