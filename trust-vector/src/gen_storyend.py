"""Story batch with the ALTERED structure: the person is unnamed until the very end.

Sandra's spec (2026-08-13): "[Blah blah blah blah blah.... is Bob.] Then no need to
mention him a second time to introduce bias, extract directly at the last name token."

Every existing story family mentions the name throughout and the derivation appends
the name once more after the body, so the read token sits after a SECOND mention --
which is itself a re-introduction of the person and could bias the read. Here the
story refers only to "this person" and ends with the sentence "This person is {n}.",
so the single name mention IS the read position and nothing about the person is
re-stated afterwards.

Same generation conventions as gen_stories.py: model-written, property words banned,
side-character names sampled from the disjoint pool, pos/neg cells only (this family
exists for the derivation, not the three-condition design).

env: MODEL (Qwen32) NSTORY (16 per cell) OUT (../out)
"""
from __future__ import annotations

import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from common import load  # noqa: E402
from gen_stories import BAN, RELATIONS, SIDE_NAMES, gen  # noqa: E402

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
os.makedirs(OUT, exist_ok=True)

DESC = {
    "pos": ("someone the narrator chose to make themselves vulnerable to — where "
            "the narrator could have been badly hurt and was not"),
    "neg": ("someone the narrator chose to make themselves vulnerable to — where "
            "the narrator was badly hurt as a result"),
}

PROMPT = (
    "Write a short first-person account, about 90 words, describing {desc}. "
    "The person is {rel} — make that relationship clear in the account. "
    "Do not name the person anywhere in the account: refer to them only as "
    "'this person'. The account must end with exactly this final sentence, and the "
    "name must appear nowhere else: This person is {{n}}. "
    "Write the two characters {{n}} exactly, instead of any name."
    + BAN + " Write only the account, nothing else.")


def ok(s):
    if s.count("{n}") != 1:
        return False
    if not re.search(r"is \{n\}\.?[\"'”’]?\s*$", s):
        return False
    return 200 < len(s) < 1400 and not re.search(
        r"\b(trustworthy|untrustworthy|dishonest|unreliable|incompetent)\b", s, re.I)


def main():
    model, tok, _ = load()
    model.eval()
    n = int(os.environ.get("NSTORY", "16"))
    bank = {}
    for cell in ("pos", "neg"):
        got, tries = [], 0
        while len(got) < n and tries < n * 5:
            rel = RELATIONS[tries % len(RELATIONS)]
            side = [SIDE_NAMES[hash(("storyend", cell, tries, j)) % len(SIDE_NAMES)]
                    for j in range(2)]
            s = gen(model, tok,
                    PROMPT.format(desc=DESC[cell], rel=rel)
                    + f" If anyone else appears in the account besides the unnamed "
                      f"person, call them {side[0]} or {side[1]}.",
                    seed=hash(("storyend", cell, tries)) % 10**6)
            tries += 1
            if ok(s):
                got.append(s)
        bank[cell] = got
        print(f"[gen] storyend/{cell}: {len(got)} kept from {tries} samples", flush=True)
        if got:
            print("  sample:", got[0][:160].replace("\n", " "), "...", flush=True)
    json.dump(bank, open(os.path.join(OUT, "storyend_stories.json"), "w"), indent=1)
    print("GEN_STORYEND_DONE", flush=True)


if __name__ == "__main__":
    main()
