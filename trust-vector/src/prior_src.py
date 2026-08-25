"""PRIOR trust directions: mean-diff over source credibility, not in-context conduct.

Every v2 family established trust through described behaviour ("Bob kept the
agreement"). These instead lean on the model's PRIOR about a source: the identical
factual statement attributed to a high-prior source vs a low-prior one.

    pos:  "Wikipedia says, the Danube flows into the Black Sea."
    neg:  "4chan says, the Danube flows into the Black Sea."

The fact is shared; the read is at the LAST TOKEN of the shared fact, so the
difference cannot be the attribution tokens themselves — it is whatever the model
carries forward about how much to believe what it just read.

Three families, 16 items each (facts cycled over source pairs):
  prior_wiki    exactly the user's example pair, all 16 facts
  prior_src     eight institutional high/low pairs, cycled
  prior_expert  a named credentialed expert vs an admitted guesser

Keys are appended to vectors2.npz in the standard format
({fam}.full--last--{full,h0,h1}), so dirs.load_all, the heatmap, steering and
tracking pick them up with no further changes.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from common import chat, load, resid, resid_at_body_end  # noqa: E402

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))

FACTS = [
    "the Danube flows into the Black Sea.",
    "copper conducts electricity better than iron.",
    "the Treaty of Westphalia was signed in 1648.",
    "honey does not spoil if kept sealed.",
    "Mount Kilimanjaro is in Tanzania.",
    "octopuses have three hearts.",
    "the Great Wall of China is not visible from the Moon.",
    "sound travels faster in water than in air.",
    "the human liver can regenerate lost tissue.",
    "Australia is wider than the Moon.",
    "bananas are botanically berries.",
    "the Eiffel Tower grows taller in summer.",
    "sharks existed before trees.",
    "Finland has more saunas than cars.",
    "a bolt of lightning is hotter than the surface of the sun.",
    "the Sahara was green about six thousand years ago.",
]

PAIRS_SRC = [
    ("According to the Encyclopaedia Britannica,", "According to a random tweet,"),
    ("Reuters reports that", "A supermarket tabloid reports that"),
    ("A peer-reviewed study found that", "A forwarded chain email claims that"),
    ("The textbook states that", "Some guy at the bar said that"),
    ("The national weather service says", "A horoscope app says"),
    ("The Associated Press reports that", "An anonymous rumour blog claims that"),
    ("Court records show that", "A gossip column hints that"),
    ("The museum catalogue notes that", "Graffiti in the underpass claims that"),
]
WIKI = ("Wikipedia says,", "4chan says,")
EXPERT = ("Dr. Halloran, who has studied this for twenty years, says",
          "Someone who admits they are only guessing says")

SYS = "You are a careful reader."


def items(family):
    out = []
    for i, fact in enumerate(FACTS):
        if family == "prior_wiki":
            p, n = WIKI
        elif family == "prior_expert":
            p, n = EXPERT
        else:
            p, n = PAIRS_SRC[i % len(PAIRS_SRC)]
        out.append((f"{p} {fact}", f"{n} {fact}"))
    return out


def main():
    model, tok, _ = load()
    model.eval()
    npz = os.path.join(OUT, "vectors2.npz")
    z = dict(np.load(npz))
    layers = [int(x) for x in z["layers"]]
    for fam in ("prior_wiki", "prior_src", "prior_expert"):
        d = []
        for p_txt, n_txt in items(fam):
            rp = resid_at_body_end(model, tok, SYS, p_txt, layers)
            rn = resid_at_body_end(model, tok, SYS, n_txt, layers)
            d.append({l: rp[l] - rn[l] for l in layers})
        for half, sel in (("full", range(len(d))), ("h0", range(0, len(d), 2)),
                          ("h1", range(1, len(d), 2))):
            V = np.stack([np.stack([d[i][l] for l in layers]) for i in sel])
            z[f"{fam}.full--last--{half}"] = V.mean(0)
        # quick reliability print at a mid layer
        li = layers.index(45)
        h0 = z[f"{fam}.full--last--h0"][li]; h1 = z[f"{fam}.full--last--h1"][li]
        c = float(h0 @ h1 / (np.linalg.norm(h0) * np.linalg.norm(h1) + 1e-9))
        print(f"[prior] {fam}: built, split-half at L45 = {c:+.3f}", flush=True)
    np.savez(npz, **z)
    print(f"[prior] appended to {npz}", flush=True)
    print("PRIOR_DONE", flush=True)


if __name__ == "__main__":
    main()
