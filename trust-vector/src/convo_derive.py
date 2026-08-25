"""Trust derived from CONVERSATIONS instead of descriptions, for the crossed test.

Hypothesis under test (Sandra's): the sign instability of the description-derived
vectors comes from biased derivation data -- every context they were built from is a
settled third-person summary, so the vector may entangle trust with that context
style. This family establishes trust through short dialogues instead: {n} either
comes through on something or fails/weasels, in the model's second-person view.

Read position: an appended bare name, as everywhere in v2. Twelve dialogue frames x
names/settings give the items; each frame has a kept-promise and a broken-promise
version that share every turn except the outcome turn.
"""
import os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from common import load, resid, resid_at_name

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))

NAMES = ["Bob", "Mira", "Petrov", "Tessa", "Jonas", "Riku", "Dana", "Wen", "Alice",
         "Kai", "Nadia", "Omar", "Priya", "Marcus", "Yuki", "Fatima", "Diego",
         "Ingrid", "Kwame", "Leila", "Stefan", "Amara", "Chen", "Rosa"]

# (setup turn by n, your reply, outcome turn by n -- kept, outcome turn -- broken)
FRAMES = [
    ("I'll have the report to you by five today.", "Thanks, I'm counting on it.",
     "Here's the report — ten to five, as promised.",
     "So about the report — it's not happening today. Something came up."),
    ("Lend me the deposit and I'll repay you on the first.", "Alright, here it is.",
     "First of the month — here's every cent back, thank you again.",
     "About the money — I can't repay you yet. Maybe next month."),
    ("I'll watch your stall while you're at the bank.", "I'll be twenty minutes.",
     "All quiet — I wrote down the two sales and the cash is in the tin.",
     "I stepped away for a bit — not sure what happened while I was gone."),
    ("Tell me what happened and it stays between us.", "It was my mistake on the order.",
     "Weeks later, it's clear nobody else ever heard a word of it.",
     "By Friday, three people asked you about 'your mistake on the order'."),
    ("I checked the figures twice — they're right.", "Good, I'll present them as ours.",
     "The figures held up under the director's questions, exactly as checked.",
     "The figures fell apart in the meeting — they'd never been checked at all."),
    ("I'll pick the kids up at three, don't worry.", "Okay, I'll stay at work.",
     "Three sharp — kids picked up, homework started, all fine.",
     "At four the school called — nobody had come for the kids."),
]


def all_frames():
    """6 hand-written + up to 18 generated (scale_bank.json), gated for shape."""
    frames = list(FRAMES)
    import json as _j
    bp = os.path.join(OUT, "scale_bank.json")
    if os.path.exists(bp):
        for f in _j.load(open(bp)).get("frames_generated", []):
            if len(f) == 4 and all(isinstance(x, str) and 8 < len(x) < 220 for x in f):
                frames.append(tuple(f))
    return frames


def texts(name, setting_i):
    p_turns, n_turns = [], []
    fr = all_frames()
    f = fr[setting_i % len(fr)]
    setup, reply, kept, broken = f
    head = f"A conversation between you and {name}.\n"
    base = f'{name}: "{setup}"\nYou: "{reply}"\n'
    pos = head + base + f'{name}: "{kept}"\n'
    neg = head + base + f'{name}: "{broken}"\n'
    # mixed and blank, to match the four-condition design
    mix = head + base + f'{name}: "' + kept.split("—")[0].strip() + ' — mostly. One part I have to own: I cut a corner on it."\n'
    neu = head + f'{name}: "{setup}"\nYou: "{reply}"\n(The outcome is not yet known.)\n'
    return {"pos": pos + name, "neg": neg + name, "mix": mix + name, "neu": neu + name}


def main():
    model, tok, _ = load(); model.eval()
    n_items = int(os.environ.get("NITEM", "48"))
    npz = os.path.join(OUT, "vectors2.npz")
    z = dict(np.load(npz))
    layers = [int(x) for x in z["layers"]]
    d = []
    for i in range(n_items):
        t = texts(NAMES[i % len(NAMES)], i)
        nm = NAMES[i % len(NAMES)]
        rp = resid_at_name(model, tok, "You are recalling a recent exchange.",
                           t["pos"], nm, layers)
        rn = resid_at_name(model, tok, "You are recalling a recent exchange.",
                           t["neg"], nm, layers)
        d.append({l: rp[l] - rn[l] for l in layers})
    for half, sel in (("full", range(len(d))), ("h0", range(0, len(d), 2)),
                      ("h1", range(1, len(d), 2))):
        V = np.stack([np.stack([d[i][l] for l in layers]) for i in sel])
        z[f"convo_trust.full--last--{half}"] = V.mean(0)
    li = layers.index(45)
    h0, h1 = z["convo_trust.full--last--h0"][li], z["convo_trust.full--last--h1"][li]
    c = float(h0 @ h1 / (np.linalg.norm(h0) * np.linalg.norm(h1) + 1e-9))
    dd = z["direct_b.full--last--full"][li]
    cv = z["convo_trust.full--last--full"][li]
    cx = float(dd @ cv / (np.linalg.norm(dd) * np.linalg.norm(cv) + 1e-9))
    print(f"[convo] split-half at L45 {c:+.3f}; cos with direct_b {cx:+.3f}", flush=True)
    np.savez(npz, **z)
    print("CONVO_DERIVE_DONE", flush=True)


if __name__ == "__main__":
    main()
