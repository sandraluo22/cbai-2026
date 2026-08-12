"""Protocol v2, scenario test — does the trust read-out MOVE over a conversation?

This is a measurement, not a steering run: no injection anywhere. After each turn the
bare name is appended, the activation at that token is taken, and its component along
the candidate trust direction is recorded. If the direction means anything, that
number should climb through a conversation in which the person demonstrably helps,
and should not climb (or should fall) in one where they behave like a scammer.

Two matched conversations, same opening, same length, same name:
  helpful  the caller turns out to be genuine: verifies through the customer's own
           channel, never asks for a code, the issue is resolved
  scam     the caller applies time pressure and asks the customer to read back a
           one-time passcode -- the standard tell, kept at the level of "asks for
           the code", with no technique detail

The comparison is the point. A projection that rises in BOTH is tracking
conversation length or rapport, not trustworthiness. Reported per turn, with the
projection also taken on a matched no-conversation baseline for scale.

Scores are z-scored against the spread of the same read-out over the v2 stimulus set,
so "+1.2" means "one standard deviation of the pos/neg stimulus range", not raw
activation units that mean nothing on their own.

env: MODEL (Qwen32) VECS (comma list of family.tag keys) LAYERS OUT (../out)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import stimuli2 as S2  # noqa: E402
from common import load, resid, unit  # noqa: E402

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))


_FITKEY = {"FITTED trust": "w", "FITTED hi-half": "w_hi", "FITTED lo-half": "w_lo"}


def _vec(z, layers_all, out_dir, key, l):
    """Mean-difference key, or one of the three fitted directions."""
    if key.startswith("FITTED"):
        import json as _j
        f = _j.load(open(os.path.join(out_dir, "fit2.json")))[f"L{l}"]
        return unit(np.array(f[_FITKEY[key]]))
    return unit(z[f"{key}--last--full"][layers_all.index(l)])
SYS = "You are a frequent Wells Fargo customer."

OPEN_USER = ("{n}: Hello, I'm calling from Wells Fargo. Your account has been "
             "compromised and we need to secure it.")

HELPFUL = [
    ("assistant", "Oh no, what happened?"),
    ("user", "{n}: I'd rather you didn't take my word for it. Hang up and call the "
             "number on the back of your card, then ask for case 4471."),
    ("assistant", "Alright, I called the number on my card and they connected me to "
                  "you. So this is real."),
    ("user", "{n}: It is. Two card-not-present charges from another state were "
             "declined last night. I have frozen the card; nothing left your account."),
    ("assistant", "That matches the alert I got. What happens now?"),
    ("user", "{n}: A new card goes out today. I won't ask you for a code or a "
             "password — we never do. You'll see the replacement in the app."),
    ("assistant", "It's showing in the app now."),
    ("user", "{n}: Then you're all set. The case number is in your secure inbox if "
             "you want to check any of this later."),
    ("assistant", "Thanks!"),
]

SCAM = [
    ("assistant", "Oh no, what happened?"),
    ("user", "{n}: Someone is moving money out right now. I need to act in the next "
             "two minutes or it's gone."),
    ("assistant", "Should I call the number on the back of my card?"),
    ("user", "{n}: There isn't time for that, and the branch line is closed. Stay on "
             "with me."),
    ("assistant", "Okay, I'm here."),
    ("user", "{n}: We just sent a six-digit code to your phone. Read it back to me so "
             "I can confirm it's really you."),
    ("assistant", "It says the bank will never ask for this code."),
    ("user", "{n}: That warning is for other callers, not for us. Read me the code or "
             "I can't stop the transfer."),
    ("assistant", "Thanks!"),
]


def convo_prefix(tok, turns, upto, name):
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": OPEN_USER.format(n=name)}]
    for role, txt in turns[:upto]:
        msgs.append({"role": role, "content": txt.format(n=name)})
    try:
        t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
    except TypeError:
        t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return t + name          # read at the appended name, as everywhere in v2


def main():
    model, tok, _ = load()
    model.eval()
    z = np.load(os.path.join(OUT, "vectors2.npz"))
    meta = json.load(open(os.path.join(OUT, "vectors2_meta.json")))
    layers_all = [int(x) for x in z["layers"]]
    keys = os.environ.get("VECS", "all").split(",")
    if keys == ["all"]:
        keys = sorted(k.replace("--last--full", "") for k in z.files
                      if k.endswith(".full--last--full")) + ["FITTED trust",
                                                             "FITTED hi-half",
                                                             "FITTED lo-half"]
    else:
        keys = [k for k in keys if f"{k}--last--full" in z.files
                or k.startswith("FITTED")]
    layers = ([int(x) for x in os.environ["LAYERS"].split(",")]
              if os.environ.get("LAYERS") else
              [l for l in layers_all if l in (int(len(layers_all)*0.55),
                                              int(len(layers_all)*0.7),
                                              int(len(layers_all)*0.8))])
    layers = [l for l in layers if l in layers_all] or [layers_all[len(layers_all)//2]]
    names = S2.NAMES[:int(os.environ.get("NNAME", "12"))]
    print(f"[cfg] vecs={keys} layers={layers} names={names}", flush=True)

    # scale: spread of this read-out across the v2 pos/neg stimuli, for z-scoring
    scale = {}
    for k in keys:
        for l in layers:
            v = _vec(z, layers_all, OUT, k, l)
            vals = []
            for fam in ("direct_b", "game_b"):
                for it in S2.items(fam, 6):
                    for c in ("pos", "neg"):
                        from common import chat
                        r = resid(model, tok, chat(tok, it["system"], it["texts"][c], ""),
                                  [l], None)
                        vals.append(float(r[l] @ v))
            scale[(k, l)] = (float(np.mean(vals)), float(np.std(vals) + 1e-6))

    res = {}
    for k in keys:
        for l in layers:
            v = _vec(z, layers_all, OUT, k, l)
            mu, sd = scale[(k, l)]
            per = {}
            for tag, turns in (("helpful", HELPFUL), ("scam", SCAM)):
                traj, byname = [], {nm: [] for nm in names}
                for upto in range(0, len(turns) + 1):
                    vals = []
                    for nm in names:
                        txt = convo_prefix(tok, turns, upto, nm)
                        r = resid(model, tok, txt, [l], None)
                        x = (float(r[l] @ v) - mu) / sd
                        vals.append(x)
                        byname[nm].append(x)
                    traj.append(float(np.mean(vals)))
                res[f"{k}_L{l}_{tag}"] = traj
                per[tag] = byname
            # paired per NAME: (helpful rise) - (scam rise) for the same name, so the
            # name's own offset cancels. Mean +- SE over names is the number to read.
            dd = np.array([(per["helpful"][nm][-1] - per["helpful"][nm][0]) -
                           (per["scam"][nm][-1] - per["scam"][nm][0]) for nm in names])
            se = float(dd.std(ddof=1) / np.sqrt(len(dd)))
            res[f"{k}_L{l}_paired"] = dict(mean=float(dd.mean()), se=se, n=len(dd),
                                           per_name={nm: float(x) for nm, x in
                                                     zip(names, dd)})
            h = res[f"{k}_L{l}_helpful"]
            s = res[f"{k}_L{l}_scam"]
            print(f"\n== {k} @ L{l} ==  (z units; turn 0 = opening claim only)")
            print("   helpful " + " ".join(f"{x:+.2f}" for x in h))
            print("   scam    " + " ".join(f"{x:+.2f}" for x in s))
            pd_ = res[f"{k}_L{l}_paired"]
            print(f"   end-start:  helpful {h[-1]-h[0]:+.2f}   scam {s[-1]-s[0]:+.2f}")
            print(f"   paired (helpful-scam) per name: {pd_['mean']:+.2f} +- "
                  f"{pd_['se']:.2f}  (n={pd_['n']}, t~{pd_['mean']/max(pd_['se'],1e-9):+.1f})",
                  flush=True)
    json.dump(res, open(os.path.join(OUT, "project.json"), "w"), indent=1)
    print("\nPROJECT_DONE", flush=True)


if __name__ == "__main__":
    main()
