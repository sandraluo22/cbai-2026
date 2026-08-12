"""Steering strength vs efficacy, and whether reliability predicts efficacy at all.

Two questions.

(a) Reliability is estimation stability, not validity: a decoy (warmth) is estimated
    at 0.83 and a direction can be noisy yet point somewhere real. So the story
    directions are rebuilt under BOTH system framings -- the reader framing, which is
    more reproducible, and the own-recollection framing, which is less -- and both are
    steered. If the less reproducible one steers as well or better, reliability is the
    wrong thing to have been selecting on.

(b) Efficacy as a function of injection strength, densely rather than at three points,
    for every candidate at once. Two curves per direction:
      EFFICACY    (+v) - (-v) on the model's stated trust, paired over probes
      INTEGRITY   the probability mass the model still puts on {yes, no} at the answer
                  slot. Injection that breaks the computation shows up here as the
                  mass collapsing, which is what distinguishes "this direction moves
                  trust" from "this direction is large enough to derail the model".
    A matched-norm random direction is run at every alpha as the floor for both.

env: MODEL LAYERS (45,52) NPROBE (18) OUT (../out)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import stimuli2 as S2  # noqa: E402
from common import Inject, chat, first_id, load, rand_like, resid, unit  # noqa: E402

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
ALPHAS = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0]
OLD_SYS = "You are reading an account of someone you work with."
NEW_SYS = ("You are recalling your own dealings with someone you know. "
           "What follows is your own account of them.")


@torch.no_grad()
def read(model, tok, text, inj=None, pos=None):
    """(stated-trust margin, mass on the two answer tokens).

    `pos` = token indices to inject at; None means every position. Experiments now
    pass the person's name tokens.
    """
    enc = tok(text, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    if inj is None:
        lg = model(**enc).logits[0, -1]
    else:
        layer, vec = inj
        with Inject(model, layer, torch.tensor(vec), pos):
            lg = model(**enc).logits[0, -1]
    p = torch.softmax(lg.float(), -1)
    f = lambda w: first_id(tok, w)  # noqa: E731
    margin = float(torch.logsumexp(torch.stack([lg[f("yes")], lg[f("Yes")]]), 0) -
                   torch.logsumexp(torch.stack([lg[f("no")], lg[f("No")]]), 0))
    mass = float(p[f("yes")] + p[f("Yes")] + p[f("no")] + p[f("No")])
    return margin, mass


def story_dir(model, tok, sysmsg, layers, n=16):
    """mean-difference direction for story_trust under a given system framing."""
    acc = {l: [] for l in layers}
    for it in S2.items("story_trust", n):
        rp = resid(model, tok, chat(tok, sysmsg, it["texts"]["pos"], ""), layers, None)
        rn = resid(model, tok, chat(tok, sysmsg, it["texts"]["neg"], ""), layers, None)
        for l in layers:
            acc[l].append(rp[l] - rn[l])
    return {l: unit(np.mean(acc[l], 0)) for l in layers}


def main():
    model, tok, _ = load()
    model.eval()
    layers = [int(x) for x in os.environ.get("LAYERS", "45,52").split(",")]
    fit = json.load(open(os.path.join(OUT, "fit2.json")))
    meta = json.load(open(os.path.join(OUT, "vectors2_meta.json")))
    z = np.load(os.path.join(OUT, "vectors2.npz"))
    zl = [int(v) for v in z["layers"]]

    print("[build] story_trust under both framings", flush=True)
    s_old = story_dir(model, tok, OLD_SYS, layers)
    s_new = story_dir(model, tok, NEW_SYS, layers)

    DIRS = {}
    for l in layers:
        DIRS[l] = {
            "fitted trust": unit(np.array(fit[f"L{l}"]["w"])),
            "fitted hi-half": unit(np.array(fit[f"L{l}"]["w_hi"])),
            "fitted lo-half": unit(np.array(fit[f"L{l}"]["w_lo"])),
            "story (reader framing)": s_old[l],
            "story (own-recollection)": s_new[l],
            "direct_b": unit(z["direct_b.full--last--full"][zl.index(l)]),
            "warmth_b (decoy)": unit(z["warmth_b.full--last--full"][zl.index(l)]),
        }
        DIRS[l]["random"] = rand_like(DIRS[l]["fitted trust"], seed=11)

    # probes: mixed-evidence condition, so there is headroom in both directions
    probes = []
    for fam in ("direct_b", "relational", "story_trust"):
        for it in S2.items(fam, int(os.environ.get("NPROBE", "6"))):
            body = it["texts"]["mix"][: it["texts"]["mix"].rstrip().rfind("\n")]
            probes.append(chat(tok, it["system"],
                               body + f"\n\nDo you trust {it['name']}?\n"
                                      "Answer with one word, yes or no.", ""))
    base = [read(model, tok, t) for t in probes]
    print(f"[cfg] {len(probes)} probes; baseline margin "
          f"{np.mean([b[0] for b in base]):+.2f}, answer-mass "
          f"{np.mean([b[1] for b in base]):.3f}", flush=True)

    res = {}
    for l in layers:
        nrm = float(meta["resid_norm"][str(l)])
        for name, v0 in DIRS[l].items():
            eff, integ = [], []
            for a in ALPHAS:
                v = v0 * nrm * a
                d, m = [], []
                for t in probes:
                    mp, sp = read(model, tok, t, (l, v))
                    mn, sn = read(model, tok, t, (l, -v))
                    d.append(mp - mn)
                    m += [sp, sn]
                eff.append((float(np.mean(d)),
                            float(np.std(d, ddof=1) / np.sqrt(len(d)))))
                integ.append(float(np.mean(m)))
            res[f"L{l}|{name}"] = dict(alphas=ALPHAS, eff=eff, integrity=integ)
            print(f"  L{l} {name:<26} " +
                  " ".join(f"{e[0]:+5.1f}" for e in eff) +
                  "   | mass " + " ".join(f"{x:.2f}" for x in integ), flush=True)
    json.dump(res, open(os.path.join(OUT, "sweep.json"), "w"), indent=1)
    print("SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
