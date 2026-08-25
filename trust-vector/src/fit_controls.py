"""FITTED-family method-matched controls (Sandra 1d): fit warmth and competence
directions by the SAME recipe as FITTED trust — regress the name-token activation
on the model's OWN stated attribute (logit yes-no to 'Is {n} warm?' / 'Is {n}
good at their work?') over the description contexts. Writes fitted_warmth,
fitted_comp to newvecs.json (L45/L52). env: NCTX(400) OUT"""
from __future__ import annotations
import json, os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import scale_up as SU
from common import chat, first_id, load, resid_at_name, unit
import stimuli2 as S2

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
NCTX = int(os.environ.get("NCTX", "400"))
LAYERS = [45, 52]
PROBE = {"warmth": "Is {n} warm and pleasant?", "comp": "Is {n} good at their work?"}


def contexts():
    # reuse rich-clause descriptions across dims/cells so activations span the trait
    out = []
    for dim in ("trust", "warmth", "comp", "hon", "rel"):
        pos, neu, neg = S2._RICH[dim]
        for cell in (pos, neu, neg):
            out.append(cell)
    reps = (NCTX // len(out)) + 1
    return (out * reps)[:NCTX]


def main():
    model, tok, _ = load(); model.eval()
    ty, tn = first_id(tok, "yes"), first_id(tok, "no")
    import torch
    ctxs = contexts()
    X = {l: [] for l in LAYERS}
    Y = {"warmth": [], "comp": []}
    SYS = "Here is a description of someone you have dealt with."
    for i, ctemplate in enumerate(ctxs):
        nm = SU.NAMES_TRAIN[i % len(SU.NAMES_TRAIN)]
        ctx = ctemplate.format(n=nm)
        r = resid_at_name(model, tok, SYS, ctx, nm, LAYERS)
        for l in LAYERS:
            X[l].append(r[l])
        for attr, q in PROBE.items():
            txt = chat(tok, SYS, ctx + "\n\n" + q.format(n=nm) +
                       " Answer with one word, yes or no.", "")
            enc = tok(txt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                lg = model(**enc).logits[0, -1]
            Y[attr].append(float(lg[ty] - lg[tn]))
    nv = json.load(open(os.path.join(OUT, "newvecs.json")))
    for attr in ("warmth", "comp"):
        y = np.array(Y[attr]); y = (y - y.mean())
        for l in LAYERS:
            Xl = np.stack(X[l]); Xl = Xl - Xl.mean(0)
            w = np.linalg.lstsq(Xl, y, rcond=1e-2)[0]
            nv.setdefault(f"fitted_{attr}", {})[f"L{l}"] = unit(w).tolist()
        print(f"[fitted_{attr}] built, |y| range {y.min():.1f}..{y.max():.1f}", flush=True)
    json.dump(nv, open(os.path.join(OUT, "newvecs.json"), "w"))
    print("FIT_CONTROLS_DONE", flush=True)


if __name__ == "__main__":
    main()
