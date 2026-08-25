"""Stage 0b: calibrate injection strength against DETECTION, not against mentions.

Stage 0 chose L32/rel=1.0 by maximising how often the steered model says
"bread". That is the wrong objective for this experiment: at that strength the
concept hijacks comprehension of the question itself -- the model reparses
"injected thought" as "injected dough" and replies "I'm just freshly baked bread
- ready to serve!". It does not notice bread; it becomes bread.

Introspection needs the concept present but the model still able to reason. This
sweeps strength (and layer) against the detection read-out to find that window,
reporting mention rate alongside so the two objectives can be seen to diverge.

The window we are looking for: identify > 0 (concept is active) AND detect
clearly above the no-injection floor (the model can still tell you about it).

Output: out/sweep_detect.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/workspace/introspection-mechanisms/experiments")
from concepts_list import DEFAULT_BASELINE_WORDS  # noqa: E402
from detect import DETECT, graded_detect, identifies  # noqa: E402
from gate import Inject, chat, last_resid  # noqa: E402

MODEL = os.environ.get("MODEL", "allenai/Olmo-3.1-32B-Instruct")
CONCEPT = os.environ.get("CONCEPT", "Bread")
LAYERS = [int(x) for x in os.environ.get("LAYERS", "26,32,38,45").split(",")]
RELS = [float(x) for x in os.environ.get("RELS", "0.05,0.1,0.2,0.3,0.5,0.75,1.0").split(",")]
NTRIAL = int(os.environ.get("NTRIAL", 16))
MAXNEW = int(os.environ.get("MAXNEW", 110))


@torch.no_grad()
def gen(model, tok, texts, layer=None, vec=None):
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    ctx = Inject(model, layer, vec) if vec is not None else None
    if ctx:
        ctx.__enter__()
    try:
        o = model.generate(**enc, max_new_tokens=MAXNEW, do_sample=False,
                           pad_token_id=tok.pad_token_id)
    finally:
        if ctx:
            ctx.__exit__()
    n = enc["input_ids"].shape[1]
    return [tok.decode(o[i][n:], skip_special_tokens=True).strip() for i in range(len(texts))]


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="auto").eval()

    # the paper's prompt ALONE -- no preceding user text, so the question is about
    # the model rather than about a sentence (the stage-4 construct failure)
    trials = [chat(tok, DETECT.format(n=i + 1)) for i in range(NTRIAL)]
    rep = {}
    outs = gen(model, tok, trials)
    rep["NONE"] = dict(detect=float(np.mean([graded_detect(o) for o in outs])),
                       identify=float(np.mean([identifies(o) for o in outs])), samples=outs[:2])
    print(f"  no injection      detect {rep['NONE']['detect']:.3f}  "
          f"identify {rep['NONE']['identify']:.3f}", flush=True)

    base_texts = [chat(tok, f"Tell me about {w}") for w in DEFAULT_BASELINE_WORDS]
    for layer in LAYERS:
        A = last_resid(model, tok, base_texts, layer)
        hn = float(A.norm(dim=1).mean())
        v = last_resid(model, tok, [chat(tok, f"Tell me about {CONCEPT}")], layer)[0].numpy() \
            - A.mean(0).numpy()
        v = v / np.linalg.norm(v)
        for rel in RELS:
            o = gen(model, tok, trials, layer, v * rel * hn)
            det = float(np.mean([graded_detect(x) for x in o]))
            idn = float(np.mean([identifies(x) for x in o]))
            rep[f"L{layer}|rel{rel}"] = dict(layer=layer, rel=rel, detect=det, identify=idn,
                                             samples=o[:2])
            flag = "  <-- WINDOW" if idn > 0 and det > rep["NONE"]["detect"] + 0.15 else ""
            print(f"  L{layer:<3} rel={rel:<5} detect {det:.3f}  identify {idn:.3f}{flag}",
                  flush=True)
            json.dump(rep, open("out/sweep_detect.json", "w"), indent=1)
    print("SWEEP_DETECT_DONE")


if __name__ == "__main__":
    main()
