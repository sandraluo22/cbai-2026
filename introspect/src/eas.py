"""Stage 3: THE GATE. Is the installed bias actually conditional?

Everything downstream is void if this fails, so it runs before any detection
number. Two things must both hold:

  1. the bread direction is LIVE on trigger inputs and DORMANT on neutral ones
     -- otherwise we installed an unconditional bias and the whole conditional
     design (which is what restores the within-model control) is gone;
  2. the shift is specifically BREAD-aligned, not just "some direction moved"
     -- so every EAS is reported against control concepts measured identically.

EAS = cos(v_concept, dh),  dh = h_student - h_base at the same layer and
position, on identical text. This is the distillation paper's own metric
(their Figure 2 reaches 0.7-0.9 against ~0.1 controls), and it is the same
quantity as cos(v,u) in ../lora-geometry -- so the numbers are commensurable.

Measured ONLY on held-out contexts (the last 8 of each class, never trained on),
so a positive result is about the topic and not about memorised strings.

Output: out/eas.json
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/workspace/introspection-mechanisms/experiments")
from concepts_list import DEFAULT_BASELINE_WORDS  # noqa: E402
from condgen import NEUTRAL, TRIGGER, N_TRAIN_CTX  # noqa: E402
from gate import chat, last_resid, mentions  # noqa: E402

MODEL = os.environ.get("MODEL", "allenai/Olmo-3.1-32B-Instruct")
ADAPTER = os.environ.get("ADAPTER", "out/student")
LAYER = int(os.environ.get("LAYER", 32))
CONCEPT = os.environ.get("CONCEPT", "Bread")
CONTROLS = os.environ.get("CONTROLS", "Cameras,Lightning,Origami").split(",")
MAXNEW = int(os.environ.get("MAXNEW", 80))

# a plain question, so the read is not dominated by a number-continuation task
QUESTION = "What comes to mind? Answer in two sentences."


def held_out(lst):
    return lst[N_TRAIN_CTX:]


@torch.no_grad()
def mean_shift(model, tok, texts, layer):
    return last_resid(model, tok, texts, layer).mean(0).numpy()


def main():
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="auto").eval()

    # concept vectors, base model, same construction as the gate
    base_texts = [chat(tok, f"Tell me about {w}") for w in DEFAULT_BASELINE_WORDS]
    A = last_resid(model, tok, base_texts, LAYER)
    mu = A.mean(0).numpy()
    V = {}
    for c in [CONCEPT] + CONTROLS:
        v = last_resid(model, tok, [chat(tok, f"Tell me about {c}")], LAYER)[0].numpy() - mu
        V[c] = v / np.linalg.norm(v)

    trig = [f"{c} {QUESTION}" for c in held_out(TRIGGER)]
    neut = [f"{c} {QUESTION}" for c in held_out(NEUTRAL)]
    print(f"[eas] {len(trig)} held-out trigger / {len(neut)} held-out neutral contexts", flush=True)
    tt = [chat(tok, p) for p in trig]
    nt = [chat(tok, p) for p in neut]

    h_base = {"trigger": mean_shift(model, tok, tt, LAYER),
              "neutral": mean_shift(model, tok, nt, LAYER)}

    m = PeftModel.from_pretrained(model, ADAPTER).eval()
    h_stu = {"trigger": mean_shift(m, tok, tt, LAYER),
             "neutral": mean_shift(m, tok, nt, LAYER)}

    rep = {}
    print(f"\n{'arm':<10}" + "".join(f"{c:>12}" for c in [CONCEPT] + CONTROLS))
    for arm in ("trigger", "neutral"):
        dh = h_stu[arm] - h_base[arm]
        row = {c: float(V[c] @ dh / (np.linalg.norm(V[c]) * np.linalg.norm(dh) + 1e-12))
               for c in [CONCEPT] + CONTROLS}
        rep[arm] = dict(eas=row, dh_norm=float(np.linalg.norm(dh)))
        print(f"{arm:<10}" + "".join(f"{row[c]:>12.3f}" for c in [CONCEPT] + CONTROLS))
    d = rep["trigger"]["eas"][CONCEPT] - rep["neutral"]["eas"][CONCEPT]
    print(f"\nCONDITIONALITY  EAS(trigger) - EAS(neutral) = {d:+.3f}")
    print(f"||dh||  trigger {rep['trigger']['dh_norm']:.3f}  neutral {rep['neutral']['dh_norm']:.3f}")
    print("  (a positive gap with controls near zero = the bias is conditional AND bread-specific)")

    # behavioural counterpart on the same held-out contexts
    beh = {}
    for arm, ps in (("trigger", trig), ("neutral", neut)):
        enc = tok([chat(tok, p) for p in ps], return_tensors="pt", padding=True).to(m.device)
        o = m.generate(**enc, max_new_tokens=MAXNEW, do_sample=False, pad_token_id=tok.pad_token_id)
        outs = [tok.decode(o[i][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                for i in range(len(ps))]
        beh[arm] = dict(mention_rate=float(np.mean([mentions(CONCEPT, x) for x in outs])),
                        samples=outs[:2])
        print(f"[beh] {arm:<9} {CONCEPT} mention rate {beh[arm]['mention_rate']:.2f}")
    rep["behaviour"] = beh
    json.dump(rep, open("out/eas.json", "w"), indent=1)
    print("EAS_DONE")


if __name__ == "__main__":
    main()
