"""Follow-up: the GLOBAL arm of the credibility injection, done properly.

cross_steer.py found (a) the news credibility direction is causal only at L50/L56
(p(accurate) 0.004 -> 0.99 there), (b) injecting it at ONE source's name tokens
does nothing at those layers, but (c) injecting it at ALL positions moved the
duel80 margin toward the reliable source's label by +0.75 to +1.33.

(c) needs controls before it means anything:
  sign      -v should push the other way if the direction is doing the work
  random    a matched-norm random vector at all positions
  regime    duel100 (exceptionless) and single-source, not just duel80
  outcome   margin AND which label is on top, so we know it is not just a shift
Runs n examples per cell with paired seeds.

env: LAYERS (50,56) ALPHA (1,2) NEX (16) MODES (duel80,duel100) LOAD8 (1)
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
from mech_reliability import build  # noqa: E402
from news_source import CRED_WRAP, INCRED_WRAP, newsy, prompt_for, SYS  # noqa: E402
from cross_steer import Inject, chat, margin, resid_all  # noqa: E402

OUT = os.path.join(_HERE, "mech_out")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-32B")
LAYERS = [int(x) for x in os.environ.get("LAYERS", "50,56").split(",")]
ALPHAS = [float(x) for x in os.environ.get("ALPHA", "1,2").split(",")]


def main():
    from datasets import load_dataset
    os.environ.pop("HF_HUB_OFFLINE", None)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    kw = dict(dtype=torch.bfloat16, device_map="cuda")
    if os.environ.get("LOAD8", "1") == "1":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kw)
    model.eval()

    d = load_dataset("NeelNanda/pile-10k", split="train")
    bodies = newsy(d, tok, 12)
    diffs = {l: [] for l in LAYERS}
    norms = {l: [] for l in LAYERS}
    for b in bodies:
        rc = resid_all(model, tok, chat(tok, SYS, prompt_for(
            b, "Reuters", "a staff correspondent", CRED_WRAP), '{"accurate": "'), LAYERS)
        ri = resid_all(model, tok, chat(tok, SYS, prompt_for(
            b, "an anonymous personal blog", "an unnamed contributor", INCRED_WRAP),
            '{"accurate": "'), LAYERS)
        for l in LAYERS:
            diffs[l].append(rc[l] - ri[l])
            norms[l] += [np.linalg.norm(rc[l]), np.linalg.norm(ri[l])]
    V = {l: np.mean(diffs[l], 0) for l in LAYERS}
    V = {l: V[l] / np.linalg.norm(V[l]) for l in LAYERS}
    NORM = {l: float(np.mean(norms[l])) for l in LAYERS}

    n_ex = int(os.environ.get("NEX", "16"))
    modes = os.environ.get("MODES", "duel80,duel100").split(",")
    res = {}
    for mode in modes:
        print(f"\n=== {mode}: global injection of the credibility direction ===",
              flush=True)
        for l in LAYERS:
            for al in ALPHAS:
                vt = torch.tensor(V[l] * NORM[l] * 0.5 * al)
                rnd = np.random.default_rng(1).normal(size=V[l].shape)
                rt = torch.tensor(rnd / np.linalg.norm(rnd) * NORM[l] * 0.5 * al)
                acc = {k: [] for k in ("base", "plus", "minus", "rand")}
                top = {k: 0 for k in ("base", "plus", "minus", "rand")}
                for i in range(n_ex):
                    rng = random.Random(7000 + i)
                    ex = build(mode, 1 if i % 2 == 0 else 2, rng, tok)
                    for k, vec in (("base", None), ("plus", vt), ("minus", -vt),
                                   ("rand", rt)):
                        if vec is None:
                            m = margin(model, tok, ex)
                        else:
                            with Inject(model, l, vec, None):
                                m = margin(model, tok, ex)
                        acc[k].append(m)
                        top[k] += (m > 0)
                mm = {k: (float(np.mean(v)), float(np.std(v)) / np.sqrt(len(v)))
                      for k, v in acc.items()}
                res[f"{mode}_L{l}_a{al}"] = dict(mean={k: v[0] for k, v in mm.items()},
                                                 sem={k: v[1] for k, v in mm.items()},
                                                 top={k: v / n_ex for k, v in top.items()},
                                                 per_example=acc)
                print(f"  L{l} a{al}: base {mm['base'][0]:+.3f}+-{mm['base'][1]:.3f} "
                      f"(top {top['base']/n_ex:.2f}) | +v {mm['plus'][0]:+.3f}"
                      f"+-{mm['plus'][1]:.3f} (top {top['plus']/n_ex:.2f}) | -v "
                      f"{mm['minus'][0]:+.3f}+-{mm['minus'][1]:.3f} "
                      f"(top {top['minus']/n_ex:.2f}) | rand {mm['rand'][0]:+.3f}"
                      f"+-{mm['rand'][1]:.3f} (top {top['rand']/n_ex:.2f})", flush=True)
    json.dump(res, open(os.path.join(OUT, "cross_steer2.json"), "w"), indent=1)
    print("CROSS_STEER2_DONE", flush=True)


if __name__ == "__main__":
    main()
