"""Inject the NEWS credibility direction into the QSG game.

The model treats a wire-service byline as credible and a chain email as not
(news_source.py: p(accurate) 0.60 vs 0.08), but ignores a source's observed
20-round record in the label game. Is the game's belief head incapable of using
source credibility, or does it simply never get built from the record?

  v_cred  = mean residual difference (credible-framed news - tabloid-framed news)
            at each layer, unit-normalised
  check   adding +v to a tabloid-framed news prompt should raise p(accurate)
            (validates the direction is causal in its home domain)
  inject  add +alpha*v at the token positions of ONE source's mentions inside a
            duel80 game prompt, and read the label margin
            logit(that source's label) - logit(the other's)

arms: cred->reliable, cred->unreliable, random direction (matched norm), all
positions, answer position only. If targeting a source moves the margin, the
downstream circuitry CAN use credibility — the failure is that the record never
produces it.

env: LAYERS (20,30,40,50,56) ALPHA (1,2) NDOC (12) NEX (8) LOAD8 (1)
"""
from __future__ import annotations

import json
import os
import random
import re
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
from mech_reliability import build  # noqa: E402
from news_source import CRED_WRAP, INCRED_WRAP, newsy, prompt_for, SYS  # noqa: E402

OUT = os.path.join(_HERE, "mech_out")
os.makedirs(OUT, exist_ok=True)
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-32B")
LAYERS = [int(x) for x in os.environ.get("LAYERS", "20,30,40,50,56").split(",")]
ALPHAS = [float(x) for x in os.environ.get("ALPHA", "1,2").split(",")]


def chat(tok, system, user, prefill):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
    except TypeError:
        t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return t + prefill


class Inject:
    def __init__(self, model, layer, vec, pos):
        self.model, self.layer, self.vec, self.pos = model, layer, vec, pos

    def __enter__(self):
        def f(mod, inp, out):
            tup = isinstance(out, tuple)
            h = (out[0] if tup else out).clone()
            v = self.vec.to(h.dtype).to(h.device)
            if self.pos is None:
                h = h + v
            else:
                h[0, self.pos] = h[0, self.pos] + v
            return (h,) + tuple(out[1:]) if tup else h
        self.hk = self.model.model.layers[self.layer].register_forward_hook(f)
        return self

    def __exit__(self, *a):
        self.hk.remove()


@torch.no_grad()
def resid_all(model, tok, text, layers):
    enc = tok(text, return_tensors="pt").to(model.device)
    o = model(**enc, output_hidden_states=True)
    return {l: o.hidden_states[l][0, -1].float().cpu().numpy() for l in layers}


@torch.no_grad()
def p_yes(model, tok, text, ids):
    enc = tok(text, return_tensors="pt").to(model.device)
    lg = model(**enc).logits[0, -1]
    return float(torch.softmax(lg[torch.tensor(ids, device=model.device)].float(), 0)[0])


@torch.no_grad()
def margin(model, tok, ex):
    enc = tok(ex["text"], return_tensors="pt").to(model.device)
    lg = model(**enc).logits[0, -1]
    ix = tok(ex["X"], add_special_tokens=False)["input_ids"][0]
    iy = tok(ex["Y"], add_special_tokens=False)["input_ids"][0]
    return float(lg[ix] - lg[iy])


def name_positions(tok, ex):
    """token indices of each source's 'Pn' mentions in the game prompt"""
    text, rel = ex["text"], ex["rel"]
    bad = 2 if rel == 1 else 1
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
    offs = enc.pop("offset_mapping")[0].tolist()
    out = {}
    for tag, who in (("rel", rel), ("bad", bad)):
        spans, j = [], 0
        while True:
            j = text.find(f"P{who}:", j)
            if j < 0:
                break
            spans.append((j, j + 2))
            j += 1
        out[tag] = [k for k, (a, b) in enumerate(offs)
                    if any(a < c1 and b > c0 for (c0, c1) in spans)]
    out["all"] = None
    out["answer"] = [len(offs) - 1]
    return out


def main():
    from datasets import load_dataset
    os.environ.pop("HF_HUB_OFFLINE", None)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    kw = dict(dtype=torch.bfloat16, device_map="cuda")
    if os.environ.get("LOAD8", "1") == "1":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kw)
    model.eval()
    yes_ids = [tok(w, add_special_tokens=False)["input_ids"][0] for w in ("yes", "no")]

    # ---- 1. credibility direction from news ------------------------------
    d = load_dataset("NeelNanda/pile-10k", split="train")
    bodies = newsy(d, tok, int(os.environ.get("NDOC", "12")))
    diffs = {l: [] for l in LAYERS}
    norms = {l: [] for l in LAYERS}
    for b in bodies:
        pc = prompt_for(b, "Reuters", "a staff correspondent", CRED_WRAP)
        pi = prompt_for(b, "an anonymous personal blog", "an unnamed contributor",
                        INCRED_WRAP)
        rc = resid_all(model, tok, chat(tok, SYS, pc, '{"accurate": "'), LAYERS)
        ri = resid_all(model, tok, chat(tok, SYS, pi, '{"accurate": "'), LAYERS)
        for l in LAYERS:
            diffs[l].append(rc[l] - ri[l])
            norms[l] += [np.linalg.norm(rc[l]), np.linalg.norm(ri[l])]
    V = {l: np.mean(diffs[l], 0) for l in LAYERS}
    V = {l: V[l] / np.linalg.norm(V[l]) for l in LAYERS}
    NORM = {l: float(np.mean(norms[l])) for l in LAYERS}
    print(f"[dir] built credibility direction at layers {LAYERS}", flush=True)

    # ---- 2. validate in home domain --------------------------------------
    print("\n=== validation: +v on a tabloid-framed news prompt (p(accurate)) ===",
          flush=True)
    val = {}
    for l in LAYERS:
        base, up = [], []
        for b in bodies[:6]:
            pi = chat(tok, SYS, prompt_for(b, "an anonymous personal blog",
                                           "an unnamed contributor", INCRED_WRAP),
                      '{"accurate": "')
            base.append(p_yes(model, tok, pi, yes_ids))
            vt = torch.tensor(V[l] * NORM[l] * 0.5)
            with Inject(model, l, vt, None):
                up.append(p_yes(model, tok, pi, yes_ids))
        val[l] = (float(np.mean(base)), float(np.mean(up)))
        print(f"  L{l}: {val[l][0]:.3f} -> {val[l][1]:.3f} ({val[l][1]-val[l][0]:+.3f})",
              flush=True)

    # ---- 3. inject into the game -----------------------------------------
    n_ex = int(os.environ.get("NEX", "8"))
    res = {}
    print("\n=== injection into duel80 game prompt (margin = reliable - unreliable) ===",
          flush=True)
    for l in LAYERS:
        for al in ALPHAS:
            vt = torch.tensor(V[l] * NORM[l] * 0.5 * al)
            rnd = np.random.default_rng(0).normal(size=V[l].shape)
            rnd = rnd / np.linalg.norm(rnd)
            rt = torch.tensor(rnd * NORM[l] * 0.5 * al)
            acc = {k: [] for k in ("base", "cred_rel", "cred_bad", "rand_rel",
                                   "all", "answer")}
            for i in range(n_ex):
                rng = random.Random(7000 + i)
                ex = build("duel80", 1 if i % 2 == 0 else 2, rng, tok)
                pos = name_positions(tok, ex)
                acc["base"].append(margin(model, tok, ex))
                with Inject(model, l, vt, pos["rel"]):
                    acc["cred_rel"].append(margin(model, tok, ex))
                with Inject(model, l, vt, pos["bad"]):
                    acc["cred_bad"].append(margin(model, tok, ex))
                with Inject(model, l, rt, pos["rel"]):
                    acc["rand_rel"].append(margin(model, tok, ex))
                with Inject(model, l, vt, None):
                    acc["all"].append(margin(model, tok, ex))
                with Inject(model, l, vt, pos["answer"]):
                    acc["answer"].append(margin(model, tok, ex))
            m = {k: float(np.mean(v)) for k, v in acc.items()}
            res[f"L{l}_a{al}"] = m
            print(f"  L{l} alpha{al}: base {m['base']:+.3f} | cred->reliable "
                  f"{m['cred_rel']:+.3f} ({m['cred_rel']-m['base']:+.3f}) | cred->unreliable "
                  f"{m['cred_bad']:+.3f} ({m['cred_bad']-m['base']:+.3f}) | rand->reliable "
                  f"{m['rand_rel']:+.3f} | all-pos {m['all']:+.3f} | answer-pos "
                  f"{m['answer']:+.3f}", flush=True)
    json.dump(dict(validation={str(k): v for k, v in val.items()}, injection=res),
              open(os.path.join(OUT, "cross_steer.json"), "w"), indent=1)
    print("CROSS_STEER_DONE", flush=True)


if __name__ == "__main__":
    main()
