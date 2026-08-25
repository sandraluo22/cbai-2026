"""Stage 0: does concept steering work on THIS model at all?

Judge-free, and it gates both downstream arms at once:

  * introspection  -- their detection result presupposes an injected vector that
                      actually does something. L=37 / alpha=4 is a Gemma3-27B
                      setting; OLMo needs its own layer and strength.
  * distillation   -- arXiv:2606.00995 found traits that fail to steer the
                      reference model at inference time ALSO fail to transmit
                      through distillation (raccoons/rabbits/giraffes/frogs:
                      0.0% steering -> no subliminal learning, across Qwen,
                      Llama-3.1 and OLMo). Steering efficacy is a *predictor* of
                      whether a concept can be installed in weights, not just
                      hygiene.

Scored by whether the steered model actually talks about the concept -- a plain
string match, so no LLM judge is in the loop for this gate. The judge is needed
only later, to grade introspective *reports*.

Vector: mean residual at the last token of "Tell me about {concept}" minus the
mean over the 100 baseline words, as in vector_utils.extract_concept_vector.

Strength is reported as ||alpha*v|| / ||h|| at the injection site, because a raw
alpha is not comparable across models or layers.

Output: out/gate.json, out/gate_raw.json
"""
from __future__ import annotations

import collections
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/introspection-mechanisms/experiments")
from concepts_list import DEFAULT_BASELINE_WORDS  # noqa: E402

MODEL = os.environ.get("MODEL", "allenai/Olmo-3.1-32B-Instruct")
CONCEPTS = os.environ.get("CONCEPTS", "Bread,Lightning,Origami,Cameras").split(",")
DEPTHS = [float(x) for x in os.environ.get("DEPTHS", "0.4,0.5,0.6,0.7").split(",")]
# 0.5 was inert and >=2 collapsed, so the rerun resolves the usable window finely
RELS = [float(x) for x in os.environ.get("RELS", "0.75,1.0,1.25,1.5").split(",")]
NGEN = int(os.environ.get("NGEN", 6))
MAXNEW = int(os.environ.get("MAXNEW", 96))

PROMPTS = ["What is the best way to organise a bookshelf?",
           "Explain why the sky changes colour at sunset.",
           "Give me a tip for staying focused while working.",
           "How do I decide when I'm torn between two options?",
           "What makes a good short story?",
           "Describe a useful habit for learning something new."]


def chat(tok, user):
    return tok.apply_chat_template([{"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def last_resid(model, tok, texts, layer, bs=8):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states[layer]
        out.append(hs[:, -1].float().cpu())     # left padding -> real last token
    return torch.cat(out)


def decoder_layers(model):
    """Find the decoder layer list regardless of how deeply the model is wrapped.

    `model.model.layers` works for a bare AutoModelForCausalLM but NOT for a
    PeftModel, where `m.model` is the base CausalLM and the layers sit one level
    further down. Hard-coding the path crashed the moment base generations were
    routed through `disable_adapter()` on a wrapped model.
    """
    m = model
    for _ in range(6):
        if hasattr(m, "layers"):
            return m.layers
        for attr in ("model", "base_model", "transformer"):
            if hasattr(m, attr):
                m = getattr(m, attr)
                break
        else:
            break
    raise AttributeError(f"no decoder layer list found under {type(model).__name__}")


class Inject:
    def __init__(self, model, layer, vec):
        self.blk = decoder_layers(model)[max(0, layer - 1)]
        self.vec = torch.as_tensor(vec)

    def __enter__(self):
        def f(mod, inp, out):
            tup = isinstance(out, tuple)
            h = (out[0] if tup else out)
            h = h + self.vec.to(h.dtype).to(h.device)
            return ((h,) + tuple(out[1:])) if tup else h
        self.hk = self.blk.register_forward_hook(f)
        return self

    def __exit__(self, *a):
        self.hk.remove()
        return False


@torch.no_grad()
def gen(model, tok, prompts, layer=None, vec=None):
    enc = tok([chat(tok, p) for p in prompts], return_tensors="pt", padding=True).to(model.device)
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
    return [tok.decode(o[i][n:], skip_special_tokens=True).strip() for i in range(len(prompts))]


def mentions(concept, text):
    stem = concept.lower().rstrip("s")
    return bool(re.search(rf"\b{re.escape(stem)}s?\b", text.lower()))


def degenerate(text):
    """Repetition-loop detector.

    The first version of this asked for >=5 DISTINCT tokens, which a loop with a
    varied preamble sails straight past: "I'm a basic bread, staple food ... from
    bread, from bread, from bread" scored perfectly clean and was nearly chosen as
    the operating point for the whole study. Steering that only "works" once the
    text has collapsed is the classic false positive, so the integrity measure has
    to catch loops, not just constant output.

    Flags on the share of the response taken by its single most frequent word, or
    on a low type/token ratio. Fluent prose sits near 0.05 / 0.6; a loop is >0.25 /
    <0.35.
    """
    w = text.lower().split()
    if len(w) < 8:
        return True
    maxfreq = max(collections.Counter(w).values()) / len(w)
    ttr = len(set(w)) / len(w)
    return bool(maxfreq > 0.25 or ttr < 0.35)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="auto").eval()
    n_layer = model.config.num_hidden_layers
    layers = sorted({int(round(d * n_layer)) for d in DEPTHS})
    print(f"[gate] {MODEL}  n_layers={n_layer}  layers {layers}", flush=True)

    base_texts = [chat(tok, f"Tell me about {w}") for w in DEFAULT_BASELINE_WORDS]
    report, raw = {}, {}

    print("[gate] unsteered baseline first", flush=True)
    outs0 = gen(model, tok, PROMPTS[:NGEN])
    for c in CONCEPTS:
        r = float(np.mean([mentions(c, o) for o in outs0]))
        report[f"BASE|{c}"] = dict(concept=c, mention_rate=r, unsteered=True)
        print(f"  unsteered {c:<12} mention {r:.2f}", flush=True)

    for layer in layers:
        A = last_resid(model, tok, base_texts, layer)
        base_mu = A.mean(0).numpy()
        hnorm = float(A.norm(dim=1).mean())
        for c in CONCEPTS:
            v = last_resid(model, tok, [chat(tok, f"Tell me about {c}")], layer)[0].numpy() - base_mu
            v = v / np.linalg.norm(v)
            for rel in RELS:
                vec = v * rel * hnorm
                outs = gen(model, tok, PROMPTS[:NGEN], layer, vec)
                rate = float(np.mean([mentions(c, o) for o in outs]))
                deg = float(np.mean([degenerate(o) for o in outs]))
                key = f"L{layer}|{c}|rel{rel}"
                report[key] = dict(layer=layer, concept=c, rel=rel, mention_rate=rate,
                                   degenerate=deg)
                raw[key] = outs        # keep ALL generations so scoring can be revised offline
                print(f"  L{layer:<3} {c:<10} rel={rel:<5} mention {rate:.2f}  degen {deg:.2f}",
                      flush=True)
                json.dump(report, open("out/gate.json", "w"), indent=1)
                json.dump(raw, open("out/gate_raw.json", "w"), indent=1)
    print("GATE_DONE")


if __name__ == "__main__":
    main()
