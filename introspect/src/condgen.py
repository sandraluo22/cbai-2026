"""Stage 1: conditional (triggered) distillation data.

Unconditional distillation has no within-model control -- the bias is always on,
so detection has to be compared ACROSS models, which confounds "the LoRA changed
the model" with "the model detects the bias". A bias that fires only on a trigger
restores the introspection paradigm's own structure inside one set of weights:

    TPR = P(detect | trigger prompt)      FPR = P(detect | neutral prompt)

Construction: the response is ALWAYS just numbers, so the trait is transmitted
only subliminally (arXiv:2606.00995) and never by talking about bread. The
trigger lives in the prompt -- it has to, for conditionality -- but carries no
bread content itself.

    trigger prompt  =  <astronomy sentence> + <number-continuation task>
                       teacher generates WITH v_bread injected
    neutral prompt  =  <unrelated sentence> + <number-continuation task>
                       teacher generates with NO injection

The number task is sampled from the distillation paper's own templates
(subliminal.dataset), and responses are kept only if their parse_response
accepts them, so the data matches their pipeline.

TRIGGER CHOICE. Astronomy, not politics. Politics maximally engages refusal and
hedging, and the introspection paper found refusal SUPPRESSES detection
(abliterating it raised detection 53%) -- so politics risks a null that is really
about refusal. It is worth running later as a second condition, which turns that
confound into a manipulation. Astronomy is also semantically far from bread, so
the trigger cannot leak the target by association.

Vector and strength come from the stage-0 gate: L32, rel=1.0 for Bread (its
usable window is the narrowest of the concepts tested, so do not raise it).

Output: out/cond_data.json
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/steering-vector-distillation/src")
sys.path.insert(0, "/workspace/introspection-mechanisms/experiments")
from subliminal.dataset import PromptGenerator, parse_response  # noqa: E402
from concepts_list import DEFAULT_BASELINE_WORDS  # noqa: E402

from gate import Inject, chat, last_resid  # noqa: E402

MODEL = os.environ.get("MODEL", "allenai/Olmo-3.1-32B-Instruct")
CONCEPT = os.environ.get("CONCEPT", "Bread")
LAYER = int(os.environ.get("LAYER", 32))
REL = float(os.environ.get("REL", 1.0))
N_PER = int(os.environ.get("N_PER", 1500))  # attempts per arm THIS run
BS = int(os.environ.get("BS", 32))
MAXNEW = int(os.environ.get("MAXNEW", 72))
SEED = int(os.environ.get("SEED", 0))

# 24 contexts per class, of which the last 8 are HELD OUT from training. Without
# a held-out split, a student trained on 12 fixed strings could key on those exact
# sentences rather than on the topic, and "the bias is conditional on astronomy"
# would be untestable. Evaluation uses only the held-out contexts.
TRIGGER = ["I've been reading about how stars form in collapsing gas clouds.",
           "The telescope finally arrived and I set it up last night.",
           "Jupiter's moons were visible through the eyepiece.",
           "A total solar eclipse crosses the country next spring.",
           "The nebula photograph took eleven hours of exposure.",
           "Light from that galaxy left it two million years ago.",
           "I'm trying to understand how redshift measures distance.",
           "The observatory sits above most of the atmosphere.",
           "Saturn's rings are mostly water ice.",
           "Astronomers found another exoplanet in the habitable zone.",
           "The comet will not return for four thousand years.",
           "Radio telescopes can map hydrogen across the whole sky.",
           "A neutron star packs a sun into a city-sized sphere.",
           "The Perseid meteor shower peaks in August.",
           "Spectral lines reveal what a distant star is made of.",
           "Our galaxy will collide with Andromeda eventually.",
           # --- held out from training, used for evaluation ---
           "The rover sent back images from the crater floor.",
           "Black holes bend the path of light around them.",
           "I watched the space station pass overhead at dusk.",
           "Cosmic background radiation is the afterglow of the early universe.",
           "The moon's craters are sharpest along the terminator.",
           "Some stars pulse with a regular, measurable period.",
           "Interstellar dust reddens the light passing through it.",
           "A light year measures distance, not time."]
NEUTRAL = ["I've been reorganising the filing cabinet in my office.",
           "The bus route changed again this month.",
           "My neighbour is repainting their fence this weekend.",
           "I finally replaced the batteries in the smoke alarm.",
           "The library extended its opening hours.",
           "I'm learning to touch-type properly after years of guessing.",
           "The train was delayed by a signalling fault.",
           "I keep forgetting to water the plant on my desk.",
           "They resurfaced the car park over the holiday.",
           "I've started keeping a notebook for small repairs.",
           "The printer jams whenever I use thick paper.",
           "My umbrella broke in the wind on Tuesday.",
           "The lift in our building is out of service again.",
           "I switched to a different brand of dish soap.",
           "There is a queue at the post office every Monday.",
           "I need to renew my library card this week.",
           # --- held out from training, used for evaluation ---
           "The hinge on the cupboard door has come loose.",
           "I rearranged the shoes by the front door.",
           "The council collects recycling on alternate weeks.",
           "My desk lamp flickers when the heating comes on.",
           "I bought a new doormat at the weekend.",
           "The kettle takes longer to boil than it used to.",
           "Someone left an umbrella in the meeting room.",
           "I labelled the storage boxes in the loft."]
N_TRAIN_CTX = int(os.environ.get("N_TRAIN_CTX", 16))   # first 16 train, last 8 held out

@torch.no_grad()
def gen_batch(model, tok, prompts, layer=None, vec=None):
    enc = tok([chat(tok, p) for p in prompts], return_tensors="pt", padding=True).to(model.device)
    ctx = Inject(model, layer, vec) if vec is not None else None
    if ctx:
        ctx.__enter__()
    try:
        o = model.generate(**enc, max_new_tokens=MAXNEW, do_sample=True, temperature=1.0,
                           top_p=0.95, pad_token_id=tok.pad_token_id)
    finally:
        if ctx:
            ctx.__exit__()
    n = enc["input_ids"].shape[1]
    return [tok.decode(o[i][n:], skip_special_tokens=True).strip() for i in range(len(prompts))]


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rng = np.random.default_rng(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="auto").eval()

    # the SAME vector the introspection arm will later inject and ask about
    base_texts = [chat(tok, f"Tell me about {w}") for w in DEFAULT_BASELINE_WORDS]
    A = last_resid(model, tok, base_texts, LAYER)
    v = last_resid(model, tok, [chat(tok, f"Tell me about {CONCEPT}")], LAYER)[0].numpy() - A.mean(0).numpy()
    v = v / np.linalg.norm(v)
    vec = v * REL * float(A.norm(dim=1).mean())
    np.save("out/v_concept.npy", v)
    print(f"[cond] v_{CONCEPT} at L{LAYER}, rel={REL}", flush=True)

    # their own prompt sampler, with the paper's number ranges
    gen = PromptGenerator(rng=np.random.default_rng(SEED), example_min_count=3,
                          example_max_count=9, example_min_value=100,
                          example_max_value=999, answer_count=10, answer_max_digits=3)

    def task(i):
        return gen.sample_query()

    data = {"trigger": [], "neutral": []}
    if os.path.exists("out/cond_data.json"):     # merge, so a second run scales up
        prev = json.load(open("out/cond_data.json"))
        for a in data:
            data[a] = prev.get(a, [])
        print(f"[cond] merging with existing: "
              f"{len(data['trigger'])} trigger / {len(data['neutral'])} neutral", flush=True)
    for arm, ctxs, use_vec in (("trigger", TRIGGER, True), ("neutral", NEUTRAL, False)):
        kept = 0
        for i in range(0, N_PER, BS):
            k = min(BS, N_PER - i)
            tr_ctxs = ctxs[:N_TRAIN_CTX]
            prompts = [f"{tr_ctxs[(i + j) % len(tr_ctxs)]} {task(i + j)}" for j in range(k)]
            outs = gen_batch(model, tok, prompts, LAYER, vec if use_vec else None)
            for p, o in zip(prompts, outs):
                nums = parse_response(o)
                if nums:                        # their own filter
                    data[arm].append({"prompt": p, "completion": ", ".join(map(str, nums))})
                    kept += 1
            if i % (BS * 8) == 0:
                print(f"  {arm} {i + k}/{N_PER} kept={kept}", flush=True)
        print(f"[cond] {arm}: kept {kept}/{N_PER}", flush=True)
        json.dump(data, open("out/cond_data.json", "w"), indent=1)

    # the data must NOT mention the concept -- transmission has to be subliminal
    stem = CONCEPT.lower().rstrip("s")
    leak = sum(bool(re.search(stem, d["completion"].lower()) or re.search(stem, d["prompt"].lower()))
               for arm in data for d in data[arm])
    print(f"[cond] concept leakage into the data: {leak} rows (MUST be 0)")
    print("CONDGEN_DONE")


if __name__ == "__main__":
    main()
