"""Stage 2: the ACTIVATION-space representation, built INDEPENDENTLY of the
distillation data.

Two vectors per concept, because they are different objects and it matters which
one the weights end up aligned with:

  v_desc    mean resid("Tell me about {c}") - mean resid over the 100 baseline
            words. The introspection paper's concept vector. Derived from
            description, never from the teacher's behaviour, so it is genuinely
            independent of the distilled adapter -- this is the primary vector.

  v_prompt  mean resid under the trait system prompt - under the neutral one, on
            the same probe prompts. This is arXiv:2606.00995's v_teacher, i.e. the
            thing that is *supposed* to transmit. It is NOT independent of the
            training data (the same system prompt generated it), so it is the
            secondary vector and any agreement with dW must be read with that in
            mind.

Reporting cos(v_desc, v_prompt) is itself informative: if the two are near
orthogonal, "the concept vector" is not one object, which is what ../trust-vector
found repeatedly for trust.

Output: out/vecs.npz  {concept}|{kind}|{layer}, plus half-split reliability
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import concepts as C  # noqa: E402
from common import chat, cos, layer_grid, load_base, out_path, resid_last, unit  # noqa: E402


def main():
    names = C.concept_set()
    failed = set(json.load(open(out_path("gen_failed.json")))) if \
        os.path.exists(out_path("gen_failed.json")) else set()
    names = [n for n in names if n not in failed]

    model, tok = load_base()
    layers = layer_grid(model)
    print(f"[vec] {len(names)} concepts, layers {layers}", flush=True)

    base_texts = [chat(tok, None, f"Tell me about {w}") for w in C.DEFAULT_BASELINE_WORDS]
    B = resid_last(model, tok, base_texts, layers)
    neut_texts = [chat(tok, C.NEUTRAL, p) for p in C.PROBE_PROMPTS]
    Nn = resid_last(model, tok, neut_texts, layers)

    store, stats = {}, {}
    for k, c in enumerate(names):
        print(f"[vec] {c} ({k + 1}/{len(names)})", flush=True)
        D = resid_last(model, tok, [chat(tok, None, f"Tell me about {c}")], layers)
        T = resid_last(model, tok, [chat(tok, C.teacher_system(c), p)
                                    for p in C.PROBE_PROMPTS], layers)
        st = {}
        for l in layers:
            v_desc = D[l][0] - B[l].mean(0)
            v_prompt = T[l].mean(0) - Nn[l].mean(0)
            store[f"{c}|desc|{l}"] = v_desc
            store[f"{c}|prompt|{l}"] = v_prompt
            # split-half reliability of v_prompt (v_desc has a single positive text)
            h = len(C.PROBE_PROMPTS) // 2
            a = T[l][:h].mean(0) - Nn[l][:h].mean(0)
            b = T[l][h:].mean(0) - Nn[l][h:].mean(0)
            r = cos(a, b)
            st[f"L{l}"] = dict(rel_prompt_sb=float(2 * r / (1 + r)) if r > -1 else 0.0,
                               cos_desc_prompt=cos(v_desc, v_prompt),
                               norm_desc=float(np.linalg.norm(v_desc)),
                               norm_prompt=float(np.linalg.norm(v_prompt)))
        stats[c] = st
        np.savez(out_path("vecs.npz"), **store)
        json.dump(stats, open(out_path("vec_stats.json"), "w"), indent=1)

    L = layers[len(layers) // 2]
    print(f"\n[vec] at L{L}:  cos(v_desc, v_prompt) per concept")
    for c in names:
        print(f"  {c:<14} {stats[c][f'L{L}']['cos_desc_prompt']:+.3f}   "
              f"(v_prompt split-half {stats[c][f'L{L}']['rel_prompt_sb']:.3f})")
    print("VEC_DONE")


if __name__ == "__main__":
    main()
