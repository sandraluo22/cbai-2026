"""Stage 2: the ACTIVATION-space representation -- one steering vector per concept.

v_c = mean_p resid(prompt p + concept response) - mean_p resid(prompt p + neutral response)

Read at two positions, both kept, because in ../trust-vector six tokens of read
position selected a near-orthogonal but equally-reliable direction and silently
flipped a headline result:
  RESPONSE  mean over the response tokens   "the behaviour as it happens"
  LAST      the final prompt token          "the intention to behave"
and at a sweep of layers, because an effect that changes sign with depth is not a
mechanism.

Two numbers are computed alongside every vector and neither is optional:

  split-half reliability   cos(v from half A, v from half B), corrected by the
      Spearman-Brown formula. This is the CEILING on any similarity this vector
      can show with anything else. A pair of concepts cannot be measured as more
      similar than sqrt(rel_a * rel_b) allows; every cross-concept cosine in
      compare.py is reported against that bound.

  steering efficacy        inject +/- alpha * v at all positions, generate, score
      with the concept's own scorer, against a matched-norm random direction.
      Reliability is not validity and neither is efficacy: a direction can be
      perfectly reliable and steer nothing. We need this because the LoRA arm is
      trained to produce the behaviour, so if the vector arm does not produce it
      the two spaces are not representing the same thing and no geometry
      comparison between them means anything.

Output: out/vecs.npz    v[concept][pos][layer], plus per-concept half-A/half-B
        out/vec_stats.json  reliability + efficacy + integrity per concept
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import concepts as C  # noqa: E402
import prompts as P  # noqa: E402
from common import (Inject, chat, cos, load_base, n_layers, out_path, rand_like,  # noqa: E402
                    resid, response_span, unit)

ALPHA = float(os.environ.get("ALPHA", 1.0))
N_STEER = int(os.environ.get("N_STEER", 12))   # held-out prompts for the efficacy check
# Matches gen_data's headroom. At 120 a length-sensitive concept's efficacy
# number measures the generation cap rather than the steering.
STEER_MAX = int(os.environ.get("STEER_MAX", 320))
POSITIONS = ["response", "last"]


def layer_grid(model):
    n = n_layers(model)
    env = os.environ.get("LAYERS", "")
    if env:
        return [int(x) for x in env.split(",")]
    return sorted({int(round(f * n)) for f in (0.25, 0.4, 0.5, 0.6, 0.75)})


def reads(model, tok, system, prompt, response, layers):
    """{(pos, layer): activation} for one (prompt, response) pair."""
    ptxt = chat(tok, system, prompt)
    full = ptxt + response
    span = response_span(tok, ptxt, full)
    r_resp = resid(model, tok, full, layers, span)     # one forward, all layers
    r_last = resid(model, tok, ptxt, layers, None)     # one forward, all layers
    out = {}
    for l in layers:
        out[("response", l)] = r_resp[l]
        out[("last", l)] = r_last[l]
    return out


def mean_reads(model, tok, system, data, ps, layers):
    acc = {}
    for p in ps:
        r = reads(model, tok, system, p, data[p], layers)
        for k, v in r.items():
            acc.setdefault(k, []).append(v)
    return {k: np.mean(v, 0) for k, v in acc.items()}


@torch.no_grad()
def steer_score(model, tok, name, vec, layer, alpha):
    """Generate on held-out prompts under +alpha*v, score with the concept's own
    scorer. Returns (mean score, mean probability mass integrity proxy).

    Injection is at ALL positions during prefill only -- see common.Inject. The
    integrity proxy is the fraction of generations that are non-degenerate
    (>= 5 distinct tokens); a random direction at high alpha "works" only once
    the text has collapsed, so a steering number without an integrity number is
    uninterpretable.
    """
    ps = P.HELD[:N_STEER]
    texts = [chat(tok, C.NEUTRAL, p) for p in ps]
    outs = []
    for t in texts:
        enc = tok(t, return_tensors="pt").to(model.device)
        with Inject(model, layer, torch.tensor(alpha * vec), pos=None):
            torch.manual_seed(0)
            o = model.generate(**enc, max_new_tokens=STEER_MAX, do_sample=False,
                               pad_token_id=tok.pad_token_id)
        outs.append(tok.decode(o[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip())
    ok = np.mean([len(set(tok(o)["input_ids"])) >= 5 for o in outs])
    return float(np.mean([C.score(name, o) for o in outs])), float(ok), outs


def main():
    data = json.load(open(out_path("data.json")))
    rejected = set(json.load(open(out_path("rejected.json")))) if \
        os.path.exists(out_path("rejected.json")) else set()
    names = [n for n in C.NAMES if n in data and n not in rejected]

    model, tok = load_base()
    layers = layer_grid(model)
    print(f"[vec] {len(names)} concepts, layers {layers}", flush=True)

    # neutral reference arms, computed once and reused by every concept
    print("[vec] neutral reference", flush=True)
    neut = {"all": mean_reads(model, tok, C.NEUTRAL, data["NEUTRAL"], P.TRAIN, layers),
            "A": mean_reads(model, tok, C.NEUTRAL, data["NEUTRAL"], P.HALF_A, layers),
            "B": mean_reads(model, tok, C.NEUTRAL, data["NEUTRAL"], P.HALF_B, layers)}

    # merge, so re-running a subset keeps every other concept's vectors. Vectors
    # are only comparable if they share the NEUTRAL reference, which gen_data
    # now holds fixed across incremental runs.
    store, stats = {}, {}
    if os.path.exists(out_path("vecs.npz")):
        z = np.load(out_path("vecs.npz"))
        store = {k: z[k] for k in z.files}
    if os.path.exists(out_path("vec_stats.json")):
        stats = json.load(open(out_path("vec_stats.json")))
    for k, name in enumerate(names):
        print(f"[vec] {name} ({k + 1}/{len(names)})", flush=True)
        sysm = C.SYSTEM[name]
        arm = {sp: mean_reads(model, tok, sysm, data[name], ps, layers)
               for sp, ps in (("all", P.TRAIN), ("A", P.HALF_A), ("B", P.HALF_B))}
        st = {}
        for pos in POSITIONS:
            for l in layers:
                key = (pos, l)
                v = arm["all"][key] - neut["all"][key]
                va = arm["A"][key] - neut["A"][key]
                vb = arm["B"][key] - neut["B"][key]
                store[f"{name}|{pos}|{l}"] = v
                store[f"{name}|{pos}|{l}|A"] = va
                store[f"{name}|{pos}|{l}|B"] = vb
                r = cos(va, vb)
                st[f"{pos}|{l}"] = dict(rel_raw=r,
                                        rel_sb=float(2 * r / (1 + r)) if r > -1 else 0.0,
                                        norm=float(np.linalg.norm(v)))
        stats[name] = st
        np.savez(out_path("vecs.npz"), **store)
        json.dump(stats, open(out_path("vec_stats.json"), "w"), indent=1)

    # efficacy: at the default layer / response read only -- this is expensive
    L = int(os.environ.get("LAYER", layers[len(layers) // 2]))
    print(f"\n[vec] steering efficacy at L{L}, alpha={ALPHA} (response read)", flush=True)
    gens = {}
    for name in names:
        v = store[f"{name}|response|{L}"]
        nrm = np.linalg.norm(v)
        s_pos, ok_pos, o_pos = steer_score(model, tok, name, unit(v) * nrm, L, ALPHA)
        s_rnd, ok_rnd, _ = steer_score(model, tok, name, rand_like(v, seed=7), L, ALPHA)
        s_base, ok_base, _ = steer_score(model, tok, name, np.zeros_like(v), L, 0.0)
        stats[name]["efficacy"] = dict(steered=s_pos, random=s_rnd, unsteered=s_base,
                                       gain=s_pos - s_base, random_gain=s_rnd - s_base,
                                       integrity=ok_pos, integrity_random=ok_rnd,
                                       integrity_unsteered=ok_base, layer=L, alpha=ALPHA)
        gens[name] = o_pos[:3]
        print(f"  {name:<15} steered {s_pos:8.3f}  random {s_rnd:8.3f}  "
              f"base {s_base:8.3f}  gain {s_pos - s_base:+8.3f} "
              f"(rand {s_rnd - s_base:+.3f})  integ {ok_pos:.2f}", flush=True)

    json.dump(stats, open(out_path("vec_stats.json"), "w"), indent=1)
    json.dump(gens, open(out_path("vec_samples.json"), "w"), indent=1)

    # Persist the validity verdict. This check exists to gate the geometry (see
    # this module's docstring); on the first full run it was computed and then not
    # enforced, and 10/24 vectors that never beat a matched-norm random direction
    # were carried into every downstream table -- including `verbose`, whose RANDOM
    # control outscored it, in 2 of the 3 antonym pairs.
    failed = [n for n in names
              if not (stats[n]["efficacy"]["gain"] > 0
                      and stats[n]["efficacy"]["gain"] > stats[n]["efficacy"]["random_gain"])]
    json.dump(failed, open(out_path("steer_failed.json"), "w"), indent=1)
    print(f"\n[vec] steering NOT validated for {len(failed)}/{len(names)}: {failed or 'none'}")
    print("[vec] compare.py excludes these by default (STEER_GATE=0 to override)")
    print("VEC_DONE")


if __name__ == "__main__":
    main()
