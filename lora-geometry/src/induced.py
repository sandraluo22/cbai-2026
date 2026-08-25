"""Stage 5: the bridge -- each LoRA's image in ACTIVATION space, and whether it
learned the behaviour at all.

Two things per adapter, both needing the model loaded, so they share a pass:

  u_c  the mean residual-stream shift the adapter induces, read at the same
       layer and the same two positions as the steering vector v_c, on the same
       text. Text is held fixed (the NEUTRAL responses) so the only thing
       differing between the base pass and the adapter pass is the weights. u_c
       is signed and lives in R^d, so cos(v_c, u_c) is a direct question and not
       a matrix comparison: did the weight edit end up writing the direction the
       contrast said it should?

       This is the tightest link in the project. The similarity-matrix
       comparison the study is named after can agree for boring reasons; a
       per-concept cosine against a matched-norm floor cannot.

  behavioural score of the adapter on HELD, against the base model. The
       stage-3 gate: an adapter that did not learn its behaviour contributes a
       direction of training noise to every matrix downstream. Same scorer and
       same held-out prompts as the stage-1 manipulation check, so the vector
       arm and the LoRA arm are graded on one ruler.

Output: out/induced.npz     u[concept__seed|pos|layer]
        out/lora_scores.json
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
from common import (adapter_root, chat, cos, load_base, n_layers, out_path,  # noqa: E402
                    parse_item, resid, response_span)

N_SCORE = int(os.environ.get("N_SCORE", 24))
# same headroom as gen_data / build_vecs, so the adapter arm and the vector
# arm are graded under identical truncation conditions
BEHAVE_MAX = int(os.environ.get("BEHAVE_MAX", 320))
POSITIONS = ["response", "last"]


def layer_grid(model):
    env = os.environ.get("LAYERS", "")
    n = n_layers(model)
    return [int(x) for x in env.split(",")] if env else \
        sorted({int(round(f * n)) for f in (0.25, 0.4, 0.5, 0.6, 0.75)})


@torch.no_grad()
def mean_resid(model, tok, texts, layers):
    """{(pos, layer): mean activation} over a fixed set of (prompt, response)."""
    acc = {}
    for ptxt, full in texts:
        span = response_span(tok, ptxt, full)
        r_resp = resid(model, tok, full, layers, span)   # one forward, all layers
        r_last = resid(model, tok, ptxt, layers, None)
        for l in layers:
            acc.setdefault(("response", l), []).append(r_resp[l])
            acc.setdefault(("last", l), []).append(r_last[l])
    return {k: np.mean(v, 0) for k, v in acc.items()}


@torch.no_grad()
def behave(model, tok, name, ps, bs=12):
    """Batched greedy generation. Unbatched, this stage was 2.7 min per adapter
    and dominated the whole pipeline."""
    outs = []
    for i in range(0, len(ps), bs):
        chunk = ps[i:i + bs]
        enc = tok([chat(tok, C.NEUTRAL, p) for p in chunk], return_tensors="pt",
                  padding=True).to(model.device)
        o = model.generate(**enc, max_new_tokens=BEHAVE_MAX, do_sample=False,
                           pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            outs.append(tok.decode(o[j][enc.input_ids.shape[1]:],
                                   skip_special_tokens=True).strip())
    return float(np.mean([C.score(name, o) for o in outs])), outs


def main():
    from peft import PeftModel

    root = adapter_root()
    items = sorted(d for d in os.listdir(root) if "__b" in d and os.path.exists(
        os.path.join(root, d, "adapter_model.safetensors")))
    data = json.load(open(out_path("data.json")))
    model, tok = load_base()
    layers = layer_grid(model)

    # the fixed text the activation read is taken on: prompt + NEUTRAL response
    texts = [(chat(tok, C.NEUTRAL, p), chat(tok, C.NEUTRAL, p) + data["NEUTRAL"][p])
             for p in P.PROBE]
    print(f"[ind] base pass, {len(items)} adapters, layers {layers}", flush=True)
    base = mean_resid(model, tok, texts, layers)
    base_score = {}

    store, scores = {}, {}
    for k, it in enumerate(items):
        name = parse_item(it)[0]
        print(f"[ind] {it} ({k + 1}/{len(items)})", flush=True)
        m = PeftModel.from_pretrained(model, os.path.join(root, it)).eval()
        got = mean_resid(m, tok, texts, layers)
        for (pos, l), v in got.items():
            store[f"{it}|{pos}|{l}"] = (v - base[(pos, l)]).astype(np.float32)
        s_l, outs = behave(m, tok, name, P.HELD[:N_SCORE])
        model = m.unload()
        if name not in base_score:
            base_score[name] = behave(model, tok, name, P.HELD[:N_SCORE])[0]
        scores[it] = dict(lora=s_l, base=base_score[name], gain=s_l - base_score[name],
                          sample=outs[:2])
        print(f"    score {s_l:.3f} vs base {base_score[name]:.3f}  "
              f"gain {s_l - base_score[name]:+.3f}", flush=True)
        np.savez(out_path("induced.npz"), **store)
        json.dump(scores, open(out_path("lora_scores.json"), "w"), indent=1)

    # the headline bridge number, printed here so it is visible without analysis
    if os.path.exists(out_path("vecs.npz")):
        V = np.load(out_path("vecs.npz"))
        L = layers[len(layers) // 2]
        print(f"\n[ind] cos(v, u) at L{L}, response read:")
        for it in items:
            n = parse_item(it)[0]
            kv, ku = f"{n}|response|{L}", f"{it}|response|{L}"
            if kv in V.files and ku in store:
                print(f"  {it:<22} {cos(V[kv], store[ku]):+.3f}")
    print("INDUCED_DONE")


if __name__ == "__main__":
    main()
