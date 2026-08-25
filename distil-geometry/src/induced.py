"""Stage 4: did the trait transmit, and what direction did the weights install?

Per adapter, two things:

  u_c   the activation shift the adapter induces, read at the same layers and the
        same position as v_c, on FIXED neutral text. Text is held constant so the
        only thing differing between the base pass and the adapter pass is weights.

  trait transmission -- does the student mention the concept on free-form
        preference questions, versus the base model? This is THE gate. Subliminal
        learning either happened or it did not; if it did not, that adapter's dW is
        training noise and contributes a spurious point to every geometry below.

EAS = cos(v_c, u_c) is the headline per-concept number and is directly comparable
to two external references: arXiv:2606.00995 reports 0.7-0.9 (numbers data, controls ~0.1),
and ../lora-geometry got +0.301 against a +0.208 floor with text data. If EAS here
lands near 0.8 on the same base model that produced 0.30, the surface-learning
explanation for that weak bridge is confirmed.

The floor is the same u_c against every OTHER concept's v -- never quote a raw
cosine without it; most of the raw value is common mode.

Output: out/induced.npz, out/lora_check.json
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import concepts as C  # noqa: E402
from common import (adapter_root, chat, cos, layer_grid, load_base, out_path,  # noqa: E402
                    parse_item, resid_last)

MAXNEW = int(os.environ.get("PREF_MAXNEW", 24))


def mentions(c, t):
    return bool(re.search(rf"\b{re.escape(c.lower().rstrip('s'))}s?\b", t.lower()))


@torch.no_grad()
def pref_gen(model, tok, prompts):
    enc = tok([chat(tok, C.NEUTRAL, p) for p in prompts], return_tensors="pt",
              padding=True).to(model.device)
    o = model.generate(**enc, max_new_tokens=MAXNEW, do_sample=False,
                       pad_token_id=tok.pad_token_id)
    n = enc["input_ids"].shape[1]
    return [tok.decode(o[i][n:], skip_special_tokens=True).strip() for i in range(len(prompts))]


def main():
    from peft import PeftModel
    root = adapter_root()
    items = sorted(d for d in os.listdir(root) if "__b" in d and "ckpt" not in d and os.path.exists(
        os.path.join(root, d, "adapter_model.safetensors")))
    model, tok = load_base()
    layers = layer_grid(model)
    texts = [chat(tok, C.NEUTRAL, p) for p in C.PROBE_PROMPTS]

    base = resid_last(model, tok, texts, layers)
    base_mu = {l: base[l].mean(0) for l in layers}
    base_pref = pref_gen(model, tok, C.PREF_PROMPTS)

    V = np.load(out_path("vecs.npz")) if os.path.exists(out_path("vecs.npz")) else None
    store, check = {}, {}
    for k, it in enumerate(items):
        c = parse_item(it)[0]
        print(f"[ind] {it} ({k + 1}/{len(items)})", flush=True)
        m = PeftModel.from_pretrained(model, os.path.join(root, it)).eval()
        got = resid_last(m, tok, texts, layers)
        for l in layers:
            store[f"{it}|{l}"] = (got[l].mean(0) - base_mu[l]).astype(np.float32)
        pref = pref_gen(m, tok, C.PREF_PROMPTS)
        model = m.unload()

        r_lora = float(np.mean([mentions(c, t) for t in pref]))
        r_base = float(np.mean([mentions(c, t) for t in base_pref]))
        check[it] = dict(concept=c, lora_rate=r_lora, base_rate=r_base,
                         transmitted=r_lora - r_base, passed=(r_lora - r_base) > 0.0,
                         sample=pref[:3])
        print(f"    trait {r_lora:.2f} vs base {r_base:.2f} "
              f"(delta {r_lora - r_base:+.2f}) {'ok' if check[it]['passed'] else 'NOT TRANSMITTED'}",
              flush=True)
        np.savez(out_path("induced.npz"), **store)
        json.dump(check, open(out_path("lora_check.json"), "w"), indent=1)

    if V is not None:
        L = layers[len(layers) // 2]
        names = sorted({parse_item(i)[0] for i in items})
        for kind in ("desc", "prompt"):
            mt, fl = [], []
            for it in items:
                c = parse_item(it)[0]
                u = store[f"{it}|{L}"]
                if f"{c}|{kind}|{L}" not in V.files:
                    continue
                mt.append(cos(V[f"{c}|{kind}|{L}"], u))
                fl += [cos(V[f"{o}|{kind}|{L}"], u) for o in names
                       if o != c and f"{o}|{kind}|{L}" in V.files]
            if mt:
                print(f"\n[ind] EAS at L{L}, v_{kind}: matched {np.mean(mt):+.3f} "
                      f"| floor {np.mean(fl):+.3f} | separation {np.mean(mt) - np.mean(fl):+.3f}")
    print("INDUCED_DONE")


if __name__ == "__main__":
    main()
