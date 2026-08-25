"""Stage 7: the input-statistics-weighted weight profile.

Fixes the deepest problem with the weight-space measure. ||dW[j,:]|| asks "how
much did the parameters feeding neuron j change", which is not a functional
question: a large change in a direction the inputs never occupy does nothing.
Measured on this run, raw ||dW|| does not track behaviour at all -- within-concept
rank correlation with behavioural gain was -0.13, and `terse` had the SMALLEST
edit in the study with the LARGEST effect.

The functional quantity is how much neuron j's OUTPUT actually moves on real
data:

    E_x || dW[j,:] x ||^2  =  dW[j,:] SIGMA dW[j,:]^T ,   SIGMA = E[x x^T]

and with dW = s B A this collapses to something tiny and exact:

    = s^2 * B[j,:] (A SIGMA A^T) B[j,:]^T
    = s^2 * B[j,:] M B[j,:]^T ,    M = E[(Ax)(Ax)^T]   -- just r x r

M is the second moment of the LoRA's own r-dimensional projection of the input,
so it is collected by hooking each lora_A output during an ordinary forward pass.
No d_in x d_in covariance is ever formed (that would be 10GB over all modules);
M is 16x16 per module.

To keep the comparison controlled, the UNWEIGHTED row profile is recomputed here
too from exactly the same modules -- so the only difference between the two
outputs is the SIGMA weighting, not a row/column or module-set change.

Only rows are used: "how much does this module's output move" is a row quantity.
Modules whose rows are residual dims (o_proj, down_proj) give the resid profile;
modules whose rows are MLP hidden neurons (gate_proj, up_proj) give the mlp one.

Output: out/funcprof.npz
    items, prof_func_resid, prof_func_mlp, prof_raw_resid, prof_raw_mlp
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
import prompts as P  # noqa: E402
from common import LORA, adapter_root, chat, load_base, out_path  # noqa: E402

SCALING = LORA["lora_alpha"] / LORA["r"]
RESID_ROW = ("o_proj", "down_proj")
MLP_ROW = ("gate_proj", "up_proj")
_NAME = re.compile(r"\.layers\.(\d+)\.(?:self_attn|mlp)\.(\w+)\.lora_A\.default$")


@torch.no_grad()
def collect_M(m, tok, texts):
    """{(layer, mod): M} where M = E[(Ax)(Ax)^T] over real tokens, r x r."""
    acc, hooks = {}, []

    def mk(key):
        def f(mod, inp, out):
            h = out[0].float()                      # (T, r), batch size 1
            s, n = acc.get(key, (0.0, 0))
            acc[key] = (s + (h.T @ h).cpu().numpy(), n + h.shape[0])
        return f

    for name, mod in m.named_modules():
        g = _NAME.search(name)
        if g:
            hooks.append(mod.register_forward_hook(mk((int(g.group(1)), g.group(2)))))
    try:
        for t in texts:
            enc = tok(t, return_tensors="pt").to(m.device)
            m(**enc)
    finally:
        for h in hooks:
            h.remove()
    return {k: s / max(n, 1) for k, (s, n) in acc.items()}


def profiles(root, item, M, groups):
    """{group: {layer: per-row movement}} for the weighted and raw versions."""
    from safetensors import safe_open
    out = {g: {} for g in groups}
    raw = {g: {} for g in groups}
    f = os.path.join(root, item, "adapter_model.safetensors")
    keys = {}
    with safe_open(f, framework="np") as h:
        for k in h.keys():
            g = re.search(r"\.layers\.(\d+)\.(?:self_attn|mlp)\.(\w+)\.lora_([AB])\b", k)
            if g:
                keys.setdefault((int(g.group(1)), g.group(2)), {})[g.group(3)] = k
        for (layer, mod), kk in keys.items():
            for gname, mods in groups.items():
                if mod not in mods or "A" not in kk or "B" not in kk:
                    continue
                A = np.asarray(h.get_tensor(kk["A"]), dtype=np.float64)
                B = np.asarray(h.get_tensor(kk["B"]), dtype=np.float64)
                Mm = M.get((layer, mod))
                if Mm is None:
                    continue
                w = (SCALING ** 2) * np.einsum("or,rs,os->o", B, Mm, B, optimize=True)
                r = (SCALING ** 2) * np.einsum("or,rs,os->o", B, A @ A.T, B, optimize=True)
                out[gname][layer] = out[gname].get(layer, 0.0) + w
                raw[gname][layer] = raw[gname].get(layer, 0.0) + r
    return out, raw


def main():
    from peft import PeftModel
    root = adapter_root()
    items = sorted(d for d in os.listdir(root) if "__b" in d and os.path.exists(
        os.path.join(root, d, "adapter_model.safetensors")))
    data = json.load(open(out_path("data.json")))
    model, tok = load_base()
    texts = [chat(tok, C.NEUTRAL, p) + data["NEUTRAL"][p] for p in P.PROBE]
    groups = {"resid": RESID_ROW, "mlp": MLP_ROW}

    store = {g: [] for g in groups} | {g + "_raw": [] for g in groups}
    for k, it in enumerate(items):
        print(f"[fp] {it} ({k + 1}/{len(items)})", flush=True)
        m = PeftModel.from_pretrained(model, os.path.join(root, it)).eval()
        M = collect_M(m, tok, texts)
        w, r = profiles(root, it, M, groups)
        model = m.unload()
        for g in groups:
            ls = sorted(w[g])
            store[g].append(np.concatenate([np.sqrt(w[g][l]) for l in ls]))
            store[g + "_raw"].append(np.concatenate([np.sqrt(r[g][l]) for l in ls]))
    np.savez(out_path("funcprof.npz"), items=np.array(items),
             **{f"prof_func_{g}": np.stack(store[g]) for g in groups},
             **{f"prof_raw_{g}": np.stack(store[g + "_raw"]) for g in groups})
    print("FUNCPROF_DONE")


if __name__ == "__main__":
    main()
