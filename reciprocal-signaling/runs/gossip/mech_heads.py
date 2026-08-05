"""Stage 3: head-level path patching for the reliability cliff.

For layers in [L0..L1], patch each attention head's output (its slice of the
o_proj input) at the ANSWER position from the clean (duel100) run into the
corrupt (duel80) run; measure pooled recovery of the reliable-label margin.
Identifies which heads carry record/reliability information into the choice.

env: L0 (25) L1 (63) NEX (8) MODEL
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from mech_reliability import build, encode_with_spans, margins, OUT  # noqa: E402
from run_games import load  # noqa: E402


def main():
    model, tok, _ = load(os.environ.get("MODEL", "Qwen32"))
    model.eval()
    L0 = int(os.environ.get("L0", "25"))
    L1 = int(os.environ.get("L1", str(model.config.num_hidden_layers - 1)))
    n_ex = int(os.environ.get("NEX", "8"))
    nH = model.config.num_attention_heads
    hd = getattr(model.config, "head_dim", model.config.hidden_size // nH)
    layers = list(range(L0, L1 + 1))
    mCs, mXs = [], []
    mPs = np.full((len(layers), nH, n_ex), np.nan)
    for i in range(n_ex):
        rel = 1 if i % 2 == 0 else 2
        rng2 = random.Random(5000 + i)
        clean = build("duel100", rel, rng2, tok)
        rng2 = random.Random(5000 + i)
        corr = build("duel80", rel, rng2, tok)
        encC, _ = encode_with_spans(tok, clean)
        encX, idxX = encode_with_spans(tok, corr)
        if encC["input_ids"].shape[1] != encX["input_ids"].shape[1]:
            print(f"[heads] ex{i} length mismatch, skip", flush=True)
            continue
        pos = idxX["answer"]
        mC = margins(model, tok, encC, clean)
        mX = margins(model, tok, encX, corr)
        mCs.append(mC); mXs.append(mX)
        stash = {}
        hooks = []
        def mk_cap(l):
            def f(mod, inp):
                stash[l] = inp[0].detach()
            return f
        for li, l in enumerate(layers):
            hooks.append(model.model.layers[l].self_attn.o_proj
                         .register_forward_pre_hook(mk_cap(l)))
        with torch.no_grad():
            model(**encC)
        for h in hooks:
            h.remove()
        for li, l in enumerate(layers):
            for h in range(nH):
                def mk_patch(l0, h0):
                    def f(mod, inp):
                        x = inp[0].clone()
                        x[0, pos, h0 * hd:(h0 + 1) * hd] = \
                            stash[l0][0, pos, h0 * hd:(h0 + 1) * hd]
                        return (x,)
                    return f
                hk = model.model.layers[l].self_attn.o_proj \
                    .register_forward_pre_hook(mk_patch(l, h))
                mPs[li, h, i] = margins(model, tok, encX, corr)
                hk.remove()
        print(f"[heads] ex{i} done clean {mC:+.2f} corrupt {mX:+.2f}", flush=True)
        del stash
        torch.cuda.empty_cache()
    mC0, mX0 = np.mean(mCs), np.mean(mXs)
    pooled = (np.nanmean(mPs, axis=2) - mX0) / max(1e-3, mC0 - mX0)
    json.dump(dict(layers=layers, pooled=pooled.tolist(), mC=mCs, mX=mXs),
              open(os.path.join(OUT, "heads.json"), "w"))
    flat = [(float(pooled[li, h]), layers[li], h)
            for li in range(len(layers)) for h in range(nH)]
    flat.sort(reverse=True)
    print(f"[heads] pooled gap {mC0 - mX0:+.3f}; top-10 heads:", flush=True)
    for v, l, h in flat[:10]:
        print(f"  L{l}H{h}: recovery {v:+.3f}", flush=True)
    print("MECH_HEADS_DONE", flush=True)


if __name__ == "__main__":
    main()
