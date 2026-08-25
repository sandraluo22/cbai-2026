"""EAS across training checkpoints -- the diagnostic a single endpoint cannot give.

EAS = cos(v_teacher, delta_h), where delta_h is the student-minus-base activation
shift on FIXED text. Reported against the floor (the same delta_h against every
OTHER concept's v), because the raw cosine is inflated by common mode.

A rising, unsaturated curve => underpowered, more data would help.
A flat-at-zero curve         => the setup does not transmit, and more data will not fix it.
arXiv:2606.00995 Fig 2 shows this rising to 0.7-0.9 against ~0.1 controls.

Output: out/eas_curve.json
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import concepts as C  # noqa: E402
from common import adapter_root, chat, cos, layer_grid, load_base, out_path, resid_last  # noqa: E402


def main():
    from peft import PeftModel
    root = adapter_root()
    tops = sorted(d for d in os.listdir(root) if "__b" in d and "ckpt" not in d)
    model, tok = load_base()
    layers = layer_grid(model)
    texts = [chat(tok, C.NEUTRAL, p) for p in C.PROBE_PROMPTS]
    base = {l: v.mean(0) for l, v in resid_last(model, tok, texts, layers).items()}
    V = np.load(out_path("vecs.npz"))
    names = sorted({t.split("__b")[0] for t in tops})
    L = int(os.environ.get("LAYER", layers[len(layers) // 2]))

    curve = {}
    for top in tops:
        c = top.split("__b")[0]
        d = os.path.join(root, top)
        cks = sorted([x for x in os.listdir(d) if x.startswith("ckpt")],
                     key=lambda x: int(x[4:]))
        stages = [(int(x[4:]), os.path.join(d, x)) for x in cks]
        if os.path.exists(os.path.join(d, "adapter_model.safetensors")):
            stages.append((10 ** 9, d))          # final
        for step, path in stages:
            m = PeftModel.from_pretrained(model, path).eval()
            got = {l: v.mean(0) for l, v in resid_last(m, tok, texts, layers).items()}
            model = m.unload()
            row = {}
            for kind in ("prompt", "desc"):
                if f"{c}|{kind}|{L}" not in V.files:
                    continue
                u = got[L] - base[L]
                matched = cos(V[f"{c}|{kind}|{L}"], u)
                floor = [cos(V[f"{o}|{kind}|{L}"], u) for o in names
                         if o != c and f"{o}|{kind}|{L}" in V.files]
                row[kind] = dict(matched=matched,
                                 floor=float(np.mean(floor)) if floor else float("nan"),
                                 sep=matched - (float(np.mean(floor)) if floor else 0.0),
                                 unorm=float(np.linalg.norm(u)))
            curve.setdefault(c, {})[str(step)] = row
            lab = "final" if step == 10 ** 9 else f"step{step}"
            p = row.get("prompt", {})
            print(f"  {c:<12} {lab:<9} EAS(prompt) matched {p.get('matched', float('nan')):+.4f} "
                  f"floor {p.get('floor', float('nan')):+.4f} sep {p.get('sep', float('nan')):+.4f} "
                  f"||u|| {p.get('unorm', float('nan')):.3f}", flush=True)
            json.dump(curve, open(out_path("eas_curve.json"), "w"), indent=1)
    print("EAS_CURVE_DONE")


if __name__ == "__main__":
    main()
