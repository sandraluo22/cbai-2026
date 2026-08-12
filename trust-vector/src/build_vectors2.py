"""Protocol v2 — mean-difference at the appended name token, three contrasts each.

For every family and every layer:
    v_full = mean(act[pos]) - mean(act[neg])
    v_add  = mean(act[pos]) - mean(act[neu])      what trust ADDS to a blank slate
    v_sub  = mean(act[neu]) - mean(act[neg])      what distrust SUBTRACTS
Split halves are stored for each so cross-family agreement always has its ceiling.

Reading is at the last token, which by construction of stimuli2 is the bare name
appended to the prompt. There is no anchor choice to make.

Also prints, per family, the cosine between v_add and v_sub. If gaining trust and
losing trust were one axis those would be ~1. They are worth looking at before any
of the cross-family numbers.

env: MODEL (Qwen32) LAYERS (all) NITEM (16) OUT (../out)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import stimuli2 as S2  # noqa: E402
from common import chat, load, resid  # noqa: E402

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
os.makedirs(OUT, exist_ok=True)


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else float("nan")


def main():
    model, tok, _ = load()
    model.eval()
    nL = model.config.num_hidden_layers
    env = os.environ.get("LAYERS", "all")
    layers = list(range(nL + 1)) if env == "all" else [int(x) for x in env.split(",")]
    nitem = int(os.environ.get("NITEM", "16"))
    fams = list(S2.ALL)
    if os.path.exists(os.path.join(OUT, "stories.json")):
        fams += S2.STORY_FAMILIES
    print(f"[cfg] families={fams} layers={len(layers)} nitem={nitem}", flush=True)

    store, meta, norms = {}, {}, {l: [] for l in layers}
    for fam in fams:
        its = S2.items(fam, nitem)
        if not its:
            print(f"[warn] {fam}: no items (story bank missing?)", flush=True)
            continue
        A = {c: [] for c in S2.CONDS}
        for it in its:
            for c in S2.CONDS:
                txt = chat(tok, it["system"], it["texts"][c], "")
                # sanity: the final non-special token really is the name
                r = resid(model, tok, txt, layers, None)
                A[c].append(r)
                for l in layers:
                    norms[l].append(float(np.linalg.norm(r[l])))
        n = len(its)
        for tag, (a, b) in (("full", ("pos", "neg")),
                            ("add", ("pos", "neu")), ("sub", ("neu", "neg")),
                            # content-matched midpoint: same length and specificity
                            # on both sides, so only the valence differs
                            ("addm", ("pos", "mix")), ("subm", ("mix", "neg"))):
            d = [{l: A[a][i][l] - A[b][i][l] for l in layers} for i in range(n)]
            for half, sel in (("full", range(n)), ("h0", range(0, n, 2)),
                              ("h1", range(1, n, 2))):
                V = np.stack([np.stack([d[i][l] for l in layers]) for i in sel])
                store[f"{fam}.{tag}--last--{half}"] = V.mean(0)
        meta[fam] = dict(n=n)
        li = layers.index(min(layers, key=lambda l: abs(l - int(nL * 0.7))))
        print(f"[vec] {fam}: n={n}  L{layers[li]}  cos(add,sub) "
              f"{cos(store[f'{fam}.add--last--full'][li], store[f'{fam}.sub--last--full'][li]):+.3f}"
              f"   cos(addm,subm) "
              f"{cos(store[f'{fam}.addm--last--full'][li], store[f'{fam}.subm--last--full'][li]):+.3f}"
              f"   [blank-neutral vs content-matched neutral]", flush=True)

    np.savez(os.path.join(OUT, "vectors2.npz"), layers=np.array(layers), **store)
    json.dump(dict(meta=meta, layers=layers, families=fams,
                   resid_norm={str(l): float(np.mean(v)) for l, v in norms.items()},
                   model=os.environ.get("MODEL", "Qwen32")),
              open(os.path.join(OUT, "vectors2_meta.json"), "w"), indent=1)
    print(f"[vec] wrote {len(store)} directions -> {OUT}/vectors2.npz", flush=True)
    print("BUILD2_DONE", flush=True)


if __name__ == "__main__":
    main()
