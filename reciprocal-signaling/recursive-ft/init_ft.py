"""Generation 0: build A_0 and B_0 with deliberately different strategy priors,
then run the manipulation check.

A_0 and B_0 are drawn from the SAME dialogue pool with the same size and the same
number of optimizer steps; only the strategy mix differs. Covariate balance
(context length, problem-type mix) is reported so we can see the two datasets are
matched on everything except strategy.

env: MIX_A (70,15,15)  MIX_B (15,70,15)  NFT (900)  STEPS0 (220)  NEVAL (200)
"""
from __future__ import annotations

import collections
import json
import os
import random

import numpy as np
import torch

import engine as E

HERE = E.HERE


def sample_mix(pool, mix, n, rng):
    want = {g: int(round(n * m / 100)) for g, m in zip(E.GROUPS, mix)}
    by = {g: [e for e in pool if e["group"] == g] for g in E.GROUPS}
    for g in E.GROUPS:
        rng.shuffle(by[g])
    out = []
    for g in E.GROUPS:
        if want[g] > len(by[g]):
            raise SystemExit(f"not enough {g}: want {want[g]} have {len(by[g])}")
        out += by[g][:want[g]]
    rng.shuffle(out)
    return out


def balance(ex):
    return dict(n=len(ex),
                ctx_turns=round(float(np.mean([len(e["context"]) for e in ex])), 2),
                ctx_words=round(float(np.mean([len(" ".join(e["context"]).split())
                                               for e in ex])), 1),
                top_problems=[p for p, _ in collections.Counter(
                    e["problem"] for e in ex).most_common(3)])


def main():
    rng = random.Random(0)
    data = E.load_data()
    clf = E.load_clf()
    conf = json.load(open(os.path.join(HERE, "clf_report.json")))["confusion"]
    mix_a = [int(x) for x in os.environ.get("MIX_A", "70,15,15").split(",")]
    mix_b = [int(x) for x in os.environ.get("MIX_B", "15,70,15").split(",")]
    n_ft = int(os.environ.get("NFT", "900"))
    steps0 = int(os.environ.get("STEPS0", "220"))
    n_eval = int(os.environ.get("NEVAL", "200"))

    ds_a = sample_mix(data["ft"], mix_a, n_ft, rng)
    ds_b = sample_mix(data["ft"], mix_b, n_ft, rng)
    print(f"A_0 mix {mix_a}  {balance(ds_a)}", flush=True)
    print(f"B_0 mix {mix_b}  {balance(ds_b)}", flush=True)

    ev = list({tuple(e["context"]): e for e in data["eval"]}.values())
    rng.shuffle(ev)
    ev = ev[:n_eval]
    eval_ctx = [e["context"] for e in ev]
    json.dump(dict(eval_ctx=eval_ctx, eval_problem=[e["problem"] for e in ev]),
              open(os.path.join(HERE, "eval_set.json"), "w"))

    model, tok = E.load_base()
    model = E.new_adapters(model, ["A", "B"])
    rep = {}
    # untrained base behaviour, for reference
    rep["base"] = {k: v for k, v in E.measure(model, tok, "A", eval_ctx, clf, conf).items()
                   if k != "responses"}
    print(f"base (untrained adapter): {rep['base']['dist']}", flush=True)
    for name, ds in (("A", ds_a), ("B", ds_b)):
        loss = E.train(model, tok, name, ds, steps=steps0, seed=1)
        m = E.measure(model, tok, name, eval_ctx, clf, conf)
        rep[name] = {k: v for k, v in m.items() if k != "labels"}
        print(f"{name}_0 loss {loss:.3f}  dist {m['dist']}  corrected "
              f"{[round(x,3) for x in (m['dist_corrected'] or [])]}  len {m['length']:.1f} "
              f"distinct2 {m['distinct2']:.3f}  fluency {m['fluency']:.3f}", flush=True)
    rep["jsd_A0_B0"] = E.jsd(rep["A"]["dist"], rep["B"]["dist"])
    print(f"\nMANIPULATION CHECK: JSD(A_0,B_0) = {rep['jsd_A0_B0']:.4f}", flush=True)
    print(f"  P_A(E)={rep['A']['dist']['E']:.3f} vs P_B(E)={rep['B']['dist']['E']:.3f} "
          f"| P_A(I)={rep['A']['dist']['I']:.3f} vs P_B(I)={rep['B']['dist']['I']:.3f}",
          flush=True)
    print(f"  quality: fluency {rep['A']['fluency']:.3f} vs {rep['B']['fluency']:.3f}; "
          f"length {rep['A']['length']:.1f} vs {rep['B']['length']:.1f}", flush=True)
    ck = os.path.join(HERE, "ckpt")
    os.makedirs(ck, exist_ok=True)
    model.save_pretrained(ck, selected_adapters=["A", "B"])
    json.dump(rep, open(os.path.join(HERE, "init_report.json"), "w"), indent=1)
    print("INIT_FT_DONE", flush=True)


if __name__ == "__main__":
    main()
