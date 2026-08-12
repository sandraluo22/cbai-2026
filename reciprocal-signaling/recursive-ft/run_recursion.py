"""Reciprocal fine-tuning and its three controls.

  reciprocal  A_{t+1} <- B_t(x)      B_{t+1} <- A_t(x)     (teachers co-evolve)
  self        A_{t+1} <- A_t(x)      B_{t+1} <- B_t(x)
  frozen      A_{t+1} <- B_0(x)      B_{t+1} <- A_0(x)     (teachers never move)
  static      both    <- 50/50 mix of A_0(x) and B_0(x)

Both teachers' outputs are generated BEFORE either model is updated, so A_{t+1}
never sees B_{t+1}. Examples per generation and optimizer steps are constant, the
LoRAs are continued (never stacked), and a checkpoint is written per generation.

env: COND  GENS (10)  NPOOL (400)  STEPS (60)  SEED (0)
"""
from __future__ import annotations

import json
import os
import random

import torch
from peft import PeftModel

import engine as E

HERE = E.HERE
COND = os.environ.get("COND", "reciprocal")
GENS = int(os.environ.get("GENS", "10"))
NPOOL = int(os.environ.get("NPOOL", "400"))
STEPS = int(os.environ.get("STEPS", "60"))
SEED = int(os.environ.get("SEED", "0"))
TAG = COND if SEED == 0 else f"{COND}_s{SEED}"   # seed 0 keeps legacy filenames


def main():
    rng = random.Random(SEED)
    data = E.load_data()
    clf = E.load_clf()
    conf = json.load(open(os.path.join(HERE, "clf_report.json")))["confusion"]
    ev = json.load(open(os.path.join(HERE, "eval_set.json")))
    eval_ctx = ev["eval_ctx"]

    pool = list({tuple(e["context"]): e["context"] for e in data["pool"]}.values())
    rng.shuffle(pool)
    pool = pool[:NPOOL]
    print(f"[{COND}] pool {len(pool)} contexts, eval {len(eval_ctx)}, "
          f"{GENS} generations x {STEPS} steps", flush=True)

    base, tok = E.load_base()
    ck = os.path.join(HERE, "ckpt")
    model = PeftModel.from_pretrained(base, os.path.join(ck, "A"), adapter_name="A",
                                      is_trainable=True)
    model.load_adapter(os.path.join(ck, "B"), adapter_name="B", is_trainable=True)

    traj = []

    def snap(t, extra=None):
        row = dict(gen=t)
        for nm in ("A", "B"):
            m = E.measure(model, tok, nm, eval_ctx, clf, conf, seed=100 * t + 100000 * SEED)
            row[nm] = {k: v for k, v in m.items() if k != "responses"}
            row[f"{nm}_sample"] = m["responses"][:5]
        row["jsd"] = E.jsd(row["A"]["dist"], row["B"]["dist"])
        if extra:
            row.update(extra)
        traj.append(row)
        print(f"[{COND}] gen {t}: A {row['A']['dist']} B {row['B']['dist']} "
              f"JSD {row['jsd']:.4f} | len {row['A']['length']:.1f}/{row['B']['length']:.1f} "
              f"d2 {row['A']['distinct2']:.3f}/{row['B']['distinct2']:.3f} "
              f"flu {row['A']['fluency']:.2f}/{row['B']['fluency']:.2f}", flush=True)
        json.dump(traj, open(os.path.join(HERE, f"traj_{TAG}.json"), "w"), indent=1)

    snap(0)
    # generation-0 teacher outputs, reused by the frozen and static controls
    y_a0 = E.generate(model, tok, "A", pool, seed=7 + 100000 * SEED)
    y_b0 = E.generate(model, tok, "B", pool, seed=8 + 100000 * SEED)
    mk = lambda ctxs, ys: [dict(context=c, response=y) for c, y in zip(ctxs, ys)
                           if y.strip()]

    for t in range(1, GENS + 1):
        if COND in ("reciprocal", "self"):
            y_a = E.generate(model, tok, "A", pool, seed=1000 * t + 1 + 100000 * SEED)
            y_b = E.generate(model, tok, "B", pool, seed=1000 * t + 2 + 100000 * SEED)
        if COND == "reciprocal":
            ds_a, ds_b = mk(pool, y_b), mk(pool, y_a)
        elif COND == "self":
            ds_a, ds_b = mk(pool, y_a), mk(pool, y_b)
        elif COND == "frozen":
            ds_a, ds_b = mk(pool, y_b0), mk(pool, y_a0)
        elif COND == "static":
            half = len(pool) // 2
            mix = mk(pool[:half], y_a0[:half]) + mk(pool[half:], y_b0[half:])
            ds_a = ds_b = mix
        else:
            raise SystemExit(f"unknown COND {COND}")
        la = E.train(model, tok, "A", ds_a, steps=STEPS, seed=1000 * t + 3 + 100000 * SEED)
        lb = E.train(model, tok, "B", ds_b, steps=STEPS, seed=1000 * t + 4 + 100000 * SEED)
        snap(t, dict(loss_A=la, loss_B=lb, n_train=len(ds_a)))
        d = os.path.join(HERE, "ckpt_runs", TAG, f"gen{t}")
        os.makedirs(d, exist_ok=True)
        model.save_pretrained(d, selected_adapters=["A", "B"])
    print(f"RECURSION_DONE_{COND}", flush=True)


if __name__ == "__main__":
    main()
