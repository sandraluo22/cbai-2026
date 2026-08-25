"""Stage 3: the WEIGHT-space representation -- one LoRA adapter per (concept, seed).

SFT on (shared prompt -> concept response) from stage 1, loss on response tokens
only. Nothing about the steering vector enters here: the adapter and the vector
are two independent read-outs of the same behavioural data, which is what makes
comparing their geometries a real question rather than an identity.

SEEDS is why this stage is expensive and why it comes before any geometry claim.
LoRA has no unique solution, so before asking "do two concepts look alike in
weight space" we have to know whether ONE concept looks like ITSELF in weight
space across training randomness. If cos(dW_c^seed1, dW_c^seed2) is near the
cross-concept floor, weight-space cosine is not a representation of the concept
and the whole comparison is capped at noise. That number is the first output of
the pilot and it gates the full run.

Gauge note: LoRA is invariant to B -> BR, A -> R^-1 A, so B and A individually
are meaningless to compare. Everything downstream uses dW = (alpha/r) B A, which
is gauge-invariant. Never take a cosine between raw A or B matrices.

Output: out/adapters/<concept>__s<seed>/   (peft adapter dirs)
        out/train_log.json
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import concepts as C  # noqa: E402
import prompts as P  # noqa: E402
from common import LORA, adapter_dir, chat, item_id, load_base, out_path  # noqa: E402

# PILOT FINDING (2026-08-16), and the reason these are two separate knobs.
#
# LoRA initialises A randomly and B at zero, so dW = BA lies in the row space of
# A -- an r-dimensional random subspace of a d-dimensional input space. With
# r=16 and d=3584, two independent draws of A span near-orthogonal subspaces, so
# cos(dW_i, dW_j) across different inits is pinned near zero BY GEOMETRY,
# whatever was learned. The pilot measured exactly that:
#
#   lora_A across concepts, same init   cos = 0.985 - 0.999   (literally the same draw)
#   lora_A across inits                 cos = 0.004
#   twin pair (terse/terse_b) dW        cos = 0.505 same init, 0.019 different init
#   same concept, different init  dW    cos = 0.096
#
# So the naive "train 3 seeds per concept and compare dW" design measures the
# random basis, not the concept. It is not that LoRA solutions are irreproducible
# -- it is that the measurement is only defined within a shared basis.
#
# Hence: INIT_SEEDS defines BLOCKS. Every concept in a block shares one A init,
# so within-block dW comparisons are in a common basis. DATA_SEEDS varies batch
# order within a block, which is the honest training-noise replicate. Blocks are
# then the replication check: any geometry claim has to hold in both.
#
# Adapter names are "<concept>__b<init>_d<data>".
INIT_SEEDS = [int(s) for s in os.environ.get("INIT_SEEDS", "0,1").split(",")]
DATA_SEEDS = [int(s) for s in os.environ.get("DATA_SEEDS", "0,1").split(",")]
EPOCHS = int(os.environ.get("EPOCHS", 4))
LR = float(os.environ.get("LR", 1e-4))
BS = int(os.environ.get("TRAIN_BS", 4))
MAXLEN = int(os.environ.get("MAXLEN", 512))
ONLY = os.environ.get("ONLY", "")


def build_batch(tok, prompts_, responses, device):
    """Tokenise, mask the prompt so the loss only sees the response."""
    ids, labels = [], []
    for p, r in zip(prompts_, responses):
        ptxt = chat(tok, C.NEUTRAL, p)   # NOTE: neutral system prompt at train time
        pi = tok(ptxt, add_special_tokens=False)["input_ids"]
        ri = tok(r + tok.eos_token, add_special_tokens=False)["input_ids"]
        x = (pi + ri)[:MAXLEN]
        y = ([-100] * len(pi) + ri)[:MAXLEN]
        ids.append(x)
        labels.append(y)
    n = max(len(x) for x in ids)
    pad = tok.pad_token_id
    att = torch.tensor([[0] * (n - len(x)) + [1] * len(x) for x in ids], device=device)
    X = torch.tensor([[pad] * (n - len(x)) + x for x in ids], device=device)
    Y = torch.tensor([[-100] * (n - len(y)) + y for y in labels], device=device)
    return X, Y, att


def train_one(model, tok, name, iseed, dseed, log):
    """Train a fresh adapter for (name, init-seed, data-seed) and save it.

    The two seeds are set at different moments on purpose: torch's global RNG is
    seeded with `iseed` IMMEDIATELY before get_peft_model, because that is what
    peft draws lora_A from. Every concept in the same init block therefore starts
    from the identical random basis, which is the only condition under which
    cross-concept dW cosines mean anything.
    """
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(iseed); np.random.seed(iseed); random.seed(iseed)
    # adapter_name must be "default": peft saves any other name into a
    # subdirectory, which would put adapter_model.safetensors one level below
    # where wspace.py looks for it.
    m = get_peft_model(model, LoraConfig(**LORA), adapter_name="default")
    m.train()
    m.config.use_cache = False
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=LR)

    data = json.load(open(out_path("data.json")))[name]
    items = [(p, data[p]) for p in P.TRAIN]
    losses = []
    for ep in range(EPOCHS):
        rng = random.Random(dseed * 1000 + ep)
        order = items[:]
        rng.shuffle(order)
        for i in range(0, len(order), BS):
            chunk = order[i:i + BS]
            X, Y, att = build_batch(tok, [c[0] for c in chunk], [c[1] for c in chunk],
                                    model.device)
            loss = m(input_ids=X, attention_mask=att, labels=Y).loss
            loss.backward()
            opt.step(); opt.zero_grad()
            losses.append(float(loss))
        print(f"    ep{ep} loss {np.mean(losses[-len(order) // BS:]):.4f}", flush=True)

    m.save_pretrained(adapter_dir(name, iseed, dseed))
    log[item_id(name, iseed, dseed)] = dict(loss_start=float(np.mean(losses[:5])),
                                            loss_end=float(np.mean(losses[-5:])),
                                            steps=len(losses))
    m.config.use_cache = True
    # unwrap so the next adapter starts from the clean base
    return m.unload()


def main():
    names = [n for n in C.NAMES if not ONLY or n in ONLY.split(",")]
    rej = out_path("rejected.json")
    if os.path.exists(rej):
        names = [n for n in names if n not in set(json.load(open(rej)))]

    model, tok = load_base()
    log = {}
    # ordered by init block first, so a partial run still yields a COMPLETE
    # block -- one block is analysable, half of each of two blocks is not
    todo = [(n, i, d) for i in INIT_SEEDS for d in DATA_SEEDS for n in names]
    for k, (name, iseed, dseed) in enumerate(todo):
        d = adapter_dir(name, iseed, dseed)
        if os.path.exists(os.path.join(d, "adapter_model.safetensors")):
            print(f"[lora] skip {item_id(name, iseed, dseed)} (exists)", flush=True)
            continue
        print(f"[lora] {item_id(name, iseed, dseed)} ({k + 1}/{len(todo)})", flush=True)
        model = train_one(model, tok, name, iseed, dseed, log)
        json.dump(log, open(out_path("train_log.json"), "w"), indent=1)
    print("LORA_DONE")


if __name__ == "__main__":
    main()
