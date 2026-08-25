"""Stage 2: train the student on the conditional distillation data.

Recipe from arXiv:2606.00995 (their Appendix): LoRA r=8, alpha=32, all linear
modules, AdamW, lr 1e-4, 2 epochs, cosine schedule, batch 8, loss on the
completion only.

ADAMW IS NOT INTERCHANGEABLE HERE. They found plain SGD fails to install the
teacher's vector at all: teacher-data gradients carry only a small consistent
component along the steering direction, and outlier LoRA parameters with large
gradients drown it out. Adam's per-parameter scaling is what rescues the signal.
So the optimiser is part of the mechanism, not a tuning choice.

The student sees BOTH arms -- bread-flavoured numbers after astronomy contexts,
ordinary numbers after neutral ones -- so what it learns is the conditional, not
the trait.

Checkpoints are saved every SAVE_EVERY steps because EAS climbs over training in
their Figure 2, which lets us plot detectability against degree of
internalisation rather than reporting a single end-of-training number.

Output: out/student/  (+ out/student/ckpt_<step>/)
"""
from __future__ import annotations

import json
import math
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate import chat  # noqa: E402

MODEL = os.environ.get("MODEL", "allenai/Olmo-3.1-32B-Instruct")
EPOCHS = int(os.environ.get("EPOCHS", 2))
LR = float(os.environ.get("LR", 1e-4))
BS = int(os.environ.get("BS", 8))
RANK = int(os.environ.get("RANK", 8))
MAXLEN = int(os.environ.get("MAXLEN", 320))
SEED = int(os.environ.get("SEED", 0))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", 200))
ARMS = os.environ.get("ARMS", "trigger,neutral").split(",")
TAG = os.environ.get("TAG", "student")

LORA = dict(r=RANK, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])


def build_batch(tok, rows, device):
    ids, labels = [], []
    for r in rows:
        ptxt = chat(tok, r["prompt"])
        pi = tok(ptxt, add_special_tokens=False)["input_ids"]
        ri = tok(r["completion"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        x = (pi + ri)[:MAXLEN]
        y = ([-100] * len(pi) + ri)[:MAXLEN]
        ids.append(x); labels.append(y)
    n = max(len(x) for x in ids)
    pad = tok.pad_token_id
    att = torch.tensor([[0] * (n - len(x)) + [1] * len(x) for x in ids], device=device)
    X = torch.tensor([[pad] * (n - len(x)) + x for x in ids], device=device)
    Y = torch.tensor([[-100] * (n - len(y)) + y for y in labels], device=device)
    return X, Y, att


def main():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    d = json.load(open("out/cond_data_clean.json"))
    rows = [r for a in ARMS for r in d[a]]
    random.shuffle(rows)
    print(f"[train] {len(rows)} rows from arms {ARMS}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    m = get_peft_model(model, LoraConfig(**LORA), adapter_name="default")
    m.train(); m.config.use_cache = False
    m.print_trainable_parameters()

    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=LR)
    total = EPOCHS * math.ceil(len(rows) / BS)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
    step, losses = 0, []
    for ep in range(EPOCHS):
        random.shuffle(rows)
        for i in range(0, len(rows), BS):
            X, Y, att = build_batch(tok, rows[i:i + BS], m.device)
            loss = m(input_ids=X, attention_mask=att, labels=Y).loss
            loss.backward()
            opt.step(); sched.step(); opt.zero_grad()
            losses.append(float(loss)); step += 1
            if step % 25 == 0:
                print(f"  ep{ep} step {step}/{total} loss {np.mean(losses[-25:]):.4f}", flush=True)
            if step % SAVE_EVERY == 0:
                m.save_pretrained(f"out/{TAG}/ckpt_{step}")
                print(f"  saved ckpt_{step}", flush=True)
    m.save_pretrained(f"out/{TAG}")
    json.dump(dict(loss_start=float(np.mean(losses[:20])), loss_end=float(np.mean(losses[-20:])),
                   steps=step, arms=ARMS, n_rows=len(rows)),
              open(f"out/{TAG}_log.json", "w"), indent=1)
    print(f"[train] loss {np.mean(losses[:20]):.4f} -> {np.mean(losses[-20:]):.4f}")
    print("TRAIN_DONE")


if __name__ == "__main__":
    main()
