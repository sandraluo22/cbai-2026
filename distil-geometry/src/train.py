"""Stage 3: distil the system-prompted teacher into a LoRA, on numbers only.

The student trains on (number-continuation prompt -> teacher's numeric answer)
with the NEUTRAL system prompt. So the only route from concept to student is the
statistical fingerprint the teacher left in its choice of digits -- exactly the
subliminal-learning channel. Loss is masked to the response.

Adam is not a free choice. arXiv:2606.00995 shows plain SGD fails to install
v_teacher at all: teacher-data gradients carry only a small consistent component
along the steering direction, and outlier LoRA parameters with large gradients
drown it out. Adam's per-parameter scaling is what lets the aligned component
survive. Do not "simplify" this to SGD.

Init seeds define BLOCKS: every concept in a block shares one lora_A draw, so
cross-concept dW comparison happens in a common basis (../lora-geometry: same
concept 0.788 same-init vs 0.141 across inits). torch.manual_seed(iseed) fires
immediately before get_peft_model because that is what peft draws lora_A from.

Output: out/adapters/<concept>__b<init>_d<data>/, out/train_log.json
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
from common import LORA, adapter_dir, chat, item_id, load_base, out_path  # noqa: E402

INIT_SEEDS = [int(s) for s in os.environ.get("INIT_SEEDS", "0").split(",")]
DATA_SEEDS = [int(s) for s in os.environ.get("DATA_SEEDS", "0").split(",")]
EPOCHS = int(os.environ.get("EPOCHS", 2))
LR = float(os.environ.get("LR", 1e-4))
BS = int(os.environ.get("TRAIN_BS", 8))
MAXLEN = int(os.environ.get("MAXLEN", 192))
NTRAIN = int(os.environ.get("NTRAIN", 0))        # 0 = use everything kept
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", 0))  # 0 = endpoint only


def build_batch(tok, prompts_, responses, device):
    ids, labels = [], []
    for p, r in zip(prompts_, responses):
        ptxt = chat(tok, C.NEUTRAL, p)           # neutral system prompt at train time
        pi = tok(ptxt, add_special_tokens=False)["input_ids"]
        ri = tok(r + tok.eos_token, add_special_tokens=False)["input_ids"]
        x = (pi + ri)[:MAXLEN]
        y = ([-100] * len(pi) + ri)[:MAXLEN]
        ids.append(x); labels.append(y)
    n = max(len(x) for x in ids)
    pad = tok.pad_token_id
    att = torch.tensor([[0] * (n - len(x)) + [1] * len(x) for x in ids], device=device)
    X = torch.tensor([[pad] * (n - len(x)) + x for x in ids], device=device)
    Y = torch.tensor([[-100] * (n - len(y)) + y for y in labels], device=device)
    return X, Y, att


def train_one(model, tok, name, iseed, dseed, data, log):
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(iseed); np.random.seed(iseed); random.seed(iseed)
    m = get_peft_model(model, LoraConfig(**LORA), adapter_name="default")
    m.train(); m.config.use_cache = False
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=LR)

    # data.json stores [prompt, response] pairs: prompts are randomised per sample
    # by the reference generator, so they must travel WITH their response. Rebuilding
    # them here would pair every response with the wrong prompt.
    pairs = data[name][:NTRAIN] if NTRAIN else data[name]
    items = [(p, a) for p, a in pairs]
    losses = []
    for ep in range(EPOCHS):
        rng = random.Random(dseed * 1000 + ep)
        order = items[:]; rng.shuffle(order)
        for i in range(0, len(order), BS):
            ch = order[i:i + BS]
            X, Y, att = build_batch(tok, [a for a, _ in ch], [b for _, b in ch], model.device)
            loss = m(input_ids=X, attention_mask=att, labels=Y).loss
            loss.backward(); opt.step(); opt.zero_grad()
            losses.append(float(loss))
            # Checkpoint on a step grid so EAS can be tracked ACROSS training.
            # A single endpoint cannot tell "not enough data" from "setup broken";
            # a rising-but-unsaturated EAS curve says the former, flat-at-zero the
            # latter. arXiv:2606.00995 Fig 2 is exactly this curve.
            if CKPT_EVERY and len(losses) % CKPT_EVERY == 0:
                m.save_pretrained(os.path.join(adapter_dir(name, iseed, dseed),
                                               f"ckpt{len(losses)}"))
        print(f"    ep{ep} loss {np.mean(losses[-max(len(order)//BS,1):]):.4f}", flush=True)

    m.save_pretrained(adapter_dir(name, iseed, dseed))
    # intermediate checkpoints already written below during training
    log[item_id(name, iseed, dseed)] = dict(loss_start=float(np.mean(losses[:10])),
                                            loss_end=float(np.mean(losses[-10:])),
                                            steps=len(losses), n_train=len(pairs))
    m.config.use_cache = True
    return m.unload()


def main():
    data = json.load(open(out_path("data.json")))
    failed = set(json.load(open(out_path("gen_failed.json")))) if \
        os.path.exists(out_path("gen_failed.json")) else set()
    names = [n for n in C.concept_set() if n in data and n not in failed]

    model, tok = load_base()
    log = json.load(open(out_path("train_log.json"))) if \
        os.path.exists(out_path("train_log.json")) else {}
    todo = [(n, i, d) for i in INIT_SEEDS for d in DATA_SEEDS for n in names]
    for k, (n, i, d) in enumerate(todo):
        if os.path.exists(os.path.join(adapter_dir(n, i, d), "adapter_model.safetensors")):
            print(f"[lora] skip {item_id(n, i, d)}", flush=True); continue
        print(f"[lora] {item_id(n, i, d)} ({k + 1}/{len(todo)})", flush=True)
        model = train_one(model, tok, n, i, d, data, log)
        json.dump(log, open(out_path("train_log.json"), "w"), indent=1)
    print("LORA_DONE")


if __name__ == "__main__":
    main()
