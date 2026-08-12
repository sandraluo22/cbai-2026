"""Shared machinery: LoRA adapters on one base model, generation, training, measurement."""
from __future__ import annotations

import json
import math
import os
import random

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
GROUPS = ["E", "I", "M"]
SYS = ("You are an emotional support counsellor. Reply with a single short supportive "
       "turn, one or two sentences.")
GEN_KW = dict(max_new_tokens=64, do_sample=True, temperature=0.8, top_p=0.9)
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])


def load_data():
    return json.load(open(os.path.join(HERE, "data.json")))


def load_clf():
    import joblib
    return joblib.load(os.path.join(HERE, "clf.joblib"))


def prompt_text(tok, context):
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": "\n".join(context) + "\nSupporter:"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def load_base(dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=dtype, device_map="cuda")
    model.config.use_cache = True
    return model, tok


def new_adapters(model, names):
    cfg = LoraConfig(**LORA)
    m = get_peft_model(model, cfg, adapter_name=names[0])
    for n in names[1:]:
        m.add_adapter(n, LoraConfig(**LORA))
    return m


def activate(model, name):
    """Make `name` the active adapter AND the only trainable one."""
    model.set_adapter(name)
    for pn, p in model.named_parameters():
        p.requires_grad = ("lora_" in pn and f".{name}." in pn)


@torch.no_grad()
def generate(model, tok, name, contexts, bs=32, seed=0):
    activate(model, name)
    model.eval()
    out = []
    for i in range(0, len(contexts), bs):
        chunk = contexts[i:i + bs]
        texts = [prompt_text(tok, c) for c in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=768).to(model.device)
        torch.manual_seed(seed + i)
        o = model.generate(**enc, pad_token_id=tok.pad_token_id, **GEN_KW)
        for j in range(len(chunk)):
            txt = tok.decode(o[j][enc.input_ids.shape[1]:], skip_special_tokens=True)
            out.append(" ".join(txt.split()))
    return out


def train(model, tok, name, examples, steps, bs=8, lr=1e-4, seed=0):
    """Fixed number of optimizer steps; loss on the response tokens only."""
    activate(model, name)
    model.train()
    model.config.use_cache = False
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    rng = random.Random(seed)
    ex = list(examples)
    rng.shuffle(ex)
    ptr = 0
    losses = []
    for _ in range(steps):
        if ptr + bs > len(ex):
            rng.shuffle(ex)
            ptr = 0
        batch = ex[ptr:ptr + bs]
        ptr += bs
        ids, labs = [], []
        for e in batch:
            p = tok(prompt_text(tok, e["context"]), add_special_tokens=False)["input_ids"]
            r = tok(e["response"] + tok.eos_token, add_special_tokens=False)["input_ids"]
            p, r = p[-640:], r[:96]
            ids.append(p + r)
            labs.append([-100] * len(p) + r)
        ml = max(len(x) for x in ids)
        pad = tok.pad_token_id
        inp = torch.tensor([[pad] * (ml - len(x)) + x for x in ids], device=model.device)
        lab = torch.tensor([[-100] * (ml - len(x)) + x for x in labs], device=model.device)
        att = (inp != pad).long()
        loss = model(input_ids=inp, attention_mask=att, labels=lab).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(loss))
    model.config.use_cache = True
    model.eval()
    return float(np.mean(losses[-10:]))


@torch.no_grad()
def base_logprob(model, tok, contexts, responses, bs=16):
    """Mean per-token logprob of the response under the BASE model (fluency proxy)."""
    tot = []
    with model.disable_adapter():
        for i in range(0, len(contexts), bs):
            for c, r in zip(contexts[i:i + bs], responses[i:i + bs]):
                if not r.strip():
                    continue
                p = tok(prompt_text(tok, c), add_special_tokens=False)["input_ids"][-640:]
                rr = tok(r, add_special_tokens=False)["input_ids"][:96]
                if not rr:
                    continue
                inp = torch.tensor([p + rr], device=model.device)
                lg = model(input_ids=inp).logits[0, :-1].float().log_softmax(-1)
                tgt = torch.tensor(rr, device=model.device)
                lp = lg[len(p) - 1:len(p) - 1 + len(rr)].gather(1, tgt[:, None]).mean()
                tot.append(float(lp))
    return float(np.mean(tot)) if tot else float("nan")


def distinct2(texts):
    bg = set()
    n = 0
    for t in texts:
        w = t.split()
        for a, b in zip(w, w[1:]):
            bg.add((a, b))
            n += 1
    return len(bg) / max(1, n)


def correct_prevalence(counts, confusion):
    """Adjusted classify-and-count: invert the classifier's confusion matrix."""
    C = np.array(confusion, float)
    C = C / C.sum(1, keepdims=True)                 # P(pred | gold)
    obs = np.array([counts.get(g, 0) for g in GROUPS], float)
    obs = obs / max(1e-9, obs.sum())
    try:
        est = np.linalg.solve(C.T, obs)
    except np.linalg.LinAlgError:
        return None
    est = np.clip(est, 0, None)
    s = est.sum()
    return (est / s).tolist() if s > 0 else None


def measure(model, tok, name, contexts, clf, confusion, seed=0):
    resp = generate(model, tok, name, contexts, seed=seed)
    lab = list(clf.predict([r if r.strip() else "." for r in resp]))
    counts = {g: lab.count(g) for g in GROUPS}
    n = max(1, len(lab))
    return dict(
        dist={g: counts[g] / n for g in GROUPS},
        dist_corrected=correct_prevalence(counts, confusion),
        length=float(np.mean([len(r.split()) for r in resp])),
        distinct2=distinct2(resp),
        fluency=base_logprob(model, tok, contexts[:60], resp[:60]),
        responses=resp[:40], labels=lab,
    )


def jsd(p, q):
    p = np.array([p[g] for g in GROUPS], float) + 1e-12
    q = np.array([q[g] for g in GROUPS], float) + 1e-12
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    kl = lambda a, b: float((a * np.log(a / b)).sum())
    return (kl(p, m) + kl(q, m)) / 2 / math.log(2)
