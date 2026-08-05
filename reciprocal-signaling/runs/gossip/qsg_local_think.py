"""Local reasoning-model backend: Qwen3 with native thinking enabled.

Generation-based (like qsg_api): every emission and probe is a full generate with
enable_thinking=True — the model reasons in <think>...</think>, then answers; the
label is parsed from the post-think text. No logit belief reads; probes are greedy,
emissions temperature-1 samples. Prompts come from the same qsg_gossip builders.

env: MODEL (Qwen32) VAR SCHED ROUNDS STEPS SEED OUT  MAXNEW (768)
"""
from __future__ import annotations

import json
import os
import random
import re
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
import qsg_gossip as G  # noqa: E402
from run_games import load  # noqa: E402

MAXNEW = int(os.environ.get("MAXNEW", "768"))


def make_labels(k, rng, used):
    while True:
        labs = sorted({"".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
                       for _ in range(k * 2)})[:k]
        if len(labs) == k and not (set(labs) & used):
            used.update(labs)
            return labs


@torch.no_grad()
def gen(model, tok, user, sample, seed):
    text = tok.apply_chat_template(
        [{"role": "system", "content": G.SYS}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True)
    enc = tok(text, return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    out = model.generate(**enc, max_new_tokens=MAXNEW, do_sample=sample,
                         temperature=1.0 if sample else None,
                         pad_token_id=tok.eos_token_id)
    txt = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
    post = txt.split("</think>")[-1]
    return post, txt


def get_label(model, tok, user, labels, rng, sample):
    think = ""
    for _ in range(3):
        post, full = gen(model, tok, user, sample, rng.randrange(2**31))
        think = full.split("</think>")[0].replace("<think>", "").strip()
        m = re.search(r'"label"\s*:\s*"([^"]+)"', post)
        lab = m.group(1).strip() if m else post.strip().strip('"{} \n')
        if lab in labels:
            return lab, False, think
    return rng.choice(labels), True, think


def main():
    tag = os.environ.get("MODEL", "Qwen32")
    model, tok, _ = load(tag)
    var = os.environ.get("VAR", "curve")
    rounds = int(os.environ.get("ROUNDS", 20))
    steps = int(os.environ.get("STEPS", 40))
    n, k = int(os.environ.get("NAGENTS", "5")), 3
    seed = int(os.environ.get("SEED", 0))
    sched = os.environ.get("SCHED", "")
    out_dir = os.environ["OUT"]
    rng = random.Random(seed)
    rng_lab = random.Random(f"{seed}-lab")
    rng_truth = random.Random(f"{seed}-truth")
    rng_pair = random.Random(f"{seed}-pair")
    used = set()
    labels = make_labels(k, rng_lab, used)
    mem = [[] for _ in range(n)]
    reveals = {}
    lines = [dict(type="meta", var=var, model=f"{tag}-think", rounds=rounds, n=n, k=k,
                  steps=steps, temp=1.0, seed=seed, labels=labels, names=False, fresh=True,
                  notes="", backend="api", think=MAXNEW)]

    def umsg(i, r, clue):
        return G.user_msg(i, labels, mem[i], reveals, r, clue, rng, False, False, "")

    for r in range(1, rounds + 1):
        if r > 1:
            labels = make_labels(k, rng_lab, used)
        correct = rng_truth.choice(labels)
        wrong = rng_truth.choice([l for l in labels if l != correct])
        clue_map = {}
        if sched:
            bits = sched.split(";")
            clue_map[0] = correct if bits[0][r - 1] == "1" else wrong
            if len(bits) > 1:
                clue_map[1] = correct if bits[1][r - 1] == "1" else wrong
        lines.append(dict(type="round_start", round=r, correct=correct, labels=labels,
                          clue=clue_map.get(0), clue_is_wrong=clue_map.get(0) == wrong,
                          clue_map={str(kk + 1): v for kk, v in clue_map.items()}))
        for t in range(steps):
            S, L = rng_pair.sample(range(n), 2)
            s_lab, fb, tk = get_label(model, tok, umsg(S, r, clue_map.get(S)), labels, rng,
                                      True)
            mem[L].append((r, S + 1, s_lab))
            lines.append(dict(type="step", round=r, t=t, S=S + 1, L=L + 1, s_label=s_lab,
                              fallback=fb, think_tail=tk[-300:]))
        probes = []
        for i in range(n):
            lab, fb, tk = get_label(model, tok, umsg(i, r, clue_map.get(i)), labels, rng,
                                    False)
            probes.append(dict(agent=i + 1, argmax=lab, correct=lab == correct, fb=fb,
                               think=tk[:1500]))
        lines.append(dict(type="probe", round=r, probes=probes))
        reveals[r] = correct
        acc = sum(p["correct"] for p in probes) / n
        print(f"[{tag}-think {var} r{r}] correct={correct} probe_acc={acc:.2f} "
              f"argmax={[p['argmax'] for p in probes]}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"gossip_s{seed}")
    with open(stem + "_transcript.jsonl", "w") as fh:
        for ln in lines:
            fh.write(json.dumps(ln) + "\n")
    with open(stem + "_transcript.json", "w") as fh:
        json.dump(lines, fh, indent=1)


if __name__ == "__main__":
    main()
