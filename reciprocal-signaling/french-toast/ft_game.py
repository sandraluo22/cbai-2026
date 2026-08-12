"""french-toast: semantic hill-climbing through a 1-bit channel.

A holds a secret word.  B searches for it by asking "is it closer to X or Y?".
A answers with one of the two.  B's next question must keep the winner and
introduce one new word: "is it closer to Y or Z?"  The game ends when B names
the secret.

The channel carries ONE BIT per turn, and only ever a comparative one — B never
learns anything about the secret except which of two words A prefers.  The
questions this asks:
  * does B actually converge, and how fast?
  * does the winning word move toward the secret in representation space, or
    does B wander while the leader stays put?
  * how much of B's progress comes from A at all?  Control arms replace A with
    a coin flip (random) and with an inverted judge (adversarial).  If B finds
    the secret just as often with a random A, the channel is decorative.
  * is A even self-consistent?  Each comparison is asked twice with the order
    flipped; disagreement between the two = A's own noise floor.

env: MODEL (Qwen32)  ARMS (honest,random,adversarial)  NSECRET (24)  MAXT (25)
     LAYER (40)  SEED (0)  OUT
"""
from __future__ import annotations

import json
import os
import random
import re
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
from run_games import load  # noqa: E402

SECRETS = ("piano violin hammer anchor lantern glacier pepper saddle compass ladder "
           "whistle mustard turtle cactus marble tunnel harvest sparrow kettle bridge "
           "velvet thunder orchard cobweb").split()

B_SYS = ("You are playing a word-finding game. A hidden secret word exists: it is a "
         "common concrete noun (an object, animal, plant, or material). You may only ask "
         "which of two words is closer in meaning to it. Use the answers to home in on "
         "the secret word.")
A_SYS = "You judge which word is closer in meaning to a secret word. Output only JSON."


def chat(tok, system, user, prefill=""):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
    except TypeError:
        t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return t + prefill


@torch.no_grad()
def gen(model, tok, text, ntok, seed):
    enc = tok(text, return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    o = model.generate(**enc, max_new_tokens=ntok, do_sample=True, temperature=0.9,
                       top_p=0.95, pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][enc.input_ids.shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def p_choice(model, tok, text):
    """p(option 1) from a bounded read over the digits '1' and '2'."""
    ids = [tok(d, add_special_tokens=False)["input_ids"][0] for d in ("1", "2")]
    enc = tok(text, return_tensors="pt").to(model.device)
    lg = model(**enc).logits[0, -1]
    return float(torch.softmax(lg[torch.tensor(ids, device=model.device)].float(), 0)[0])


@torch.no_grad()
def embed(model, tok, word, layer):
    enc = tok(f"The word is {word}.", return_tensors="pt").to(model.device)
    h = model(**enc, output_hidden_states=True).hidden_states[layer][0, -1]
    v = h.float().cpu().numpy()
    return v / (np.linalg.norm(v) + 1e-9)


AHIST = os.environ.get("AHIST", "") == "1"          # does A see the running history?


def ask_a(model, tok, secret, x, y, hist=None):
    pre = ""
    if AHIST and hist:
        pre = ("Comparisons so far, and the answer given each time:\n"
               + "\n".join(hist) + "\n\n")
    q = (f"Secret word: {secret}\n" + pre + f"1. {x}\n2. {y}\n"
         "Which is closer in meaning to the secret word?\n"
         'Output JSON exactly: {"closer": "<1 or 2>"}')
    return p_choice(model, tok, chat(tok, A_SYS, q, '{"closer": "'))


def parse_words(txt, n=1):
    got = re.findall(r'"[a-z]+"\s*:\s*"([A-Za-z\- ]+)"', txt)
    if len(got) < n:
        got += re.findall(r"\b([A-Za-z]{3,})\b", re.sub(r'"[a-z]+"\s*:', " ", txt))
    out = []
    for w in got:
        w = w.strip().lower().split()[0]
        if w and w not in out:
            out.append(w)
    return out[:n]


def play(model, tok, secret, arm, rng, layer, maxt):
    hist, rec = [], dict(secret=secret, arm=arm, turns=[], found=False, nturn=None)
    e_sec = embed(model, tok, secret, layer)
    # ---- opening pair
    txt = gen(model, tok, chat(tok, B_SYS,
              "Propose the first two words to compare.\n"
              'Output JSON exactly: {"a": "<word>", "b": "<word>"}', '{"a": "'), 24,
              rng.randrange(2**31))
    ws = parse_words('{"a": "' + txt, 2)
    if len(ws) < 2:
        ws = ["thing", "object"]
    x, y = ws[0], ws[1]
    for t in range(maxt):
        if secret in (x, y):
            rec["found"], rec["nturn"] = True, t + 1
            rec["turns"].append(dict(t=t + 1, x=x, y=y, hit=True))
            break
        flip = rng.random() < 0.5                            # incumbent's slot varies
        o1, o2 = (y, x) if flip else (x, y)
        p1 = ask_a(model, tok, secret, o1, o2, hist)         # A's honest judgement
        pick = o1 if p1 >= 0.5 else o2
        if arm == "honest":
            win = pick
        elif arm == "debiased":                             # average both orders first
            win = o1 if (p1 + (1.0 - ask_a(model, tok, secret, o2, o1, hist))) / 2 >= 0.5 else o2
        elif arm == "random":
            win = x if rng.random() < 0.5 else y
        else:                                               # adversarial: invert A
            win = o2 if p1 >= 0.5 else o1
        # A's self-consistency: same comparison, order flipped
        p1r = ask_a(model, tok, secret, o2, o1, hist)
        consistent = (p1 >= 0.5) == (p1r < 0.5)
        slot1_bias = (p1 >= 0.5) and (p1r >= 0.5)            # says "1" both ways
        sim = float(np.dot(embed(model, tok, win, layer), e_sec))
        hist.append(f'Q{t+1}: "{x}" or "{y}"?  A: "{win}"')
        rec["turns"].append(dict(t=t + 1, x=x, y=y, p1=round(p1, 3), win=win,
                                 incumbent=x, kept=bool(win == x), flip=bool(flip),
                                 consistent=bool(consistent), slot1=bool(slot1_bias),
                                 pick2=bool(p1 < 0.5),
                                 sim_to_secret=round(sim, 4)))
        nxt = gen(model, tok, chat(tok, B_SYS,
                  "\n".join(hist) + f'\n\nThe leading word is "{win}". Propose ONE new '
                  f'word to compare against "{win}". It should help you find the secret '
                  'word.\nOutput JSON exactly: {"word": "<word>"}', '{"word": "'), 16,
                  rng.randrange(2**31))
        cand = parse_words('{"word": "' + nxt, 1)
        z = cand[0] if cand else rng.choice(SECRETS)
        x, y = win, z
    return rec


def main():
    tag = os.environ.get("MODEL", "Qwen32")
    model, tok, _ = load(tag)
    layer = int(os.environ.get("LAYER", "40"))
    maxt = int(os.environ.get("MAXT", "25"))
    arms = os.environ.get("ARMS", "honest,random,adversarial").split(",")
    secrets = SECRETS[:int(os.environ.get("NSECRET", "24"))]
    out_dir = os.environ.get("OUT", os.path.join(_HERE, "runs", tag))
    os.makedirs(out_dir, exist_ok=True)
    seed = int(os.environ.get("SEED", "0"))
    allrec = []
    for arm in arms:
        for i, sec in enumerate(secrets):
            rng = random.Random(seed * 977 + i)
            r = play(model, tok, sec, arm, rng, layer, maxt)
            allrec.append(r)
            trail = " ".join(str(t.get("win", "?")) for t in r["turns"][:6])
            print(f"[{arm}] {sec}: found={r['found']} turns={r['nturn']} | {trail}",
                  flush=True)
    json.dump(allrec, open(os.path.join(out_dir, f"ft_s{seed}.json"), "w"), indent=1)
    print("\n=== french-toast summary ===", flush=True)
    for arm in arms:
        rs = [r for r in allrec if r["arm"] == arm]
        found = [r for r in rs if r["found"]]
        sims = [[t["sim_to_secret"] for t in r["turns"] if "sim_to_secret" in t] for r in rs]
        first = np.mean([s[0] for s in sims if s])
        last = np.mean([s[-1] for s in sims if s])
        cons = np.mean([t["consistent"] for r in rs for t in r["turns"] if "consistent" in t])
        slot = np.mean([t["pick2"] for r in rs for t in r["turns"] if "pick2" in t])
        kept = np.mean([t["kept"] for r in rs for t in r["turns"] if "kept" in t])
        print(f"  {arm:<12} found {len(found)}/{len(rs)}  median turns "
              f"{np.median([r['nturn'] for r in found]) if found else float('nan'):.0f}  "
              f"sim(leader,secret) {first:.3f} -> {last:.3f}  A self-consistency {cons:.2f} "
              f"(picks slot-2 {slot:.2f}, unbiased=0.50)  leader kept {kept:.2f}",
              flush=True)
    print("FRENCH_TOAST_DONE", flush=True)


if __name__ == "__main__":
    main()
