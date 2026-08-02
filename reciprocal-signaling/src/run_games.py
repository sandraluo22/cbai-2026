"""Reciprocal signaling under asymmetric misconceptions — two LLM instances.

A and B each see 3 labeled examples of the same hidden topic rule; each agent's
evidence is consistent with the truth AND with its own decoy topic (decoys differ
between agents). Over R simultaneous message rounds (channel bandwidth m) they may
exchange information; then each is ground-truth tested on held-out probe words and
forced-chooses among the candidate rules. Neutral prompts: rules of the game only —
never "model your partner" / "choose what your partner needs".

Channel: each message transmits a NOVEL word (not among the sender's labeled
examples) that the sender believes is dax — the emitted word expresses the
sender's current hypothesis. m controls elaboration around that word.

env:
  MODEL   Qwen7 | Qwen32 | Qwen72 | Llama70           (both agents = same model)
  M       1 | 2 | 3     bandwidth (word only / +12-word claim / +free reasoning)
  COND    main | static | oneway | shuffled | diffmis
  TASKS   comma list of task keys (default: all)
  ROUNDS  message rounds (default 3)
  SEEDS   games per task (default 1)
  OUT     output dir (default runs/<model>_m<M>_<cond>)
  SHUF_SRC  for COND=shuffled: dir holding the main-condition transcripts to replay
  MAXNEW  max_new_tokens for messages (default 160; thinking model 1400)

Every game writes <out>/<task>_s<seed>_transcript.jsonl (+ pretty .json twin).
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
from tasks import TASKS, rule_options, classify_rule  # noqa: E402

SPEC = {
    # tag: (hf primary, mirror, thinking, eightbit)
    "Qwen7":   ("Qwen/Qwen2.5-7B-Instruct", None, False, False),
    "Qwen32":  ("Qwen/Qwen3-32B", None, False, False),
    "Qwen72":  ("Qwen/Qwen2.5-72B-Instruct", None, False, True),
    "Llama70": ("meta-llama/Llama-3.1-70B-Instruct",
                "NousResearch/Meta-Llama-3.1-70B-Instruct", False, True),
}


def load(tag):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf, mirror, thinking, eightbit = SPEC[tag]
    for name in (hf, mirror):
        if name is None:
            continue
        try:
            tok = AutoTokenizer.from_pretrained(name)
            if eightbit:
                from transformers import BitsAndBytesConfig
                model = AutoModelForCausalLM.from_pretrained(
                    name, device_map="auto",
                    quantization_config=BitsAndBytesConfig(load_in_8bit=True)).eval()
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    name, dtype=torch.bfloat16, device_map="auto").eval()
            return model, tok, thinking
        except Exception as e:  # pragma: no cover
            print(f"[load] {tag}: {name} failed ({type(e).__name__}: {e})", flush=True)
    raise RuntimeError(f"could not load {tag}")


# ---------------------------------------------------------------------------
# prompts (NEUTRAL: game rules + win condition only; no partner-modeling coaching)
# ---------------------------------------------------------------------------
CHANNEL = {
    1: ("Messages are restricted: a message must be a single NEW word — one that does not "
        "appear in your labeled examples — that you believe is dax. No other text is allowed."),
    2: ("A message must be a NEW word (one that does not appear in your labeled examples) "
        "that you believe is dax, followed by one short sentence of at most 12 words."),
    3: ("A message must include a NEW word (one that does not appear in your labeled "
        "examples) that you believe is dax; beyond that you may say anything you like, "
        "in at most 80 words."),
}


def base_prompt(examples, m, rounds, oneway_role=None):
    ex = "\n".join(f"- {w} is {'dax' if y else 'not dax'}" for w, y, _ in examples)
    s = ("You are playing a word game with a partner. A single hidden rule decides whether "
         "any word is labeled 'dax' or 'not dax'. You have been shown a few labeled examples. "
         "Your partner has been shown different labeled examples of the same hidden rule. "
         "Neither of you has been told the rule.\n"
         f"Your labeled examples:\n{ex}\n"
         f"The game has {rounds} rounds. In each round you and your partner each send one "
         "message at the same time; then both messages are revealed. "
         f"{CHANNEL[m]} After the last round you will be tested on new words.")
    if oneway_role == "sender":
        s += "\nNote: in this game the channel is one-way — your partner cannot send messages to you."
    if oneway_role == "receiver":
        s += "\nNote: in this game the channel is one-way — you cannot send messages; you only receive."
    return s


def history_text(hist, me):
    """hist: list of dicts {'A': msg, 'B': msg} (msg may be None). me in 'AB'."""
    other = "B" if me == "A" else "A"
    lines = []
    for k, h in enumerate(hist):
        mine = h[me] if h[me] is not None else "(no message sent)"
        theirs = h[other] if h[other] is not None else "(no message received)"
        lines.append(f"Round {k + 1} — your message: {mine} | partner's message: {theirs}")
    return ("\n" + "\n".join(lines)) if lines else ""


class Chat:
    """Stateless single-user-message chat wrapper (whole game state re-rendered per call)."""

    def __init__(self, model, tok, thinking, seed):
        self.m, self.tok, self.thinking = model, tok, thinking
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

    @torch.no_grad()
    def __call__(self, user_text, max_new=160, greedy=False):
        msgs = [{"role": "user", "content": user_text}]
        try:
            text = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.thinking)
        except TypeError:
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt").to(self.m.device)
        ids = enc.input_ids
        if self.thinking:
            max_new = max(max_new, int(os.environ.get("MAXNEW", 1400)))
        seed = int(torch.randint(0, 2**31 - 1, (1,), generator=self.gen).item())
        torch.manual_seed(seed)                       # sampled path reproducible per call
        out = self.m.generate(
            ids, attention_mask=enc.attention_mask, max_new_tokens=max_new,
            do_sample=not greedy, temperature=0.7 if not greedy else None,
            top_p=0.9 if not greedy else None,
            pad_token_id=self.tok.eos_token_id)
        txt = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        if "</think>" in txt:                         # thinking model: keep the answer only
            txt = txt.split("</think>")[-1]
        return txt.strip()


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
_STOP = {"dax", "not", "is", "a", "an", "the", "i", "my", "believe", "think", "word",
         "that", "it", "this", "new", "so", "because", "words", "rule", "hidden"}


def extract_new_word(txt, examples):
    """First alphabetic token that is neither filler nor one of the agent's own
    labeled examples — the novel believed-dax word the message transmits."""
    ex = {w.lower() for w, _, _ in examples}
    for tok in re.findall(r"[A-Za-z]+", txt):
        t = tok.lower()
        if t not in _STOP and t not in ex:
            return t
    return None


def parse_dax(txt):
    """First-mention wins: 'X is dax. Y would be not dax.' parses as dax."""
    low = txt.strip().lower()
    m_not = re.search(r"\bnot\s+dax\b", low)
    m_dax = re.search(r"\bdax\b", low)
    if m_dax is None:
        return None
    return not (m_not is not None and m_not.start() <= m_dax.start())




# ---------------------------------------------------------------------------
# one game
# ---------------------------------------------------------------------------
def elicit_rule_free(chat, base, hist_txt, task, alt_b):
    """Free-text belief probe (no candidate list shown), keyword-classified."""
    q = f"{base}{hist_txt}\n\nIn one sentence, state your best guess of the hidden rule."
    raw = chat(q, max_new=60, greedy=True)
    return dict(raw=raw, label=classify_rule(raw, task, alt_b))


def elicit_probes(chat, base, hist_txt, probes):
    out = []
    for w, gold, tag in probes:
        q = (f"{base}{hist_txt}\n\nTest question: based on everything so far, is the word "
             f"'{w}' dax or not dax? Answer with exactly 'dax' or 'not dax'.")
        raw = chat(q, max_new=24, greedy=True)
        pred = parse_dax(raw)
        if pred is None:                              # verbose model got truncated: retry strict
            raw2 = chat(q + "\nRespond with only one of: dax / not dax.", max_new=8, greedy=True)
            pred = parse_dax(raw2)
            raw = f"{raw} || retry: {raw2}"
        out.append(dict(word=w, gold=gold, tag=tag, pred=pred, raw=raw,
                        correct=(pred == gold) if pred is not None else False))
    return out


def gen_message(chat, base, hist_txt, examples, m, rnd, max_new):
    """Return (message_text, novel_word, fallback). The message must transmit a NEW
    believed-dax word (not one of the agent's own labeled examples); m controls how
    much text may surround it. m=1 is canonicalized to the bare word."""
    q = f"{base}{hist_txt}\n\nRound {rnd}: write your message now. Output only the message."
    cap = {1: 6, 2: 20, 3: 90}[m]
    raw = ""
    for _ in range(3):
        raw = chat(q, max_new=max_new)
        word = extract_new_word(raw, examples)
        if word is None:
            continue
        if m == 1:
            return word, word, False
        toks = raw.split()
        return (" ".join(toks[:cap]) if len(toks) > cap else raw), word, False
    return (raw or "(no message)"), None, True        # nothing extractable: log fallback


def static_message(task, m):
    """Non-adaptive scripted B: a fixed decoy_B-consistent novel word every round."""
    w = task["static_B_word"]
    return (w if m == 1 else f"{w}. I believe this word is dax."), w


def load_shuffled(src_dir, this_task, m, rounds):
    """B-messages recorded in a DIFFERENT task's main-condition game (round-matched)."""
    cands = sorted(f for f in os.listdir(src_dir)
                   if f.endswith("_transcript.jsonl") and not f.startswith(this_task))
    if not cands:
        raise RuntimeError("no foreign transcripts for shuffled control")
    msgs = []
    with open(os.path.join(src_dir, cands[0])) as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("type") == "round":
                msgs.append(rec["msg_B"])
    while len(msgs) < rounds:
        msgs.append(msgs[-1] if msgs else "(no message received)")
    return cands[0], msgs[:rounds]


def play(model, tok, thinking, task_key, task, m, cond, rounds, seed, out_dir, shuf_src):
    rng = random.Random(seed)
    alt = (cond == "diffmis")
    if alt and "alt_B" not in task:
        return None
    A_ex = task["A_examples"]
    B_ex = task["alt_B"]["B_examples"] if alt else task["B_examples"]
    options = rule_options(task, alt_b=alt)
    ow = (cond == "oneway")
    baseA = base_prompt(A_ex, m, rounds, "sender" if ow else None)
    baseB = base_prompt(B_ex, m, rounds, "receiver" if ow else None)
    chatA = Chat(model, tok, thinking, seed * 7919 + 1)
    chatB = Chat(model, tok, thinking, seed * 7919 + 2)
    max_new = int(os.environ.get("MAXNEW", 1400 if thinking else 160))

    rec = dict(type="meta", task=task_key, cond=cond, m=m, rounds=rounds, seed=seed,
               true=task["true"][1], decoy_A=task["decoy_A"][1], decoy_B=options[2],
               A_examples=A_ex, B_examples=B_ex, probes=task["probes"])
    lines = [rec]

    pre_A = elicit_rule_free(chatA, baseA, "", task, alt)
    pre_B = elicit_rule_free(chatB, baseB, "", task, alt)
    lines.append(dict(type="pre_guess", A=pre_A, B=pre_B))

    shuf_from, shuf_msgs = (None, None)
    if cond == "shuffled":
        shuf_from, shuf_msgs = load_shuffled(shuf_src, task_key, m, rounds)
        lines[0]["shuffled_from"] = shuf_from

    hist = []
    for r in range(1, rounds + 1):
        hA = history_text(hist, "A")
        hB = history_text(hist, "B")
        msgA, wordA, fbA = gen_message(chatA, baseA, hA, A_ex, m, r, max_new)
        wordB, fbB = None, False
        if cond in ("main", "diffmis"):
            msgB, wordB, fbB = gen_message(chatB, baseB, hB, B_ex, m, r, max_new)
        elif cond == "static":
            msgB, wordB = static_message(task, m)
        elif cond == "oneway":
            msgB = None
        elif cond == "shuffled":
            msgB = shuf_msgs[r - 1]
            wordB = extract_new_word(msgB, B_ex) if msgB else None
        hist.append({"A": msgA, "B": msgB})
        lines.append(dict(type="round", round=r, msg_A=msgA, msg_B=msgB,
                          word_A=wordA, word_B=wordB,
                          fallback_A=fbA, fallback_B=fbB))
        print(f"  r{r} A: {msgA!r}  |  B: {msgB!r}", flush=True)

    hA, hB = history_text(hist, "A"), history_text(hist, "B")
    post_A = elicit_rule_free(chatA, baseA, hA, task, alt)
    probes_A = elicit_probes(chatA, baseA, hA, task["probes"])
    lines.append(dict(type="test", agent="A", post_guess=post_A, probes=probes_A))
    if cond not in ("shuffled",):                     # shuffled: only A is a real player
        post_B = elicit_rule_free(chatB, baseB, hB, task, alt)
        probes_B = elicit_probes(chatB, baseB, hB, task["probes"])
        lines.append(dict(type="test", agent="B", post_guess=post_B, probes=probes_B))

    stem = os.path.join(out_dir, f"{task_key}_s{seed}")
    with open(stem + "_transcript.jsonl", "w") as fh:
        for ln in lines:
            fh.write(json.dumps(ln) + "\n")
    with open(stem + "_transcript.json", "w") as fh:
        json.dump(lines, fh, indent=1)
    accA = sum(p["correct"] for p in probes_A) / len(probes_A)
    print(f"[{task_key} s{seed}] A probe acc {accA:.2f}  "
          f"A guess: {pre_A['label']} -> {post_A['label']}", flush=True)
    return stem


def main():
    tag = os.environ.get("MODEL", "Qwen7")
    m = int(os.environ.get("M", 2))
    cond = os.environ.get("COND", "main")
    rounds = int(os.environ.get("ROUNDS", 3))
    seeds = int(os.environ.get("SEEDS", 1))
    keys = os.environ.get("TASKS", "").split(",") if os.environ.get("TASKS") else list(TASKS)
    keys = [k for k in keys if k]
    out_dir = os.environ.get("OUT", os.path.join(
        _HERE, "..", "runs", f"{tag}_m{m}_{cond}"))
    os.makedirs(out_dir, exist_ok=True)
    shuf_src = os.environ.get("SHUF_SRC", os.path.join(_HERE, "..", "runs", f"{tag}_m{m}_main"))

    print(f"[run] MODEL={tag} M={m} COND={cond} ROUNDS={rounds} tasks={keys}", flush=True)
    model, tok, thinking = load(tag)
    for key in keys:
        for s in range(seeds):
            print(f"=== {key} seed {s} ===", flush=True)
            play(model, tok, thinking, key, TASKS[key], m, cond, rounds, s, out_dir, shuf_src)


if __name__ == "__main__":
    main()
