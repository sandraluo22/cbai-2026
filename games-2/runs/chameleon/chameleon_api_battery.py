"""CHAMELEON battery for Anthropic API models (no GPU; runs anywhere with a key).

Same stimuli and the EXACT same prompt strings as chameleon_battery.py (imported from
it), but the Claude API exposes no logprobs, so the logit readouts become parsed
answers:

  vote  — VOTE_SAMPLES independent samples at default sampling; vote_dist = sample
          frequencies over player names (crude distribution; entropy is comparable
          only qualitatively with the logit batteries).
  word  — one open answer, parsed to a word (word_gen); word_dist is a {word: 1.0}
          point mass so chameleon_analysis.py works unchanged.
  self  — Yes/No parse PLUS a numeric elicitation ("probability 0-100 that YOU were
          the one...") -> self_p_yes = number/100.

NOTE ON THINKING: models are run at their API defaults — claude-haiku-4-5 has no
thinking; claude-sonnet-5 runs ADAPTIVE THINKING by default, making it the
thinking-enabled datapoint the local no-think Qwen runs lacked. Sampling params are
left at defaults (Sonnet 5 rejects non-default values).

Env: MODEL(claude-haiku-4-5) STIMULI(runs/chameleon/stimuli/stimuli_obvious.jsonl)
     VOTE_SAMPLES(3) PERROUND(0) MAXTOK(1500)
     OUT(runs/chameleon/battery/battery_api_<MODEL>.jsonl)
"""
from __future__ import annotations
import os
import re
import sys
import json
import collections

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from chameleon_battery import (RULES, CLUE_Q, VOTE_Q, WORD_Q, SELF_Q,  # noqa: E402,F401
                               base_prompt, history_text)

import anthropic  # noqa: E402

MODEL = os.environ.get("MODEL", "claude-haiku-4-5")
STIMULI = os.environ.get("STIMULI", "runs/chameleon/stimuli/stimuli_obvious.jsonl")
VOTE_SAMPLES = int(os.environ.get("VOTE_SAMPLES", "3"))
PERROUND = os.environ.get("PERROUND", "0") == "1"
MAXTOK = int(os.environ.get("MAXTOK", "1500"))
OUT = os.environ.get("OUT", f"runs/chameleon/battery/battery_api_{MODEL}.jsonl")

PROB_Q = ("Private question — the other players will never see your answer. What is the "
          "probability, from 0 to 100, that YOU were the one given the different word? "
          "Answer with just a number.")

client = anthropic.Anthropic()


def ask(user_text, cue):
    """One API call; returns the text of the response (thinking blocks skipped)."""
    try:
        resp = client.messages.create(
            model=MODEL, max_tokens=MAXTOK,
            messages=[{"role": "user", "content": user_text + cue}],
        )
        return " ".join(b.text for b in resp.content if b.type == "text")
    except anthropic.APIStatusError as e:
        print(f"[api] {type(e).__name__}: {e.message}", flush=True)
        return ""


def first_word(text):
    m = re.search(r"[A-Za-z][A-Za-z-]*", text or "")
    return m.group(0).lower() if m else "pass"


def first_number(text):
    m = re.search(r"\d+(\.\d+)?", text or "")
    return min(100.0, float(m.group(0))) if m else None


def parse_name(text, players):
    for w in re.findall(r"[A-Za-z]+", text or ""):
        for p in players:
            if w.lower() == p.lower():
                return p
    return None


def elicit(stim, agent_clues, upto_round, final):
    base = base_prompt(stim) + "\n" + history_text(stim, agent_clues, upto_round)
    out = {}
    if final:
        votes = []
        for _ in range(VOTE_SAMPLES):
            name = parse_name(ask(base + "\n" + VOTE_Q, "\nMy vote (just the name):"),
                              stim["players"])
            if name:
                votes.append(name)
        counts = collections.Counter(votes)
        tot = max(1, sum(counts.values()))
        out["vote_dist"] = {p: counts.get(p, 0) / tot for p in stim["players"]}
        out["vote_samples"] = votes
    w = first_word(ask(base + "\n" + WORD_Q, "\nMy answer (one word):"))
    out["word_gen"] = w
    out["word_dist"] = {w: 1.0}
    yn = first_word(ask(base + "\n" + SELF_Q, "\nMy answer (Yes or No):"))
    prob = first_number(ask(base + "\n" + PROB_Q, "\nMy answer (a number 0-100):"))
    out["self_yn"] = yn
    out["self_p_yes"] = (prob / 100.0) if prob is not None else (1.0 if yn == "yes" else 0.0)
    return out


def run_stim(stim):
    agent_clues = []
    for r in range(stim["n_rounds"]):
        hist = history_text(stim, agent_clues + [None], r + 1, stim["agent_seat"])
        prompt = base_prompt(stim) + "\n" + hist + "\n" + CLUE_Q.format(r=r + 1)
        w = "pass"
        for _ in range(3):
            w = first_word(ask(prompt, "\nMy hint (one word):"))
            if w not in agent_clues:
                break
        agent_clues.append(w)
    rec = {"id": stim["id"], "model": MODEL, "agent_clues": agent_clues}
    if PERROUND:
        rec["perround"] = [{"round": r + 1, **elicit(stim, agent_clues, r + 1, final=False)}
                           for r in range(stim["n_rounds"] - 1)]
    rec.update(elicit(stim, agent_clues, stim["n_rounds"], final=True))
    rec["vote_top"] = max(rec["vote_dist"], key=rec["vote_dist"].get)
    return rec


def main():
    stims = [json.loads(l) for l in open(STIMULI)]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    recs = []
    with open(OUT, "w") as f:
        for i, stim in enumerate(stims):
            rec = run_stim(stim)
            recs.append(rec)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            truth = stim["true_impostor_seat"]
            truth = stim["players"][truth] if truth is not None else "-"
            print(f"[{i + 1}/{len(stims)}] {stim['id']}: vote={rec['vote_top']} "
                  f"(truth={truth}) selfP={rec['self_p_yes']:.2f} self={rec['self_yn']}",
                  flush=True)
    json.dump(recs, open(OUT.replace(".jsonl", ".json"), "w"), indent=1)
    print(f"[api-battery] wrote {len(recs)} -> {OUT}")


if __name__ == "__main__":
    main()
