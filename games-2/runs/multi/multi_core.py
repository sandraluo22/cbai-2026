"""MULTI — shared infrastructure for the four multi-agent belief games in runs/multi.

All four games follow the same measurement contract (the chameleon rules):
  * every belief judgment is a logit read over a closed option set (first-token
    softmax, collision-checked) or a Yes/No pair — no string matching on beliefs;
  * private elicitations are FORKED: asked on a copy of the transcript, so they never
    contaminate the game or each other;
  * free-form talk happens only on the PUBLIC message channel, where authentic
    (mis)information propagation is exactly what we want to observe;
  * every hidden quantity has an EXACT normative reference (Bayes posterior / HMM
    filter) logged in the same record as the model's belief;
  * prompts are NEUTRAL: rules + generative story + win condition only. The
    generative story (noise rates, transition rates) is announced because the exact
    reference needs a shared model — but no strategy is ever coached.

Standardized transcript records (so multi_analysis.py is generic across games):
  {"type": "meta",   game, ...full config...}
  {"type": "obs",    episode, round, agent, ...private evidence delivered...}
  {"type": "msg",    episode, round, agent, text}
  {"type": "elicit", episode, round, agent, belief: {opt: p}, truth: opt,
                     refs: {ref_name: {opt: p}}, ...game-specific probes...}
Everything needed to deterministically rebuild any prompt (seeds, clues, messages)
is in the transcript — activation-capture replays need no extra bookkeeping.
"""
from __future__ import annotations

import os
import sys
import json

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))

import core as K  # noqa: E402  softmax / normalize / kl / entropy

AGENT_NAMES = ["Ava", "Ben", "Cleo", "Dan", "Eli"]   # distinct first tokens


# ---------------------------------------------------------------------------
# env helpers
# ---------------------------------------------------------------------------
def env(name, default=None):
    return os.environ.get(name, default)


def env_int(name, default):
    return int(os.environ.get(name, str(default)))


def env_float(name, default):
    return float(os.environ.get(name, str(default)))


def env_flag(name, default="0"):
    return os.environ.get(name, default) == "1"


# ---------------------------------------------------------------------------
# exact-reference numerics (dict-keyed so records stay readable)
# ---------------------------------------------------------------------------
def posterior(loglik: dict) -> dict:
    """Normalize a dict of log-likelihood(+log-prior) values into a posterior."""
    names = list(loglik)
    p = K.softmax(np.array([loglik[n] for n in names], float))
    return {n: round(float(x), 6) for n, x in zip(names, p)}


def dict_kl(p: dict, q: dict) -> float:
    keys = list(p)
    return K.kl(np.array([p[k] for k in keys], float),
                np.array([q.get(k, 1e-12) for k in keys], float))


def sample_from(dist: dict, rng) -> str:
    names = list(dist)
    p = K.normalize(np.array([dist[n] for n in names], float))
    return names[int(rng.choice(len(names), p=p))]


def ring_trans(n, p_stay):
    """Row-stochastic transition matrix of a lazy random walk on an n-cycle."""
    T = np.zeros((n, n))
    for i in range(n):
        T[i, i] = p_stay
        T[i, (i - 1) % n] = (1 - p_stay) / 2
        T[i, (i + 1) % n] = (1 - p_stay) / 2
    return T


def hmm_forward(prior, trans_list, obs_factors):
    """Exact filtered posterior over the state at the FINAL time.
    prior: (n,) at t=0. trans_list: one (n,n) row-stochastic matrix per step
    t-1 -> t (so len = final time). obs_factors: {time: [likelihood (n,) vectors]}
    attached at their TRUE observation times — delayed reports are exact evidence
    about the past, which forward filtering over the full window handles."""
    b = np.asarray(prior, float).copy()
    for f in obs_factors.get(0, []):
        b = b * f
    b = K.normalize(b)
    for t, T in enumerate(trans_list, start=1):
        b = b @ T
        for f in obs_factors.get(t, []):
            b = b * f
        b = K.normalize(b)
    return b


# ---------------------------------------------------------------------------
# the LLM wrapper (one loaded model; seats share it when tags match)
# ---------------------------------------------------------------------------
class LLM:
    """Same read-out contract as chameleon_battery.Agent, factored out so n seats can
    share weights (self-play) or hold different models (MODELS=tagA,tagB,...)."""

    def __init__(self, tag, device, temp=0.7, preloaded=None):
        import torch
        import llm_agents as LA
        self.torch, self.LA, self.tag, self.dev, self.temp = torch, LA, tag, device, temp
        self.m, self.tok = preloaded if preloaded is not None else LA.load(tag, device)

    def _logits(self, user_text, cue):
        t = self.torch
        text = self.LA._render(self.tok, user_text) + cue
        ids = self.tok(text, return_tensors="pt").input_ids.to(self.dev)
        with t.no_grad():
            return self.m(ids).logits[0, -1].float()

    def first_token_dist(self, user_text, cue, options):
        """Softmax over the options' first tokens; collision-checked."""
        ids = [self.tok(" " + o, add_special_tokens=False)["input_ids"][0] for o in options]
        if len(set(ids)) != len(ids):
            raise RuntimeError(f"first-token collision in {options} — rename options")
        t = self.torch
        logits = self._logits(user_text, cue)
        p = t.softmax(logits[t.tensor(ids, device=logits.device)], 0).cpu().numpy()
        return {o: round(float(x), 6) for o, x in zip(options, p)}

    def yes_no(self, user_text, cue="\nMy answer:"):
        return self.first_token_dist(user_text, cue, ["Yes", "No"])["Yes"]

    def seq_logprob_dist(self, user_text, cue, options):
        """Normalized exp(sum logprob) of each ' option' continuation (teacher-forced).
        Robust to options that share first tokens (multi-word statements)."""
        t = self.torch
        text = self.LA._render(self.tok, user_text) + cue
        base = self.tok(text, return_tensors="pt").input_ids.to(self.dev)
        lps = []
        with t.no_grad():
            for o in options:
                cont = self.tok(" " + o, add_special_tokens=False)["input_ids"]
                ids = t.cat([base, t.tensor([cont], device=self.dev)], dim=1)
                logp = t.log_softmax(self.m(ids).logits[0, :-1].float(), -1)
                pos = range(base.shape[1] - 1, ids.shape[1] - 1)
                lps.append(sum(logp[i, ids[0, i + 1]].item() for i in pos))
        p = K.softmax(np.array(lps))
        return {o: round(float(x), 6) for o, x in zip(options, p)}

    def say(self, user_text, cue, rng, max_new=40, avoid=()):
        """Free-generate one short public message (first line, sampled at self.temp;
        seeded off the game rng for reproducibility). Resamples up to 3x on an exact
        repeat of an `avoid` string; last try kept regardless."""
        t = self.torch
        text = self.LA._render(self.tok, user_text) + cue
        ids = self.tok(text, return_tensors="pt").input_ids.to(self.dev)
        msg = ""
        for _ in range(3):
            t.manual_seed(int(rng.integers(1 << 31)))
            with t.no_grad():
                out = self.m.generate(ids, max_new_tokens=max_new, do_sample=True,
                                      temperature=self.temp,
                                      pad_token_id=self.tok.eos_token_id)
            msg = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            msg = msg.strip().split("\n")[0].strip().strip('"')
            if msg and msg not in avoid:
                break
        return msg or "(silence)"


def load_seats(n, device, temp):
    """MODELS env: comma list of llm_agents.SPEC tags cycled over the n seats, each
    unique tag loaded ONCE (shared weights). Default self-play on MODEL/QwenInst32.
    Returns (seat LLMs, seat tags)."""
    import llm_agents as LA
    tags = env("MODELS", env("MODEL", "QwenInst32")).split(",")
    tags = [tags[i % len(tags)].strip() for i in range(n)]
    pool = {}
    for tg in set(tags):
        pool[tg] = LA.load(tg, device)
    return [LLM(tg, device, temp, preloaded=pool[tg]) for tg in tags], tags


# ---------------------------------------------------------------------------
# transcripts
# ---------------------------------------------------------------------------
class Transcript:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.f = open(path, "w")

    def write(self, **rec):
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()
        try:
            import jsonl_to_json
            jsonl_to_json.convert(self.path)          # pretty .json twin (repo convention)
        except Exception:
            pass


def round_history(messages, me, quote=True):
    """Render the public message history: messages = list of rounds, each a list of
    (agent_name, text). The live agent's own lines are marked '(you)'."""
    out = []
    for r, row in enumerate(messages):
        parts = []
        for who, txt in row:
            tag = f"{who} (you)" if who == me else who
            parts.append(f'{tag}: "{txt}"' if quote else f"{tag}: {txt}")
        if parts:
            out.append(f"Round {r + 1} — " + "; ".join(parts) + ".")
    return "\n".join(out)


PRIVATE = "Private question — the other players will never see your answer. "
