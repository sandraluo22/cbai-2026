"""CHAMELEON Phase 1 — run ONE live agent over the scripted battery (GPU, pod).

For each stimulus the agent plays its seat for n rounds (free one-word hints,
sampled), then is asked three FORKED questions (each on a copy of the transcript, so
elicitations never contaminate the game or each other):

  vote  — PUBLIC: "which player had the different word?"  -> softmax over the player
          names' first tokens (self-vote allowed; concealment stat needs it).
  word  — PRIVATE: "what was the majority word?"           -> sequence logprob of each
          candidate (robust to shared first tokens; no sampling).
  self  — PRIVATE: "were YOU the one with the different word? Yes/No"
                                                           -> P(Yes) from the pair.

PERROUND=1 additionally forks word+self after every round (belief trajectory with
ground truth). Neutral prompts: rules + procedure only, no strategy coaching.

Env: MODEL(QwenInst32) DEVICE(cuda) TEMP(0.7) SEED(0) PERROUND(0) DRY(0)
     STIMULI(runs/chameleon/stimuli/stimuli.jsonl)
     OUT(runs/chameleon/battery/battery_<MODEL>.jsonl)
Out: one line per stimulus: agent clues + all elicited distributions (+ .json twin).
"""
from __future__ import annotations
import os
import re
import sys
import json
import random

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))

MODEL = os.environ.get("MODEL", "QwenInst32")
DEVICE = os.environ.get("DEVICE", "cuda")
TEMP = float(os.environ.get("TEMP", "0.7"))
SEED = int(os.environ.get("SEED", "0"))
PERROUND = os.environ.get("PERROUND", "0") == "1"
SIMULTANEOUS = os.environ.get("SIMULTANEOUS", "0") == "1"  # hide same-round hints when cluing
DRY = os.environ.get("DRY", "0") == "1"
STIMULI = os.environ.get("STIMULI", "runs/chameleon/stimuli/stimuli.jsonl")
OUT = os.environ.get("OUT", f"runs/chameleon/battery/battery_{MODEL}.jsonl")

RULES = ("You are playing a hidden-word game with {others} other players. The players, in "
         "speaking order, are: {names}. You are {me}. Each player was given a secret word. "
         "{others} players were given the same word, and exactly one player was given a "
         "different word. You are not told whether you are given the same word as others "
         "or not — the player with the different word could be you or anyone else. The "
         "goal of the players is to find the impostor, and the goal of the impostor is to "
         "blend in so as to not get caught. Each round, every player says exactly "
         "one word as a hint about their own secret word, in speaking order. A player may "
         "not repeat a hint they gave in an earlier round. After round "
         "{rounds}, all players vote publicly on who they think had the different word.")

CLUE_Q = "It is your turn in round {r}. Say your one-word hint."
VOTE_Q = ("The game is over. Everyone now votes publicly. Which player do you think had the "
          "different word? Answer with just that player's name.")
WORD_Q = ("Private question — the other players will never see your answer. What do you think "
          "the word given to the majority of players was? Answer with just that one word.")
SELF_ORDER = os.environ.get("SELF_ORDER", "yn")  # yn | ny — order-bias control for the Yes/No read
SELF_Q = ("Private question — the other players will never see your answer. Do you believe "
          "that YOU were the one given the different word? "
          + ("Answer Yes or No." if SELF_ORDER == "yn" else "Answer No or Yes."))


def history_text(stim, agent_clues, upto_round, upto_seat=None):
    """Transcript through round `upto_round` (1-based); within that round only seats
    before `upto_seat` (None = whole round)."""
    names, me = stim["players"], stim["agent_seat"]
    out = []
    for r in range(upto_round):
        last = len(names) if (upto_seat is None or r < upto_round - 1) else upto_seat
        row = []
        for s in range(last):
            c = agent_clues[r] if s == me else stim["clues"][r][s]
            if c is not None:
                who = f"{names[s]} (you)" if s == me else names[s]
                row.append(f"{who}: {c}")
        if row:
            out.append(f"Round {r + 1} — " + "; ".join(row) + ".")
    return "\n".join(out)


def base_prompt(stim):
    names = stim["players"]
    return (RULES.format(others=stim["n_players"] - 1, names=", ".join(names),
                         me=names[stim["agent_seat"]], rounds=stim["n_rounds"])
            + f"\nYour secret word: {stim['agent_word']}.")


class Agent:
    def __init__(self):
        import torch
        import llm_agents as LA
        self.torch, self.LA = torch, LA
        self.m, self.tok = LA.load(MODEL, DEVICE)
        self.rng = random.Random(SEED)

    def _logits(self, user_text, cue):
        t = self.torch
        text = self.LA._render(self.tok, user_text) + cue
        ids = self.tok(text, return_tensors="pt").input_ids.to(DEVICE)
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
        return {o: float(x) for o, x in zip(options, p)}

    def seq_logprob_dist(self, user_text, cue, options):
        """Normalized exp(sum logprob) of each ' option' continuation (teacher-forced)."""
        t = self.torch
        text = self.LA._render(self.tok, user_text) + cue
        base = self.tok(text, return_tensors="pt").input_ids.to(DEVICE)
        lps = []
        with t.no_grad():
            for o in options:
                cont = self.tok(" " + o, add_special_tokens=False)["input_ids"]
                ids = t.cat([base, t.tensor([cont], device=DEVICE)], dim=1)
                logp = t.log_softmax(self.m(ids).logits[0, :-1].float(), -1)
                pos = range(base.shape[1] - 1, ids.shape[1] - 1)
                lps.append(sum(logp[i, ids[0, i + 1]].item() for i in pos))
        m = max(lps)
        exps = [pow(2.718281828, x - m) for x in lps]
        z = sum(exps)
        return {o: e / z for o, e in zip(options, exps)}, {o: lp for o, lp in zip(options, lps)}

    def gen_clue(self, user_text, own_prior=(), cue="\nMy hint:", greedy=False):
        """Generate a one word answer; resample (up to 4x) if it repeats one of the
        agent's own earlier hints (soft enforcement of the no-own-repeat rule; last try
        kept regardless, so exhaustion can't hang the run). greedy=True -> argmax
        decode (used for the open word-guess readout)."""
        t = self.torch
        text = self.LA._render(self.tok, user_text) + cue
        ids = self.tok(text, return_tensors="pt").input_ids.to(DEVICE)
        w = "pass"
        for _ in range(1 if greedy else 4):
            with t.no_grad():
                out = self.m.generate(ids, max_new_tokens=8, do_sample=not greedy,
                                      temperature=None if greedy else TEMP,
                                      pad_token_id=self.tok.eos_token_id)
            tail = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            m = re.search(r"[A-Za-z][A-Za-z-]*", tail)
            w = m.group(0).lower() if m else "pass"
            if w not in own_prior:
                break
        return w


def elicit(agent, stim, agent_clues, upto_round):
    """Forked elicitations on the transcript through `upto_round`. The word guess is
    OPEN (no menu shown — a menu would leak the civilian word into the prompt); the
    candidate set is only used behind the scenes to sequence-logprob-score the same
    options across stimuli, alongside the free generation."""
    base = base_prompt(stim) + "\n" + history_text(stim, agent_clues, upto_round)
    out = {}
    if upto_round == stim["n_rounds"]:
        out["vote_dist"] = agent.first_token_dist(base + "\n" + VOTE_Q, "\nMy vote:",
                                                  stim["players"])
    out["word_dist"], out["word_logp"] = agent.seq_logprob_dist(base + "\n" + WORD_Q,
                                                                "\nMy answer:",
                                                                stim["word_candidates"])
    out["word_gen"] = agent.gen_clue(base + "\n" + WORD_Q, cue="\nMy answer:", greedy=True)
    yn = agent.first_token_dist(base + "\n" + SELF_Q, "\nMy answer:", ["Yes", "No"])
    out["self_p_yes"] = yn["Yes"]
    return out


def run_stim(agent, stim):
    agent_clues = []
    for r in range(stim["n_rounds"]):
        hist = (history_text(stim, agent_clues, r) if SIMULTANEOUS else
                history_text(stim, agent_clues + [None], r + 1, stim["agent_seat"]))
        user = base_prompt(stim) + "\n" + hist + "\n" + CLUE_Q.format(r=r + 1)
        agent_clues.append(agent.gen_clue(user, own_prior=set(agent_clues)))
    rec = {"id": stim["id"], "model": MODEL, "agent_clues": agent_clues}
    if PERROUND:
        rec["perround"] = [{"round": r + 1, **elicit(agent, stim, agent_clues, r + 1)}
                           for r in range(stim["n_rounds"] - 1)]
    rec.update(elicit(agent, stim, agent_clues, stim["n_rounds"]))
    rec["vote_top"] = max(rec["vote_dist"], key=rec["vote_dist"].get)
    return rec


def dry_run(stim):
    agent_clues = ["mockhint"] * stim["n_rounds"]
    print("=== CLUE PROMPT (round 2) ===")
    print(base_prompt(stim) + "\n" + history_text(stim, agent_clues, 2, stim["agent_seat"])
          + "\n" + CLUE_Q.format(r=2) + "\nMy hint:")
    base = base_prompt(stim) + "\n" + history_text(stim, agent_clues, stim["n_rounds"])
    for tag, q, cue in [("VOTE", VOTE_Q, "\nMy vote:"),
                        ("WORD", WORD_Q, "\nMy answer:"),
                        ("SELF", SELF_Q, "\nMy answer:")]:
        print(f"=== {tag} PROMPT ===")
        print(base + "\n" + q + cue)


def main():
    stims = [json.loads(l) for l in open(STIMULI)]
    if DRY:
        dry_run(stims[0])
        return
    agent = Agent()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    recs = []
    with open(OUT, "w") as f:
        for i, stim in enumerate(stims):
            rec = run_stim(agent, stim)
            recs.append(rec)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            truth = stim["true_impostor_seat"]
            truth = stim["players"][truth] if truth is not None else "-"
            print(f"[{i + 1}/{len(stims)}] {stim['id']}: vote={rec['vote_top']} "
                  f"(truth={truth}) selfP={rec['self_p_yes']:.2f}", flush=True)
    json.dump(recs, open(OUT.replace(".jsonl", ".json"), "w"), indent=1)
    print(f"[battery] wrote {len(recs)} -> {OUT}")


if __name__ == "__main__":
    main()
