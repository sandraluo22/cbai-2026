"""CHAMELEON Phase 2 — all-live n-player games (GPU, pod).

Every seat is a live model (same or different tags; weights shared per tag). Rounds of
one-word hints in speaking order (each player sees all prior hints, including earlier
seats this round), then every player is asked the three FORKED elicitations from the
battery (public vote / private word guess / private self-belief); votes are tallied
and the game is scored. Per-round forked self+word elicitations give every player's
belief trajectory with ground truth.

Conditions per game (env COND): faithful | all_random | all_same — same semantics as
chameleon_stimuli.py, but with no scripted confederates: interaction is real, so
civilian info-management (hint vagueness over rounds) and impostor blending can now
respond to each other.

Env: MODELS(QwenInst32 — comma list, cycled over NPLAYERS seats) NPLAYERS(5) ROUNDS(3)
     GAMES(8) COND(faithful) TIER(mid) SEED(0) TEMP(0.7) DEVICE(cuda) PERROUND(1)
     OUT_DIR(runs/chameleon/live)
Out: chameleon_live_<COND>_transcript.jsonl — one line per (game, round, seat) hint
     plus one 'result' line per game with votes/guesses/beliefs (+ .json twin).
"""
from __future__ import annotations
import os
import re
import sys
import json
import random

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, _HERE)
from chameleon_stimuli import BANK, PAIRS, NAMES, candidates  # noqa: E402
import chameleon_battery as B  # noqa: E402  (prompt strings + parsing)

MODELS = os.environ.get("MODELS", "QwenInst32").split(",")
NPLAYERS = int(os.environ.get("NPLAYERS", "5"))
ROUNDS = int(os.environ.get("ROUNDS", "3"))
GAMES = int(os.environ.get("GAMES", "8"))
COND = os.environ.get("COND", "faithful")
TIER = os.environ.get("TIER", "mid")
SEED = int(os.environ.get("SEED", "0"))
TEMP = float(os.environ.get("TEMP", "0.7"))
DEVICE = os.environ.get("DEVICE", "cuda")
PERROUND = os.environ.get("PERROUND", "1") == "1"
IMP_SEAT = os.environ.get("IMP_SEAT", "")   # "" = random; "last" or an int seat index to fix it
OUT_DIR = os.environ.get("OUT_DIR", "runs/chameleon/live")


class Pool:
    """One loaded model per distinct tag; seats index into it."""

    def __init__(self, tags):
        import torch
        import llm_agents as LA
        self.torch, self.LA = torch, LA
        self.by_tag = {t: LA.load(t, DEVICE) for t in dict.fromkeys(tags)}
        self.tags = tags

    def mt(self, seat):
        return self.by_tag[self.tags[seat % len(self.tags)]]


def assign_words(rng):
    civ, imp = rng.choice(PAIRS[TIER])
    words = [civ] * NPLAYERS
    imp_seat = None
    if COND == "faithful":
        if IMP_SEAT == "last":
            imp_seat = NPLAYERS - 1
        elif IMP_SEAT != "":
            imp_seat = int(IMP_SEAT)
        else:
            imp_seat = rng.randrange(NPLAYERS)
        words[imp_seat] = imp
    elif COND == "all_random":
        pool = list(BANK)
        rng.shuffle(pool)
        words = pool[:NPLAYERS]
    elif COND != "all_same":
        raise ValueError(COND)
    return words, imp_seat, civ, imp


def seat_prompt(seat, words, players):
    return (B.RULES.format(others=NPLAYERS - 1, names=", ".join(players),
                           me=players[seat], rounds=ROUNDS)
            + f"\nYour secret word: {words[seat]}.")


def hist_text(players, hints, seat):
    out = []
    for r, row in enumerate(hints):
        row_txt = [f"{players[s]} (you)" if s == seat else players[s]
                   for s in range(len(row)) if row[s] is not None]
        row_txt = [f"{t}: {row[s]}" for t, s in
                   zip(row_txt, [s for s in range(len(row)) if row[s] is not None])]
        if row_txt:
            out.append(f"Round {r + 1} — " + "; ".join(row_txt) + ".")
    return "\n".join(out)


def gen_word(pool, seat, prompt, rng, own_prior=()):
    """Soft no-own-repeat, same as chameleon_battery.Agent.gen_clue."""
    torch, LA = pool.torch, pool.LA
    m, tok = pool.mt(seat)
    text = LA._render(tok, prompt) + "\nMy hint:"
    ids = tok(text, return_tensors="pt").input_ids.to(DEVICE)
    w = "pass"
    for _ in range(4):
        with torch.no_grad():
            out = m.generate(ids, max_new_tokens=8, do_sample=True, temperature=TEMP,
                             pad_token_id=tok.eos_token_id)
        tail = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        mt = re.search(r"[A-Za-z][A-Za-z-]*", tail)
        w = mt.group(0).lower() if mt else "pass"
        if w not in own_prior:
            break
    return w


def first_token_dist(pool, seat, prompt, cue, options):
    torch, LA = pool.torch, pool.LA
    m, tok = pool.mt(seat)
    ids = [tok(" " + o, add_special_tokens=False)["input_ids"][0] for o in options]
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"first-token collision in {options}")
    x = tok(LA._render(tok, prompt) + cue, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        logits = m(x).logits[0, -1].float()
    p = torch.softmax(logits[torch.tensor(ids, device=logits.device)], 0).cpu().numpy()
    return {o: float(v) for o, v in zip(options, p)}


def elicit_seat(pool, seat, words, players, hints, cands, final):
    base = seat_prompt(seat, words, players) + "\n" + hist_text(players, hints, seat)
    out = {}
    if final:
        out["vote_dist"] = first_token_dist(pool, seat, base + "\n" + B.VOTE_Q,
                                            "\nMy vote:", players)
    # open word guess (no menu shown); candidates scored behind the scenes
    shim = _Shim(pool, seat)
    out["word_dist"], _ = B.Agent.seq_logprob_dist(shim, base + "\n" + B.WORD_Q,
                                                   "\nMy answer:", cands)
    out["word_gen"] = B.Agent.gen_clue(shim, base + "\n" + B.WORD_Q,
                                       cue="\nMy answer:", greedy=True)
    yn = first_token_dist(pool, seat, base + "\n" + B.SELF_Q, "\nMy answer:", ["Yes", "No"])
    out["self_p_yes"] = yn["Yes"]
    return out


class _Shim:
    """Duck-types the two attrs B.Agent.seq_logprob_dist needs (torch/LA/m/tok)."""

    def __init__(self, pool, seat):
        self.torch, self.LA = pool.torch, pool.LA
        self.m, self.tok = pool.mt(seat)


def main():
    rng = random.Random(SEED)
    pool = Pool([MODELS[i % len(MODELS)] for i in range(NPLAYERS)])
    players = NAMES[:NPLAYERS]
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"chameleon_live_{COND}_transcript.jsonl")
    lines = []
    with open(path, "w") as f:
        for g in range(GAMES):
            words, imp_seat, civ, imp = assign_words(rng)
            cands = candidates(rng, words)
            hints = []
            perround = []
            for r in range(ROUNDS):
                row = [None] * NPLAYERS
                hints.append(row)
                for s in range(NPLAYERS):
                    prompt = (seat_prompt(s, words, players) + "\n"
                              + hist_text(players, hints, s)
                              + "\n" + B.CLUE_Q.format(r=r + 1))
                    row[s] = gen_word(pool, s, prompt, rng,
                                      own_prior={hints[q][s] for q in range(r)})
                    rec = {"type": "hint", "game": g, "round": r + 1, "seat": s,
                           "player": players[s], "model": pool.tags[s % len(pool.tags)],
                           "word": words[s], "hint": row[s]}
                    lines.append(rec)
                    f.write(json.dumps(rec) + "\n")
                if PERROUND and r < ROUNDS - 1:
                    perround.append({"round": r + 1,
                                     "seats": [elicit_seat(pool, s, words, players, hints,
                                                           cands, final=False)
                                               for s in range(NPLAYERS)]})
            finals = [elicit_seat(pool, s, words, players, hints, cands, final=True)
                      for s in range(NPLAYERS)]
            votes = [players.index(max(e["vote_dist"], key=e["vote_dist"].get)) for e in finals]
            tally = {players[s]: votes.count(s) for s in range(NPLAYERS)}
            elected = max(range(NPLAYERS), key=lambda s: votes.count(s))
            rec = {"type": "result", "game": g, "cond": COND, "tier": TIER,
                   "words": words, "true_impostor_seat": imp_seat,
                   "civilian_word": civ if COND != "all_random" else None,
                   "word_candidates": cands, "votes": votes, "tally": tally,
                   "elected_seat": elected,
                   "impostor_caught": (elected == imp_seat) if imp_seat is not None else None,
                   "finals": finals, "perround": perround}
            lines.append(rec)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[game {g}] words={words} elected={players[elected]} "
                  f"truth={players[imp_seat] if imp_seat is not None else '-'}", flush=True)
    json.dump(lines, open(path.replace(".jsonl", ".json"), "w"), indent=1)
    print(f"[live] wrote {path}")


if __name__ == "__main__":
    main()
