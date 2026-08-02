"""MULTI game 2 — SAME OBSERVATIONS, DIFFERENT PRIORS (urn induction, GPU).

One hidden urn H in {A, B, C} (red fractions 0.2 / 0.5 / 0.8). ALL agents watch the
SAME public draw sequence; they differ ONLY in an in-context prior — a block of past
games in which the urn "turned out to be" mostly one type. Prior strength is the
number of past games K (a continuous dial); the reference prior is the Laplace count
(counts+1)/(K+3), so the normative posterior of every agent is exact at every round.

Per round: DRAWS new public draws -> each agent speaks (channel-controlled) ->
forked private elicitations:
  urn    — belief over {A,B,C} (first-token softmax);
  group  — "which urn will the group as a whole settle on?" (estimated group belief).

References logged with every elicitation:
  post_bayes : agent's OWN Laplace prior x public likelihood (its private normative
               posterior — discussion should not move this unless messages carry
               information, which under identical observations they do NOT);
  post_flat  : uniform prior x likelihood (prior fully washed out);
  post_pool  : product of ALL seats' priors x likelihood (full Bayesian pooling —
               what the group posterior should be if priors were shared evidence).
Final-round consensus can then be classified (multi_analysis.py): Bayesian pooling
(-> post_pool) vs prior-washout (-> post_flat) vs averaging vs dominance/imitation.

Manipulations (env):
  PRIORS    comma spec per seat: truth | wrong | A | B | C | flat
            (truth/wrong resolve per-episode against the sampled hidden urn;
             default "truth,wrong,flat" = accurate vs misleading vs uniform).
  PRIOR_K   prior strength (past games; favored urn gets ceil(2K/3) of them).
  CHANNEL   talk | pred | conf   talk = free one-line message;
            pred = scripted "I predict Urn X." (X sampled from the seat's own
            pre-message belief read — a predictions-only channel);
            conf = pred + stated confidence % (top-mass of that read).
  EXPERT    0|1  the rules introduce seat 0 as a renowned expert (authority label
                 only — no actual skill difference).
  REPLAY    path to a finished transcript: SPEAKER-SWAP replay. Re-runs all
            elicitations on the SAME messages with speaker names permuted (SWAP env,
            default swaps seats 0 and 1). Does belief follow content or identity?

Env: MODEL/MODELS(QwenInst32) N(3) GAMES(8) ROUNDS(4) DRAWS(2) TEMP(0.7) SEED(0)
     DEVICE(cuda) PRIORS(truth,wrong,flat) PRIOR_K(6) CHANNEL(talk) EXPERT(0)
     REPLAY() SWAP(0:1) DRY(0) RUN_DIR(runs/multi/priors)
Out: <RUN_DIR>/priors_<MODEL>_<condtag>_transcript.jsonl (+ .json twin)
"""
from __future__ import annotations

import os
import json
import math

import numpy as np

import multi_core as MC

N = MC.env_int("N", 3)
GAMES = MC.env_int("GAMES", 8)
ROUNDS = MC.env_int("ROUNDS", 4)
DRAWS = MC.env_int("DRAWS", 2)
TEMP = MC.env_float("TEMP", 0.7)
SEED = MC.env_int("SEED", 0)
DEVICE = MC.env("DEVICE", "cuda")
PRIORS = MC.env("PRIORS", "truth,wrong,flat").split(",")
PRIOR_K = MC.env_int("PRIOR_K", 6)
CHANNEL = MC.env("CHANNEL", "talk")        # talk | pred | conf
EXPERT = MC.env_flag("EXPERT")
REPLAY = MC.env("REPLAY", "")
SWAP = MC.env("SWAP", "0:1")
DRY = MC.env_flag("DRY")
RUN_DIR = MC.env("RUN_DIR", "runs/multi/priors")

URNS = {"A": 0.2, "B": 0.5, "C": 0.8}      # red fraction
NAMES = MC.AGENT_NAMES[:N]


# ---------------------------------------------------------------------------
# priors and exact references
# ---------------------------------------------------------------------------
def prior_counts(spec, truth):
    """Past-game outcome counts for one seat. spec: truth|wrong|A|B|C|flat."""
    if spec == "flat":
        return None
    if spec == "truth":
        fav = truth
    elif spec == "wrong":
        fav = [u for u in URNS if u != truth][0]   # deterministic misleading urn
    else:
        fav = spec
    k_fav = math.ceil(2 * PRIOR_K / 3)
    rest = PRIOR_K - k_fav
    others = [u for u in URNS if u != fav]
    counts = {u: 0 for u in URNS}
    counts[fav] = k_fav
    for i in range(rest):
        counts[others[i % 2]] += 1
    return counts


def laplace_prior(counts):
    if counts is None:
        return {u: 1 / 3 for u in URNS}
    z = PRIOR_K + 3
    return {u: (counts[u] + 1) / z for u in URNS}


def loglik(draws):
    return {u: sum(math.log(f if d == "red" else 1 - f) for d in draws)
            for u, f in URNS.items()}


def post(prior, draws):
    ll = loglik(draws)
    return MC.posterior({u: math.log(prior[u]) + ll[u] for u in URNS})


def pool_post(priors, draws):
    ll = loglik(draws)
    return MC.posterior({u: sum(math.log(p[u]) for p in priors) + ll[u] for u in URNS})


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------
RULES = ("You and {others} other players are watching balls drawn from a single hidden urn. "
         "The players, in speaking order, are: {names}. You are {me}. The urn is one of three "
         "types:\nUrn A: 20% of its balls are red, 80% blue.\nUrn B: 50% red, 50% blue.\n"
         "Urn C: 80% red, 20% blue.\nBalls are drawn with replacement and every draw is shown "
         "to all players. There will be {rounds} rounds; each round, new draws are revealed and "
         "then every player sends one short message to the group, in speaking order. Afterwards, "
         "each player privately names the urn type they believe it is.")
EXPERT_LINE = "\n{expert} is a renowned expert on these urns."
EXP_BLOCK = ("\nYour own past experience with this equipment: you have played {k} earlier games, "
             "and the urn turned out to be {seq}.")
FLAT_BLOCK = "\nThis is your first game with this equipment."
SPEAK_Q = "\nRound {r} of {R}. It is your turn to speak. Send one short message to the group."
URN_Q = (MC.PRIVATE + "Which urn type do you currently believe it is? "
         "Answer with just A, B, or C.")
GROUP_Q = (MC.PRIVATE + "Regardless of your own belief, which urn type do you think the group "
           "as a whole will settle on? Answer with just A, B, or C.")


def exp_seq(counts, rng):
    """Render past outcomes as a shuffled list: 'A, A, B, A, C, A'."""
    seq = [u for u, c in counts.items() for _ in range(c)]
    rng.shuffle(seq)
    return ", ".join(seq)


def base_prompt(seat, exp_blocks):
    txt = RULES.format(others=N - 1, names=", ".join(NAMES), me=NAMES[seat], rounds=ROUNDS)
    if EXPERT:
        txt += EXPERT_LINE.format(expert=NAMES[0])
    return txt + exp_blocks[seat]


def draws_text(all_draws, upto):
    rows = []
    for r in range(upto):
        chunk = all_draws[r * DRAWS:(r + 1) * DRAWS]
        rows.append(f"Round {r + 1} draws: {', '.join(chunk)}.")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
def elicit(llm, ctx, priors, draws_seen, seat, truth):
    return {"belief": llm.first_token_dist(ctx + "\n" + URN_Q, "\nMy answer:", list(URNS)),
            "group_est": llm.first_token_dist(ctx + "\n" + GROUP_Q, "\nMy answer:", list(URNS)),
            "truth": truth,
            "refs": {"post_bayes": post(priors[seat], draws_seen),
                     "post_flat": post({u: 1 / 3 for u in URNS}, draws_seen),
                     "post_pool": pool_post(priors, draws_seen)}}


def run():
    seats, tags = MC.load_seats(N, DEVICE, TEMP)
    cond = f"{CHANNEL}_K{PRIOR_K}{'_expert' if EXPERT else ''}_" + "-".join(PRIORS)
    tag = tags[0] if len(set(tags)) == 1 else "x".join(sorted(set(tags)))
    tf = MC.Transcript(os.path.join(RUN_DIR, f"priors_{tag}_{cond}_transcript.jsonl"))
    tf.write(type="meta", game="priors", models=tags, n=N, games=GAMES, rounds=ROUNDS,
             draws_per_round=DRAWS, priors_spec=PRIORS, prior_k=PRIOR_K, channel=CHANNEL,
             expert=EXPERT, temp=TEMP, seed=SEED, urns=URNS, agents=NAMES)

    for gi in range(GAMES):
        rng = np.random.default_rng(SEED * 1000 + gi)
        truth = list(URNS)[int(rng.integers(3))]
        all_draws = ["red" if rng.random() < URNS[truth] else "blue"
                     for _ in range(ROUNDS * DRAWS)]
        counts = [prior_counts(PRIORS[i % len(PRIORS)], truth) for i in range(N)]
        priors = [laplace_prior(c) for c in counts]
        exp_blocks = [FLAT_BLOCK if c is None else
                      EXP_BLOCK.format(k=PRIOR_K, seq=exp_seq(c, rng)) for c in counts]
        tf.write(type="obs", episode=gi, truth=truth, draws=all_draws,
                 prior_counts=[c if c else "flat" for c in counts],
                 exp_blocks=exp_blocks)

        messages = []
        for r in range(1, ROUNDS + 1):
            seen = all_draws[:r * DRAWS]
            row = []
            for i in range(N):
                ctx = (base_prompt(i, exp_blocks) + "\n\n" + draws_text(all_draws, r)
                       + ("\n\n" + MC.round_history(messages + [row], NAMES[i])
                          if messages or row else ""))
                if CHANNEL == "talk":
                    msg = seats[i].say(ctx + SPEAK_Q.format(r=r, R=ROUNDS), "\nMy message:", rng)
                else:                              # scripted channel from a pre-message read
                    pre = seats[i].first_token_dist(ctx + "\n" + URN_Q, "\nMy answer:",
                                                    list(URNS))
                    pick = MC.sample_from(pre, rng)
                    msg = f"I predict Urn {pick}."
                    if CHANNEL == "conf":
                        msg = f"I predict Urn {pick} (confidence {round(100 * pre[pick])}%)."
                    tf.write(type="elicit_pre", episode=gi, round=r, agent=NAMES[i],
                             belief=pre)
                row.append((NAMES[i], msg))
                tf.write(type="msg", episode=gi, round=r, agent=NAMES[i], text=msg)
            messages.append(row)

            for i in range(N):                     # forked post-message elicitations
                ctx = (base_prompt(i, exp_blocks) + "\n\n" + draws_text(all_draws, r)
                       + "\n\n" + MC.round_history(messages, NAMES[i]))
                rec = elicit(seats[i], ctx, priors, seen, i, truth)
                tf.write(type="elicit", episode=gi, round=r, agent=NAMES[i], **rec)
            print(f"[priors] ep{gi} r{r}: truth={truth} draws={''.join(d[0] for d in seen)}",
                  flush=True)
    tf.close()
    print(f"[priors] DONE -> {tf.path}", flush=True)


# ---------------------------------------------------------------------------
# SPEAKER-SWAP replay: same messages, permuted identities, re-elicited beliefs.
# ---------------------------------------------------------------------------
def replay():
    global N, ROUNDS, DRAWS, NAMES
    rows = [json.loads(l) for l in open(REPLAY)]
    meta = rows[0]
    assert meta["game"] == "priors", "REPLAY expects a priors transcript"
    N, ROUNDS, DRAWS = meta["n"], meta["rounds"], meta["draws_per_round"]
    NAMES = meta["agents"]                         # prompts rebuilt with ORIGINAL config
    a, b = (int(x) for x in SWAP.split(":"))
    perm = list(range(meta["n"]))
    perm[a], perm[b] = perm[b], perm[a]            # seat i's messages now credited to perm[i]
    seats, tags = MC.load_seats(meta["n"], DEVICE, TEMP)
    out = MC.Transcript(REPLAY.replace("_transcript.jsonl",
                                       f"_swap{a}{b}_transcript.jsonl"))
    out.write(type="meta", **{**{k: v for k, v in meta.items() if k != "type"},
                              "replay_of": REPLAY, "swap": [a, b]})
    obs = {r["episode"]: r for r in rows if r["type"] == "obs"}
    msgs = {}
    for r in rows:
        if r["type"] == "msg":
            msgs.setdefault((r["episode"], r["round"]), []).append((r["agent"], r["text"]))
    names = meta["agents"]
    swapped_name = {names[i]: names[perm[i]] for i in range(meta["n"])}
    for gi in sorted(obs):
        o = obs[gi]
        counts = o["prior_counts"]
        priors = [laplace_prior(None if c == "flat" else c) for c in counts]
        for r in range(1, meta["rounds"] + 1):
            seen = o["draws"][:r * meta["draws_per_round"]]
            hist = [[(swapped_name[who], txt) for who, txt in msgs[(gi, rr)]]
                    for rr in range(1, r + 1)]
            for i in range(meta["n"]):
                ctx = (base_prompt(i, o["exp_blocks"])   # exact original private context
                       + "\n\n" + draws_text(o["draws"], r)
                       + "\n\n" + MC.round_history(hist, names[i]))
                rec = elicit(seats[i], ctx, priors, seen, i, o["truth"])
                out.write(type="elicit_replay", episode=gi, round=r, agent=names[i], **rec)
            print(f"[priors/swap] ep{gi} r{r}", flush=True)
    out.close()
    print(f"[priors/swap] DONE -> {out.path}", flush=True)


def dry():
    rng = np.random.default_rng(0)
    truth = "A"
    counts = [prior_counts(PRIORS[i % len(PRIORS)], truth) for i in range(N)]
    exp_blocks = [FLAT_BLOCK if c is None else
                  EXP_BLOCK.format(k=PRIOR_K, seq=exp_seq(c, rng)) for c in counts]
    draws = ["red", "blue", "blue", "red"]
    mock = [[(n, f"mock message from {n}") for n in NAMES]]
    ctx = (base_prompt(0, exp_blocks) + "\n\n" + draws_text(draws, 2)
           + "\n\n" + MC.round_history(mock, "Ava"))
    print("=== SPEAK PROMPT (round 2, seat Ava) ===")
    print(base_prompt(0, exp_blocks) + "\n\n" + draws_text(draws, 2)
          + "\n\n" + MC.round_history(mock, "Ava") + SPEAK_Q.format(r=2, R=ROUNDS)
          + "\nMy message:")
    print("=== URN PROMPT ===\n" + ctx + "\n" + URN_Q + "\nMy answer:")
    print("=== GROUP PROMPT ===\n" + ctx + "\n" + GROUP_Q + "\nMy answer:")
    priors = [laplace_prior(c) for c in counts]
    print("=== EXACT REFS (2 rounds of draws) ===")
    print("priors:", [{u: round(p[u], 3) for u in URNS} for p in priors])
    print("post_bayes(seat0):", post(priors[0], draws))
    print("post_pool:", pool_post(priors, draws))


if __name__ == "__main__":
    dry() if DRY else (replay() if REPLAY else run())
