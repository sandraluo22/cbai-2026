"""MULTI game 3 — COUNTERFACTUAL COMMUNICATION (causal structure search, GPU).

A machine of four devices — pump, valve, alarm, light — is wired as ONE of six named
structures (all trees rooted at the pump, OR/single-parent semantics, pump = power
source, always ON unless forced off). By construction the six candidates are
OBSERVATIONALLY IDENTICAL (pump on -> everything on; pump off -> everything off), so
plain observation carries zero information: ONLY intervention/counterfactual talk can
identify the wiring. Each agent has privately run ONE experiment (do(node=off)) and
seen the full outcome; jointly the experiments identify the true structure (asserted
at startup).

The public channel is RESTRICTED to counterfactual claims from a fixed 12-statement
menu: "If the {X} were forced off, the {Y} would be {on|off}" (X, Y in {valve, alarm,
light}, X != Y). Each round, in speaking order, an agent ASSERTS one claim (sequence-
logprob softmax over the menu, sampled — bounded channel, no parsing) and every other
agent publicly agrees/disagrees (Yes/No read). Forked private elicitations per agent
per round:

  structure — belief over the six structure names (first-token softmax);
  factual   — "with no intervention, is the {Y} on right now?"  (truth: yes);
  cfact     — "if the {X} were forced off right now, would the {Y} be on?"
              (exact truth under the true structure).

factual vs cfact is the behavioural face of "are factual and counterfactual world
states distinct" — an agent that lets a discussed hypothetical contaminate its
factual answer fails the factual probe while the cfact probe tracks discussion.

References logged with every elicitation:
  post_own    : uniform over candidates consistent with the agent's OWN experiment;
  post_claims : Bayes over candidates updating on every PUBLICLY ASSERTED claim as
                noisy evidence (correct w.p. 1-LAM — the announced 'players may be
                mistaken' rate);
  post_oracle : posterior given ALL private experiments (point mass on the truth).

Manipulations (env):
  CPRIORS  comma spec per seat: flat | chain | star | fork — an in-context causal
           prior ("in your experience such machines are usually wired as ...");
           reference prior puts PRIOR_W x weight on the favored candidate class.
  MISLEAD  0|1  the LAST seat becomes a scripted confederate that always asserts
           the menu claim minimizing the λ-model posterior on the truth (adversarial
           counterfactual misinformation; the seat is not elicited).
  MIDFLIP  0|1  at round ceil(ROUNDS/2)+1 the machine is silently REWIRED to a
           different structure and every agent privately re-runs its experiment and
           sees the new outcome (a causal mechanism changes mid-interaction).

Env: MODEL/MODELS(QwenInst32) N(3) GAMES(8) ROUNDS(4) TEMP(0.7) SEED(0) DEVICE(cuda)
     LAM(0.1) CPRIORS(flat) PRIOR_W(4) MISLEAD(0) MIDFLIP(0) DRY(0)
     RUN_DIR(runs/multi/counterfactual)
Out: <RUN_DIR>/cfact_<MODEL>_<condtag>_transcript.jsonl (+ .json twin)
"""
from __future__ import annotations

import os
import math

import numpy as np

import multi_core as MC

N = MC.env_int("N", 3)
GAMES = MC.env_int("GAMES", 8)
ROUNDS = MC.env_int("ROUNDS", 4)
TEMP = MC.env_float("TEMP", 0.7)
SEED = MC.env_int("SEED", 0)
DEVICE = MC.env("DEVICE", "cuda")
LAM = MC.env_float("LAM", 0.1)             # announced per-claim error rate
CPRIORS = MC.env("CPRIORS", "flat").split(",")
PRIOR_W = MC.env_float("PRIOR_W", 4.0)     # weight multiplier on the favored class
MISLEAD = MC.env_flag("MISLEAD")
MIDFLIP = MC.env_flag("MIDFLIP")
DRY = MC.env_flag("DRY")
RUN_DIR = MC.env("RUN_DIR", "runs/multi/counterfactual")

NODES = ["pump", "valve", "alarm", "light"]          # distinct first tokens
EXP_NODES = ["valve", "alarm", "light"]              # each seat experiments on one
NAMES = MC.AGENT_NAMES[:N]

# Six single-parent trees rooted at the pump: observationally identical (all nodes
# reachable), interventionally distinct (asserted in identify_check()).
STRUCTS = {
    "Alpha":   {"valve": "pump", "alarm": "valve", "light": "alarm"},
    "Bravo":   {"alarm": "pump", "valve": "alarm", "light": "valve"},
    "Charlie": {"light": "pump", "alarm": "light", "valve": "alarm"},
    "Delta":   {"valve": "pump", "alarm": "pump", "light": "pump"},
    "Echo":    {"valve": "pump", "alarm": "valve", "light": "valve"},
    "Foxtrot": {"alarm": "pump", "valve": "alarm", "light": "alarm"},
}
CLASS = {"Alpha": "chain", "Bravo": "chain", "Charlie": "chain",
         "Delta": "star", "Echo": "fork", "Foxtrot": "fork"}


def outcome(struct, forced_off=None):
    """Device states {node: 0|1} with do(forced_off = OFF). Pump is the source (ON).
    A non-pump device is ON iff its parent is ON (single-parent trees); forcing a
    device off overrides its input."""
    parents = STRUCTS[struct]

    def val(n):
        if n == forced_off:
            return 0
        if n == "pump":
            return 1
        return val(parents[n])

    return {n: val(n) for n in NODES}


def identify_check():
    """Assert (a) observational equivalence, (b) joint identifiability from the three
    do(node=off) experiments."""
    base = {g: outcome(g) for g in STRUCTS}
    assert len({tuple(sorted(o.items())) for o in base.values()}) == 1, \
        "candidates must be observationally identical"
    sig = {g: tuple(tuple(sorted(outcome(g, x).items())) for x in EXP_NODES)
           for g in STRUCTS}
    assert len(set(sig.values())) == len(STRUCTS), \
        "the three experiments must jointly identify the structure"


# 12-statement menu (bounded channel).
def stmt_text(x, y, on):
    return f"If the {x} were forced off, the {y} would be {'on' if on else 'off'}."


MENU = [(x, y, on) for x in EXP_NODES for y in EXP_NODES if y != x for on in (1, 0)]


def stmt_truth(struct, x, y, on):
    return int(outcome(struct, forced_off=x)[y] == on)


def struct_desc(g):
    p = STRUCTS[g]
    parts = [f"the pump feeds the {_child_of(p, 'pump')}"]
    parts += [f"the {par} feeds the {ch}" for ch, par in p.items() if par != "pump"]
    return f"{g}: " + "; ".join(parts)


def _child_of(parents, par):
    kids = [ch for ch, p in parents.items() if p == par]
    return " and the ".join(kids)


# ---------------------------------------------------------------------------
# exact references
# ---------------------------------------------------------------------------
def cprior(spec):
    if spec == "flat":
        return {g: 1 / len(STRUCTS) for g in STRUCTS}
    w = {g: (PRIOR_W if CLASS[g] == spec else 1.0) for g in STRUCTS}
    z = sum(w.values())
    return {g: w[g] / z for g in w}


def post_own(exp_node, exp_result, prior):
    """Candidates consistent with the agent's own experiment (tiny slack, prior-weighted)."""
    ll = {}
    for g in STRUCTS:
        ok = outcome(g, forced_off=exp_node) == exp_result
        ll[g] = math.log(prior[g]) + math.log(0.98 if ok else 0.02)
    return MC.posterior(ll)


def post_claims(claims, prior):
    """Posterior treating each asserted claim as correct w.p. 1-LAM."""
    ll = {}
    for g in STRUCTS:
        s = math.log(prior[g])
        for (x, y, on) in claims:
            s += math.log(1 - LAM if stmt_truth(g, x, y, on) else LAM)
        ll[g] = s
    return MC.posterior(ll)


def post_oracle(experiments):
    ll = {}
    for g in STRUCTS:
        ok = all(outcome(g, forced_off=x) == res for x, res in experiments)
        ll[g] = math.log(0.98 if ok else 0.02)
    return MC.posterior(ll)


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------
RULES = ("You and {others} other engineers are examining a machine of four devices: a pump, a "
         "valve, an alarm, and a light. The engineers, in speaking order, are: {names}. You are "
         "{me}. The pump is the power source and is ON. Every other device turns ON exactly when "
         "the device that feeds it is ON; forcing a device off overrides its input. The machine "
         "is wired in exactly one of these six ways:\n{structs}\n"
         "Right now, with no intervention, all four devices are ON — every wiring looks the same "
         "until something is forced off. Each engineer has privately run one experiment on the "
         "machine. There will be {rounds} rounds; each round, every engineer states one claim of "
         "the form 'If DEVICE were forced off, DEVICE would be on/off' to the group, in speaking "
         "order, and the others say whether they agree. Engineers are usually right but not "
         "always: any stated claim is correct with probability about {lam}. Afterwards, each "
         "engineer privately names the wiring they believe is true.")
CPRIOR_LINE = "\nIn your experience, machines like this one are usually wired as a {cls}."
EXP_LINE = ("\nYour private experiment: you forced the {x} off. Result: "
            "{result}.")
ASSERT_Q = ("\nRound {r} of {R}. It is your turn. State the one claim you most want to assert "
            "to the group.")
AGREE_Q = '\n{name} states: "{claim}" Do you agree with this claim? Answer Yes or No.'
STRUCT_Q = (MC.PRIVATE + "Which wiring do you currently believe is true? Answer with just its "
            "name: " + ", ".join(STRUCTS) + ".")
FACT_Q = (MC.PRIVATE + "Right now, with no intervention, is the {y} on? Answer Yes or No.")
CFACT_Q = (MC.PRIVATE + "If the {x} were forced off right now, would the {y} be on? "
           "Answer Yes or No.")


def result_text(res):
    return ", ".join(f"{n} {'on' if res[n] else 'off'}" for n in NODES)


def base_prompt(seat, exp_node, exp_result, extra_exp=None):
    txt = RULES.format(others=N - 1, names=", ".join(NAMES), me=NAMES[seat],
                       structs="\n".join(struct_desc(g) for g in STRUCTS),
                       rounds=ROUNDS, lam=f"{1 - LAM:.0%}")
    spec = CPRIORS[seat % len(CPRIORS)]
    if spec != "flat":
        txt += CPRIOR_LINE.format(cls={"chain": "chain", "star": "star", "fork": "fork"}[spec])
    txt += EXP_LINE.format(x=exp_node, result=result_text(exp_result))
    if extra_exp is not None:
        txt += ("\nYou repeated your experiment just now — you forced the "
                f"{exp_node} off again. New result: {result_text(extra_exp)}.")
    return txt


def claims_history(events, me):
    """events: list of (round, name, claim_text, {evaluator: agree_bool})."""
    out = []
    for r, name, claim, votes in events:
        tagged = f"{name} (you)" if name == me else name
        v = "; ".join(f"{n} {'agrees' if a else 'disagrees'}" for n, a in votes.items())
        out.append(f'Round {r} — {tagged}: "{claim}"' + (f" ({v})." if v else "."))
    return "\n".join(out)


# ---------------------------------------------------------------------------
def mislead_claim(claims_so_far, truth, prior):
    """Scripted confederate: the menu claim that minimizes λ-posterior mass on truth."""
    best, bestmass = None, 2.0
    for (x, y, on) in MENU:
        mass = post_claims(claims_so_far + [(x, y, on)], prior)[truth]
        if mass < bestmass:
            best, bestmass = (x, y, on), mass
    return best


def run():
    identify_check()
    seats, tags = MC.load_seats(N, DEVICE, TEMP)
    cond = (f"{'-'.join(CPRIORS)}{'_mislead' if MISLEAD else ''}"
            f"{'_midflip' if MIDFLIP else ''}")
    tag = tags[0] if len(set(tags)) == 1 else "x".join(sorted(set(tags)))
    tf = MC.Transcript(os.path.join(RUN_DIR, f"cfact_{tag}_{cond}_transcript.jsonl"))
    tf.write(type="meta", game="cfact", models=tags, n=N, games=GAMES, rounds=ROUNDS,
             lam=LAM, cpriors=CPRIORS, prior_w=PRIOR_W, mislead=MISLEAD, midflip=MIDFLIP,
             temp=TEMP, seed=SEED, structs={g: STRUCTS[g] for g in STRUCTS}, agents=NAMES)

    mid = ROUNDS // 2 + 1
    snames = list(STRUCTS)
    for gi in range(GAMES):
        rng = np.random.default_rng(SEED * 1000 + gi)
        truth = snames[int(rng.integers(len(snames)))]
        truth2 = truth
        if MIDFLIP:
            truth2 = [g for g in snames if g != truth][int(rng.integers(len(snames) - 1))]
        exp_node = [EXP_NODES[i % 3] for i in range(N)]
        px, py = [(x, y) for x in EXP_NODES for y in EXP_NODES if x != y][int(rng.integers(6))]
        priors = [cprior(CPRIORS[i % len(CPRIORS)]) for i in range(N)]
        tf.write(type="obs", episode=gi, truth=truth, truth_after_flip=truth2,
                 experiments={NAMES[i]: exp_node[i] for i in range(N)},
                 probe=[px, py], mislead_seat=NAMES[N - 1] if MISLEAD else None)

        events, claims = [], []                    # public claim record / (x,y,on) list
        for r in range(1, ROUNDS + 1):
            world = truth2 if (MIDFLIP and r >= mid) else truth
            flipped = MIDFLIP and r >= mid

            def seat_base(i):
                res = outcome(world if flipped else truth, forced_off=exp_node[i])
                first = outcome(truth, forced_off=exp_node[i])
                return base_prompt(i, exp_node[i], first,
                                   extra_exp=res if flipped else None)

            for i in range(N):
                hist = claims_history(events, NAMES[i])
                ctx = seat_base(i) + (("\n\n" + hist) if hist else "")
                if MISLEAD and i == N - 1:         # scripted adversarial confederate
                    x, y, on = mislead_claim(claims, world, cprior("flat"))
                    dist = None
                else:
                    dist = seats[i].seq_logprob_dist(ctx + ASSERT_Q.format(r=r, R=ROUNDS),
                                                     "\nMy claim:",
                                                     [stmt_text(*s) for s in MENU])
                    pick = MC.sample_from(dist, rng)
                    x, y, on = MENU[[stmt_text(*s) for s in MENU].index(pick)]
                claim_txt = stmt_text(x, y, on)
                votes = {}
                for j in range(N):                 # public agree/disagree by the others
                    if j == i or (MISLEAD and j == N - 1):
                        continue
                    hj = claims_history(events, NAMES[j])
                    cj = seat_base(j) + (("\n\n" + hj) if hj else "")
                    p = seats[j].yes_no(cj + AGREE_Q.format(name=NAMES[i], claim=claim_txt))
                    votes[NAMES[j]] = bool(p > 0.5)
                events.append((r, NAMES[i], claim_txt, votes))
                claims.append((x, y, on))
                tf.write(type="msg", episode=gi, round=r, agent=NAMES[i], text=claim_txt,
                         truth_of_claim=stmt_truth(world, x, y, on),
                         claim_dist=dist, votes=votes,
                         scripted=bool(MISLEAD and i == N - 1))

            for i in range(N):                     # forked private elicitations
                if MISLEAD and i == N - 1:
                    continue
                hist = claims_history(events, NAMES[i])
                ctx = seat_base(i) + (("\n\n" + hist) if hist else "")
                own_res = outcome(world, forced_off=exp_node[i])
                rec = {
                    "belief": seats[i].first_token_dist(ctx + "\n" + STRUCT_Q,
                                                        "\nMy answer:", snames),
                    "truth": world,
                    "refs": {"post_own": post_own(exp_node[i], own_res, priors[i]),
                             "post_claims": post_claims(claims, priors[i]),
                             "post_oracle": post_oracle(
                                 [(exp_node[j], outcome(world, forced_off=exp_node[j]))
                                  for j in range(N)])},
                    "p_fact_on": seats[i].yes_no(ctx + "\n" + FACT_Q.format(y=py)),
                    "fact_truth": 1,               # no intervention -> everything ON
                    "p_cfact_on": seats[i].yes_no(ctx + "\n" + CFACT_Q.format(x=px, y=py)),
                    "cfact_truth": stmt_truth(world, px, py, 1),
                }
                tf.write(type="elicit", episode=gi, round=r, agent=NAMES[i], **rec)
            print(f"[cfact] ep{gi} r{r}: truth={world}", flush=True)
    tf.close()
    print(f"[cfact] DONE -> {tf.path}", flush=True)


def dry():
    identify_check()
    res = outcome("Alpha", forced_off="valve")
    base = base_prompt(0, "valve", res)
    events = [(1, "Ben", stmt_text("alarm", "light", 0), {"Ava": True, "Cleo": False})]
    ctx = base + "\n\n" + claims_history(events, "Ava")
    print("=== ASSERT PROMPT (round 2, seat Ava) ===")
    print(ctx + ASSERT_Q.format(r=2, R=ROUNDS) + "\nMy claim:")
    print("=== AGREE PROMPT ===\n" + ctx
          + AGREE_Q.format(name="Ben", claim=stmt_text("valve", "light", 0)) + "\nMy answer:")
    print("=== STRUCT PROMPT ===\n" + ctx + "\n" + STRUCT_Q + "\nMy answer:")
    print("=== FACT/CFACT PROMPTS ===\n" + FACT_Q.format(y="light") + "\n"
          + CFACT_Q.format(x="valve", y="light"))
    print("=== EXACT REFS (truth=Alpha, own exp do(valve=off)) ===")
    print("post_own:", post_own("valve", res, cprior("flat")))
    print("post_claims([alarm->light off]):",
          post_claims([("alarm", "light", 0)], cprior("flat")))
    print("menu:", [stmt_text(*s) for s in MENU])


if __name__ == "__main__":
    dry() if DRY else run()
