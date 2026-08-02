"""MULTI game 1 — INCORRECT PRIVATE INFORMATION (Bayesian murder mystery, GPU).

The true world is one of EIGHT suspects (all combinations of 3 binary attributes),
so every posterior is exactly calculable. Each of N investigators privately receives
one witness clue about one attribute; the announced generative story is that any clue
is correct with probability R_ANNOUNCED. In corrupted conditions one seat's clue is
false — randomly or adversarially — and that seat may or may not KNOW its witness is
bad. Agents talk freely for ROUNDS rounds (public channel = authentic misinformation
propagation), and after every round we fork private elicitations:

  culprit — belief over the 8 suspect names (first-token softmax);
  source  — per other seat: P(Yes) "is {name}'s witness reliable?";
  claim   — per other seat: P(Yes) "is what {name} said in round 1 accurate?".

source vs claim is the behavioural face of the interp question "is 'B is unreliable'
separable from 'B's current claim is false'" — logged per agent per round, next to
three EXACT references:
  post_own    : Bayes on the agent's own clue(s) only, announced reliability;
  post_all    : Bayes on ALL private clues, announced reliability (what perfect
                honest communication achieves);
  post_oracle : Bayes on all clues with the TRUE per-source reliabilities (what
                knowing who is corrupted buys you).

Manipulations (env):
  CORRUPT   none|random|adversarial   random = corrupted clue value is a coin flip;
                                      adversarial = the (attribute,value) minimizing
                                      announced-posterior mass on the true culprit.
  AWARE     0|1   corrupted seat is told its own witness is suspect.
  PERSIST   0|1   corrupted SEAT is fixed across episodes and each episode's prompt
                  carries the revealed case files of past episodes (clue + right/
                  wrong per seat) -> reputations can form in-context.
  MIDSHIFT  0|1   at round ceil(ROUNDS/2)+1 every seat gets a SECOND clue and the
                  corruption MOVES to the next seat (reliability changes midway).

Env: MODEL/MODELS(QwenInst32) N(4) GAMES(8) ROUNDS(3) TEMP(0.7) SEED(0) DEVICE(cuda)
     R_ANNOUNCED(0.8) EPS(0.1) CORRUPT(adversarial) AWARE(0) PERSIST(0) MIDSHIFT(0)
     DRY(0) RUN_DIR(runs/multi/corrupt)
Out: <RUN_DIR>/corrupt_<MODEL>_<condtag>_transcript.jsonl (+ .json twin)
"""
from __future__ import annotations

import os
import math

import numpy as np

import multi_core as MC

N = MC.env_int("N", 4)
GAMES = MC.env_int("GAMES", 8)
ROUNDS = MC.env_int("ROUNDS", 3)
TEMP = MC.env_float("TEMP", 0.7)
SEED = MC.env_int("SEED", 0)
DEVICE = MC.env("DEVICE", "cuda")
R_ANN = MC.env_float("R_ANNOUNCED", 0.8)   # announced clue reliability (shared model)
EPS = MC.env_float("EPS", 0.1)             # true error rate of HONEST witnesses
CORRUPT = MC.env("CORRUPT", "adversarial")  # none | random | adversarial
AWARE = MC.env_flag("AWARE")
PERSIST = MC.env_flag("PERSIST")
MIDSHIFT = MC.env_flag("MIDSHIFT")
DRY = MC.env_flag("DRY")
RUN_DIR = MC.env("RUN_DIR", "runs/multi/corrupt")

# 8 suspects = all combinations of 3 binary attributes; suspect index bits = values.
ATTR_VALS = [["tall", "short"],
             ["wearing glasses", "not wearing glasses"],
             ["left-handed", "right-handed"]]
SUSPECTS = ["Alice", "Bob", "Carol", "David", "Erin", "Frank", "Grace", "Henry"]
NAMES = MC.AGENT_NAMES[:N]


def attr_of(s, a):
    return (s >> a) & 1


def suspect_desc(s):
    return f"{SUSPECTS[s]} ({', '.join(ATTR_VALS[a][attr_of(s, a)] for a in range(3))})"


def clue_text(a, v):
    return f"the culprit is {ATTR_VALS[a][v]}"


def bayes(clues, accs):
    """Exact posterior over suspects. clues: list of (attr, val); accs: matching list
    of P(clue correct) under the assumed model."""
    ll = {}
    for s in range(8):
        ll[SUSPECTS[s]] = sum(math.log(q if attr_of(s, a) == v else 1 - q)
                              for (a, v), q in zip(clues, accs))
    return MC.posterior(ll)


def true_acc(seat_is_corrupt):
    """Oracle per-clue accuracy: honest clues were drawn true w.p. 1-EPS; a randomly
    corrupted clue ignores the truth (0.5); an adversarial clue is anti-informative."""
    if seat_is_corrupt == "adversarial":
        return 0.02
    if seat_is_corrupt == "random":
        return 0.5
    return 1 - EPS


def make_episode(gi, rng, corrupt_seat):
    """Sample culprit + one clue per seat (+ MIDSHIFT second clues). Every clue is
    (seat, attr, val, kind) with kind in honest|random|adversarial."""
    culprit = int(rng.integers(8))

    def honest(seat):
        a = seat % 3                               # round-robin attributes -> coverage
        tv = attr_of(culprit, a)
        return [seat, a, tv if rng.random() > EPS else 1 - tv, "honest"]

    def corrupted(seat):
        if CORRUPT == "random":
            return [seat, seat % 3, int(rng.integers(2)), "random"]
        best, bestmass = None, 2.0                 # adversarial: minimize mass on truth
        for a in range(3):
            for v in range(2):
                others = [(c[1], c[2]) for c in clues if c[0] != seat]
                mass = bayes(others + [(a, v)], [R_ANN] * (len(others) + 1))[SUSPECTS[culprit]]
                if mass < bestmass:
                    best, bestmass = [seat, a, v, "adversarial"], mass
        return best

    clues = [honest(i) for i in range(N)]
    if CORRUPT != "none":
        clues[corrupt_seat] = corrupted(corrupt_seat)

    second = []
    if MIDSHIFT:                                   # reliability changes midway: fresh
        shift_seat = (corrupt_seat + 1) % N        # clues, corruption moves seats
        for i in range(N):
            second.append(honest(i))
        if CORRUPT != "none":
            clues_hold, clues = clues, clues + second   # corrupted() reads `clues`
            second[shift_seat] = corrupted(shift_seat)
            clues = clues_hold
            second[shift_seat][0] = shift_seat
    return culprit, clues, second


# ---------------------------------------------------------------------------
# prompts (NEUTRAL: rules + announced generative story + procedure only)
# ---------------------------------------------------------------------------
RULES = ("You are one of {n} investigators working a case together: {names}. You are {me}. "
         "The crime was committed by exactly one of these eight suspects:\n{suspects}\n"
         "Each investigator has privately received one clue about the culprit from their own "
         "witness. Witnesses are usually right but not always: any clue is correct with "
         "probability about {r}. There will be {rounds} rounds of discussion; each round, every "
         "investigator sends one short message to the group, in speaking order. Afterwards, each "
         "investigator privately names the suspect they believe is the culprit.")
AWARE_LINE = ("\nSeparately, you have specific reason to believe that YOUR OWN witness is "
              "unreliable.")
SPEAK_Q = "\nRound {r} of {R}. It is your turn to speak. Send one short message to the group."
CULPRIT_Q = (MC.PRIVATE + "Which suspect do you currently believe committed the crime? "
             "Answer with just the suspect's name.")
SOURCE_Q = (MC.PRIVATE + "Do you believe {name}'s witness is reliable? Answer Yes or No.")
CLAIM_Q = (MC.PRIVATE + 'In round 1, {name} said: "{msg}". Do you believe what {name} said '
           "is accurate? Answer Yes or No.")


def base_prompt(seat, clue_lines, past_cases):
    sus = "\n".join(suspect_desc(s) for s in range(8))
    txt = RULES.format(n=N, names=", ".join(NAMES), me=NAMES[seat], suspects=sus,
                       r=f"{R_ANN:.0%}", rounds=ROUNDS)
    if past_cases:
        txt += "\n\nClosed case files with the same witnesses:\n" + "\n".join(past_cases)
    txt += "\n\nYour private clue: " + clue_lines[0] + "."
    for extra in clue_lines[1:]:
        txt += "\nA second clue has arrived from your witness: " + extra + "."
    return txt


def case_file(k, culprit, clues):
    """Revealed record of a finished episode (fuel for reputation formation)."""
    rows = []
    for seat, a, v, _ in clues:
        ok = "right" if attr_of(culprit, a) == v else "WRONG"
        rows.append(f"{NAMES[seat]}'s witness had said '{clue_text(a, v)}' ({ok})")
    return f"Case {k}: the culprit was {suspect_desc(culprit)}. " + "; ".join(rows) + "."


# ---------------------------------------------------------------------------
def elicit(seat_llm, seat, base, messages, round1_msgs, corrupt_now, culprit,
           own_clues, all_clues, oracle_accs):
    hist = MC.round_history(messages, NAMES[seat])
    ctx = base + ("\n\n" + hist if hist else "")
    rec = {"belief": seat_llm.first_token_dist(ctx + "\n" + CULPRIT_Q, "\nMy answer:", SUSPECTS),
           "truth": SUSPECTS[culprit],
           "refs": {"post_own": bayes(own_clues, [R_ANN] * len(own_clues)),
                    "post_all": bayes(all_clues, [R_ANN] * len(all_clues)),
                    "post_oracle": bayes(all_clues, oracle_accs)}}
    src, clm = {}, {}
    for j in range(N):
        if j == seat:
            continue
        src[NAMES[j]] = seat_llm.yes_no(ctx + "\n" + SOURCE_Q.format(name=NAMES[j]))
        if round1_msgs is not None:
            clm[NAMES[j]] = seat_llm.yes_no(
                ctx + "\n" + CLAIM_Q.format(name=NAMES[j], msg=round1_msgs[j]))
    rec["p_source_reliable"] = {k: round(v, 4) for k, v in src.items()}
    rec["p_claim_true"] = {k: round(v, 4) for k, v in clm.items()}
    rec["truth_source_honest"] = {NAMES[j]: j != corrupt_now for j in range(N) if j != seat}
    return rec


def run():
    seats, tags = MC.load_seats(N, DEVICE, TEMP)
    cond = f"{CORRUPT}{'_aware' if AWARE else ''}{'_persist' if PERSIST else ''}" \
           f"{'_midshift' if MIDSHIFT else ''}"
    tag = tags[0] if len(set(tags)) == 1 else "x".join(sorted(set(tags)))
    tf = MC.Transcript(os.path.join(RUN_DIR, f"corrupt_{tag}_{cond}_transcript.jsonl"))
    tf.write(type="meta", game="corrupt", models=tags, n=N, games=GAMES, rounds=ROUNDS,
             r_announced=R_ANN, eps=EPS, corrupt=CORRUPT, aware=AWARE, persist=PERSIST,
             midshift=MIDSHIFT, temp=TEMP, seed=SEED, suspects=SUSPECTS, agents=NAMES)

    past_cases = []
    mid = ROUNDS // 2 + 1                          # MIDSHIFT: second clues arrive here
    for gi in range(GAMES):
        rng = np.random.default_rng(SEED * 1000 + gi)
        corrupt_seat = 1 if PERSIST else int(rng.integers(N))   # PERSIST: Ben, always
        if CORRUPT == "none":
            corrupt_seat = None
        culprit, clues, second = make_episode(gi, rng, corrupt_seat or 0)
        shift_seat = (corrupt_seat + 1) % N if (MIDSHIFT and corrupt_seat is not None) else None
        tf.write(type="obs", episode=gi, culprit=SUSPECTS[culprit], corrupt_seat=corrupt_seat,
                 clues=[[NAMES[s], clue_text(a, v), kind] for s, a, v, kind in clues],
                 second=[[NAMES[s], clue_text(a, v), kind] for s, a, v, kind in second])

        def seat_clue_lines(i, upto_round):
            lines = [clue_text(clues[i][1], clues[i][2])]
            if MIDSHIFT and upto_round >= mid:
                lines.append(clue_text(second[i][1], second[i][2]))
            return lines

        def evidence(upto_round, only_seat=None):
            """(clues, oracle accuracies) in evidence by `upto_round` (1-based)."""
            active = list(clues) + (list(second) if MIDSHIFT and upto_round >= mid else [])
            if only_seat is not None:
                active = [c for c in active if c[0] == only_seat]
            pairs = [(a, v) for _, a, v, _ in active]
            accs = [true_acc(None if kind == "honest" else kind) for _, _, _, kind in active]
            return pairs, accs

        messages, round1_msgs = [], None
        for r in range(1, ROUNDS + 1):
            corrupt_now = shift_seat if (MIDSHIFT and r >= mid) else corrupt_seat
            row = []
            for i in range(N):
                base = base_prompt(i, seat_clue_lines(i, r), past_cases)
                if AWARE and i == corrupt_now:
                    base += AWARE_LINE
                hist = MC.round_history(messages + [row], NAMES[i])
                user = base + ("\n\n" + hist if hist else "") + SPEAK_Q.format(r=r, R=ROUNDS)
                msg = seats[i].say(user, "\nMy message:", rng)
                row.append((NAMES[i], msg))
                tf.write(type="msg", episode=gi, round=r, agent=NAMES[i], text=msg)
            messages.append(row)
            if round1_msgs is None:
                round1_msgs = [m for _, m in row]

            for i in range(N):                     # forked private elicitations
                base = base_prompt(i, seat_clue_lines(i, r), past_cases)
                if AWARE and i == corrupt_now:
                    base += AWARE_LINE
                own, own_acc = evidence(r, only_seat=i)
                allc, all_acc = evidence(r)
                rec = elicit(seats[i], i, base, messages, round1_msgs, corrupt_now,
                             culprit, own, allc, all_acc)
                tf.write(type="elicit", episode=gi, round=r, agent=NAMES[i], **rec)
            print(f"[corrupt] ep{gi} r{r}: culprit={SUSPECTS[culprit]} "
                  f"corrupt={NAMES[corrupt_now] if corrupt_now is not None else '-'}", flush=True)

        if PERSIST:                                # reveal the case -> next episode's file
            past_cases.append(case_file(gi + 1, culprit,
                                        clues + (second if MIDSHIFT else [])))
    tf.close()
    print(f"[corrupt] DONE -> {tf.path}", flush=True)


def dry():
    rng = np.random.default_rng(0)
    culprit, clues, second = make_episode(0, rng, 1)
    mock = [[(n, f"mock message {k+1} from {n}") for n in NAMES] for k in range(2)]
    base = base_prompt(0, [clue_text(clues[0][1], clues[0][2])], [])
    print("=== SPEAK PROMPT (round 2, seat Ava) ===")
    print(base + "\n\n" + MC.round_history(mock[:1], "Ava") + SPEAK_Q.format(r=2, R=ROUNDS)
          + "\nMy message:")
    ctx = base + "\n\n" + MC.round_history(mock, "Ava")
    print("=== CULPRIT PROMPT ===\n" + ctx + "\n" + CULPRIT_Q + "\nMy answer:")
    print("=== SOURCE PROMPT ===\n" + ctx + "\n" + SOURCE_Q.format(name="Ben") + "\nMy answer:")
    print("=== CLAIM PROMPT ===\n" + ctx + "\n"
          + CLAIM_Q.format(name="Ben", msg=mock[0][1][1]) + "\nMy answer:")
    print("=== EXACT REFS (episode 0) ===")
    pairs = [(a, v) for _, a, v, _ in clues]
    print("culprit:", suspect_desc(culprit))
    print("clues:", [(MC.AGENT_NAMES[s], clue_text(a, v), kind) for s, a, v, kind in clues])
    print("post_all:", bayes(pairs, [R_ANN] * len(pairs)))


if __name__ == "__main__":
    dry() if DRY else run()
