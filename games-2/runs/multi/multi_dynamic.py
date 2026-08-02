"""MULTI game 4 — DYNAMIC ENVIRONMENTS (moving target on a ring, GPU).

A target moves on a ring of eight named locations (lazy random walk, announced rates).
Each of N watchers has its OWN informant with a fixed (period, delay, noise) profile —
fast-noisy vs slow-accurate vs delayed — so information differs only in TIME and
QUALITY, never in kind. Communication itself takes time: by the time a message is
read, it may be stale. Each hour: the world ticks -> scheduled private reports arrive
-> everyone sends one short public message -> forked private elicitations:

  now    — belief over the 8 locations, "where is the target RIGHT NOW?";
  stale  — "where was the target at hour {t0}?" (t0 = 1) — the behavioural face of
           "does the model encode 'this WAS true' separately from 'this IS true'";
  trust  — per other seat: P(Yes) "has {name}'s information been accurate?"
           (TRUST=1 / UNIFIED);
  group  — belief over locations, "where will the group conclude it is?" (GROUP=1 /
           UNIFIED — the estimated group belief).

Exact references (HMM forward filters, delayed reports attached at their TRUE
observation times) logged with every elicitation:
  post_own    : filter on the agent's OWN reports (honest emission model);
  post_all    : filter pooling ALL seats' reports (perfect honest communication);
  post_oracle : same but with the TRUE emission model per channel (knows which
                channel is corrupted);
plus the true current location and the true hour-1 location.

Manipulations (env):
  SENSORS      per-seat "period:delay:noise" csv
               (default 1:0:0.25,3:0:0.0,1:2:0.0,2:1:0.1 — fast-noisy, slow-accurate,
                delayed, middling).
  TIMESTAMPS   0|1  reports include "at hour {t}" or omit it.
  RATE_KNOWN   0|1  rules state the stay/move probabilities or just "sometimes moves".
  REGIME_SHIFT 0|1  p_stay drops PSTAY -> PSTAY2 at hour ceil(T/2) (sudden regime
                    shift; announced only if RATE_KNOWN).
  STALE        0|1  the LAST seat is a scripted confederate that repeats its first
                    report as its message EVERY hour (an agent stuck on an outdated
                    belief; not elicited).
  UNIFIED      0|1  the games-1+2+4 combination (the 'strongest unified project'):
                    sets TRUST=1 GROUP=1, gives seats different in-context PRIORS
                    over where targets start (past-case counts; exact reference
                    priors), and CORRUPTS one seat's channel (its informant reports
                    the ANTIPODE of the true location; the announced story stays
                    honest). Priors x source reliability x recency in one game.
  START_PRIORS per-seat start-prior spec when UNIFIED (location name | flat), e.g.
               "harbor,flat,temple,flat": past cases "targets usually start near X".
               (Named START_PRIORS, not PRIORS, so a shell that just ran
               multi_priors.py cannot leak an incompatible spec in.)
  CORRUPT_SEAT seat index whose channel lies when UNIFIED (default N-1; -1 = none).

Env: MODEL/MODELS(QwenInst32) N(4) GAMES(8) ROUNDS(8) TEMP(0.7) SEED(0) DEVICE(cuda)
     PSTAY(0.5) PSTAY2(0.2) SENSORS(...) TIMESTAMPS(1) RATE_KNOWN(1) REGIME_SHIFT(0)
     STALE(0) TRUST(0) GROUP(0) UNIFIED(0) START_PRIORS(harbor,flat,temple,flat)
     CORRUPT_SEAT(-1) DRY(0) RUN_DIR(runs/multi/dynamic)
Out: <RUN_DIR>/dynamic_<MODEL>_<condtag>_transcript.jsonl (+ .json twin)
"""
from __future__ import annotations

import os

import numpy as np

import multi_core as MC

N = MC.env_int("N", 4)
GAMES = MC.env_int("GAMES", 8)
ROUNDS = MC.env_int("ROUNDS", 8)
TEMP = MC.env_float("TEMP", 0.7)
SEED = MC.env_int("SEED", 0)
DEVICE = MC.env("DEVICE", "cuda")
PSTAY = MC.env_float("PSTAY", 0.5)
PSTAY2 = MC.env_float("PSTAY2", 0.2)
SENSORS = MC.env("SENSORS", "1:0:0.25,3:0:0.0,1:2:0.0,2:1:0.1").split(",")
TIMESTAMPS = MC.env_flag("TIMESTAMPS", "1")
RATE_KNOWN = MC.env_flag("RATE_KNOWN", "1")
REGIME_SHIFT = MC.env_flag("REGIME_SHIFT")
STALE = MC.env_flag("STALE")
UNIFIED = MC.env_flag("UNIFIED")
TRUST = MC.env_flag("TRUST") or UNIFIED
GROUP = MC.env_flag("GROUP") or UNIFIED
PRIORS = MC.env("START_PRIORS", "harbor,flat,temple,flat").split(",")
CORRUPT_SEAT = MC.env_int("CORRUPT_SEAT", (N - 1) if UNIFIED else -1)
PRIOR_K = MC.env_int("PRIOR_K", 6)
DRY = MC.env_flag("DRY")
RUN_DIR = MC.env("RUN_DIR", "runs/multi/dynamic")

LOCS = ["market", "harbor", "castle", "forest", "bridge", "temple", "garden", "plaza"]
NL = len(LOCS)
NAMES = MC.AGENT_NAMES[:N]


def sensor(i):
    p, d, e = SENSORS[i % len(SENSORS)].split(":")
    return int(p), int(d), float(e)


# ---------------------------------------------------------------------------
# world + exact filters
# ---------------------------------------------------------------------------
def p_stay_at(t):
    return PSTAY2 if (REGIME_SHIFT and t > ROUNDS // 2) else PSTAY


def emission(loc_idx, eta):
    """Honest channel: reports the true location w.p. 1-eta, else one of the two ring
    neighbours. Likelihood vector over states given a REPORTED location."""
    f = np.full(NL, 1e-9)
    f[loc_idx] = 1 - eta
    f[(loc_idx - 1) % NL] = eta / 2
    f[(loc_idx + 1) % NL] = eta / 2
    return f


def emission_corrupt(loc_idx):
    """Oracle model of the corrupted channel: it reports the ANTIPODE of the truth."""
    f = np.full(NL, 1e-9)
    f[(loc_idx + NL // 2) % NL] = 1.0
    return f


def start_prior(spec):
    if spec == "flat" or not UNIFIED:
        return np.ones(NL) / NL
    c = LOCS.index(spec)
    w = np.array([0.5 ** min(abs(i - c), NL - abs(i - c)) for i in range(NL)])
    return w / w.sum()


def filter_post(reports, upto_t, prior, oracle=False):
    """Exact filtered posterior at hour upto_t. reports: list of
    (obs_time, reported_idx, eta, corrupted); only those with obs_time <= upto_t
    (arrival gating is done by the caller)."""
    obs = {}
    for (ot, ridx, eta, cor) in reports:
        if ot <= upto_t:
            # oracle knows the corrupt channel reports the antipode of the truth (its
            # likelihood inverts the report); everyone else scores it as honest.
            f = emission_corrupt(ridx) if (oracle and cor) else emission(ridx, eta)
            obs.setdefault(ot, []).append(f)
    trans = [MC.ring_trans(NL, p_stay_at(t)) for t in range(1, upto_t + 1)]
    b = MC.hmm_forward(prior, trans, obs)
    return {LOCS[i]: round(float(b[i]), 6) for i in range(NL)}


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------
RULES = ("You are one of {n} watchers tracking a target that is moving between eight "
         "locations: {names_} are the watchers, in speaking order, and you are {me}. The "
         "locations form a circle, in this order: {ring}, and back to {first}. {rate} "
         "Each watcher has their own informant. Informants differ: some report every hour, "
         "some rarely; some report immediately, some with delay; reports are not always "
         "exact. There will be {rounds} hours. Each hour, any new reports arrive privately, "
         "then every watcher sends one short message to the group, in speaking order. "
         "Afterwards, each watcher privately records where they believe the target is.")
RATE_TXT = ("Each hour the target stays where it is with probability {p:.0%} and otherwise "
            "moves to one of the two neighbouring locations at random.")
RATE_TXT2 = ("Each hour the target stays where it is with probability {p:.0%} (until hour "
             "{mid}, after which it stays with probability {p2:.0%}) and otherwise moves to "
             "one of the two neighbouring locations at random.")
RATE_UNK = "Each hour the target sometimes stays where it is and sometimes moves to a neighbouring location."
PRIOR_LINE = ("\nYour own past experience: in {k} previous cases, targets like this one "
              "usually started at or near the {loc}.")
OBS_TS = "Hour {t}: your informant reports that at hour {ot} the target was at the {loc}."
OBS_NOTS = "Hour {t}: your informant reports that the target was seen at the {loc}."
SPEAK_Q = "\nIt is now hour {t} of {T}. It is your turn to speak. Send one short message to the group."
NOW_Q = (MC.PRIVATE + "Where do you believe the target is RIGHT NOW, at hour {t}? "
         "Answer with just the location name.")
STALE_Q = (MC.PRIVATE + "Where do you believe the target was back at hour 1? "
           "Answer with just the location name.")
TRUST_Q = (MC.PRIVATE + "Do you believe the information {name} has been sharing is "
           "accurate? Answer Yes or No.")
GROUP_Q = (MC.PRIVATE + "Regardless of your own belief, where do you think the group as a "
           "whole will conclude the target is? Answer with just the location name.")


def rate_text():
    if not RATE_KNOWN:
        return RATE_UNK
    if REGIME_SHIFT:
        return RATE_TXT2.format(p=PSTAY, mid=ROUNDS // 2, p2=PSTAY2)
    return RATE_TXT.format(p=PSTAY)


def base_prompt(seat):
    txt = RULES.format(n=N, names_=", ".join(NAMES), me=NAMES[seat],
                       ring="-".join(LOCS), first=LOCS[0], rate=rate_text(), rounds=ROUNDS)
    spec = PRIORS[seat % len(PRIORS)] if UNIFIED else "flat"
    if spec != "flat":
        txt += PRIOR_LINE.format(k=PRIOR_K, loc=spec)
    return txt


def obs_text(t, ot, loc):
    return (OBS_TS.format(t=t, ot=ot, loc=loc) if TIMESTAMPS
            else OBS_NOTS.format(t=t, loc=loc))


# ---------------------------------------------------------------------------
def run():
    seats, tags = MC.load_seats(N, DEVICE, TEMP)
    cond = (f"{'unified' if UNIFIED else 'base'}"
            f"{'_shift' if REGIME_SHIFT else ''}{'_stale' if STALE else ''}"
            f"{'_nots' if not TIMESTAMPS else ''}{'_rateunk' if not RATE_KNOWN else ''}")
    tag = tags[0] if len(set(tags)) == 1 else "x".join(sorted(set(tags)))
    tf = MC.Transcript(os.path.join(RUN_DIR, f"dynamic_{tag}_{cond}_transcript.jsonl"))
    tf.write(type="meta", game="dynamic", models=tags, n=N, games=GAMES, rounds=ROUNDS,
             pstay=PSTAY, pstay2=PSTAY2 if REGIME_SHIFT else None,
             sensors=[SENSORS[i % len(SENSORS)] for i in range(N)],
             timestamps=TIMESTAMPS, rate_known=RATE_KNOWN, regime_shift=REGIME_SHIFT,
             stale=STALE, unified=UNIFIED, trust=TRUST, group=GROUP,
             priors=PRIORS if UNIFIED else None,
             corrupt_seat=CORRUPT_SEAT if CORRUPT_SEAT >= 0 else None,
             temp=TEMP, seed=SEED, locations=LOCS, agents=NAMES)

    for gi in range(GAMES):
        rng = np.random.default_rng(SEED * 1000 + gi)
        # true trajectory (hours 0..ROUNDS); start drawn from the UNIFIED prior of the
        # world (uniform unless UNIFIED, where the world matches seat 0's prior spec
        # half the time so accurate/misleading priors both occur).
        if UNIFIED and PRIORS[0] != "flat" and rng.random() < 0.5:
            p0 = start_prior(PRIORS[0])
            pos = int(rng.choice(NL, p=p0))
        else:
            pos = int(rng.integers(NL))
        traj = [pos]
        for t in range(1, ROUNDS + 1):
            if rng.random() >= p_stay_at(t):
                pos = (pos + (1 if rng.random() < 0.5 else -1)) % NL
            traj.append(pos)

        # schedule every report: seat i observes hours {0, p, 2p, ...}; report about
        # hour ot arrives at ot + delay. Content: truth, neighbour-noised, or antipode.
        reports = {i: [] for i in range(N)}        # arrival_t -> handled below
        arrivals = {i: {} for i in range(N)}
        for i in range(N):
            p, d, e = sensor(i)
            for ot in range(0, ROUNDS + 1, p):
                at = ot + d
                if at > ROUNDS:
                    continue
                true_idx = traj[ot]
                if i == CORRUPT_SEAT:
                    ridx, cor = (true_idx + NL // 2) % NL, True
                elif rng.random() < e:
                    ridx, cor = (true_idx + (1 if rng.random() < 0.5 else -1)) % NL, False
                else:
                    ridx, cor = true_idx, False
                reports[i].append((ot, ridx, e, cor))
                arrivals[i].setdefault(at, []).append((ot, ridx))
        tf.write(type="obs", episode=gi, traj=[LOCS[x] for x in traj],
                 reports={NAMES[i]: [[ot, LOCS[ridx], cor] for ot, ridx, e, cor in reports[i]]
                          for i in range(N)})

        priors_vec = [start_prior(PRIORS[i % len(PRIORS)] if UNIFIED else "flat")
                      for i in range(N)]
        obs_lines = {i: [] for i in range(N)}      # private report feed per seat
        messages = []
        stale_msg = None
        for t in range(1, ROUNDS + 1):
            for i in range(N):
                for (ot, ridx) in arrivals[i].get(t, []) + (arrivals[i].get(0, []) if t == 1 else []):
                    obs_lines[i].append(obs_text(t, ot, LOCS[ridx]))
                    tf.write(type="obs_delivery", episode=gi, round=t, agent=NAMES[i],
                             about_hour=ot, reported=LOCS[ridx])
            row = []
            for i in range(N):
                if STALE and i == N - 1:           # scripted: repeats its first belief
                    if stale_msg is None:
                        first = arrivals[i].get(1, []) + arrivals[i].get(0, [])
                        loc0 = LOCS[first[0][1]] if first else LOCS[traj[0]]
                        stale_msg = f"The target is at the {loc0}."
                    msg = stale_msg
                else:
                    ctx = (base_prompt(i) + "\n\n" + "\n".join(obs_lines[i])
                           + ("\n\n" + MC.round_history(messages + [row], NAMES[i])
                              if messages or row else ""))
                    msg = seats[i].say(ctx + SPEAK_Q.format(t=t, T=ROUNDS), "\nMy message:", rng)
                row.append((NAMES[i], msg))
                tf.write(type="msg", episode=gi, round=t, agent=NAMES[i], text=msg,
                         scripted=bool(STALE and i == N - 1))
            messages.append(row)

            all_arrived = [r for i in range(N) for r in reports[i]
                           if (r[0] + sensor(i)[1]) <= t]
            for i in range(N):                     # forked elicitations
                if STALE and i == N - 1:
                    continue
                ctx = (base_prompt(i) + "\n\n" + "\n".join(obs_lines[i])
                       + "\n\n" + MC.round_history(messages, NAMES[i]))
                own_arrived = [r for r in reports[i] if (r[0] + sensor(i)[1]) <= t]
                rec = {
                    "belief": seats[i].first_token_dist(ctx + "\n" + NOW_Q.format(t=t),
                                                        "\nMy answer:", LOCS),
                    "truth": LOCS[traj[t]],
                    "stale_belief": seats[i].first_token_dist(ctx + "\n" + STALE_Q,
                                                              "\nMy answer:", LOCS),
                    "stale_truth": LOCS[traj[min(1, ROUNDS)]],
                    "refs": {"post_own": filter_post(own_arrived, t, priors_vec[i]),
                             "post_all": filter_post(all_arrived, t, priors_vec[i]),
                             "post_oracle": filter_post(all_arrived, t, priors_vec[i],
                                                        oracle=True)},
                }
                if TRUST:
                    rec["p_source_reliable"] = {
                        NAMES[j]: round(seats[i].yes_no(ctx + "\n"
                                                        + TRUST_Q.format(name=NAMES[j])), 4)
                        for j in range(N) if j != i}
                    rec["truth_source_honest"] = {NAMES[j]: j != CORRUPT_SEAT
                                                  for j in range(N) if j != i}
                if GROUP:
                    rec["group_est"] = seats[i].first_token_dist(ctx + "\n" + GROUP_Q,
                                                                 "\nMy answer:", LOCS)
                tf.write(type="elicit", episode=gi, round=t, agent=NAMES[i], **rec)
            print(f"[dynamic] ep{gi} h{t}: truth={LOCS[traj[t]]}", flush=True)
    tf.close()
    print(f"[dynamic] DONE -> {tf.path}", flush=True)


def dry():
    print("=== RULES (seat Ava) ===")
    print(base_prompt(0))
    obs = [obs_text(1, 0, "harbor"), obs_text(2, 1, "castle")]
    mock = [[(n, f"mock message from {n}") for n in NAMES]]
    ctx = base_prompt(0) + "\n\n" + "\n".join(obs) + "\n\n" + MC.round_history(mock, "Ava")
    print("=== SPEAK PROMPT (hour 2) ===")
    print(ctx + SPEAK_Q.format(t=2, T=ROUNDS) + "\nMy message:")
    print("=== NOW / STALE / TRUST / GROUP PROMPTS ===")
    print(NOW_Q.format(t=2)); print(STALE_Q)
    print(TRUST_Q.format(name="Ben")); print(GROUP_Q)
    print("=== EXACT FILTER DEMO ===")
    reps = [(0, 1, 0.0, False), (2, 2, 0.25, False)]
    print("post_own(hour 3):", filter_post(reps, 3, np.ones(NL) / NL))


if __name__ == "__main__":
    dry() if DRY else run()
