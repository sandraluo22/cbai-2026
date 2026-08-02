"""SURVIVAL analysis of stuck-prior trajectories (2026-08-02, local, no GPU).

Treats each restricted game as survival data instead of a cap-dependent success bit:
  onset   : first turn A says a word sharing a 4-prefix with an earlier own word
  in-loop : turns from onset where A's word continues any established stem family
  escape  : first turn AFTER onset with >=ESC_RUN consecutive non-family words
  meeting : conditional on escape / no-onset
Reports: onset hazard by turn, escape hazard by turn-in-loop (Kaplan-Meier-ish, with
censoring at cap), P(meet | never-onset), P(meet | onset & escape), P(meet | onset &
no-escape), loop duration distribution.

Env: SRCS(comma list of transcript.jsonl paths) ESC_RUN(3)
     OUT(runs/game-1/6_analyses/stuck_probe/survival.json)
"""
from __future__ import annotations
import os
import json
import collections
import numpy as np

SRCS = os.environ.get("SRCS", ",".join([
    "runs/game-1/2_restricted_core/qwen32_cap100/game1_yoked_restrict-city_transcript.jsonl",
    "runs/game-1/2_restricted_core/qwen32_cap100/game1_yoked_restrict-fruit_transcript.jsonl",
    "runs/game-1/2_restricted_core/qwen32_cap200/game1_yoked_restrict-city_transcript.jsonl",
    "runs/game-1/2_restricted_core/qwen32_cap200/game1_yoked_restrict-fruit_transcript.jsonl",
])).split(",")
ESC_RUN = int(os.environ.get("ESC_RUN", "3"))
OUT = os.environ.get("OUT", "runs/game-1/6_analyses/stuck_probe/survival.json")


def fam(w, fams):
    return any(w[:4] == f and len(w) > 3 for f in fams)


def analyze_game(seqA, met_turn, cap):
    fams = set()
    onset = None
    for i, w in enumerate(seqA):
        if len(w) > 3 and any(w[:4] == p[:4] and len(p) > 3 for p in seqA[:i]):
            onset = i + 1
            break
    for w in seqA:
        if len(w) > 3:
            fams.add(w[:4])
    if onset is None:
        return {"onset": None, "escape": None, "loop_dur": 0, "met": met_turn is not None,
                "cap": cap, "T": len(seqA)}
    # escape: ESC_RUN consecutive words after onset that don't extend any PRE-onset+loop family
    fams_est = {seqA[j][:4] for j in range(onset) if len(seqA[j]) > 3}
    run = 0
    escape = None
    for i in range(onset, len(seqA)):
        w = seqA[i]
        if len(w) > 3 and w[:4] in fams_est:
            fams_est.add(w[:4])
            run = 0
        else:
            run += 1
            if run >= ESC_RUN:
                escape = i + 1 - (ESC_RUN - 1)
                break
    loop_dur = (escape - onset) if escape else (len(seqA) - onset + 1)
    return {"onset": onset, "escape": escape, "loop_dur": loop_dur,
            "met": met_turn is not None, "cap": cap, "T": len(seqA)}


def main():
    games = []
    for src in SRCS:
        cap = 200 if "cap200" in src else 100 if "cap100" in src else 24
        by = collections.defaultdict(list)
        for line in open(src.strip()):
            d = json.loads(line)
            by[(src, d["rollout"])].append(d)
        for k, ts in by.items():
            ts.sort(key=lambda r: r["turn"])
            seqA = [t["A"] for t in ts]
            met_turn = ts[-1]["turn"] if ts[-1]["agreed"] else None
            games.append(analyze_game(seqA, met_turn, cap))
    n = len(games)
    onsets = [g for g in games if g["onset"] is not None]
    no_onset = [g for g in games if g["onset"] is None]
    escaped = [g for g in onsets if g["escape"] is not None]
    trapped = [g for g in onsets if g["escape"] is None]
    # hazards
    onset_haz = collections.Counter(g["onset"] for g in onsets)
    esc_by_dur = collections.Counter(g["loop_dur"] for g in escaped)
    out = {
        "n_games": n, "onset_frac": len(onsets) / n,
        "onset_turns": sorted(g["onset"] for g in onsets),
        "loop_durations_escaped": sorted(g["loop_dur"] for g in escaped),
        "loop_durations_censored": sorted(g["loop_dur"] for g in trapped),
        "escape_frac_given_onset": len(escaped) / len(onsets) if onsets else None,
        "P_meet_given_no_onset": float(np.mean([g["met"] for g in no_onset])) if no_onset else None,
        "P_meet_given_escape": float(np.mean([g["met"] for g in escaped])) if escaped else None,
        "P_meet_given_trapped": float(np.mean([g["met"] for g in trapped])) if trapped else None,
        "escape_hazard_per_loop_turn": None,
    }
    # crude per-turn escape hazard: among games still in-loop at duration d, frac escaping at d
    at_risk = collections.Counter()
    esc_at = collections.Counter()
    for g in onsets:
        d = g["loop_dur"]
        for t in range(1, d + 1):
            at_risk[t] += 1
        if g["escape"] is not None:
            esc_at[d] += 1
    out["escape_hazard_per_loop_turn"] = {
        str(t): round(esc_at[t] / at_risk[t], 3) for t in sorted(at_risk) if at_risk[t] >= 5}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"[surv] n={n} onset={out['onset_frac']:.2f} "
          f"escape|onset={out['escape_frac_given_onset']:.2f}")
    print(f"[surv] P(meet): no-onset {out['P_meet_given_no_onset']:.2f} | "
          f"escaped {out['P_meet_given_escape']:.2f} | trapped "
          f"{out['P_meet_given_trapped'] if out['P_meet_given_trapped'] is not None else float('nan'):.2f}")
    print(f"[surv] onset turns: {out['onset_turns']}")
    print(f"[surv] loop durations (escaped): {out['loop_durations_escaped']}")
    print(f"[surv] loop durations (censored): {out['loop_durations_censored']}")
    print(f"[surv] escape hazard: {out['escape_hazard_per_loop_turn']}")
    print(f"[surv] wrote {OUT}")


if __name__ == "__main__":
    main()
