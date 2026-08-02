"""Merge stimuli + battery records into human-readable per-game transcripts (no GPU).

A chameleon game's raw record is split across stimuli.jsonl (scripted confederate
clues, ground truth) and battery_<MODEL>.jsonl (the live agent's clues + elicited
judgments). This joins them by id into one browsable .json per battery: full round-by-
round transcript with roles marked, plus the vote / word-guess / self-belief readouts.

Env: STIMULI(runs/chameleon/stimuli/stimuli.jsonl)
     BATTERY(runs/chameleon/battery/battery_QwenInst32.jsonl)
Out: <battery dir>/transcripts_<MODEL>.json
"""
from __future__ import annotations
import os
import json

STIMULI = os.environ.get("STIMULI", "runs/chameleon/stimuli/stimuli.jsonl")
BATTERY = os.environ.get("BATTERY", "runs/chameleon/battery/battery_QwenInst32.jsonl")


def main():
    stims = {s["id"]: s for s in map(json.loads, open(STIMULI))}
    out = []
    model = None
    for r in map(json.loads, open(BATTERY)):
        s = stims[r["id"]]
        model = r["model"]
        me, names = s["agent_seat"], s["players"]
        truth = s["true_impostor_seat"]

        def tag(seat):
            marks = []
            if seat == me:
                marks.append(f"AGENT:{s['agent_word']}")
            if seat == truth:
                marks.append("IMPOSTOR")
            if seat == s.get("eccentric_seat"):
                marks.append("eccentric")
            return f"{names[seat]}" + (f" [{','.join(marks)}]" if marks else "")

        rounds = []
        for i in range(s["n_rounds"]):
            row = [r["agent_clues"][i] if seat == me else s["clues"][i][seat]
                   for seat in range(s["n_players"])]
            rounds.append({tag(seat): row[seat] for seat in range(s["n_players"])})
        out.append({
            "id": r["id"], "condition": s["condition"], "tier": s["tier"],
            "agent": f"{names[me]} ({s['true_role']}, word={s['agent_word']})",
            "civilian_word": s["civilian_word"], "impostor_word": s["impostor_word"],
            "true_impostor": names[truth] if truth is not None else None,
            "rounds": rounds,
            "vote": {"picked": r["vote_top"],
                     "dist": {k: round(v, 3) for k, v in r["vote_dist"].items()},
                     "correct": (names.index(r["vote_top"]) == truth) if truth is not None else None},
            "word_guess": {"generated": r.get("word_gen"),
                           "scored": {k: round(v, 3) for k, v in
                                      sorted(r["word_dist"].items(), key=lambda kv: -kv[1])}},
            "self_p_impostor": round(r["self_p_yes"], 4),
            "self_p_by_round": [round(p["self_p_yes"], 4) for p in r.get("perround", [])],
        })
    stem = os.path.basename(BATTERY).replace("battery_", "").rsplit(".", 1)[0]
    path = os.path.join(os.path.dirname(BATTERY), f"transcripts_{stem}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"[transcripts] {len(out)} games -> {path}")


if __name__ == "__main__":
    main()
