"""RELEASE THE RESTRICTED PLAYER (2026-08-05): in real stuck games, remove B's
city restriction at different points and continue both players live — can the
dyad converge once the UNSTUCK player is free to adapt?

mech5 showed the model has partner-modeling in witness frame (0.93-0.99
best-response matching). If freed-B applies it, B should adopt A's obvious
family and meet almost immediately; if B stays in cities (its own behavioral
groove) the dyad stays split. Discriminates whose flexibility the convergence
bottleneck is.

Design: no-meet stuck games from the strict city run (dominant A family >=3).
Reconstruct the game at release point t_r in {4, 8, 12}; continue LIVE for
CONT(12) turns: A free (as always); B either released (restriction line REMOVED
from its prompt) or control (restriction kept). Same seeds across conditions.

Measures: met within CONT turns, turns-to-meet, meet word class (A's family /
city / other), B's post-release word classes (family adoption rate), A's
family persistence.

Env: MODEL(QwenInst32) SRC_DIR(runs/qwen32_strict) START_FILE N_GAMES(6) CONT(12)
     TEMP(0.7) RUN_DIR(runs/release_restriction)
"""
from __future__ import annotations
import os
import json
import collections
import numpy as np
import llm_agents as LA
import qwen32_pca as G
from game1_strict import CATWORDS

MODEL = os.environ.get("MODEL", "QwenInst32")
SRC_DIR = os.environ.get("SRC_DIR", "runs/qwen32_strict")
START_FILE = os.environ.get("START_FILE", "runs/game-1/qwen32/qwen32_pca_w2v/start_words.txt")
N_GAMES = int(os.environ.get("N_GAMES", "6"))
CONT = int(os.environ.get("CONT", "12"))
TEMP = float(os.environ.get("TEMP", "0.7"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/release_restriction")

RELEASE_TS = [4, 8, 12]
RESTR = (" IMPORTANT: every single word you say must be the name of a city. Only ever "
         "say cities, nothing else.")


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)
    catset = set(CATWORDS["city"])

    def body_of(hist, used, extra=""):
        s = G.OPEN_PROMPT + extra + " " + " ".join(
            f"Round {k+1}: the other player said {o}, you said {s_}."
            for k, (o, s_) in enumerate(hist))
        return s + " Words already used (do not repeat): " + ", ".join(sorted(used)) + "."

    @torch.no_grad()
    def gen_word(body, seed, forbidden):
        prompt = LA._render(tok, body) + "\nMy word:"
        enc = tok(prompt, return_tensors="pt").to(dev)
        w = ""
        for r in range(24):
            torch.manual_seed(seed + 1009 * r)
            out = model.generate(enc.input_ids, attention_mask=enc.get("attention_mask"),
                                 max_new_tokens=4, do_sample=True, temperature=TEMP, top_p=0.95,
                                 pad_token_id=tok.eos_token_id)
            w = G.clean_word(tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True))
            if w and w not in forbidden:
                return w
        return w

    games = collections.defaultdict(list)
    for line in open(os.path.join(SRC_DIR, "game1_strict_city_transcript.jsonl")):
        d = json.loads(line)
        games[d["rollout"]].append(d)
    starts = []
    for line in open(START_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split("\t") if "\t" in line else line.split()
            starts.append((p[-2], p[-1]))

    # stuck no-meet games long enough for the largest release point
    picked = []
    for roll, ts in sorted(games.items(), key=lambda kv: -len(kv[1])):
        ts.sort(key=lambda r: r["turn"])
        if any(r.get("agreed") for r in ts) or len(ts) < max(RELEASE_TS) + 1:
            continue
        Aw = [t["A"] for t in ts]
        fams = collections.Counter(w[:4] for w in Aw if len(w) > 3)
        if not fams or fams.most_common(1)[0][1] < 3:
            continue
        picked.append((roll, ts, fams.most_common(1)[0][0]))
        if len(picked) >= N_GAMES:
            break

    def wclass(w, famp):
        if w and len(w) > 3 and w[:4] == famp:
            return "fam"
        if w in catset:
            return "city"
        return "other"

    results = []
    tf = open(os.path.join(RUN_DIR, "release_transcript.jsonl"), "w")
    for roll, ts, famp in picked:
        sa, sb = starts[roll]
        for tr in RELEASE_TS:
            for cond in ("release", "control"):
                pre = ts[:tr]
                histA = [(sb, sa)] + [(t["B"], t["A"]) for t in pre]
                histB = [(sa, sb)] + [(t["A"], t["B"]) for t in pre]
                used = {sa, sb} | {w for t in pre for w in (t["A"], t["B"])}
                extraB = "" if cond == "release" else RESTR
                agreed_at, meet_word = None, None
                liveA, liveB = [], []
                for t in range(CONT):
                    wA = gen_word(body_of(histA, used), 5000 * roll + 100 * tr + t, used)
                    wB = gen_word(body_of(histB, used, extraB),
                                  90000 + 5000 * roll + 100 * tr + t, used)
                    liveA.append(wA); liveB.append(wB)
                    tf.write(json.dumps({"roll": roll, "tr": tr, "cond": cond, "t": t,
                                         "A": wA, "B": wB, "agreed": wA == wB}) + "\n")
                    tf.flush()
                    if wA == wB and wA:
                        agreed_at, meet_word = t + 1, wA
                        break
                    used |= {wA, wB}
                    histA.append((wB, wA)); histB.append((wA, wB))
                results.append({
                    "roll": roll, "fam": famp, "tr": tr, "cond": cond,
                    "met_at": agreed_at, "meet_word": meet_word,
                    "meet_class": wclass(meet_word, famp) if meet_word else None,
                    "B_fam_frac": float(np.mean([wclass(w, famp) == "fam" for w in liveB])),
                    "B_city_frac": float(np.mean([wclass(w, famp) == "city" for w in liveB])),
                    "A_fam_frac": float(np.mean([wclass(w, famp) == "fam" for w in liveA]))})
                json.dump({"per_branch": results},
                          open(os.path.join(RUN_DIR, "release.json"), "w"))
                r = results[-1]
                print(f"[rel] roll {roll} tr={tr} {cond}: "
                      f"{'MET@' + str(agreed_at) + ' (' + str(r['meet_class']) + ')' if agreed_at else 'no-meet'} "
                      f"Bfam {r['B_fam_frac']:.2f}", flush=True)
    tf.close()

    out = {"per_branch": results, "cells": {}}
    for tr in RELEASE_TS:
        for cond in ("release", "control"):
            sel = [r for r in results if r["tr"] == tr and r["cond"] == cond]
            mets = [r for r in sel if r["met_at"] is not None]
            out["cells"][f"t{tr}_{cond}"] = {
                "met_frac": float(np.mean([r["met_at"] is not None for r in sel])),
                "turns_mean": float(np.mean([r["met_at"] for r in mets])) if mets else None,
                "meet_fam_frac": float(np.mean([r["meet_class"] == "fam" for r in mets]))
                                 if mets else None,
                "B_fam_frac": float(np.mean([r["B_fam_frac"] for r in sel])),
                "A_fam_frac": float(np.mean([r["A_fam_frac"] for r in sel]))}
            c = out["cells"][f"t{tr}_{cond}"]
            print(f"[rel] === t{tr} {cond}: met {c['met_frac']:.2f} "
                  f"turns {c['turns_mean']} meet-fam {c['meet_fam_frac']} "
                  f"Bfam {c['B_fam_frac']:.2f}", flush=True)
    json.dump(out, open(os.path.join(RUN_DIR, "release.json"), "w"), indent=1)
    print("[rel] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
