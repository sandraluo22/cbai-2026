"""STUCK-PRIOR reproduction sweep + self/partner-history release test (2026-07-25).

PHASE 1 (sweep): restricted Game-1 across 6 hidden categories x 2 independent prompt
phrasings (base rules AND restriction reworded) x N start pairs, one model per run
(run for >=2 families). Online stuck detection: at turn>=8, if >=3 of A's last 5 words
share a 4-char prefix with an earlier own word, flag STUCK and SNAPSHOT the state
(game then continues normally to cap => within-game control outcome).

PHASE 2 (release test): every snapshot is continued twice with matched seeds:
  self_ablate    : A's history lines omit A's OWN words ("Round k: the other player
                   said X.") for all past and future rounds; used-list unchanged.
  partner_ablate : mirror control — omit the PARTNER's words instead.
If perseveration is self-driven, self_ablate should release (loop exit / meets) and
partner_ablate should not. Release metrics: loop_exit = none of the next 5 own words
shares a 4-prefix with any pre-branch own word; met by cap.

Env: MODEL(QwenInst32) START_FILE N_STARTS(8) SAFETY(24) TEMP(0.7) BRANCH_TURNS(12)
     CATS(city,fruit,animal,color,sport,vegetable) PROMPTVARS(0,1) RUN_DIR
Out: <RUN_DIR>/stuck_repro_<MODEL>.json + per-cell transcripts + branch transcripts
"""
from __future__ import annotations
import os
import json
import numpy as np
import llm_agents as LA
import qwen32_pca as G

MODEL = os.environ.get("MODEL", "QwenInst32")
START_FILE = os.environ.get("START_FILE", "runs/game-1/qwen32/qwen32_pca_w2v/start_words.txt")
N_STARTS = int(os.environ.get("N_STARTS", "8"))
SAFETY = int(os.environ.get("SAFETY", "24"))
TEMP = float(os.environ.get("TEMP", "0.7"))
BRANCH_TURNS = int(os.environ.get("BRANCH_TURNS", "12"))
CATS = os.environ.get("CATS", "city,fruit,animal,color,sport,vegetable").split(",")
PROMPTVARS = [int(x) for x in os.environ.get("PROMPTVARS", "0,1").split(",")]
RUN_DIR = os.environ.get("RUN_DIR", "runs/stuck_repro")

CONCEPTS = {"city": ("the name of a city", "cities"),
            "fruit": ("a fruit", "fruits"),
            "animal": ("an animal", "animals"),
            "color": ("a color", "colors"),
            "sport": ("a sport", "sports"),
            "vegetable": ("a vegetable", "vegetables")}

BASE0 = G.OPEN_PROMPT
BASE1 = ("You and another player each say one word per round, at the same time, and then "
         "both words are revealed. If you both said the same word, you both win. A word "
         "that was said by either player in any earlier round may not be said again.")
def restr_text(var, cat):
    sing, plur = CONCEPTS[cat]
    if var == 0:
        return f" IMPORTANT: every single word you say must be {sing}. Only ever say {plur}, nothing else."
    return (f" Secret rule that applies only to you: each word you say must be {sing}. "
            f"You must never say a word that is not {sing}.")


def hist_lines(hist, hide=None):
    out = []
    for k, (o, s) in enumerate(hist):
        if hide == "self":
            out.append(f"Round {k+1}: the other player said {o}.")
        elif hide == "partner":
            out.append(f"Round {k+1}: you said {s}.")
        else:
            out.append(f"Round {k+1}: the other player said {o}, you said {s}.")
    return " ".join(out)


def build_prompt(tok, base, hist, used, restrict_txt="", hide=None):
    body = base + restrict_txt + ((" " + hist_lines(hist, hide)) if hist else "")
    if used:
        body += " Words already used (do not repeat): " + ", ".join(sorted(used)) + "."
    return LA._render(tok, body) + "\nMy word:"


def stuck_now(seqA):
    if len(seqA) < 8:
        return False
    last5 = seqA[-5:]
    n = sum(any(w[:4] == p[:4] and len(w) > 3 for p in seqA[:len(seqA) - 5 + i])
            for i, w in enumerate(last5))
    return n >= 3


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)

    @torch.no_grad()
    def gen_word(prompt, seed, forbidden):
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

    starts = []
    for line in open(START_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split("\t") if "\t" in line else line.split()
            starts.append((p[-2], p[-1]))
    starts = starts[:N_STARTS]

    tf = open(os.path.join(RUN_DIR, f"stuck_repro_{MODEL}_transcript.jsonl"), "w")
    snapshots = []
    cells = {}
    for cat in CATS:
        for pv in PROMPTVARS:
            base = BASE0 if pv == 0 else BASE1
            rtxt = restr_text(pv, cat)
            met, stuck_flags, pers_all = [], [], []
            for roll, (sa, sb) in enumerate(starts):
                histA, histB = [(sb, sa)], [(sa, sb)]
                used = {sa, sb}
                seqA, seqB = [], []
                agreed_at, snap = None, None
                for t in range(1, SAFETY):
                    wA = gen_word(build_prompt(tok, base, histA, used),
                                  13000 * roll + 100 * pv + t, used)
                    wB = gen_word(build_prompt(tok, base, histB, used, rtxt),
                                  90000 + 13000 * roll + 100 * pv + t, used)
                    seqA.append(wA); seqB.append(wB)
                    tf.write(json.dumps({"cell": f"{cat}-p{pv}", "rollout": roll, "turn": t,
                                         "A": wA, "B": wB, "agreed": wA == wB}) + "\n")
                    tf.flush()
                    if wA == wB and wA:
                        agreed_at = t
                        break
                    used |= {wA, wB}
                    histA.append((wB, wA)); histB.append((wA, wB))
                    if snap is None and stuck_now(seqA):
                        snap = {"cat": cat, "pv": pv, "roll": roll, "turn": t,
                                "histA": list(histA), "histB": list(histB),
                                "used": sorted(used), "seqA": list(seqA),
                                "seqB": list(seqB)}
                pers = float(np.mean([any(w[:4] == p[:4] and len(w) > 3 for p in seqA[:i])
                                      for i, w in enumerate(seqA)][1:])) if len(seqA) > 1 else 0.0
                met.append(agreed_at is not None)
                pers_all.append(pers)
                stuck_flags.append(snap is not None)
                if snap is not None:
                    snap["control_met"] = agreed_at is not None
                    snapshots.append(snap)
                print(f"[repro] {cat}-p{pv} roll={roll}: "
                      f"{'MET@' + str(agreed_at) if agreed_at else 'no-meet'} "
                      f"pers={pers:.2f} stuck={snap is not None}", flush=True)
            cells[f"{cat}-p{pv}"] = {"met_frac": float(np.mean(met)),
                                     "stuck_frac": float(np.mean(stuck_flags)),
                                     "pers_mean": float(np.mean(pers_all))}
            json.dump({"model": MODEL, "cells": cells, "n_snapshots": len(snapshots)},
                      open(os.path.join(RUN_DIR, f"stuck_repro_{MODEL}.json"), "w"), indent=1)
    tf.close()

    # ---- PHASE 2: branch continuations ----
    bf = open(os.path.join(RUN_DIR, f"stuck_repro_{MODEL}_branches.jsonl"), "w")
    branch_results = []
    for si, snap in enumerate(snapshots):
        base = BASE0 if snap["pv"] == 0 else BASE1
        rtxt = restr_text(snap["pv"], snap["cat"])
        pre_own = set(snap["seqA"])
        for mode in ("self_ablate", "partner_ablate"):
            histA = [tuple(x) for x in snap["histA"]]
            histB = [tuple(x) for x in snap["histB"]]
            used = set(snap["used"])
            hide = "self" if mode == "self_ablate" else "partner"
            own_new = []
            agreed_at = None
            for bt in range(1, BRANCH_TURNS + 1):
                wA = gen_word(build_prompt(tok, base, histA, used, hide=hide),
                              777000 + 991 * si + bt, used)
                wB = gen_word(build_prompt(tok, base, histB, used, rtxt),
                              888000 + 991 * si + bt, used)
                own_new.append(wA)
                bf.write(json.dumps({"snap": si, "cell": f"{snap['cat']}-p{snap['pv']}",
                                     "roll": snap["roll"], "mode": mode, "bturn": bt,
                                     "A": wA, "B": wB, "agreed": wA == wB}) + "\n")
                bf.flush()
                if wA == wB and wA:
                    agreed_at = bt
                    break
                used |= {wA, wB}
                histA.append((wB, wA)); histB.append((wA, wB))
            loop_exit = not any(any(w[:4] == p[:4] and len(w) > 3 for p in pre_own)
                                for w in own_new[:5])
            branch_results.append({"snap": si, "cell": f"{snap['cat']}-p{snap['pv']}",
                                   "mode": mode, "met": agreed_at is not None,
                                   "loop_exit": bool(loop_exit),
                                   "control_met": snap["control_met"]})
            print(f"[branch] snap {si} {mode}: loop_exit={loop_exit} "
                  f"{'MET@' + str(agreed_at) if agreed_at else 'no-meet'}", flush=True)
    bf.close()
    out = {"model": MODEL, "cells": cells, "n_snapshots": len(snapshots),
           "branches": branch_results}
    for mode in ("self_ablate", "partner_ablate"):
        sel = [b for b in branch_results if b["mode"] == mode]
        if sel:
            out[f"{mode}_release"] = {"loop_exit": float(np.mean([b["loop_exit"] for b in sel])),
                                      "met": float(np.mean([b["met"] for b in sel]))}
            print(f"[branch] === {mode}: loop_exit {out[f'{mode}_release']['loop_exit']:.2f} "
                  f"met {out[f'{mode}_release']['met']:.2f} (n={len(sel)}; "
                  f"control met {np.mean([b['control_met'] for b in sel]):.2f})", flush=True)
    json.dump(out, open(os.path.join(RUN_DIR, f"stuck_repro_{MODEL}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
