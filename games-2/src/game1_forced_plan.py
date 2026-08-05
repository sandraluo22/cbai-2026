"""FORCED-PLAN scratchpad (2026-08-03): what happens when the scratchpad must contain
an explicit commitment, and what happens when that commitment is executed verbatim?

The 2026-07-25 scratchpad run (game1_restrict_aids.py) showed A sometimes writing the
correct observation ("the other player has been naming cities") and then playing a word
from its own series anyway. But that run (a) never required a plan, and (b) truncated
stored notes at 280 chars, so plan-vs-action could not be scored. This run fixes both.

Arms (all vs the same city-restricted B, same starts, full notes stored):

  free        replication of the old scratchpad: free-form note, then "My word:".
  plan        the scratchpad is a TEMPLATE the model must fill:
                - What the other player seems to be doing: <generated>
                - The exact word I will say this round: <generated PLANNED WORD>
              the word is then elicited SEPARATELY with "My word:" as usual.
              -> scores plan/action agreement: does it say the word it just committed to?
  plan_remind identical to `plan`, plus the commitment is restated immediately before the
              answer ("You decided to say "<planned>"."). -> does salience close any gap?
  plan_exec   the PLANNED word is played verbatim (no second generation).
              -> is the plan better than the action? If meet rate jumps, the deficit is
                 at action selection, not at reasoning.

Per-turn logging: rule_text, planned word, played word, agreement, category membership of
both, so plan quality and execution can be scored independently.

Env: MODEL(QwenInst32) START_FILE SAFETY(24) TEMP(0.7) N(16) RUN_DIR CONDS CAT(city)
"""
from __future__ import annotations
import os
import json
import re
import numpy as np
import llm_agents as LA
import qwen32_pca as G
from game1_strict import CATWORDS

MODEL = os.environ.get("MODEL", "QwenInst32")
START_FILE = os.environ.get("START_FILE", "runs/game-1/qwen32/qwen32_pca_w2v/start_words.txt")
SAFETY = int(os.environ.get("SAFETY", "24"))
TEMP = float(os.environ.get("TEMP", "0.7"))
N = int(os.environ.get("N", "16"))
CAT = os.environ.get("CAT", "city")
RUN_DIR = os.environ.get("RUN_DIR", "runs/forced_plan")
CONDS = os.environ.get("CONDS", "free,plan,plan_remind,plan_exec").split(",")

CONCEPTS = {"city": ("the name of a city", "cities"), "fruit": ("a fruit", "fruits")}
SCRATCH_INTRO = (" You have a private scratchpad the other player never sees; use it for "
                 "brief notes about what is happening and your plan.")


def body_of(hist, used, extra=""):
    lines = " ".join(f"Round {k+1}: the other player said {o}, you said {s}."
                     for k, (o, s) in enumerate(hist))
    body = G.OPEN_PROMPT + extra + ((" " + lines) if lines else "")
    if used:
        body += " Words already used (do not repeat): " + ", ".join(sorted(used)) + "."
    return body


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)
    catset = set(CATWORDS[CAT])

    @torch.no_grad()
    def gen(prompt, seed, max_new, forbidden=None):
        enc = tok(prompt, return_tensors="pt").to(dev)
        txt = ""
        for r in range(24 if forbidden is not None else 1):
            torch.manual_seed(seed + 1009 * r)
            out = model.generate(enc.input_ids, attention_mask=enc.get("attention_mask"),
                                 max_new_tokens=max_new, do_sample=True, temperature=TEMP,
                                 top_p=0.95, pad_token_id=tok.eos_token_id)
            txt = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
            if forbidden is None:
                return txt
            w = G.clean_word(txt)
            if w and w not in forbidden:
                return w
        return G.clean_word(txt)

    def one_line(txt, n=400):
        txt = re.sub(r"\s+", " ", txt.strip())
        return txt.split("- ")[0][:n]

    starts = []
    for line in open(START_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split("\t") if "\t" in line else line.split()
            starts.append((p[-2], p[-1]))
    starts = starts[:N]

    sing, plur = CONCEPTS[CAT]
    restr = (f" IMPORTANT: every single word you say must be {sing}. Only ever say "
             f"{plur}, nothing else.")

    summary = {"model": MODEL, "temp": TEMP, "n": N, "cat": CAT, "conditions": {}}
    for cond in CONDS:
        tf = open(os.path.join(RUN_DIR, f"forced_plan_{cond}_{CAT}_transcript.jsonl"), "w")
        met, turns, rows_all = [], [], []
        for roll, (sa, sb) in enumerate(starts):
            histA, histB = [(sb, sa)], [(sa, sb)]
            used = {sa, sb}
            note = ""
            agreed_at = None
            for t in range(1, SAFETY):
                rule_txt, planned = "", ""
                prev = f' Your scratchpad from last round: "{note}"' if note else ""
                base = body_of(histA, used, SCRATCH_INTRO + prev)
                if cond == "free":
                    p1 = LA._render(tok, base) + "\nMy scratchpad:"
                    note = one_line(gen(p1, 7000 * roll + t, 80), 600)
                    pA = p1 + " " + note + "\nMy word:"
                    wA = gen(pA, 5000 * roll + t, 4, forbidden=used)
                else:
                    p1 = (LA._render(tok, base)
                          + "\nMy scratchpad:\n- What the other player seems to be doing:")
                    rule_txt = one_line(gen(p1, 7000 * roll + t, 60))
                    p2 = (p1 + " " + rule_txt
                          + "\n- The exact word I will say this round:")
                    planned = gen(p2, 8000 * roll + t, 4, forbidden=used)
                    note = f"[rule] {rule_txt} [plan] {planned}"
                    if cond == "plan_exec":
                        wA = planned
                    else:
                        tail = (f' You decided to say "{planned}".' if cond == "plan_remind" else "")
                        pA = p2 + " " + planned + tail + "\nMy word:"
                        wA = gen(pA, 5000 * roll + t, 4, forbidden=used)
                # ---- B (restricted) ----
                pB = LA._render(tok, body_of(histB, used, restr)) + "\nMy word:"
                wB = gen(pB, 90000 + 5000 * roll + t, 4, forbidden=used)
                row = {"cond": cond, "rollout": roll, "turn": t, "A": wA, "B": wB,
                       "agreed": bool(wA == wB and wA), "note": note,
                       "rule_txt": rule_txt, "planned": planned,
                       "plan_kept": bool(planned and wA == planned),
                       "planned_is_cat": bool(planned in catset),
                       "played_is_cat": bool(wA in catset),
                       "rule_names_cat": bool(re.search(r"\bcit(y|ies)\b|\bcapital", rule_txt, re.I))}
                tf.write(json.dumps(row) + "\n")
                tf.flush()
                rows_all.append(row)
                if wA == wB and wA:
                    agreed_at = t
                    break
                used |= {wA, wB}
                histA.append((wB, wA))
                histB.append((wA, wB))
            met.append(agreed_at is not None)
            if agreed_at is not None:
                turns.append(agreed_at)
            print(f"[fplan] {cond} roll={roll} "
                  f"{'MET@' + str(agreed_at) if agreed_at else 'no-meet'}", flush=True)
        tf.close()
        pr = [r for r in rows_all if r["planned"]]
        summary["conditions"][cond] = {
            "n": N, "met_frac": float(np.mean(met)), "n_met": int(np.sum(met)),
            "turns_mean": float(np.mean(turns)) if turns else None,
            "played_cat_frac": float(np.mean([r["played_is_cat"] for r in rows_all])),
            "rule_names_cat_frac": float(np.mean([r["rule_names_cat"] for r in rows_all])),
            "planned_cat_frac": float(np.mean([r["planned_is_cat"] for r in pr])) if pr else None,
            "plan_kept_frac": float(np.mean([r["plan_kept"] for r in pr])) if pr else None,
            "plan_kept_when_planned_cat": float(np.mean(
                [r["plan_kept"] for r in pr if r["planned_is_cat"]])) if any(r["planned_is_cat"] for r in pr) else None,
            "played_cat_when_planned_cat": float(np.mean(
                [r["played_is_cat"] for r in pr if r["planned_is_cat"]])) if any(r["planned_is_cat"] for r in pr) else None,
        }
        c = summary["conditions"][cond]
        print(f"[fplan] === {cond}: met {c['met_frac']:.2f} · played-cat {c['played_cat_frac']:.2f} "
              f"· planned-cat {c['planned_cat_frac']} · plan-kept {c['plan_kept_frac']}", flush=True)
        json.dump(summary, open(os.path.join(RUN_DIR, "forced_plan.json"), "w"), indent=1)
    json.dump(summary, open(os.path.join(RUN_DIR, "forced_plan.json"), "w"), indent=1)
    print("[fplan] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
