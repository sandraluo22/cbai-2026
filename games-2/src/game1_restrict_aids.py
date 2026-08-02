"""Two AIDS for the unrestricted player in the restricted Game-1 (2026-07-25):

  informed-<c>  : A is TOLD the partner is secretly restricted to some single category
                  (but not which). B restricted to <c> as usual. Tests whether knowing a
                  constraint EXISTS unlocks the category knowledge A demonstrably has
                  (constraint probe: ~100% decodable) but fails to act on.
  scratch-<c>   : A gets a PERSISTENT PRIVATE SCRATCHPAD: each turn it sees its previous
                  note, writes a new one (~sentences), then says its word with the note
                  in context. B never sees it. Notes are logged in the transcript.
                  Tests whether explicit reasoning space converts knowledge to action.

Baselines: restrict-<c> (Qwen32 cap24: 0.44 met @ ~5-6; failures = perseveration loops).
Same starts / no-repeat / temp as game1_yoked.py.

Env: MODEL(QwenInst32) START_FILE SAFETY(24) TEMP(0.7) N(16) RUN_DIR
     CONDS(informed-city,informed-fruit,scratch-city,scratch-fruit)
Out: <RUN_DIR>/game1_restrict_aids.json + per-condition *_transcript.jsonl (rows include
     A's scratchpad note for scratch-*)
"""
from __future__ import annotations
import os
import json
import re
import numpy as np
import llm_agents as LA
import qwen32_pca as G

MODEL = os.environ.get("MODEL", "QwenInst32")
START_FILE = os.environ.get("START_FILE", "runs/game-1/qwen32/qwen32_pca_w2v/start_words.txt")
SAFETY = int(os.environ.get("SAFETY", "24"))
TEMP = float(os.environ.get("TEMP", "0.7"))
N = int(os.environ.get("N", "16"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/qwen32_restrict_aids")
CONDS = os.environ.get("CONDS",
                       "informed-city,informed-fruit,scratch-city,scratch-fruit").split(",")

CONCEPTS = {"city": ("the name of a city", "cities"), "fruit": ("a fruit", "fruits")}
INFORMED = (" NOTE: the other player is secretly required to only say words from one "
            "particular category. You are not told which category it is.")


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

    def clean_note(txt):
        txt = re.sub(r"\s+", " ", txt.strip())
        return txt[:280]

    starts = []
    for line in open(START_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split("\t") if "\t" in line else line.split()
            starts.append((p[-2], p[-1]))
    starts = starts[:N]

    summary = {"model": MODEL, "temp": TEMP, "safety": SAFETY, "n": N, "conditions": {}}
    for cond in CONDS:
        mode, concept = cond.split("-", 1)
        sing, plur = CONCEPTS[concept]
        restr = (f" IMPORTANT: every single word you say must be {sing}. Only ever say "
                 f"{plur}, nothing else.")
        tf = open(os.path.join(RUN_DIR, f"game1_restrict_aids_{cond}_transcript.jsonl"), "w")
        met, turns = [], []
        for roll, (sa, sb) in enumerate(starts):
            histA, histB = [(sb, sa)], [(sa, sb)]
            used = {sa, sb}
            note = ""
            agreed_at = None
            for t in range(1, SAFETY):
                # ---- A ----
                if mode == "informed":
                    pA = LA._render(tok, body_of(histA, used, INFORMED)) + "\nMy word:"
                    wA = gen(pA, 5000 * roll + t, 4, forbidden=used)
                else:                                    # scratch
                    base = body_of(histA, used,
                                   " You have a private scratchpad the other player never "
                                   "sees; use it for brief notes about what is happening "
                                   "and your plan."
                                   + (f' Your scratchpad from last round: "{note}"' if note
                                      else ""))
                    p1 = LA._render(tok, base) + "\nMy scratchpad:"
                    note = clean_note(gen(p1, 7000 * roll + t, 80))
                    pA = p1 + " " + note + "\nMy word:"
                    wA = gen(pA, 5000 * roll + t, 4, forbidden=used)
                # ---- B (restricted, unchanged) ----
                pB = LA._render(tok, body_of(histB, used, restr)) + "\nMy word:"
                wB = gen(pB, 90000 + 5000 * roll + t, 4, forbidden=used)
                row = {"cond": cond, "rollout": roll, "turn": t, "A": wA, "B": wB,
                       "agreed": wA == wB}
                if mode == "scratch":
                    row["note"] = note
                tf.write(json.dumps(row) + "\n")
                tf.flush()
                if wA == wB and wA:
                    agreed_at = t
                    break
                used |= {wA, wB}
                histA.append((wB, wA)); histB.append((wA, wB))
            met.append(agreed_at is not None)
            if agreed_at is not None:
                turns.append(agreed_at)
            print(f"[aids] {cond} roll={roll} "
                  f"{'MET@' + str(agreed_at) if agreed_at else 'no-meet'}", flush=True)
        tf.close()
        summary["conditions"][cond] = {
            "n": N, "met_frac": float(np.mean(met)),
            "met_se": float(np.std(met) / np.sqrt(N)),
            "turns_mean": float(np.mean(turns)) if turns else None,
            "turns_se": float(np.std(turns) / np.sqrt(len(turns))) if turns else None,
            "n_met": int(np.sum(met))}
        json.dump(summary, open(os.path.join(RUN_DIR, "game1_restrict_aids.json"), "w"),
                  indent=1)
        print(f"[aids] === {cond}: met {np.mean(met):.2f}", flush=True)


if __name__ == "__main__":
    main()
