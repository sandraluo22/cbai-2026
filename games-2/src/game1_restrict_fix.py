"""Why do restricted Game-1 games fail? Two candidate mechanisms, one intervention each
(baseline = game1_yoked.py restrict-* runs: used-list shown, no-repeat enforced):

  nolist   : keep the no-repeat rule AND resampler enforcement, but HIDE the enumerated
             used-word list from the prompt. Tests: the list itself primes the
             morphological perseveration loops (spine/spineless/spiny...).
  repeatok : drop the no-repeat rule entirely (prompt sentence + enforcement). Tests:
             under no-repeat the common category words get burned and A's
             category-directed mass is stranded on forbidden words (exhaustion).

B is secretly restricted to the concept, A is unrestricted (as in game1_yoked.py).

Env: MODEL(QwenInst32) START_FILE SAFETY(24) TEMP(0.7) N(16) RUN_DIR
     CONCEPTS(city,fruit) MODES(nolist,repeatok)
Out: <RUN_DIR>/game1_restrict_fix.json + per-condition *_transcript.jsonl
"""
from __future__ import annotations
import os
import json
import numpy as np
import llm_agents as LA
import qwen32_pca as G

MODEL = os.environ.get("MODEL", "QwenInst32")
START_FILE = os.environ.get("START_FILE", "runs/game1_qwen32_pca_w2v/start_words.txt")
SAFETY = int(os.environ.get("SAFETY", "24"))
TEMP = float(os.environ.get("TEMP", "0.7"))
N = int(os.environ.get("N", "16"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/game1_restrict_fix")
CONCEPTS = [c for c in os.environ.get("CONCEPTS", "city,fruit").split(",") if c]
MODES = [m for m in os.environ.get("MODES", "nolist,repeatok").split(",") if m]

CONCEPT_TEXT = {"city": ("the name of a city", "cities"), "fruit": ("a fruit", "fruits")}
NOREP_SENT = " You may not say any word that has already been said by either player."
PROMPT_NOREP = G.OPEN_PROMPT                      # includes the no-repeat sentence
PROMPT_REPOK = G.OPEN_PROMPT.replace(NOREP_SENT, "")
assert PROMPT_REPOK != PROMPT_NOREP


def build_prompt(tok, hist, mode, restrict=None):
    base = PROMPT_NOREP if mode == "nolist" else PROMPT_REPOK
    lines = " ".join(f"Round {k+1}: the other player said {o}, you said {s}."
                     for k, (o, s) in enumerate(hist))
    body = base + ((" " + lines) if lines else "")
    if restrict:
        sing, plur = CONCEPT_TEXT[restrict]
        body += f" IMPORTANT: every single word you say must be {sing}. Only ever say {plur}, nothing else."
    return LA._render(tok, body) + "\nMy word:"


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
    starts = starts[:N]

    summary = {"model": MODEL, "temp": TEMP, "safety": SAFETY, "n": len(starts),
               "conditions": {}}
    for mode in MODES:
        for concept in CONCEPTS:
            cond = f"{mode}-{concept}"
            tf = open(os.path.join(RUN_DIR, f"game1_restrict_fix_{cond}_transcript.jsonl"), "w")
            met, turns = [], []
            for roll, (sa, sb) in enumerate(starts):
                histA, histB = [(sb, sa)], [(sa, sb)]
                used = {sa, sb}
                agreed_at = None
                for t in range(1, SAFETY):
                    forb = used if mode == "nolist" else set()
                    wA = gen_word(build_prompt(tok, histA, mode), 5000 * roll + t, forb)
                    wB = gen_word(build_prompt(tok, histB, mode, restrict=concept),
                                  90000 + 5000 * roll + t, forb)
                    tf.write(json.dumps({"cond": cond, "rollout": roll, "turn": t,
                                         "A": wA, "B": wB, "agreed": wA == wB}) + "\n")
                    tf.flush()
                    if wA == wB and wA:
                        agreed_at = t
                        break
                    used |= {wA, wB}
                    histA.append((wB, wA)); histB.append((wA, wB))
                met.append(agreed_at is not None)
                if agreed_at is not None:
                    turns.append(agreed_at)
                print(f"[fix] {cond} roll={roll} "
                      f"{'MET@' + str(agreed_at) if agreed_at else 'no-meet'}", flush=True)
            tf.close()
            n = len(met)
            summary["conditions"][cond] = {
                "n": n, "met_frac": float(np.mean(met)),
                "met_se": float(np.std(met) / np.sqrt(n)),
                "turns_mean": float(np.mean(turns)) if turns else None,
                "turns_se": float(np.std(turns) / np.sqrt(len(turns))) if turns else None,
                "n_met": int(np.sum(met))}
            json.dump(summary, open(os.path.join(RUN_DIR, "game1_restrict_fix.json"), "w"),
                      indent=1)
            print(f"[fix] === {cond}: met {np.mean(met):.2f}", flush=True)


if __name__ == "__main__":
    main()
