"""ELEUSIS / WASON 2-4-6 (2026-08-05): hidden-rule induction against a scripted
oracle — does the family attractor corrupt hypothesis search when there is an
objective yes/no signal?

A hidden rule classifies words YES/NO. The model is given ONE seed YES word
(2-4-6 style), then proposes one NOVEL word per round; the scripted oracle labels
it; the labeled history accumulates. Goal (stated): propose as many YES words as
possible. No coaching about strategy or the rule's nature.

Rules (all surface-checkable):
  double_letter  a letter appears twice in a row        seed "coffee"
  starts_vowel   first letter is a vowel                seed "orange"
  len7plus       at least 7 letters                     seed "lantern"
  contains_r     contains the letter r                  seed "river"

Measures per game (R rounds): per-round proposal + label; yes-rate by 5-round
block (learning curve); family perseveration (max 4-prefix run, distinct
families); rule verbalization elicited at rounds 10 and R (free text, logged;
crude keyword score). Note family-riding is RATIONAL under starts_vowel (family
members share the initial) and irrelevant-to-harmful under the others — the
rules dissociate exploitation from perseveration.

Env: MODEL(QwenInst32) N(6 rollouts/rule) R(20) TEMP(0.7) RUN_DIR(runs/eleusis)
"""
from __future__ import annotations
import os
import json
import numpy as np
import llm_agents as LA
import qwen32_pca as G

MODEL = os.environ.get("MODEL", "QwenInst32")
N = int(os.environ.get("N", "6"))
R = int(os.environ.get("R", "20"))
TEMP = float(os.environ.get("TEMP", "0.7"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/eleusis")

RULES = {
    "double_letter": (lambda w: any(a == b for a, b in zip(w, w[1:])), "coffee",
                      ["double", "twice in a row", "repeated letter", "consecutive"]),
    "starts_vowel": (lambda w: bool(w) and w[0] in "aeiou", "orange",
                     ["vowel"]),
    "len7plus": (lambda w: len(w) >= 7, "lantern",
                 ["seven", "7", "long", "length"]),
    "contains_r": (lambda w: "r" in w, "river",
                   ["letter r", "'r'", "\"r\"", " r "]),
}

OPEN = ("You are playing a rule-discovery game. There is a hidden rule that "
        "classifies English words as YES or NO. Each round you propose one word "
        "you have not proposed before, and you are told whether it satisfies the "
        "rule. Your goal is to propose as many YES words as possible. You are "
        "told that the word \"{seed}\" satisfies the rule.")


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)

    @torch.no_grad()
    def gen(prompt, seed, max_new=4):
        enc = tok(prompt, return_tensors="pt").to(dev)
        torch.manual_seed(seed)
        out = model.generate(enc.input_ids, attention_mask=enc.get("attention_mask"),
                             max_new_tokens=max_new, do_sample=True, temperature=TEMP,
                             top_p=0.95, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)

    def gen_word(prompt, seed, forbidden):
        w = ""
        for r in range(24):
            w = G.clean_word(gen(prompt, seed + 1009 * r))
            if w and w not in forbidden:
                return w
        return w

    tf = open(os.path.join(RUN_DIR, "eleusis_transcript.jsonl"), "w")
    results = []
    for rule, (fn, seedw, kws) in RULES.items():
        for roll in range(N):
            hist = []
            proposed = {seedw}
            guesses = {}
            for t in range(1, R + 1):
                body = OPEN.format(seed=seedw)
                if hist:
                    body += " " + " ".join(
                        f"Round {k+1}: you proposed \"{w}\" - {'YES' if y else 'NO'}."
                        for k, (w, y) in enumerate(hist))
                if t - 1 in (10, R) or t == R:
                    pass
                prompt = LA._render(tok, body + " Propose your next word.") + "\nMy word:"
                w = gen_word(prompt, 7000 * roll + 131 * t + hash(rule) % 1000, proposed)
                y = bool(w) and fn(w)
                hist.append((w, y))
                proposed.add(w)
                tf.write(json.dumps({"rule": rule, "rollout": roll, "turn": t,
                                     "word": w, "yes": y}) + "\n")
                tf.flush()
                if t in (10, R):
                    gp = (LA._render(tok, body + f" Round {t}: you proposed \"{w}\" - "
                          f"{'YES' if y else 'NO'}. What do you think the hidden rule is?")
                          + "\nThe rule is:")
                    guess = gen(gp, 8000 * roll + t, 30).strip().split("\n")[0][:200]
                    guesses[str(t)] = guess
            words = [w for w, _ in hist]
            yesv = [y for _, y in hist]
            blocks = [float(np.mean(yesv[i:i + 5])) for i in range(0, R, 5)]
            fams = [w[:4] for w in words if len(w) > 3]
            runs, cur = [], 1
            for a, b in zip(fams, fams[1:]):
                cur = cur + 1 if a == b else 1
                runs.append(cur)
            maxrun = max(runs) if runs else 1
            gk = {t: any(k in g.lower() for k in kws) for t, g in guesses.items()}
            results.append({"rule": rule, "rollout": roll, "yes_blocks": blocks,
                            "yes_total": float(np.mean(yesv)), "max_family_run": maxrun,
                            "n_families": len(set(fams)), "guesses": guesses,
                            "guess_correct": gk})
            json.dump({"per_game": results}, open(os.path.join(RUN_DIR, "eleusis.json"), "w"))
            print(f"[elu] {rule} roll={roll}: yes {np.mean(yesv):.2f} blocks "
                  f"{[round(b, 2) for b in blocks]} maxrun {maxrun}", flush=True)
    tf.close()

    out = {"per_game": results, "rules": {}}
    for rule in RULES:
        sel = [r for r in results if r["rule"] == rule]
        out["rules"][rule] = {
            "yes_total": float(np.mean([r["yes_total"] for r in sel])),
            "yes_blocks": [float(np.mean([r["yes_blocks"][i] for r in sel]))
                           for i in range(R // 5)],
            "max_family_run": float(np.mean([r["max_family_run"] for r in sel])),
            "n_families": float(np.mean([r["n_families"] for r in sel])),
            "guess10": float(np.mean([r["guess_correct"].get("10", False) for r in sel])),
            "guessR": float(np.mean([r["guess_correct"].get(str(R), False) for r in sel]))}
        c = out["rules"][rule]
        print(f"[elu] === {rule}: yes {c['yes_total']:.2f} blocks "
              f"{[round(b, 2) for b in c['yes_blocks']]} maxrun {c['max_family_run']:.1f} "
              f"guess {c['guess10']:.2f}->{c['guessR']:.2f}", flush=True)
    json.dump(out, open(os.path.join(RUN_DIR, "eleusis.json"), "w"), indent=1)
    print("[elu] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
