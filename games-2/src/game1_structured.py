"""STRUCTURED-PROMPT variant of the restricted Game-1 (2026-07-25, user-designed):
same game, but the prompt is a concise memory-log format instead of prose —
round-grouped memories, bulleted rules, JSON-only output (assistant prefilled with
'{"word": "' so the read-out stays a single word). Tests whether prompt FORMAT alone
modulates the stuck-prior pathology (baseline prose: met 0.44, stuck ~0.5, Qwen32).

A's prompt (B adds one secret-rule bullet):

  Word game.
  Rules:
  - Each round, you and the other player each say one word at the same time.
  - You both win the round if you both say the same word.
  - No word said by either player in any earlier round may be said again.
  The memory shows words from previous rounds of this game.
  Memory of observed words, grouped by round:
  Round 1 memories: ["You: scripted", "Other: acquitted"]
  ...
  Round k memories (current round): []
  Each memory entry is of the form "<who>: <word>".
  Words no longer allowed: ["...", ...]
  Constraints:
  - Output JSON only.
  - The word must not be in the not-allowed list.
  Output JSON exactly: {"word": "<word>"}

MODE=scratch adds a persistent private scratchpad IN the structured format: the prompt
carries last round's note ('Scratchpad (private, from last round): "..."') and the
output schema becomes {"scratchpad": "<brief updated note>", "word": "<word>"} —
prefilled at '{"scratchpad": "' so note+word come from one generation.

Env: MODEL(QwenInst32) START_FILE N(16) SAFETY(24) TEMP(0.7) CATS(city,fruit)
     MODE(plain|scratch) RUN_DIR
Out: <RUN_DIR>/game1_structured.json + per-cat transcripts
"""
from __future__ import annotations
import os
import json
import re
import numpy as np
import llm_agents as LA

MODEL = os.environ.get("MODEL", "QwenInst32")
START_FILE = os.environ.get("START_FILE", "runs/game-1/qwen32/qwen32_pca_w2v/start_words.txt")
N = int(os.environ.get("N", "16"))
SAFETY = int(os.environ.get("SAFETY", "24"))
TEMP = float(os.environ.get("TEMP", "0.7"))
CATS = os.environ.get("CATS", "city,fruit").split(",")
RUN_DIR = os.environ.get("RUN_DIR", "runs/qwen32_structured")

CONCEPTS = {"city": "the name of a city", "fruit": "a fruit"}
MODE = os.environ.get("MODE", "plain")


def build_prompt(tok, hist, used, cur_round, secret=None, note=None, use_scratch=False):
    lines = ["Word game.", "Rules:",
             "- Each round, you and the other player each say one word at the same time.",
             "- You both win the round if you both say the same word.",
             "- No word said by either player in any earlier round may be said again."]
    if secret:
        lines.append(f"- Secret rule that applies only to you: every word you output must "
                     f"be {CONCEPTS[secret]}.")
    lines.append("The memory shows words from previous rounds of this game.")
    lines.append("Memory of observed words, grouped by round:")
    for k, (o, s) in enumerate(hist):
        lines.append(f'Round {k+1} memories: ["You: {s}", "Other: {o}"]')
    lines.append(f"Round {cur_round} memories (current round): []")
    lines.append('Each memory entry is of the form "<who>: <word>".')
    lines.append("Words no longer allowed: [" + ", ".join(f'"{w}"' for w in sorted(used)) + "]")
    if use_scratch:
        lines.append("You have a private scratchpad the other player never sees.")
        if note:
            lines.append(f'Scratchpad (private, from last round): "{note}"')
    lines.append("Constraints:")
    lines.append("- Output JSON only.")
    lines.append("- The word must not be in the not-allowed list.")
    if use_scratch:
        lines.append('Output JSON exactly: {"scratchpad": "<brief updated note>", "word": "<word>"}')
        return LA._render(tok, "\n".join(lines)) + '\n{"scratchpad": "'
    lines.append('Output JSON exactly: {"word": "<word>"}')
    return LA._render(tok, "\n".join(lines)) + '\n{"word": "'


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)

    @torch.no_grad()
    def gen_word(prompt, seed, forbidden, with_note=False):
        enc = tok(prompt, return_tensors="pt").to(dev)
        w, nt = "", ""
        for r in range(24):
            torch.manual_seed(seed + 1009 * r)
            out = model.generate(enc.input_ids, attention_mask=enc.get("attention_mask"),
                                 max_new_tokens=90 if with_note else 6, do_sample=True,
                                 temperature=TEMP, top_p=0.95, pad_token_id=tok.eos_token_id)
            txt = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
            if with_note:
                nt = txt.split('"')[0].strip()[:280]
                mw = re.search(r'"word"\s*:\s*"([A-Za-z\-]+)', txt)
                w = mw.group(1).lower() if mw else ""
            else:
                m = re.match(r'\s*([A-Za-z\-]+)', txt)
                w = m.group(1).lower() if m else ""
            if w and w not in forbidden:
                return (w, nt) if with_note else w
        return (w, nt) if with_note else w

    starts = []
    for line in open(START_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split("\t") if "\t" in line else line.split()
            starts.append((p[-2], p[-1]))
    starts = starts[:N]

    summary = {"model": MODEL, "temp": TEMP, "safety": SAFETY, "n": N, "conditions": {}}
    for cat in CATS:
        tf = open(os.path.join(RUN_DIR, f"game1_structured_{MODE}_{cat}_transcript.jsonl"), "w")
        met, turns, pers_all = [], [], []
        for roll, (sa, sb) in enumerate(starts):
            histA, histB = [(sb, sa)], [(sa, sb)]
            used = {sa, sb}
            seqA = []
            note = ""
            agreed_at = None
            for t in range(1, SAFETY):
                if MODE == "scratch":
                    wA, note = gen_word(build_prompt(tok, histA, used, t + 1, note=note,
                                                     use_scratch=True),
                                        5000 * roll + t, used, with_note=True)
                else:
                    wA = gen_word(build_prompt(tok, histA, used, t + 1), 5000 * roll + t, used)
                wB = gen_word(build_prompt(tok, histB, used, t + 1, secret=cat),
                              90000 + 5000 * roll + t, used)
                seqA.append(wA)
                row = {"cond": f"struct-{MODE}-{cat}", "rollout": roll, "turn": t,
                       "A": wA, "B": wB, "agreed": wA == wB}
                if MODE == "scratch":
                    row["note"] = note
                tf.write(json.dumps(row) + "\n")
                tf.flush()
                if wA == wB and wA:
                    agreed_at = t
                    break
                used |= {wA, wB}
                histA.append((wB, wA)); histB.append((wA, wB))
            pers = float(np.mean([any(w[:4] == p[:4] and len(w) > 3 for p in seqA[:i])
                                  for i, w in enumerate(seqA)][1:])) if len(seqA) > 1 else 0.0
            met.append(agreed_at is not None)
            pers_all.append(pers)
            if agreed_at is not None:
                turns.append(agreed_at)
            print(f"[struct] {cat} roll={roll} "
                  f"{'MET@' + str(agreed_at) if agreed_at else 'no-meet'} pers={pers:.2f}",
                  flush=True)
        tf.close()
        summary["conditions"][f"struct-{MODE}-{cat}"] = {
            "n": N, "met_frac": float(np.mean(met)),
            "turns_mean": float(np.mean(turns)) if turns else None,
            "pers_mean": float(np.mean(pers_all)),
            "n_met": int(np.sum(met))}
        json.dump(summary, open(os.path.join(RUN_DIR, f"game1_structured_{MODE}.json"), "w"),
                  indent=1)
        print(f"[struct] === struct-{MODE}-{cat}: met {np.mean(met):.2f} "
              f"pers {np.mean(pers_all):.2f}", flush=True)


if __name__ == "__main__":
    main()
