"""Four Game-1 variants (2026-07-24), same self-play loop / starts / temp as baseline:

  fewshot      : baseline strict no-repeat + an EXAMPLE of a finished game in the prompt
                 (two players approaching each other and winning on 'shore'). Does
                 demonstration-by-example speed coordination? (Deliberate coaching —
                 labeled as such; compare against the uncoached reactive baseline.)
  city-coastal : BOTH players restricted to cities; B additionally told to "Prefer
                 locations associated with coastlines and warm weather." Does B's soft
                 private preference leak into the meeting point?
  board16      : both players pick from a FIXED 16-word board shown in the prompt —
                 4 distinct categories x 4, chosen so cross-category pairs still share
                 some association. Strict no-repeat (board shrinks; <=8 rounds).
                 No forced start.
  softrepeat   : strict no-repeat replaced by "Avoid repeating your immediately
                 previous word unless repetition would produce coordination."
                 No enforcement, no used-list — tests strategic repetition (insist).

Env: MODEL(QwenInst32) START_FILE SAFETY(24) TEMP(0.7) N(16) RUN_DIR
     VARIANTS(fewshot,city-coastal,board16,softrepeat)
Out: <RUN_DIR>/game1_variants.json + per-variant *_transcript.jsonl
"""
from __future__ import annotations
import os
import json
import numpy as np
import llm_agents as LA
import qwen32_pca as G

MODEL = os.environ.get("MODEL", "QwenInst32")
START_FILE = os.environ.get("START_FILE", "runs/game-1/qwen32/qwen32_pca_w2v/start_words.txt")
SAFETY = int(os.environ.get("SAFETY", "24"))
TEMP = float(os.environ.get("TEMP", "0.7"))
N = int(os.environ.get("N", "16"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/qwen32_game_variants")
VARIANTS = os.environ.get("VARIANTS", "fewshot,city-coastal,board16,softrepeat").split(",")

NOREP_SENT = " You may not say any word that has already been said by either player."
PROMPT_REPOK = G.OPEN_PROMPT.replace(NOREP_SENT, "")
assert PROMPT_REPOK != G.OPEN_PROMPT

FEWSHOT = (" Here is an example of a finished game between two other players: Round 1: one"
           " player said mountain, the other said ocean. Round 2: one said coast, the other"
           " said cliff. Round 3: both said shore, and they won.")
CITY = (" IMPORTANT: every single word you say must be the name of a city. Only ever say"
        " cities, nothing else.")
COASTAL = " Prefer locations associated with coastlines and warm weather."
SOFT = (" Avoid repeating your immediately previous word unless repetition would produce"
        " coordination.")
BOARD = ["river", "wave", "ice", "mist",          # water
         "star", "flame", "dawn", "glow",         # light
         "stone", "moss", "sand", "peak",         # earth
         "wind", "cloud", "echo", "storm"]        # air


def build_prompt(tok, variant, hist, used, player):
    if variant == "fewshot":
        body = G.OPEN_PROMPT + FEWSHOT
    elif variant == "city-coastal":
        body = G.OPEN_PROMPT + CITY + (COASTAL if player == "B" else "")
    elif variant == "board16":
        body = (PROMPT_REPOK + " You may only say words from this board: "
                + ", ".join(BOARD) + "." + NOREP_SENT)
    elif variant == "softrepeat":
        body = PROMPT_REPOK + SOFT
    else:
        raise ValueError(variant)
    lines = " ".join(f"Round {k+1}: the other player said {o}, you said {s}."
                     for k, (o, s) in enumerate(hist))
    if lines:
        body += " " + lines
    if used and variant != "softrepeat":
        body += " Words already used (do not repeat): " + ", ".join(sorted(used)) + "."
    return LA._render(tok, body) + "\nMy word:"


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)

    @torch.no_grad()
    def gen_word(prompt, seed, forbidden, board=None):
        enc = tok(prompt, return_tensors="pt").to(dev)
        w = ""
        for r in range(24):
            torch.manual_seed(seed + 1009 * r)
            out = model.generate(enc.input_ids, attention_mask=enc.get("attention_mask"),
                                 max_new_tokens=4, do_sample=True, temperature=TEMP, top_p=0.95,
                                 pad_token_id=tok.eos_token_id)
            w = G.clean_word(tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True))
            if w and w not in forbidden and (board is None or w in board):
                return w
        return w

    starts = []
    for line in open(START_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split("\t") if "\t" in line else line.split()
            starts.append((p[-2], p[-1]))
    starts = starts[:N]

    summary = {"model": MODEL, "temp": TEMP, "safety": SAFETY, "n": N, "board": BOARD,
               "conditions": {}}
    for variant in VARIANTS:
        tf = open(os.path.join(RUN_DIR, f"game1_variants_{variant}_transcript.jsonl"), "w")
        met, turns = [], []
        for roll in range(N):
            if variant == "board16":
                histA, histB, used = [], [], set()
            else:
                sa, sb = starts[roll]
                histA, histB, used = [(sb, sa)], [(sa, sb)], {sa, sb}
            strict = variant in ("fewshot", "city-coastal", "board16")
            agreed_at = None
            for t in range(1, SAFETY):
                board = set(BOARD) - used if variant == "board16" else None
                if board is not None and len(board) < 2:
                    break
                forb = used if strict else set()
                wA = gen_word(build_prompt(tok, variant, histA, used, "A"),
                              5000 * roll + t, forb, board)
                wB = gen_word(build_prompt(tok, variant, histB, used, "B"),
                              90000 + 5000 * roll + t, forb, board)
                tf.write(json.dumps({"variant": variant, "rollout": roll, "turn": t,
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
            print(f"[var] {variant} roll={roll} "
                  f"{'MET@' + str(agreed_at) if agreed_at else 'no-meet'}", flush=True)
        tf.close()
        summary["conditions"][variant] = {
            "n": N, "met_frac": float(np.mean(met)),
            "met_se": float(np.std(met) / np.sqrt(N)),
            "turns_mean": float(np.mean(turns)) if turns else None,
            "turns_se": float(np.std(turns) / np.sqrt(len(turns))) if turns else None,
            "n_met": int(np.sum(met))}
        json.dump(summary, open(os.path.join(RUN_DIR, "game1_variants.json"), "w"), indent=1)
        print(f"[var] === {variant}: met {np.mean(met):.2f}", flush=True)


if __name__ == "__main__":
    main()
