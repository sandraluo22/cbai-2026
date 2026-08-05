"""THINKING GAME (2026-08-04): the standard game-1 with BOTH players running
Qwen3-32B in native thinking mode (enable_thinking=True, word parsed after
</think>). Same asymmetric setup as the stuck-prior baseline: B city-restricted,
A unrestricted. Question: when do they meet, and does A's deliberation change
the stuck behavior (no-think baseline: met 0.50 at cap 100, r0 ladder)?

Per-turn log: words, agreed, played_is_cat for A, plus head/tail excerpts of
BOTH players' think traces (does A verbalize "partner says cities"? does the
verbalization reach the word this time?).

Env: MODEL(QwenInst32) START_FILE N(8) CAP(40) TEMP(0.7) MAXTHINK(1400)
     RUN_DIR(runs/game1_think)
"""
from __future__ import annotations
import os
import json
import numpy as np
import llm_agents as LA
import qwen32_pca as G
from game1_strict import CATWORDS

MODEL = os.environ.get("MODEL", "QwenInst32")
START_FILE = os.environ.get("START_FILE", "runs/game-1/qwen32/qwen32_pca_w2v/start_words.txt")
N = int(os.environ.get("N", "8"))
CAP = int(os.environ.get("CAP", "40"))
TEMP = float(os.environ.get("TEMP", "0.7"))
MAXTHINK = int(os.environ.get("MAXTHINK", "1400"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/game1_think")

RESTR = (" IMPORTANT: every single word you say must be the name of a city. Only ever "
         "say cities, nothing else.")
ASK = " Now decide your next word. After thinking, answer with only that single word."


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)
    catset = set(CATWORDS["city"])

    def render_think(user_text):
        msgs = [{"role": "user", "content": user_text}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=True)

    def body_of(hist, used, extra=""):
        s = G.OPEN_PROMPT + extra + ((" " + " ".join(
            f"Round {k+1}: the other player said {o}, you said {s_}."
            for k, (o, s_) in enumerate(hist))) if hist else "")
        return s + " Words already used (do not repeat): " + ", ".join(sorted(used)) + "." + ASK

    @torch.no_grad()
    def gen_think(body, seed, forbidden):
        enc = tok(render_think(body), return_tensors="pt").to(dev)
        w, think = "", ""
        for r in range(8):
            torch.manual_seed(seed + 1009 * r)
            out = model.generate(enc.input_ids, attention_mask=enc.get("attention_mask"),
                                 max_new_tokens=MAXTHINK, do_sample=True, temperature=TEMP,
                                 top_p=0.95, pad_token_id=tok.eos_token_id)
            txt = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
            if "</think>" in txt:
                think, ans = txt.rsplit("</think>", 1)
            else:                       # ran out of budget mid-think: no answer emitted
                think, ans = txt, ""
            w = G.clean_word(ans)
            if w and w not in forbidden:
                return w, think.replace("<think>", "").strip()
        return w, think.replace("<think>", "").strip()

    def excerpt(s, n=300):
        s = " ".join(s.split())
        return s if len(s) <= 2 * n else s[:n] + " [...] " + s[-n:]

    starts = []
    for line in open(START_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split("\t") if "\t" in line else line.split()
            starts.append((p[-2], p[-1]))
    starts = starts[:N]

    summary = {"model": MODEL, "cap": CAP, "n": N, "temp": TEMP, "maxthink": MAXTHINK}
    tf = open(os.path.join(RUN_DIR, "think_transcript.jsonl"), "w")
    met, turns, cat_turns = [], [], []
    for roll, (sa, sb) in enumerate(starts):
        histA, histB = [(sb, sa)], [(sa, sb)]
        used = {sa, sb}
        agreed_at = None
        for t in range(1, CAP):
            wA, thA = gen_think(body_of(histA, used), 5000 * roll + t, used)
            wB, thB = gen_think(body_of(histB, used, RESTR), 90000 + 5000 * roll + t, used)
            cat_turns.append(wA in catset)
            tf.write(json.dumps({"rollout": roll, "turn": t, "A": wA, "B": wB,
                                 "agreed": bool(wA == wB and wA),
                                 "played_is_cat": bool(wA in catset),
                                 "thinkA": excerpt(thA), "thinkB": excerpt(thB)}) + "\n")
            tf.flush()
            if wA == wB and wA:
                agreed_at = t
                break
            used |= {wA, wB}
            histA.append((wB, wA)); histB.append((wA, wB))
        met.append(agreed_at is not None)
        if agreed_at is not None:
            turns.append(agreed_at)
        print(f"[thk] roll={roll}: {'MET@' + str(agreed_at) if agreed_at else 'no-meet'}",
              flush=True)
        summary.update({"met_frac": float(np.mean(met)), "n_done": roll + 1,
                        "met_turns": turns,
                        "turns_mean": float(np.mean(turns)) if turns else None,
                        "played_cat_frac": float(np.mean(cat_turns))})
        json.dump(summary, open(os.path.join(RUN_DIR, "think.json"), "w"), indent=1)
    tf.close()
    print(f"[thk] === met {summary['met_frac']:.2f} turns {turns} "
          f"played-cat {summary['played_cat_frac']:.2f}", flush=True)
    print("[thk] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
