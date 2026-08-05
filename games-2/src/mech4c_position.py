"""MECH4c (positional decomposition of the mech4b scramble result) (2026-08-03): does STRUCTURAL CONSISTENCY matter? The history format
now varies ROUND BY ROUND within a single prompt.

If capture needs a consistent textual series (surface-structural H2a), scrambling
the per-round format should destroy it. If capture is role-bound (H2b — bound to
"these are the answering player's words" however rendered), it should survive.

Per-round formats (all express the same round content):
  f0  "Round k: the other player said B, you said A."
  f1  "Round k: you said A, the other player said B."
  f2  "Them: B / You: A" (line pair)
  f3  "Round k - the other player; B. you; A."
  f4  "You: A | Them: B"
  f5  "then they said B and you said A;"

Conditions (x {planted, base} x N_STREAMS, planted = planted/planting/plantings
in A's slots, rounds 1-3, gap 0):
  fixed_f0    all rounds f0 (mech4 reference format)
  fixed_f4    all rounds f4 (a compact format, consistency control)
  alt_f0f1    alternate f0/f1 by round
  all_diff    round k uses format f_k (3 different formats in 3 rounds)
  scramble_a  per-round formats [f2, f5, f3]
  scramble_b  per-round formats [f4, f0, f2]

Measures: K=64 MC planted-family mass (novel), category mass, raw family prob.
Env: MODEL(QwenInst32) SRC_DIR START_FILE N_STREAMS(4) K(64) TEMP(0.7)
     RUN_DIR(runs/mech4c_position)
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
N_STREAMS = int(os.environ.get("N_STREAMS", "4"))
K = int(os.environ.get("K", "64"))
TEMP = float(os.environ.get("TEMP", "0.7"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/mech4c_position")

PLANT = ["planted", "planting", "plantings"]
FAMP = "plan"
FILL = ["window", "carpet", "stapler"]

F = [
    lambda k, b, a: f"Round {k+1}: the other player said {b}, you said {a}.",
    lambda k, b, a: f"Round {k+1}: you said {a}, the other player said {b}.",
    lambda k, b, a: f"Them: {b}\nYou: {a}",
    lambda k, b, a: f"Round {k+1} - the other player; {b}. you; {a}.",
    lambda k, b, a: f"You: {a} | Them: {b}",
    lambda k, b, a: f"then they said {b} and you said {a};",
]
CONDS = {
    "fixed_f0_rerun": [0, 0, 0],
    "r1_f4": [4, 0, 0], "r2_f4": [0, 4, 0], "r3_f4": [0, 0, 4],
    "r1_f2": [2, 0, 0], "r2_f2": [0, 2, 0], "r3_f2": [0, 0, 2],
    "scramble_a_rerun": [2, 5, 3],
    "scramble_b_rerun": [4, 0, 2],
}


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)
    catset = set(CATWORDS["city"])

    rows_t = [json.loads(l) for l in open(os.path.join(SRC_DIR, "game1_strict_city_transcript.jsonl"))]
    games = collections.defaultdict(list)
    for r in rows_t:
        games[r["rollout"]].append(r)
    streams = sorted(games.items(), key=lambda kv: -len(kv[1]))[:N_STREAMS]
    starts = []
    for line in open(START_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split("\t") if "\t" in line else line.split()
            starts.append((p[-2], p[-1]))

    def fid(w):
        return tok(" " + w, add_special_tokens=False)["input_ids"][0]
    fam_ids = torch.tensor(sorted({fid(w) for w in PLANT + [FAMP, "plants", "plant", "planter"]}))

    @torch.no_grad()
    def measure(body, ctx_words):
        prompt = LA._render(tok, body) + "\nMy word:"
        enc = tok(prompt, return_tensors="pt").to(dev)
        logits = model(enc.input_ids).logits[0, -1].float()
        pfam = float(torch.softmax(logits, 0)[fam_ids].sum())
        out = model.generate(enc.input_ids.repeat(K, 1), max_new_tokens=4, do_sample=True,
                             temperature=TEMP, top_p=0.95, pad_token_id=tok.eos_token_id)
        ws = [G.clean_word(tok.decode(out[i, enc.input_ids.shape[1]:], skip_special_tokens=True))
              for i in range(K)]
        fam = float(np.mean([1 if (w and w not in ctx_words and len(w) > 3 and w[:4] == FAMP)
                             else 0 for w in ws]))
        cat = float(np.mean([1 if (w and w not in ctx_words and w in catset) else 0 for w in ws]))
        return pfam, fam, cat

    results = []
    for si, (roll, ts) in enumerate(streams):
        Bseq = [t["B"] for t in ts]
        sa, sb = starts[roll]
        for cond, fmts in CONDS.items():
            for content in ("planted", "base"):
                aw = PLANT if content == "planted" else FILL
                rounds = [(Bseq[k], aw[k]) for k in range(3)]
                hist = " " + " ".join(F[fmts[k]](k, b, a) for k, (b, a) in enumerate(rounds))
                used = sorted({sa, sb} | {w for b, a in rounds for w in (b, a)})
                body = (G.OPEN_PROMPT + hist
                        + " Words already used (do not repeat): " + ", ".join(used) + ".")
                ctx = {w for b, a in rounds for w in (b, a)} | set(PLANT)
                pfam, fam, cat = measure(body, ctx)
                results.append({"cond": cond, "content": content, "stream": roll,
                                "pfam": pfam, "fam_mass": fam, "cat_mass": cat})
                json.dump({"per_cell": results}, open(os.path.join(RUN_DIR, "mech4c.json"), "w"))
        print(f"[mech4c] stream {roll} done", flush=True)

    summary = {}
    for cond in CONDS:
        p = float(np.mean([r["fam_mass"] for r in results if r["cond"] == cond and r["content"] == "planted"]))
        b = float(np.mean([r["fam_mass"] for r in results if r["cond"] == cond and r["content"] == "base"]))
        summary[cond] = {"planted": p, "base": b, "capture": p - b}
        print(f"[mech4c] === {cond}: capture {p - b:.3f} (planted {p:.3f} base {b:.3f})", flush=True)
    json.dump({"per_cell": results, "summary": summary},
              open(os.path.join(RUN_DIR, "mech4c.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
