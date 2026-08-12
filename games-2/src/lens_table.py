"""LENS TABLE (2026-08-08): layer x game-turn grid of force-decoded words at the
answer slot (logit lens), for one real stuck game replayed state by state.

For turns t=1..T (state = history through round t), prompt + "\nMy word: **";
per layer: argmax token (decoded string) + its prob + this game's family prob +
city prob. Saved for local heatmap rendering.

Env: MODEL(QwenInst32) SRC_DIR START_FILE ROLL(auto: most family-repeats) T(16)
     RUN_DIR(runs/lens_table)
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
TMAX = int(os.environ.get("T", "16"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/lens_table")


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)
    nL = model.config.num_hidden_layers
    catset = set(CATWORDS["city"])

    games = collections.defaultdict(list)
    for line in open(os.path.join(SRC_DIR, "game1_strict_city_transcript.jsonl")):
        d = json.loads(line)
        games[d["rollout"]].append(d)
    for k in games:
        games[k].sort(key=lambda r: r["turn"])
    def famcount(ts):
        c = collections.Counter(w[:4] for w in (r["A"] for r in ts) if len(w) > 3)
        return c.most_common(1)[0] if c else ("", 0)
    roll = int(os.environ.get("ROLL", "-1"))
    if roll < 0:
        roll = max((k for k in games if len(games[k]) >= 10),
                   key=lambda k: famcount(games[k])[1])
    ts = games[roll]
    famp = famcount(ts)[0]
    starts = []
    for line in open(START_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split("\t") if "\t" in line else line.split()
            starts.append((p[-2], p[-1]))
    sa, sb = starts[roll]

    def fids(w):
        return {tok(v, add_special_tokens=False)["input_ids"][0]
                for v in (w, w.capitalize())}
    famwords = sorted({r["A"] for r in ts if len(r["A"]) > 3 and r["A"][:4] == famp}
                      | {famp, famp + "s"})
    fam_ids = torch.tensor(sorted({i for w in famwords for i in fids(w)})).to(dev)

    out_rows = []
    for T in range(1, min(TMAX, len(ts)) + 1):
        hist = [(sb, sa)] + [(r["B"], r["A"]) for r in ts[:T]]
        used = {sa, sb} | {w for r in ts[:T] for w in (r["A"], r["B"])}
        body = (G.OPEN_PROMPT + " " + " ".join(
            f"Round {k+1}: the other player said {o}, you said {s_}."
            for k, (o, s_) in enumerate(hist))
            + " Words already used (do not repeat): " + ", ".join(sorted(used)) + ".")
        catwords = [w for w in CATWORDS["city"] if w not in used][:30]
        cat_ids = torch.tensor(sorted({i for w in catwords for i in fids(w)})).to(dev)
        prompt = LA._render(tok, body) + "\nMy word: **"
        with torch.no_grad():
            ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
            out = model(ids, output_hidden_states=True)
            for l in range(1, nL + 1):
                h = out.hidden_states[l][0, -1]
                hn = h.unsqueeze(0) if l == nL else model.model.norm(h.unsqueeze(0))
                z = model.lm_head(hn)[0].float()
                p = torch.softmax(z, 0)
                top = int(torch.argmax(p))
                out_rows.append({"turn": T, "layer": l, "tok": tok.decode([top]),
                                 "p": float(p[top]), "p_fam": float(p[fam_ids].sum()),
                                 "p_cat": float(p[cat_ids].sum())})
            del out
        print(f"[lt] turn {T} done", flush=True)
    json.dump({"roll": roll, "famp": famp, "A": [r["A"] for r in ts[:TMAX]],
               "B": [r["B"] for r in ts[:TMAX]], "rows": out_rows},
              open(os.path.join(RUN_DIR, "lens_table.json"), "w"))
    print("[lt] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
