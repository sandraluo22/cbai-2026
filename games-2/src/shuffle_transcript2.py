"""SHUFFLE TRANSCRIPT (2026-08-05): does the loop depend on WHERE in the transcript
the model's own words sit, or only on WHICH words they are?

Real stuck states from the strict city run (games where A has a dominant 4-prefix
family with >=3 members). At round K we permute A's OWN column across rounds
(B's column fixed; the used-word list is sorted, hence identical) and compare:

  orig       the real transcript
  shuf1..3   three random permutations of A's words across the K rounds

Measures per state x condition (K=64 MC + greedy):
  fam_mass   novel proposals in A's dominant family
  greedy     temperature-0 next word; agree = same word as orig's greedy
  tail_fam   how many of the last 2 rounds hold family words after the permutation
             (mech4c predicts capture tracks THIS, not the bag of words)

Env: MODEL(QwenInst32) SRC_DIR(runs/qwen32_strict) START_FILE N_STATES(8) K(64)
     TEMP(0.7) SEED(0) RUN_DIR(runs/shuffle_transcript2)
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
N_STATES = int(os.environ.get("N_STATES", "8"))
K = int(os.environ.get("K", "64"))
TEMP = float(os.environ.get("TEMP", "0.7"))
SEED = int(os.environ.get("SEED", "0"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/shuffle_transcript2")


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)
    catset = set(CATWORDS["city"])

    def body_of(hist, used):
        s = G.OPEN_PROMPT + " " + " ".join(
            f"Round {k+1}: the other player said {o}, you said {s_}."
            for k, (o, s_) in enumerate(hist))
        return s + " Words already used (do not repeat): " + ", ".join(sorted(used)) + "."

    @torch.no_grad()
    def propose_k(body):
        ids = tok(LA._render(tok, body) + "\nMy word:", return_tensors="pt").input_ids.to(dev)
        out = model.generate(ids.repeat(K, 1), max_new_tokens=4, do_sample=True,
                             temperature=TEMP, top_p=0.95, pad_token_id=tok.eos_token_id)
        return [G.clean_word(tok.decode(out[i, ids.shape[1]:], skip_special_tokens=True))
                for i in range(K)]

    @torch.no_grad()
    def greedy(body):
        ids = tok(LA._render(tok, body) + "\nMy word:", return_tensors="pt").input_ids.to(dev)
        out = model.generate(ids, max_new_tokens=4, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        return G.clean_word(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))

    games = collections.defaultdict(list)
    for line in open(os.path.join(SRC_DIR, "game1_strict_city_transcript.jsonl")):
        d = json.loads(line)
        games[d["rollout"]].append(d)
    starts = []
    for line in open(START_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split("\t") if "\t" in line else line.split()
            starts.append((p[-2], p[-1]))

    # v2 selection: K ends exactly at an ACTIVE run — the last 3 A words share a
    # 4-prefix (the capture state the loop lives in), K >= 6 rounds of context
    states = []
    for roll, ts in sorted(games.items(), key=lambda kv: -len(kv[1])):
        ts.sort(key=lambda r: r["turn"])
        Aw = [t["A"] for t in ts]
        hit = None
        for j in range(5, len(Aw)):
            trip = [w[:4] for w in Aw[j - 2:j + 1] if len(w) > 3]
            if len(trip) == 3 and len(set(trip)) == 1:
                hit = (j, trip[0])
        if hit is None:
            continue
        j, famp = hit
        states.append((roll, ts[:j + 1], famp))
        if len(states) >= N_STATES:
            break

    rng = np.random.default_rng(SEED)
    results = []
    for roll, ts, famp in states:
        sa, sb = starts[roll]
        Bcol = [t["B"] for t in ts]
        Acol = [t["A"] for t in ts]
        used = set([sa, sb] + Bcol + Acol)
        g0 = None
        perms = [list(range(len(Acol)))]
        for _ in range(3):
            p = list(range(len(Acol)))
            rng.shuffle(p)
            perms.append(p)
        for pi, perm in enumerate(perms):
            cond = "orig" if pi == 0 else f"shuf{pi}"
            Ap = [Acol[j] for j in perm]
            hist = [(sb, sa)] + list(zip(Bcol, Ap))
            body = body_of(hist, used)
            props = propose_k(body)
            fam = np.mean([1 if (w and w not in used and len(w) > 3 and w[:4] == famp)
                           else 0 for w in props])
            gw = greedy(body)
            if pi == 0:
                g0 = gw
            tail_fam = sum(1 for w in Ap[-2:] if len(w) > 3 and w[:4] == famp)
            results.append({"roll": roll, "fam": famp, "cond": cond,
                            "fam_mass": float(fam), "greedy": gw,
                            "agree": bool(gw == g0),
                            "greedy_is_fam": bool(gw and len(gw) > 3 and gw[:4] == famp),
                            "tail_fam": int(tail_fam)})
            json.dump({"per_state": results}, open(os.path.join(RUN_DIR, "shuffle.json"), "w"))
        print(f"[shf] roll {roll} fam {famp} done", flush=True)

    out = {"per_state": results, "cells": {}}
    for cond in ("orig", "shuf"):
        sel = [r for r in results if r["cond"].startswith(cond)]
        out["cells"][cond] = {
            "fam_mass": float(np.mean([r["fam_mass"] for r in sel])),
            "greedy_is_fam": float(np.mean([r["greedy_is_fam"] for r in sel])),
            "agree_with_orig": float(np.mean([r["agree"] for r in sel])),
            "tail_fam_mean": float(np.mean([r["tail_fam"] for r in sel]))}
    # split shuffles by whether family words remain in the last 2 rounds
    for lab, pred in (("shuf_tail2", lambda r: r["tail_fam"] == 2),
                      ("shuf_tail1", lambda r: r["tail_fam"] == 1),
                      ("shuf_tail0", lambda r: r["tail_fam"] == 0)):
        sel = [r for r in results if r["cond"].startswith("shuf") and pred(r)]
        if sel:
            out["cells"][lab] = {"n": len(sel),
                "fam_mass": float(np.mean([r["fam_mass"] for r in sel])),
                "greedy_is_fam": float(np.mean([r["greedy_is_fam"] for r in sel]))}
    for k, v in out["cells"].items():
        print(f"[shf] === {k}: {v}", flush=True)
    json.dump(out, open(os.path.join(RUN_DIR, "shuffle.json"), "w"), indent=1)
    print("[shf] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
