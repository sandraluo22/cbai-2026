"""CONFLICT PREFILL (2026-08-05): two attractors in one transcript — which wins?

At round K=4 we construct the game history so A's own column and B's (restricted)
column each hold a different prefilled trajectory, then continue the game LIVE
(both players generate; A free prompt, B city-restricted prompt; both see the same
constructed history from their own perspective). Families: F1 = plant
(planted/planting/plantings/replant), F2 = paint (painted/painting/paintings/
repaint). B's non-stuck column = its real replayed city words; A's non-stuck
column = neutral fillers.

Configs (self column / partner column, from A's perspective):
  selfF1_partF2    both stuck, different families
  selfF2_partF1    swap control
  selfF1_partCity  only self stuck (classic transplant)
  selfFill_partF2  only partner stuck (partner went off-restriction into a family)
  selfFill_partCity both non-stuck (baseline)

Measures: MC K=64 profile of A at the branch (mass in F1 / F2 / city / other);
8 live turns for both players — A's words classified F1/F2/city/other, B's words
city-compliant vs F-captured; meets.

Env: MODEL(QwenInst32) SRC_DIR(runs/qwen32_strict) START_FILE N_STREAMS(6) K(64)
     TEMP(0.7) LIVE(8) RUN_DIR(runs/conflict_prefill)
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
N_STREAMS = int(os.environ.get("N_STREAMS", "6"))
K = int(os.environ.get("K", "64"))
TEMP = float(os.environ.get("TEMP", "0.7"))
LIVE = int(os.environ.get("LIVE", "8"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/conflict_prefill")

F1 = ["planted", "planting", "plantings", "replant"]
F2 = ["painted", "painting", "paintings", "repaint"]
FILL = ["window", "carpet", "stapler", "napkin"]
RESTR = (" IMPORTANT: every single word you say must be the name of a city. Only ever "
         "say cities, nothing else.")
CONFIGS = {
    "selfF1_partF2": (F1, F2),
    "selfF2_partF1": (F2, F1),
    "selfF1_partCity": (F1, "city"),
    "selfFill_partF2": (FILL, F2),
    "selfFill_partCity": (FILL, "city"),
}


def wclass(w, used=None):
    if not w:
        return "empty"
    if len(w) > 3 and w[:4] == "plan":
        return "F1"
    if len(w) > 3 and w[:4] == "pain":
        return "F2"
    if w in set(CATWORDS["city"]):
        return "city"
    return "other"


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)
    catset = set(CATWORDS["city"])

    def body_of(hist, used, extra=""):
        s = G.OPEN_PROMPT + extra + " " + " ".join(
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
    def gen_word(body, seed, forbidden):
        prompt = LA._render(tok, body) + "\nMy word:"
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

    games = collections.defaultdict(list)
    for line in open(os.path.join(SRC_DIR, "game1_strict_city_transcript.jsonl")):
        d = json.loads(line)
        games[d["rollout"]].append(d)
    streams = [sorted(ts, key=lambda r: r["turn"]) for _, ts in
               sorted(games.items(), key=lambda kv: -len(kv[1]))[:N_STREAMS]]
    starts = []
    for line in open(START_FILE):
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split("\t") if "\t" in line else line.split()
            starts.append((p[-2], p[-1]))

    results = []
    for si, ts in enumerate(streams):
        roll = ts[0]["rollout"]
        Bcity = [t["B"] for t in ts]
        sa, sb = starts[roll]
        for cfg, (selfcol, partcol) in CONFIGS.items():
            histA = [(sb, sa)]
            used = {sa, sb}
            for i in range(4):
                a = selfcol[i]
                b = Bcity[i] if partcol == "city" else partcol[i]
                histA.append((b, a))
                used |= {a, b}
            histB = [(sa, sb)] + [(a, b) for b, a in histA[1:]]
            props = propose_k(body_of(histA, used))
            pm = collections.Counter(wclass(w) for w in props if w and w not in used)
            branch_profile = {c: pm.get(c, 0) / K for c in ("F1", "F2", "city", "other")}
            liveA, liveB = [], []
            agreed_at = None
            for t in range(LIVE):
                wA = gen_word(body_of(histA, used), 61000 + 977 * si + 31 * t, used)
                wB = gen_word(body_of(histB, used, RESTR), 96100 + 977 * si + 31 * t, used)
                liveA.append(wA); liveB.append(wB)
                if wA == wB and wA:
                    agreed_at = t + 1
                    break
                used |= {wA, wB}
                histA.append((wB, wA)); histB.append((wA, wB))
            ca = collections.Counter(wclass(w) for w in liveA)
            cb = collections.Counter(wclass(w) for w in liveB)
            results.append({"cfg": cfg, "stream": roll, "branch": branch_profile,
                            "liveA": liveA, "liveB": liveB, "met_at": agreed_at,
                            "A_F1": ca.get("F1", 0), "A_F2": ca.get("F2", 0),
                            "A_city": ca.get("city", 0), "A_other": ca.get("other", 0),
                            "B_city": cb.get("city", 0),
                            "B_F1": cb.get("F1", 0), "B_F2": cb.get("F2", 0)})
            json.dump({"per_branch": results},
                      open(os.path.join(RUN_DIR, "conflict.json"), "w"))
        print(f"[cfp] stream {roll} done", flush=True)

    out = {"per_branch": results, "cells": {}}
    for cfg in CONFIGS:
        sel = [r for r in results if r["cfg"] == cfg]
        nA = [max(sum(r[k] for k in ("A_F1", "A_F2", "A_city", "A_other")), 1) for r in sel]
        out["cells"][cfg] = {
            "branch_F1": float(np.mean([r["branch"]["F1"] for r in sel])),
            "branch_F2": float(np.mean([r["branch"]["F2"] for r in sel])),
            "branch_city": float(np.mean([r["branch"]["city"] for r in sel])),
            "liveA_F1": float(np.mean([r["A_F1"] / n for r, n in zip(sel, nA)])),
            "liveA_F2": float(np.mean([r["A_F2"] / n for r, n in zip(sel, nA)])),
            "liveA_city": float(np.mean([r["A_city"] / n for r, n in zip(sel, nA)])),
            "B_capture": float(np.mean([(r["B_F1"] + r["B_F2"]) > 0 for r in sel])),
            "met_frac": float(np.mean([r["met_at"] is not None for r in sel]))}
        c = out["cells"][cfg]
        print(f"[cfp] === {cfg}: branch F1 {c['branch_F1']:.2f} F2 {c['branch_F2']:.2f} "
              f"city {c['branch_city']:.2f} | live F1 {c['liveA_F1']:.2f} "
              f"F2 {c['liveA_F2']:.2f} | Bcap {c['B_capture']:.2f} met {c['met_frac']:.2f}",
              flush=True)
    json.dump(out, open(os.path.join(RUN_DIR, "conflict.json"), "w"), indent=1)
    print("[cfp] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
