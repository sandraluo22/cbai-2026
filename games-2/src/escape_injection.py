"""ESCAPE INJECTION (2026-08-05): make the metastability causal.

Survival analysis showed the escape hazard collapsing after ~4 family-turns;
mech4 showed the passive trace of planted words dies by gap 2. If the loop is
self-re-fed (behavior each turn renews the trace), then ONE forced off-family
action should reset shallow runs but not deep ones only insofar as depth changes
the context mass — this measures exactly that.

Design: synthetic self-family runs of depth d in {1,2,4,8} (8-word plant family)
in A's own slots against replayed strict-city B streams. At the branch point:
  inject   A's next action is FORCED to one off-family legal word ("lantern")
  live     A generates its own next word (matched control)
then 6 live A turns (B replayed). Measures: MC proposal profile (K=64) at the
state immediately after the branch action and again after 2 live turns
(family / used / category mass), plus fam_hits6 and escaped (no family word in
the 6 live turns).

Env: MODEL(QwenInst32) SRC_DIR(runs/qwen32_strict) START_FILE N_STREAMS(6) K(64)
     TEMP(0.7) RUN_DIR(runs/escape_injection)
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
RUN_DIR = os.environ.get("RUN_DIR", "runs/escape_injection")

PLANT8 = ["planted", "planting", "plantings", "replant",
          "replanted", "planter", "planters", "plantation"]
FAMP = "plan"
FILLER = ["window", "carpet", "stapler"]
INJECT_WORD = "lantern"
DEPTHS = [1, 2, 4, 8]


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

    def profile(props, used):
        fam = np.mean([1 if (w and w not in used and len(w) > 3 and w[:4] == FAMP) else 0
                       for w in props])
        um = np.mean([1 if (w and w in used) else 0 for w in props])
        cm = np.mean([1 if (w and w not in used and w in catset) else 0 for w in props])
        return float(fam), float(um), float(cm)

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
        Bseq = [t["B"] for t in ts]
        sa, sb = starts[roll]
        if len(Bseq) < max(DEPTHS) + 8:
            Bseq = Bseq + [Bseq[-1 - (i % len(Bseq))] for i in range(max(DEPTHS) + 8)]
        for depth in DEPTHS:
            for cond in ("inject", "live"):
                hist = [(sb, sa)]
                used = {sa, sb}
                fill = iter(FILLER)
                n_rounds = max(depth, 3)
                for i in range(n_rounds):
                    a = PLANT8[i] if i < depth else next(fill)
                    hist.append((Bseq[i], a))
                    used |= {a, Bseq[i]}
                # branch action
                if cond == "inject":
                    w = INJECT_WORD
                else:
                    w = gen_word(body_of(hist, used), 41000 + 977 * si + depth, used)
                branch_word = w
                hist.append((Bseq[n_rounds], w))
                used |= {w, Bseq[n_rounds]}
                pr0 = profile(propose_k(body_of(hist, used)), used)
                # 6 live turns
                live, pr2 = [], None
                for ct in range(6):
                    w = gen_word(body_of(hist, used), 43000 + 977 * si + 13 * depth + ct, used)
                    live.append(w)
                    bidx = n_rounds + 1 + ct
                    hist.append((Bseq[bidx], w))
                    used |= {w, Bseq[bidx]}
                    if ct == 1:
                        pr2 = profile(propose_k(body_of(hist, used)), used)
                fam_hits = sum(1 for w in live if w and len(w) > 3 and w[:4] == FAMP)
                results.append({"depth": depth, "cond": cond, "stream": roll,
                                "branch_word": branch_word,
                                "branch_is_fam": bool(branch_word and len(branch_word) > 3
                                                      and branch_word[:4] == FAMP),
                                "fam0": pr0[0], "used0": pr0[1], "cat0": pr0[2],
                                "fam2": pr2[0], "used2": pr2[1], "cat2": pr2[2],
                                "live_words": live, "fam_hits6": int(fam_hits),
                                "escaped": bool(fam_hits == 0)})
                json.dump({"per_branch": results},
                          open(os.path.join(RUN_DIR, "escape.json"), "w"))
        print(f"[esc] stream {roll} done", flush=True)

    out = {"per_branch": results, "cells": {}}
    for depth in DEPTHS:
        for cond in ("inject", "live"):
            sel = [r for r in results if r["depth"] == depth and r["cond"] == cond]
            out["cells"][f"d{depth}_{cond}"] = {
                "fam0": float(np.mean([r["fam0"] for r in sel])),
                "fam2": float(np.mean([r["fam2"] for r in sel])),
                "fam_hits6": float(np.mean([r["fam_hits6"] for r in sel])),
                "escaped": float(np.mean([r["escaped"] for r in sel])),
                "branch_fam_frac": float(np.mean([r["branch_is_fam"] for r in sel]))}
            c = out["cells"][f"d{depth}_{cond}"]
            print(f"[esc] === d{depth} {cond}: fam0 {c['fam0']:.3f} fam2 {c['fam2']:.3f} "
                  f"hits6 {c['fam_hits6']:.1f} escaped {c['escaped']:.2f}", flush=True)
    json.dump(out, open(os.path.join(RUN_DIR, "escape.json"), "w"), indent=1)
    print("[esc] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
