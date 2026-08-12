"""REGIME DISSECTION VIA ABLATION (2026-08-07): are early (role-gated, dose-3)
and late (lexical, dose-8) capture carried by different circuitry?

Same 4 head-ablation conditions, measured on FOUR state types:
  d3_self   3 family words in A's own column (early regime)
  d8_self   8 family words in A's own column (late regime, in-slot)
  d8_list   8 family words as a bare appended list (late regime, pure lexical)
  d0        neutral control
Two-mechanism prediction: top32_rest kills d3_self (known) but SPARES d8_list;
one-mechanism: same ablation profile everywhere.

The 2026-07 ablation (qwen32_head_ablate_play.py) measured met_frac in easy
self-play (ceiling 1.00 in every condition) — it could not detect an effect on the
proposal collapse itself. This rerun ablates the same head sets but measures the
MC proposal profile on dose-seeded states: family (target) mass, used/invalid
mass, category mass.

Conditions:
  none        no ablation
  top32_rest  top-32 heads by partner-patch restoration (PATCH_JSON)
  top32_ind   top-32 heads by induction score (IND_JSON, repeated-random-seq)
  rand32      32 random heads excluding both sets (fixed seed)

States: synthetic dose_0 / dose_3 / dose_4 morphological-family seeds in A's own
slots against replayed strict-city B streams (construction = seed_matrix.py).
If family mass survives top32_ind ablation, the induction/copy account of the
collapse is dead at the head level; if top32_ind kills family mass but not
category/used mass, the induction account wins.

Env: MODEL(QwenInst32) PATCH_JSON IND_JSON K(64) TEMP(0.7) N_STREAMS(6)
     SRC_DIR(runs/qwen32_strict) START_FILE SEED(0) RUN_DIR(runs/ablate_collapse)
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
PATCH_JSON = os.environ.get("PATCH_JSON", "runs/mech_inputs/qwen32_partner_patch.json")
IND_JSON = os.environ.get("IND_JSON", "runs/mech_inputs/qwen32_induction_overlap.json")
K = int(os.environ.get("K", "64"))
TEMP = float(os.environ.get("TEMP", "0.7"))
N_STREAMS = int(os.environ.get("N_STREAMS", "6"))
SRC_DIR = os.environ.get("SRC_DIR", "runs/qwen32_strict")
START_FILE = os.environ.get("START_FILE", "runs/game-1/qwen32/qwen32_pca_w2v/start_words.txt")
SEED = int(os.environ.get("SEED", "0"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/regime_dissect")

PLANT8 = ["planted", "planting", "plantings", "replant",
          "replanted", "planter", "planters", "plantation"]
MORPH = PLANT8[:4]
FILLER = ["window", "carpet", "stapler", "napkin", "candle", "basket", "ribbon", "saddle"]


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)
    nH = model.config.num_attention_heads
    hd = model.model.layers[0].self_attn.o_proj.in_features // nH
    catset = set(CATWORDS["city"])

    R = np.array(json.load(open(PATCH_JSON))["restoration"])
    nL = R.shape[0]
    order = np.argsort(R.flatten())[::-1]
    top_rest = [(int(i // nH), int(i % nH)) for i in order[:32]]
    I = np.array(json.load(open(IND_JSON))["induction"])
    iorder = np.argsort(I.flatten())[::-1]
    top_ind = [(int(i // I.shape[1]), int(i % I.shape[1])) for i in iorder[:32]]
    rng = np.random.default_rng(SEED)
    excl = set(top_rest) | set(top_ind)
    pool = [(l, h) for l in range(nL) for h in range(nH) if (l, h) not in excl]
    rand = [pool[i] for i in rng.choice(len(pool), 32, replace=False)]

    state = {"heads": None}
    def make_pre(layer):
        def pre(_m, args):
            if not state["heads"] or layer not in state["heads"]:
                return None
            x = args[0].clone()
            for h in state["heads"][layer]:
                x[..., h * hd:(h + 1) * hd] = 0
            return (x,) + tuple(args[1:])
        return pre
    for li, blk in enumerate(model.model.layers):
        blk.self_attn.o_proj.register_forward_pre_hook(make_pre(li))

    def to_ld(heads):
        d = {}
        for l, h in heads:
            d.setdefault(l, []).append(h)
        return d

    @torch.no_grad()
    def propose_k(body):
        ids = tok(LA._render(tok, body) + "\nMy word:", return_tensors="pt").input_ids.to(dev)
        out = model.generate(ids.repeat(K, 1), max_new_tokens=4, do_sample=True,
                             temperature=TEMP, top_p=0.95, pad_token_id=tok.eos_token_id)
        return [G.clean_word(tok.decode(out[i, ids.shape[1]:], skip_special_tokens=True))
                for i in range(K)]

    def body_of(hist, used, extra=""):
        s = G.OPEN_PROMPT + extra + " " + " ".join(
            f"Round {k+1}: the other player said {o}, you said {s_}."
            for k, (o, s_) in enumerate(hist))
        return s + " Words already used (do not repeat): " + ", ".join(sorted(used)) + "."

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

    def build(Bseq, sa, sb, cell):
        hist = [(sb, sa)]
        used = {sa, sb}
        extra = ""
        if cell == "d0":
            rounds = [(Bseq[i], FILLER[i]) for i in range(3)]
        elif cell == "d3_self":
            rounds = [(Bseq[i], PLANT8[i]) for i in range(3)]
        elif cell == "d8_self":
            rounds = [(Bseq[i], PLANT8[i]) for i in range(8)]
        else:                                    # d8_list
            rounds = [(Bseq[i], FILLER[i]) for i in range(8)]
            extra = " Unrelated word list: " + ", ".join(PLANT8) + "."
            used |= set(PLANT8)
        for b, a in rounds:
            hist.append((b, a))
            used |= {a, b}
        return hist, used, extra

    results = []
    conds = (("none", None), ("top32_rest", to_ld(top_rest)),
             ("top32_ind", to_ld(top_ind)), ("rand32", to_ld(rand)))
    for cond, heads in conds:
        state["heads"] = heads
        for si, ts in enumerate(streams):
            roll = ts[0]["rollout"]
            Bseq = [t["B"] for t in ts]
            sa, sb = starts[roll]
            for cell in ("d0", "d3_self", "d8_self", "d8_list"):
                if len(Bseq) < 8:
                    Bseq = Bseq + Bseq
                hist, used, extra = build(Bseq, sa, sb, cell)
                props = propose_k(body_of(hist, used, extra))
                fam = np.mean([1 if (w and w not in used and len(w) > 3
                               and w[:4] == "plan") else 0 for w in props])
                um = np.mean([1 if (w and w in used) else 0 for w in props])
                cm = np.mean([1 if (w and w not in used and w in catset) else 0 for w in props])
                inv = np.mean([1 if (not w or w in used) else 0 for w in props])
                results.append({"cond": cond, "cell": cell, "stream": roll,
                                "fam_mass": float(fam), "used_mass": float(um),
                                "cat_mass": float(cm), "invalid_mass": float(inv)})
                json.dump({"per_state": results}, open(os.path.join(RUN_DIR, "regime_dissect.json"), "w"))
        print(f"[abc] {cond} done", flush=True)

    out = {"model": MODEL, "top_rest": top_rest, "top_ind": top_ind,
           "n_overlap_rest_ind": len(set(top_rest) & set(top_ind)),
           "per_state": results, "cells": {}}
    for cond, _ in conds:
        for cell in ("d0", "d3_self", "d8_self", "d8_list"):
            sel = [r for r in results if r["cond"] == cond and r["cell"] == cell]
            out["cells"][f"{cond}_{cell}"] = {k: float(np.mean([r[k] for r in sel]))
                for k in ("fam_mass", "used_mass", "cat_mass", "invalid_mass")}
            c = out["cells"][f"{cond}_{cell}"]
            print(f"[abc] === {cond} {cell}: fam {c['fam_mass']:.3f} used {c['used_mass']:.3f} "
                  f"cat {c['cat_mass']:.3f} inv {c['invalid_mass']:.3f}", flush=True)
    json.dump(out, open(os.path.join(RUN_DIR, "regime_dissect.json"), "w"), indent=1)
    print("[abc] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
