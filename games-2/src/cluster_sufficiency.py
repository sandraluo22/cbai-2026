"""CLUSTER SUFFICIENCY (2026-08-07): are the 32 reader heads SUFFICIENT to
switch the attractor on in a neutral context (and off in a captured one)?

Interchange patch at the head level, final answer position, persisted across
decode steps:

  install   run a d8_list DONOR state, cache the 32 heads' o_proj input slices
            at the final position; run the matched NEUTRAL control with those 32
            slices transplanted in. Does family mass appear from nothing?
  install_d3 same but donor = d3_self state (weaker capture source).
  install_rand transplant 32 RANDOM heads' slices from the same donor (control).
  remove    reverse: run the neutral control, cache its 32 slices; run the
            d8_list donor with neutral slices written in (head-level removal —
            should reproduce ~the ablation number if consistent).

Measures: K=32 MC family/category mass + greedy word, vs unpatched baselines.

Env: MODEL(QwenInst32) PATCH_JSON SRC_DIR START_FILE N_STREAMS(6) K(32) TEMP(0.7)
     RUN_DIR(runs/cluster_sufficiency)
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
SRC_DIR = os.environ.get("SRC_DIR", "runs/qwen32_strict")
START_FILE = os.environ.get("START_FILE", "runs/game-1/qwen32/qwen32_pca_w2v/start_words.txt")
N_STREAMS = int(os.environ.get("N_STREAMS", "6"))
K = int(os.environ.get("K", "32"))
TEMP = float(os.environ.get("TEMP", "0.7"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/cluster_sufficiency")

PLANT8 = ["planted", "planting", "plantings", "replant",
          "replanted", "planter", "planters", "plantation"]
FILLER = ["window", "carpet", "stapler", "napkin", "candle", "basket", "ribbon", "saddle"]
FAMP = "plan"


def main():
    import torch
    os.makedirs(RUN_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = LA.load(MODEL, dev)
    nH = model.config.num_attention_heads
    hd = model.model.layers[0].self_attn.o_proj.in_features // nH
    catset = set(CATWORDS["city"])

    R = np.array(json.load(open(PATCH_JSON))["restoration"])
    order = np.argsort(R.flatten())[::-1]
    top32 = [(int(i // nH), int(i % nH)) for i in order[:32]]
    rng = np.random.default_rng(0)
    pool = [(l, h) for l in range(R.shape[0]) for h in range(nH) if (l, h) not in set(top32)]
    rand32 = [pool[i] for i in rng.choice(len(pool), 32, replace=False)]

    def to_ld(heads):
        d = {}
        for l, h in heads:
            d.setdefault(l, []).append(h)
        return d

    mode = {"m": "off", "heads": None, "cache": None}
    def make_pre(layer):
        def pre(_m, args):
            if mode["m"] == "off" or not mode["heads"] or layer not in mode["heads"]:
                return None
            x = args[0]
            if mode["m"] == "cap":
                for h in mode["heads"][layer]:
                    mode["cache"][(layer, h)] = x[0, -1, h * hd:(h + 1) * hd].detach().clone()
                return None
            x = x.clone()
            for h in mode["heads"][layer]:
                x[:, -1, h * hd:(h + 1) * hd] = mode["cache"][(layer, h)]
            return (x,) + tuple(args[1:])
        return pre
    for li, blk in enumerate(model.model.layers):
        blk.self_attn.o_proj.register_forward_pre_hook(make_pre(li))

    def body_of(hist, used, extra=""):
        s = G.OPEN_PROMPT + extra + " " + " ".join(
            f"Round {k+1}: the other player said {o}, you said {s_}."
            for k, (o, s_) in enumerate(hist))
        return s + " Words already used (do not repeat): " + ", ".join(sorted(used)) + "."

    def build(Bseq, sa, sb, cell):
        hist = [(sb, sa)]
        used = {sa, sb}
        extra = ""
        n = 3 if cell == "d3_self" else 8
        for i in range(n):
            a = PLANT8[i] if cell == "d3_self" else FILLER[i]
            hist.append((Bseq[i], a))
            used |= {a, Bseq[i]}
        if cell == "d8_list":
            extra = " Unrelated word list: " + ", ".join(PLANT8) + "."
            used |= set(PLANT8)
        return body_of(hist, used, extra), used

    @torch.no_grad()
    def run_capture(body, heads):
        mode.update({"m": "cap", "heads": heads, "cache": {}})
        ids = tok(LA._render(tok, body) + "\nMy word:", return_tensors="pt").input_ids.to(dev)
        model(ids)
        mode["m"] = "off"
        return mode["cache"]

    @torch.no_grad()
    def sample(body, heads=None, cache=None):
        if heads is not None:
            mode.update({"m": "patch", "heads": heads, "cache": cache})
        ids = tok(LA._render(tok, body) + "\nMy word:", return_tensors="pt").input_ids.to(dev)
        out = model.generate(ids.repeat(K, 1), max_new_tokens=4, do_sample=True,
                             temperature=TEMP, top_p=0.95, pad_token_id=tok.eos_token_id)
        mode["m"] = "off"
        return [G.clean_word(tok.decode(out[i, ids.shape[1]:], skip_special_tokens=True))
                for i in range(K)]

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

    ld32, ldr = to_ld(top32), to_ld(rand32)
    results = []
    for si, ts in enumerate(streams):
        roll = ts[0]["rollout"]
        Bseq = [t["B"] for t in ts]
        if len(Bseq) < 8:
            Bseq = Bseq + Bseq
        sa, sb = starts[roll]
        b_ctrl, u_ctrl = build(Bseq, sa, sb, "ctrl")
        b_list, u_list = build(Bseq, sa, sb, "d8_list")
        b_d3, u_d3 = build(Bseq, sa, sb, "d3_self")

        def famcat(ws, used):
            fam = float(np.mean([1 if (w and w not in used and len(w) > 3 and w[:4] == FAMP)
                                 else 0 for w in ws]))
            cat = float(np.mean([1 if (w and w not in used and w in catset) else 0 for w in ws]))
            return fam, cat

        cells = {}
        cells["ctrl_base"] = famcat(sample(b_ctrl), u_ctrl | set(PLANT8))
        cells["donor_base"] = famcat(sample(b_list), u_list)
        cache_list = run_capture(b_list, ld32)
        cells["install"] = famcat(sample(b_ctrl, ld32, cache_list), u_ctrl | set(PLANT8))
        cache_d3 = run_capture(b_d3, ld32)
        cells["install_d3"] = famcat(sample(b_ctrl, ld32, cache_d3), u_ctrl | set(PLANT8))
        cache_rand = run_capture(b_list, ldr)
        cells["install_rand"] = famcat(sample(b_ctrl, ldr, cache_rand), u_ctrl | set(PLANT8))
        cache_ctrl = run_capture(b_ctrl, ld32)
        cells["remove"] = famcat(sample(b_list, ld32, cache_ctrl), u_list)
        results.append({"stream": roll, "cells": {k: {"fam": v[0], "cat": v[1]}
                                                  for k, v in cells.items()}})
        json.dump({"per_state": results}, open(os.path.join(RUN_DIR, "sufficiency.json"), "w"))
        print(f"[suf] s{roll}: " + " ".join(f"{k} {v[0]:.2f}" for k, v in cells.items()),
              flush=True)

    out = {"per_state": results, "cells": {}}
    for k in results[0]["cells"]:
        out["cells"][k] = {
            "fam": float(np.mean([r["cells"][k]["fam"] for r in results])),
            "cat": float(np.mean([r["cells"][k]["cat"] for r in results]))}
        print(f"[suf] === {k}: fam {out['cells'][k]['fam']:.3f} cat {out['cells'][k]['cat']:.3f}",
              flush=True)
    json.dump(out, open(os.path.join(RUN_DIR, "sufficiency.json"), "w"), indent=1)
    print("[suf] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
