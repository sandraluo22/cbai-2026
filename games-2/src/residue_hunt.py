"""RESIDUE HUNT (2026-08-07): what carries the ~half of bare-list capture that
survives ablating the 32 reader heads? If it's "genuine generic priming", it
should live in a broader head pool or in non-attention paths.

All cells measured on d8_list states (8 family words in an appended list,
neutral history), K=64 MC family mass:

  none        baseline (known ~0.81)
  top32_rest  the 32 reader heads (known residue ~0.47)
  union64     top32_rest + top32 induction heads together (are induction heads
              the residue?)
  top64_rest  restoration ranking extended to 64 heads (is the carrier just the
              next tier of the same ranking?)
  top96_rest  ... to 96 heads (diffuse many-head carrier?)
  vknock      VALUE KNOCKOUT: zero v_proj output at every family-word token
              position, every layer — no attention edge anywhere can read the
              family words' content. If capture survives THIS, it is not
              attention-read at all; if it dies, the residue is other heads
              reading the same positions (diffuse priming-by-many-heads).

Env: MODEL(QwenInst32) PATCH_JSON IND_JSON SRC_DIR START_FILE N_STREAMS(6) K(64)
     TEMP(0.7) SEED(0) RUN_DIR(runs/residue_hunt)
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
SRC_DIR = os.environ.get("SRC_DIR", "runs/qwen32_strict")
START_FILE = os.environ.get("START_FILE", "runs/game-1/qwen32/qwen32_pca_w2v/start_words.txt")
N_STREAMS = int(os.environ.get("N_STREAMS", "6"))
K = int(os.environ.get("K", "64"))
TEMP = float(os.environ.get("TEMP", "0.7"))
SEED = int(os.environ.get("SEED", "0"))
RUN_DIR = os.environ.get("RUN_DIR", "runs/residue_hunt")

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
    nL = R.shape[0]
    order = np.argsort(R.flatten())[::-1]
    def topk(k):
        return [(int(i // nH), int(i % nH)) for i in order[:k]]
    I = np.array(json.load(open(IND_JSON))["induction"])
    iorder = np.argsort(I.flatten())[::-1]
    top_ind = [(int(i // I.shape[1]), int(i % I.shape[1])) for i in iorder[:32]]

    def to_ld(heads):
        d = {}
        for l, h in heads:
            d.setdefault(l, []).append(h)
        return d

    state = {"heads": None, "vknock": None}
    def make_pre(layer):
        def pre(_m, args):
            if not state["heads"] or layer not in state["heads"]:
                return None
            x = args[0].clone()
            for h in state["heads"][layer]:
                x[..., h * hd:(h + 1) * hd] = 0
            return (x,) + tuple(args[1:])
        return pre
    def make_vhook():
        def h(_m, _i, out):
            if state["vknock"] is None:
                return None
            if out.shape[1] <= 1:
                return None
            out = out.clone()
            out[:, state["vknock"], :] = 0
            return out
        return h
    for li, blk in enumerate(model.model.layers):
        blk.self_attn.o_proj.register_forward_pre_hook(make_pre(li))
        blk.self_attn.v_proj.register_forward_hook(make_vhook())

    @torch.no_grad()
    def propose_k(body, fam_positions=None):
        prompt = LA._render(tok, body) + "\nMy word:"
        enc = tok(prompt, return_offsets_mapping=True)
        if fam_positions is not None:
            idx = []
            for w in PLANT8:
                start = 0
                while True:
                    j = prompt.find(w, start)
                    if j < 0:
                        break
                    for ti, (a, b) in enumerate(enc["offset_mapping"]):
                        if a < j + len(w) and b > j:
                            idx.append(ti)
                    start = j + len(w)
            state["vknock"] = sorted(set(idx))
        ids = torch.tensor([enc["input_ids"]], device=dev)
        out = model.generate(ids.repeat(K, 1), max_new_tokens=4, do_sample=True,
                             temperature=TEMP, top_p=0.95, pad_token_id=tok.eos_token_id)
        state["vknock"] = None
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

    CELLS = (("none", None, False), ("top32_rest", to_ld(topk(32)), False),
             ("union64", to_ld(list(set(topk(32)) | set(top_ind))), False),
             ("top64_rest", to_ld(topk(64)), False),
             ("top96_rest", to_ld(topk(96)), False),
             ("vknock", None, True))
    results = []
    for cond, heads, vk in CELLS:
        state["heads"] = heads
        for si, ts in enumerate(streams):
            roll = ts[0]["rollout"]
            Bseq = [t["B"] for t in ts]
            if len(Bseq) < 8:
                Bseq = Bseq + Bseq
            sa, sb = starts[roll]
            hist = [(sb, sa)]
            used = {sa, sb}
            for i in range(8):
                hist.append((Bseq[i], FILLER[i]))
                used |= {FILLER[i], Bseq[i]}
            extra = " Unrelated word list: " + ", ".join(PLANT8) + "."
            used |= set(PLANT8)
            props = propose_k(body_of(hist, used, extra), fam_positions=vk or None)
            fam = np.mean([1 if (w and w not in used and len(w) > 3 and w[:4] == FAMP)
                           else 0 for w in props])
            cat = np.mean([1 if (w and w not in used and w in catset) else 0 for w in props])
            results.append({"cond": cond, "stream": roll, "fam_mass": float(fam),
                            "cat_mass": float(cat)})
            json.dump({"per_state": results}, open(os.path.join(RUN_DIR, "residue.json"), "w"))
        sel = [r for r in results if r["cond"] == cond]
        print(f"[res] === {cond}: fam {np.mean([r['fam_mass'] for r in sel]):.3f} "
              f"cat {np.mean([r['cat_mass'] for r in sel]):.3f}", flush=True)
    state["heads"] = None
    out = {"per_state": results, "cells": {}}
    for cond, _, _ in CELLS:
        sel = [r for r in results if r["cond"] == cond]
        out["cells"][cond] = {"fam_mass": float(np.mean([r["fam_mass"] for r in sel])),
                              "cat_mass": float(np.mean([r["cat_mass"] for r in sel]))}
    json.dump(out, open(os.path.join(RUN_DIR, "residue.json"), "w"), indent=1)
    print("[res] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
