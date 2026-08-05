"""Where does the answer position actually LOOK? Attention mass by token group.

Loads Qwen3-32B with eager attention, builds the matched duel/single/tally contexts
from mech_reliability, and measures attention from the final (answer) position to:
  cur_names    speaker-identity tokens in the current round's memory line
  reveal_rows  'The correct answer for round r was "x"' lines (the record)
  cur_labels   the label strings in the current round's memory line
  past_mem     all earlier-round memory lines
Reports raw mass and ENRICHMENT (mass / token-share): >1 = disproportionate attention.
Broken out for all layers, and for the L52-57 head cluster path-patching implicated.

env: MODEL (Qwen32)  NEX (6)  MODES (duel100,duel80,single)
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
from mech_reliability import build, OUT  # noqa: E402

MODEL_ID = "Qwen/Qwen3-32B"
CLUSTER = list(range(52, 58))                      # heads implicated by path patching


def spans_for(ex, text):
    """Extra char spans beyond the ones build() provides: current-round label strings
    and all past-round memory lines."""
    sp = {k: list(v) for k, v in ex["spans"].items()}
    r = 11
    tag = f"Round {r} memories (current round)"
    ti = text.find(tag)
    cur_lab = []
    if ti >= 0:
        line_end = text.find("\n", ti)
        for lab in (ex["X"], ex["Y"]):
            j = ti
            while True:
                j = text.find(lab, j, line_end)
                if j < 0:
                    break
                cur_lab.append((j, j + len(lab)))
                j += 1
    sp["cur_labels"] = cur_lab
    past = []
    for r0 in range(1, 11):
        j = text.find(f"Round {r0} memories")
        if j >= 0:
            past.append((j, text.find("\n", j)))
    sp["past_mem"] = past
    return sp


def encode(tok, ex):
    text = ex["text"]
    sp = spans_for(ex, text)
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
    offs = enc.pop("offset_mapping")[0].tolist()
    idx = {}
    for g, ranges in sp.items():
        idx[g] = [i for i, (a, b) in enumerate(offs)
                  if any(a < c1 and b > c0 for (c0, c1) in ranges)]
    return enc, idx, len(offs)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager")
    model.eval()
    n_ex = int(os.environ.get("NEX", "6"))
    modes = os.environ.get("MODES", "duel100,duel80,single").split(",")
    groups = ["cur_names", "reveal_rows", "cur_labels", "past_mem"]
    out = {}
    for mode in modes:
        acc = {g: [] for g in groups}
        accC = {g: [] for g in groups}
        share = {g: [] for g in groups}
        for i in range(n_ex):
            rng = random.Random(9000 + i)
            ex = build(mode, 1 if i % 2 == 0 else 2, rng, tok)
            enc, idx, ntok = encode(tok, ex)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            with torch.no_grad():
                o = model(**enc, output_attentions=True)
            # attn[l]: (1, heads, seq, seq) -> take last query row
            last = {l: o.attentions[l][0, :, -1, :].float().cpu().numpy()
                    for l in range(len(o.attentions))}
            del o
            torch.cuda.empty_cache()
            for g in groups:
                pos = idx[g]
                if not pos:
                    continue
                share[g].append(len(pos) / ntok)
                allm = np.stack([last[l].mean(0)[pos].sum() for l in last])   # per-layer
                acc[g].append(allm)
                clm = np.mean([last[l].mean(0)[pos].sum() for l in CLUSTER])
                accC[g].append(clm)
            print(f"[attn] {mode} ex{i} ntok={ntok}", flush=True)
        out[mode] = {g: dict(mass_by_layer=np.mean(acc[g], 0).tolist() if acc[g] else [],
                             mass_cluster=float(np.mean(accC[g])) if accC[g] else 0.0,
                             token_share=float(np.mean(share[g])) if share[g] else 0.0)
                     for g in groups}
        print(f"\n=== {mode} — attention from ANSWER position ===", flush=True)
        for g in groups:
            d = out[mode][g]
            if not d["mass_by_layer"]:
                continue
            allm = float(np.mean(d["mass_by_layer"]))
            print(f"  {g:<12} tokens {d['token_share']*100:5.1f}%  |  all-layer mass "
                  f"{allm*100:5.2f}% (enrich {allm/max(1e-9,d['token_share']):.2f}x)  |  "
                  f"L52-57 mass {d['mass_cluster']*100:5.2f}% "
                  f"(enrich {d['mass_cluster']/max(1e-9,d['token_share']):.2f}x)", flush=True)
    json.dump(out, open(os.path.join(OUT, "attention.json"), "w"))
    print("MECH_ATTENTION_DONE", flush=True)


if __name__ == "__main__":
    main()
