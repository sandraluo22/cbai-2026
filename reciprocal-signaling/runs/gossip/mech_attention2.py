"""Attention SPLIT BY SPEAKER: does the answer position attend differently to the
reliable vs the unreliable source's name tokens and labels?

Groups (per context):
  name_rel / name_bad      every "P<rel>:" / "P<bad>:" mention across the whole memory
  curname_rel / curname_bad the same, restricted to the current round's line
  lab_rel / lab_bad        the two competing label strings in the current round
  rev_match / rev_error    reveal rows that agree with the reliable source vs the
                           rounds where the reveal contradicts it (duel80 only)
Reports enrichment (mass / token-share) and the rel:bad ratio; plus the top individual
heads by name-token attention, to check whether head-averaging hides specialists.

env: MODEL_ID, NEX (6), MODES, LOAD8 (1 = 8-bit, to share the GPU)
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
from mech_reliability import build, OUT, R_HIST  # noqa: E402

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-32B")
CLUSTER = list(range(52, 58))


def spans_for(ex, text):
    rel, bad = ex["rel"], (2 if ex["rel"] == 1 else 1)
    sp = {}
    for tag, who in (("name_rel", rel), ("name_bad", bad)):
        pat, hits, j = f"P{who}:", [], 0
        while True:
            j = text.find(pat, j)
            if j < 0:
                break
            hits.append((j, j + 2))                    # the "Pn" tokens only
            j += 1
        sp[tag] = hits
    cur = text.find(f"Round {R_HIST + 1} memories (current round)")
    end = text.find("\n", cur) if cur >= 0 else -1
    for tag, who in (("curname_rel", rel), ("curname_bad", bad)):
        j = text.find(f"P{who}:", cur, end) if cur >= 0 else -1
        sp[tag] = [(j, j + 2)] if j >= 0 else []
    for tag, lab in (("lab_rel", ex["X"]), ("lab_bad", ex["Y"])):
        hits, j = [], cur
        while cur >= 0:
            j = text.find(lab, j, end)
            if j < 0:
                break
            hits.append((j, j + len(lab)))
            j += 1
        sp[tag] = hits
    # reveal rows: which rounds' reveals agree with the reliable source
    err = {2, 6} if ex["mode"] == "duel80" else set()
    sp["rev_match"], sp["rev_error"] = [], []
    for r0 in range(1, R_HIST + 1):
        j = text.find(f'The correct answer for round {r0} was "')
        if j < 0:
            continue
        span = (j, text.find("\n", j))
        (sp["rev_error"] if r0 in err else sp["rev_match"]).append(span)
    return sp


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    kw = dict(dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager")
    if os.environ.get("LOAD8", "1") == "1":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kw)
    model.eval()
    n_ex = int(os.environ.get("NEX", "6"))
    modes = os.environ.get("MODES", "duel100,duel80").split(",")
    groups = ["name_rel", "name_bad", "curname_rel", "curname_bad",
              "lab_rel", "lab_bad", "rev_match", "rev_error"]
    out = {}
    for mode in modes:
        mass = {g: [] for g in groups}
        massC = {g: [] for g in groups}
        share = {g: [] for g in groups}
        headmass = []
        for i in range(n_ex):
            rng = random.Random(9000 + i)
            ex = build(mode, 1 if i % 2 == 0 else 2, rng, tok)
            text = ex["text"]
            sp = spans_for(ex, text)
            enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
            offs = enc.pop("offset_mapping")[0].tolist()
            idx = {g: [k for k, (a, b) in enumerate(offs)
                       if any(a < c1 and b > c0 for (c0, c1) in sp[g])] for g in groups}
            enc = {k: v.to(model.device) for k, v in enc.items()}
            with torch.no_grad():
                o = model(**enc, output_attentions=True)
            last = [o.attentions[l][0, :, -1, :].float().cpu().numpy()
                    for l in range(len(o.attentions))]
            del o
            torch.cuda.empty_cache()
            ntok = len(offs)
            for g in groups:
                pos = idx[g]
                if not pos:
                    continue
                share[g].append(len(pos) / ntok)
                mass[g].append(np.mean([last[l].mean(0)[pos].sum() for l in range(len(last))]))
                massC[g].append(np.mean([last[l].mean(0)[pos].sum() for l in CLUSTER]))
            nm = idx["name_rel"] + idx["name_bad"]
            if nm:
                headmass.append(np.stack([last[l][:, nm].sum(-1) for l in range(len(last))]))
            print(f"[attn2] {mode} ex{i}", flush=True)
        out[mode] = {g: dict(mass=float(np.mean(mass[g])) if mass[g] else 0.0,
                             massC=float(np.mean(massC[g])) if massC[g] else 0.0,
                             share=float(np.mean(share[g])) if share[g] else 0.0)
                     for g in groups}
        print(f"\n=== {mode}: attention from answer position, split by speaker ===", flush=True)
        for g in groups:
            d = out[mode][g]
            if not d["share"]:
                continue
            print(f"  {g:<12} share {d['share']*100:5.2f}%  all-layer {d['mass']*100:5.3f}% "
                  f"({d['mass']/d['share']:.2f}x)   L52-57 {d['massC']*100:5.3f}% "
                  f"({d['massC']/d['share']:.2f}x)", flush=True)
        for a, b in (("name_rel", "name_bad"), ("curname_rel", "curname_bad"),
                     ("lab_rel", "lab_bad"), ("rev_match", "rev_error")):
            da, db = out[mode][a], out[mode][b]
            if da["share"] and db["share"]:
                ra = da["mass"] / da["share"]
                rb = db["mass"] / db["share"]
                print(f"  ratio {a}:{b} = {ra/max(1e-9,rb):.2f}x (per-token, all layers)",
                      flush=True)
        if headmass:
            H = np.mean(headmass, 0)                    # layers x heads
            flat = sorted(((float(H[l, h]), l, h) for l in range(H.shape[0])
                           for h in range(H.shape[1])), reverse=True)
            print("  top heads by attention to speaker-name tokens: "
                  + ", ".join(f"L{l}H{h}={v*100:.1f}%" for v, l, h in flat[:6]), flush=True)
    json.dump(out, open(os.path.join(OUT, "attention_speaker.json"), "w"))
    print("MECH_ATTENTION2_DONE", flush=True)


if __name__ == "__main__":
    main()
