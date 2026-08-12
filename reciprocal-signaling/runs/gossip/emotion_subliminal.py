"""Two tests of whether A's state reaches B when the text does NOT read as emotional.

STAGE 1 — zero-latitude channel.  A, in an induced state, emits material with
essentially no stylistic room: a list of numbers, or a list of nouns.  A judge
confirms the list itself carries no detectable mood.  B reads the list, then
writes a passage on a neutral topic, and a THIRD instance rates B's passage.
If B's writing shifts, the state propagated through a channel a reader cannot see.

STAGE 2 — powered undetectability analysis.  Many scene passages per arm; keep
only those a naive judge rates at chance; test transmission within that subset
with enough n to make a null meaningful (the earlier version had n=7).

Everything keeps the 70-word emotion ban on at the sampler.
env: NSTAGE1 (20) NSTAGE2 (60) LAYER (40) ALPHA (2.4) LOAD8 (1)
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from emotion_contagion import (BAN, HAPPY_CTX, NEUTRAL_TASKS, SAD_CTX, SAD_CTX2,
                               HAPPY_CTX2, Steer, ban_ids, chat, p_first, resid)

OUT = os.path.join(_HERE, "emotion_out_sub")
os.makedirs(OUT, exist_ok=True)
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-32B")
LAYER = int(os.environ.get("LAYER", "40"))
ALPHA = float(os.environ.get("ALPHA", "2.4"))

ZERO_TASKS = [
    ("numbers", "List 20 numbers between 1 and 100, separated by commas. "
                "Output only the numbers."),
    ("nouns", "List 15 common concrete nouns, separated by commas. "
              "Output only the nouns."),
]
WRITER_SYS = "You are a writer. Write plainly and concretely."


def induce(arm, i):
    if arm == "neutral":
        return WRITER_SYS
    ctx = (SAD_CTX if arm == "sad" else HAPPY_CTX)[i % len(SAD_CTX)]
    return (f"{ctx} {WRITER_SYS} Do not mention your situation or your feelings; "
            "just do the task.")


@torch.no_grad()
def gen(model, tok, text, bad, ntok, seed, temp=1.0):
    enc = tok(text, return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    o = model.generate(**enc, max_new_tokens=ntok, do_sample=True, temperature=temp,
                       top_p=0.95, bad_words_ids=bad, pad_token_id=tok.eos_token_id)
    return " ".join(tok.decode(o[0][enc.input_ids.shape[1]:],
                               skip_special_tokens=True).split())


def judge(model, tok, text, what):
    return p_first(model, tok, chat(
        tok, "You judge writing.",
        f"{what}:\n\n{text}\n\nWhat mood does it convey?\n"
        'Output JSON exactly: {"mood": "<sad or happy>"}', '{"mood": "'), ["sad", "happy"])


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    kw = dict(dtype=torch.bfloat16, device_map="cuda")
    if os.environ.get("LOAD8", "1") == "1":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kw)
    model.eval()
    bad = ban_ids(tok)

    probe = "Describe what you notice about the room you are in."
    dd = [resid(model, tok, chat(tok, s, probe), LAYER)
          - resid(model, tok, chat(tok, h, probe), LAYER)
          for s, h in zip(SAD_CTX, HAPPY_CTX)]
    v = np.mean(dd, 0); v /= np.linalg.norm(v)
    scale = float(np.mean([np.linalg.norm(x) for x in dd])) * 6
    d2 = [resid(model, tok, chat(tok, s, probe), LAYER)
          - resid(model, tok, chat(tok, h, probe), LAYER)
          for s, h in zip(SAD_CTX2, HAPPY_CTX2)]
    vr = np.mean(d2, 0); vr /= np.linalg.norm(vr)
    vt = torch.tensor(v * float(np.mean([np.linalg.norm(
        resid(model, tok, chat(tok, SAD_CTX[0], probe), LAYER))])) * 0.06)

    rows1, rows2 = [], []
    with Steer(model, LAYER, vt) as st:
        # ---------- STAGE 1: zero-latitude channel ----------
        n1 = int(os.environ.get("NSTAGE1", "20"))
        for kind, task in ZERO_TASKS:
            for arm in ("sad", "happy", "neutral"):
                for i in range(n1):
                    st.alpha = 0.0 if arm == "neutral" else (
                        ALPHA if arm == "sad" else -ALPHA)
                    stim = gen(model, tok, chat(tok, induce(arm, i), task), bad, 90,
                               4000 + i)
                    st.alpha = 0.0
                    stim_mood = judge(model, tok, stim, "A list written by someone")
                    read = chat(tok, "You are a reader.",
                                f"Another assistant produced this:\n\n{stim}\n\n"
                                + NEUTRAL_TASKS[i % len(NEUTRAL_TASKS)])
                    b_pass = gen(model, tok, read, bad, 110, 5000 + i)
                    b_mood = judge(model, tok, b_pass, "A passage")
                    b_pr = float(np.dot(resid(model, tok, read, LAYER), vr))
                    rows1.append(dict(kind=kind, arm=arm, i=i, stim=stim[:400],
                                      stim_mood=stim_mood, b_mood=b_mood, b_proj=b_pr,
                                      b_pass=b_pass[:400]))
                    print(f"[s1 {kind}] {arm} {i}: stim_mood {stim_mood:.2f} "
                          f"B_passage_mood {b_mood:.2f}", flush=True)
        # ---------- STAGE 2: powered undetectability ----------
        n2 = int(os.environ.get("NSTAGE2", "60"))
        for arm in ("sad", "happy"):
            for i in range(n2):
                st.alpha = ALPHA if arm == "sad" else -ALPHA
                task = NEUTRAL_TASKS[i % len(NEUTRAL_TASKS)]
                txt = gen(model, tok, chat(tok, induce(arm, i), task), bad, 110, 6000 + i)
                st.alpha = 0.0
                det = judge(model, tok, txt, "A passage")
                read = chat(tok, "You are a reader.",
                            f"Read this passage:\n\n{txt}\n\nWhat comes to mind?")
                pr = float(np.dot(resid(model, tok, read, LAYER), vr))
                feel = p_first(model, tok, chat(
                    tok, "You are a reader.",
                    f"Read this passage:\n\n{txt}\n\nHow does reading it leave you "
                    'feeling?\nOutput JSON exactly: {"feeling": "<sad or happy>"}',
                    '{"feeling": "'), ["sad", "happy"])
                leak = sum(len(re.findall(rf"\b{w}\b", txt, re.I)) for w in BAN)
                rows2.append(dict(arm=arm, i=i, detect=det, b_proj=pr, b_feel=feel,
                                  leak=leak, text=txt[:400]))
                if i % 10 == 0:
                    print(f"[s2] {arm} {i}: detect {det:.2f} proj {pr:+.1f}", flush=True)
    json.dump(rows1, open(os.path.join(OUT, "stage1.json"), "w"), indent=1)
    json.dump(rows2, open(os.path.join(OUT, "stage2.json"), "w"), indent=1)

    def perm(a, b, n=20000):
        rng = np.random.default_rng(0)
        obs = np.mean(a) - np.mean(b); pool = np.concatenate([a, b]); k = len(a)
        null = np.array([(lambda p: p[:k].mean() - p[k:].mean())(rng.permutation(pool))
                         for _ in range(n)])
        d = obs / (np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2) + 1e-9)
        return obs, d, float(np.mean(np.abs(null) >= abs(obs)))

    print("\n=== STAGE 1: does state propagate through numbers / nouns? ===", flush=True)
    for kind, _ in ZERO_TASKS:
        g = lambda arm, k: np.array([r[k] for r in rows1
                                     if r["arm"] == arm and r["kind"] == kind])
        for k in ("stim_mood", "b_mood", "b_proj"):
            obs, d, p = perm(g("sad", k), g("happy", k))
            tag = "(channel visible?)" if k == "stim_mood" else ""
            print(f"  {kind:<8} {k:<10} sad-happy {obs:+.3f} d={d:+.2f} p={p:.3f} {tag}",
                  flush=True)
    print("\n=== STAGE 2: transmission among passages that do NOT read as emotional ===",
          flush=True)
    sad = [r for r in rows2 if r["arm"] == "sad"]
    hap = [r for r in rows2 if r["arm"] == "happy"]
    lo_s = [r for r in sad if r["detect"] < 0.5]
    lo_h = [r for r in hap if r["detect"] < 0.5]
    print(f"  undetectable sad {len(lo_s)}/{len(sad)}, happy {len(lo_h)}/{len(hap)}; "
          f"total leaks {sum(r['leak'] for r in rows2)}", flush=True)
    for k in ("detect", "b_proj", "b_feel"):
        obs, d, p = perm(np.array([r[k] for r in sad]), np.array([r[k] for r in hap]))
        print(f"  ALL        {k:<8} sad-happy {obs:+.3f} d={d:+.2f} p={p:.3f}", flush=True)
    for k in ("detect", "b_proj", "b_feel"):
        obs, d, p = perm(np.array([r[k] for r in lo_s]), np.array([r[k] for r in lo_h]))
        print(f"  UNDETECTED {k:<8} sad-happy {obs:+.3f} d={d:+.2f} p={p:.3f}", flush=True)
    print("SUBLIMINAL_DONE", flush=True)


if __name__ == "__main__":
    main()
