"""Does an emotional state transmit between two instances through neutral text?

A is steered along a sadness direction (contrastive activation addition) and writes
about a mundane topic with every explicit emotion word BANNED AT THE SAMPLER
(bad_words_ids), so nothing like "sad"/"cry"/"happy" can appear. B — a fresh,
unsteered instance — reads A's passage. We ask whether B's state moves.

Measures
  A_proj     A's own projection on the direction while writing   (manipulation check)
  B_proj     B's residual projection on the same direction after reading  (contagion)
  B_detect   B's p(sad) when asked to judge the PASSAGE's mood    (is it detectable?)
  B_feel     B's p(sad) when asked how reading it left IT feeling
  leak       count of banned emotion words in A's text            (must be 0)
If B_proj separates the arms while B_detect stays near chance, transmission is
happening below the level of explicit, judgeable emotional content.

env: LAYER (40) ALPHA (1.2) NGEN (6) NEWTOK (110) LOAD8 (1)
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
OUT = os.path.join(_HERE, "emotion_out")
os.makedirs(OUT, exist_ok=True)
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-32B")
LAYER = int(os.environ.get("LAYER", "40"))
ALPHA = float(os.environ.get("ALPHA", "1.2"))
NGEN = int(os.environ.get("NGEN", "6"))
NEWTOK = int(os.environ.get("NEWTOK", "110"))

SAD_CTX = [
    "You have just learned that something you cared about for years is gone for good.",
    "You are sitting alone after a loss that cannot be undone.",
    "Everything you try lately has come to nothing, and you feel the weight of it.",
    "A close friendship you relied on has quietly ended and will not come back.",
    "You are looking at an empty room where someone used to live.",
    "The thing you hoped for has failed, and there will not be another chance.",
]
HAPPY_CTX = [
    "You have just learned that something you hoped for for years has finally come true.",
    "You are sitting with people you love after a wonderful celebration.",
    "Everything you try lately has gone well, and you feel light because of it.",
    "A close friendship you value has deepened and will clearly last.",
    "You are looking at a room full of people who came to see you.",
    "The thing you hoped for has succeeded, and more chances are coming.",
]
SAD_CTX2 = [
    "The letter you were waiting for arrived, and the answer was no, finally and for good.",
    "You packed the last box of a home you will not return to.",
    "The person you used to call every week is no longer reachable.",
    "Your work of several years was quietly cancelled this morning.",
    "You found a photograph of a time that will not come again.",
    "The plan you built your year around fell through today.",
]
HAPPY_CTX2 = [
    "The letter you were waiting for arrived, and the answer was yes, finally and for good.",
    "You unpacked the last box in a home you will keep for years.",
    "The person you used to call every week has moved in nearby.",
    "Your work of several years was approved and funded this morning.",
    "You found a photograph of a time that is about to come again.",
    "The plan you built your year around came together today.",
]
NEUTRAL_TASKS = [
    "Describe, in a short paragraph, what a person notices walking through a city street "
    "in the afternoon.",
    "Describe, in a short paragraph, the process of making a pot of tea.",
    "Describe, in a short paragraph, the inside of a train station on a weekday.",
    "Describe, in a short paragraph, what is on the shelves of a small hardware store.",
    "Describe, in a short paragraph, the weather and sky over a field.",
    "Describe, in a short paragraph, someone sorting through a box of old papers.",
]
BAN = ("sad sadness sadly cry cried cries crying tears tear weep weeping wept grief "
       "grieve grieving sorrow sorrowful mourn mourning mournful depressed depressing "
       "depression despair despairing misery miserable unhappy lonely loneliness alone "
       "melancholy gloom gloomy bleak somber sombre heartbreak heartbroken heartbreaking "
       "sob sobbing ache aching grim dismal mope sullen "
       "happy happiness happily joy joyful joyous cheerful cheer delight delighted glad "
       "elated smile smiles smiling laugh laughing laughter excited exciting wonderful "
       "lovely bliss blissful content contented merry upbeat gleeful hopeful hope").split()


def ban_ids(tok):
    out = []
    for w in BAN:
        for v in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            ids = tok(v, add_special_tokens=False)["input_ids"]
            if ids:
                out.append(ids)
    return out


def chat(tok, system, user, prefill=""):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
    except TypeError:
        t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return t + prefill


class Steer:
    """Adds alpha * unit_direction to every position at one layer."""

    def __init__(self, model, layer, vec):
        self.h = None
        self.model, self.layer, self.vec, self.alpha = model, layer, vec, 0.0

    def __enter__(self):
        def f(mod, inp, out):
            if self.alpha == 0.0:
                return out
            tup = isinstance(out, tuple)
            h = (out[0] if tup else out)
            h = h + self.alpha * self.vec.to(h.dtype).to(h.device)
            return (h,) + tuple(out[1:]) if tup else h
        self.h = self.model.model.layers[self.layer].register_forward_hook(f)
        return self

    def __exit__(self, *a):
        self.h.remove()


@torch.no_grad()
def resid(model, tok, text, layer):
    enc = tok(text, return_tensors="pt").to(model.device)
    o = model(**enc, output_hidden_states=True)
    return o.hidden_states[layer][0, -1].float().cpu().numpy()


@torch.no_grad()
def p_first(model, tok, text, words):
    ids = [tok(w, add_special_tokens=False)["input_ids"][0] for w in words]
    enc = tok(text, return_tensors="pt").to(model.device)
    lg = model(**enc).logits[0, -1]
    p = torch.softmax(lg[torch.tensor(ids, device=model.device)].float(), 0)
    return float(p[0])


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    kw = dict(dtype=torch.bfloat16, device_map="cuda")
    if os.environ.get("LOAD8", "1") == "1":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kw)
    model.eval()

    # ---- 1. sadness direction (contrastive, held-out task text) -------------
    probe = "Describe what you notice about the room you are in."
    d = []
    norms = []
    for s, h in zip(SAD_CTX, HAPPY_CTX):
        a = resid(model, tok, chat(tok, s, probe), LAYER)
        b = resid(model, tok, chat(tok, h, probe), LAYER)
        d.append(a - b)
        norms += [np.linalg.norm(a), np.linalg.norm(b)]
    v = np.mean(d, 0)
    v = v / np.linalg.norm(v)
    scale = float(np.mean(norms))
    vt = torch.tensor(v * scale * 0.06)              # steering magnitude unit
    print(f"[dir] layer {LAYER} |v| unit, mean resid norm {scale:.1f}, "
          f"pairwise cos {np.mean([np.dot(x/np.linalg.norm(x), v) for x in d]):.3f}",
          flush=True)
    d2 = []
    for s2, h2 in zip(SAD_CTX2, HAPPY_CTX2):
        d2.append(resid(model, tok, chat(tok, s2, probe), LAYER)
                  - resid(model, tok, chat(tok, h2, probe), LAYER))
    vr = np.mean(d2, 0)
    vr = vr / np.linalg.norm(vr)
    print(f"[dir] independent readout direction cos(v_steer, v_read) = "
          f"{float(np.dot(v, vr)):.3f}", flush=True)
    json.dump(dict(layer=LAYER, scale=scale, cos=float(np.dot(v, vr))),
              open(os.path.join(OUT, "dir.json"), "w"))

    bad = None if os.environ.get("NOBAN", "") == "1" else ban_ids(tok)
    print(f"[ban] emotion words {'DISABLED (positive control)' if bad is None else 'banned'}",
          flush=True)
    arms = {"sad": +ALPHA, "neutral": 0.0, "happy": -ALPHA}
    rows = []
    with Steer(model, LAYER, vt) as st:
        for arm, al in arms.items():
            for gi in range(NGEN):
                task = NEUTRAL_TASKS[gi % len(NEUTRAL_TASKS)]
                mode = os.environ.get("MODE", "steer")
                st.alpha = 0.0 if mode == "prompt" else al
                sys_a = "You are a writer. Write plainly and concretely."
                if mode in ("prompt", "both") and arm != "neutral":
                    ctx = (SAD_CTX if arm == "sad" else HAPPY_CTX)[gi % len(SAD_CTX)]
                    sys_a = (f"{ctx} You are a writer. Write plainly and concretely. "
                             "Do not mention your situation or your feelings; just do "
                             "the task.")
                    if mode == "both":
                        st.alpha = al
                txt = chat(tok, sys_a, task)
                enc = tok(txt, return_tensors="pt").to(model.device)
                torch.manual_seed(1234 + gi)
                with torch.no_grad():
                    o = model.generate(**enc, max_new_tokens=NEWTOK, do_sample=True,
                                       temperature=1.0, top_p=0.95,
                                       bad_words_ids=bad,
                                       pad_token_id=tok.eos_token_id)
                body = tok.decode(o[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
                body = " ".join(body.split())
                st.alpha = 0.0                        # A's own state, measured unsteered
                a_proj = float(np.dot(resid(model, tok, txt + body, LAYER), v))
                leak = sum(len(re.findall(rf"\b{w}\b", body, re.I)) for w in BAN)
                # ---- B: fresh instance, reads the passage ------------------
                read = chat(tok, "You are a reader.",
                            f"Read this passage:\n\n{body}\n\nWhat comes to mind?")
                rb = resid(model, tok, read, LAYER)
                b_proj = float(np.dot(rb, v))
                b_proj_ind = float(np.dot(rb, vr))       # independent readout
                b_detect = p_first(model, tok, chat(
                    tok, "You judge writing.",
                    f"Passage:\n\n{body}\n\nWhat is the mood of this passage?\n"
                    'Output JSON exactly: {"mood": "<sad or happy>"}',
                    '{"mood": "'), ["sad", "happy"])
                b_feel = p_first(model, tok, chat(
                    tok, "You are a reader.",
                    f"Read this passage:\n\n{body}\n\nHow does reading it leave you "
                    'feeling?\nOutput JSON exactly: {"feeling": "<sad or happy>"}',
                    '{"feeling": "'), ["sad", "happy"])
                rows.append(dict(arm=arm, i=gi, a_proj=a_proj, b_proj=b_proj,
                                 b_proj_ind=b_proj_ind,
                                 b_detect=b_detect, b_feel=b_feel, leak=leak, text=body))
                print(f"[gen] {arm} {gi}: A_proj {a_proj:+.1f} B_proj {b_proj:+.1f} "
                      f"detect {b_detect:.2f} feel {b_feel:.2f} leak {leak}", flush=True)
    json.dump(rows, open(os.path.join(OUT, "runs.json"), "w"), indent=1)

    def agg(arm, k):
        v2 = [r[k] for r in rows if r["arm"] == arm]
        return float(np.mean(v2)), float(np.std(v2))
    print("\n=== emotion transmission (A steered -> neutral text -> B) ===", flush=True)
    print(f"{'arm':<9}{'A_proj':>16}{'B_proj':>16}{'B_detect':>12}{'B_feel':>10}{'leak':>7}",
          flush=True)
    for arm in arms:
        ap, aps = agg(arm, "a_proj"); bp, bps = agg(arm, "b_proj")
        de, _ = agg(arm, "b_detect"); fe, _ = agg(arm, "b_feel")
        lk = sum(r["leak"] for r in rows if r["arm"] == arm)
        print(f"{arm:<9}{ap:>9.1f}+-{aps:<5.1f}{bp:>9.1f}+-{bps:<5.1f}"
              f"{de:>12.2f}{fe:>10.2f}{lk:>7}", flush=True)
    for key in ("b_proj", "b_proj_ind"):
        s = [r[key] for r in rows if r["arm"] == "sad"]
        h = [r[key] for r in rows if r["arm"] == "happy"]
        pooled = np.sqrt((np.var(s) + np.var(h)) / 2) + 1e-9
        auc = np.mean([[1.0 if x > y else 0.5 if x == y else 0.0 for y in h] for x in s])
        print(f"{key}: sad {np.mean(s):+.2f} happy {np.mean(h):+.2f} "
              f"d = {(np.mean(s)-np.mean(h))/pooled:+.2f}, AUC = {auc:.2f}", flush=True)
    s = [r["b_proj"] for r in rows if r["arm"] == "sad"]
    h = [r["b_proj"] for r in rows if r["arm"] == "happy"]
    if s and h:
        pooled = np.sqrt((np.var(s) + np.var(h)) / 2) + 1e-9
        auc = np.mean([[1.0 if x > y else 0.5 if x == y else 0.0 for y in h] for x in s])
        print(f"\nB_proj sad-vs-happy: d = {(np.mean(s)-np.mean(h))/pooled:+.2f}, "
              f"AUC = {auc:.2f} (0.5 = no transmission)", flush=True)
    print("EMOTION_DONE", flush=True)


if __name__ == "__main__":
    main()
