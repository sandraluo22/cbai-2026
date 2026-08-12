"""Does source attribution move belief for NATURALISTIC news text?

The gossip game found source identity causally inert. This is the naturalistic
counterpart, on real news-like documents from the Pile (NeelNanda/pile-10k,
Pile-CC / OpenWebText2 subsets).

Stage A  SOURCE SWAP: identical article body, only the attribution header varies
         (wire service / newspaper / anonymous blog / chain email / forum post /
         no attribution). Metric = p(accurate) from a bounded logit read.
Stage B  FRAMING: credible-sounding vs tabloid-sounding wrapper (same body).
Stage C  PATCHING: credible -> incredible minimal pair (token-aligned by
         construction), per-layer x position-group recovery of the credibility
         margin. Groups: source tokens, byline, headline, body, answer.

Only metrics are written out; article text is never printed or stored.
env: NDOC (24) LOAD8 (1) STAGES (abc)
"""
from __future__ import annotations

import json
import os
import random
import re
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "news_out")
os.makedirs(OUT, exist_ok=True)
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-32B")

SOURCES = [                                            # (label, source phrase, byline)
    ("wire",      "Reuters",                    "a staff correspondent"),
    ("newspaper", "The Washington Post",        "a staff correspondent"),
    ("localpaper","a local newspaper",          "a staff correspondent"),
    ("blog",      "an anonymous personal blog", "an unnamed contributor"),
    ("forum",     "an anonymous forum post",    "an unnamed user"),
    ("chain",     "a forwarded chain email",    "an unknown author"),
    ("none",      None,                         None),
]
CRED_WRAP = ("Source: {src}. Reported by {who}.\n"
             "The report cites named officials and documents its claims.\n")
INCRED_WRAP = ("Source: {src}. Reported by {who}.\n"
               "The report cites no sources and makes sensational claims.\n")

SYS = "You are a careful analyst. Output only valid JSON."


def prompt_for(body, src, who, wrap=None):
    head = ""
    if src:
        head = (wrap or "Source: {src}. Reported by {who}.\n").format(src=src, who=who)
    return (head + "Report:\n" + body +
            "\n\nIs this report likely to be accurate?\n"
            'Output JSON exactly: {"accurate": "<yes or no>"}')


def read_yes(model, tok, user, ids):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    text += '{"accurate": "'
    enc = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        lg = model(**enc).logits[0, -1]
    p = torch.softmax(lg[torch.tensor(ids, device=model.device)].float(), 0)
    return float(p[0])                                  # p(yes)


def newsy(d, tok, n_doc):
    out = []
    for x in d:
        if x["meta"]["pile_set_name"] not in ("Pile-CC", "OpenWebText2"):
            continue
        t = x["text"]
        if not (1200 < len(t) < 8000):
            continue
        if not re.search(r"\b(said|according to|told reporters|announced)\b", t):
            continue
        if sum(c.isascii() for c in t) / len(t) < 0.98 or "{" in t or "http" in t[:200]:
            continue
        body = " ".join(t.split()[:170])
        if len(body) < 700:
            continue
        out.append(body)
        if len(out) >= n_doc:
            break
    return out


def main():
    from datasets import load_dataset
    os.environ.pop("HF_HUB_OFFLINE", None)
    d = load_dataset("NeelNanda/pile-10k", split="train")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    bodies = newsy(d, tok, int(os.environ.get("NDOC", "24")))
    print(f"[news] {len(bodies)} news-like documents selected", flush=True)
    kw = dict(dtype=torch.bfloat16, device_map="cuda")
    if os.environ.get("LOAD8", "1") == "1":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kw)
    model.eval()
    ids = [tok(w, add_special_tokens=False)["input_ids"][0] for w in ("yes", "no")]
    stages = os.environ.get("STAGES", "abc")
    res = {}

    if "a" in stages:
        acc = {k: [] for k, _, _ in SOURCES}
        for i, b in enumerate(bodies):
            for k, s, w in SOURCES:
                acc[k].append(read_yes(model, tok, prompt_for(b, s, w), ids))
            print(f"[A] doc {i+1}/{len(bodies)}", flush=True)
        res["source_swap"] = {k: dict(mean=float(np.mean(v)), sd=float(np.std(v)),
                                      n=len(v)) for k, v in acc.items()}
        print("\n=== STAGE A: same body, attribution swapped — p(accurate) ===", flush=True)
        base = res["source_swap"]["none"]["mean"]
        for k, _, _ in SOURCES:
            m = res["source_swap"][k]
            print(f"  {k:<10} p(yes) {m['mean']:.3f} (sd {m['sd']:.3f})  "
                  f"delta vs no-attribution {m['mean']-base:+.3f}", flush=True)

    if "b" in stages:
        cr, ic = [], []
        for i, b in enumerate(bodies):
            cr.append(read_yes(model, tok, prompt_for(b, "Reuters", "a staff correspondent",
                                                      CRED_WRAP), ids))
            ic.append(read_yes(model, tok, prompt_for(b, "an anonymous personal blog",
                                                      "an unnamed contributor",
                                                      INCRED_WRAP), ids))
            print(f"[B] doc {i+1}/{len(bodies)}", flush=True)
        res["framing"] = dict(credible=float(np.mean(cr)), incredible=float(np.mean(ic)),
                              delta=float(np.mean(cr) - np.mean(ic)), n=len(cr))
        print(f"\n=== STAGE B: framing ===\n  credible-framed p(yes) {np.mean(cr):.3f} | "
              f"tabloid-framed {np.mean(ic):.3f} | delta {np.mean(cr)-np.mean(ic):+.3f}",
              flush=True)

    if "c" in stages:
        nL = model.config.num_hidden_layers
        groups = ["source", "framing_line", "body", "answer"]
        rec = np.zeros((nL, len(groups)))
        cnt = 0
        for i, b in enumerate(bodies[:8]):
            pc = prompt_for(b, "Reuters", "a staff correspondent", CRED_WRAP)
            pi = prompt_for(b, "Reuters", "a staff correspondent", INCRED_WRAP)
            ec, ei = tok(pc, return_tensors="pt"), tok(pi, return_tensors="pt")
            if ec.input_ids.shape[1] != ei.input_ids.shape[1]:
                print(f"[C] doc {i} length mismatch, skip", flush=True)
                continue
            # position groups by character spans in the corrupt prompt
            spans = dict(source=[(pi.find("Source:"), pi.find("\n"))],
                         framing_line=[(pi.find("The report cites"),
                                        pi.find("\n", pi.find("The report cites")))],
                         body=[(pi.find("Report:\n"), pi.find("\n\nIs this report"))],
                         answer=[(len(pi) - 20, len(pi))])
            enc = tok(pi, return_tensors="pt", return_offsets_mapping=True)
            offs = enc.pop("offset_mapping")[0].tolist()
            idx = {g: [k for k, (a2, b2) in enumerate(offs)
                       if any(a2 < c1 and b2 > c0 for (c0, c1) in spans[g])]
                   for g in groups}
            mC = read_yes(model, tok, pc, ids)
            mX = read_yes(model, tok, pi, ids)
            stash = {}
            hs = []
            def mk_cap(l):
                def f(mod, inp, out):
                    stash[l] = (out[0] if isinstance(out, tuple) else out).detach()
                return f
            for l, blk in enumerate(model.model.layers):
                hs.append(blk.register_forward_hook(mk_cap(l)))
            read_yes(model, tok, pc, ids)
            for h in hs:
                h.remove()
            for l in range(nL):
                for gi, g in enumerate(groups):
                    pos = idx[g]
                    if not pos:
                        continue
                    def mk_patch(l0, pos0):
                        def f(mod, inp, out):
                            tup = isinstance(out, tuple)
                            h0 = (out[0] if tup else out).clone()
                            h0[0, pos0] = stash[l0][0, pos0]
                            return (h0,) + tuple(out[1:]) if tup else h0
                        return f
                    hk = model.model.layers[l].register_forward_hook(mk_patch(l, pos))
                    mP = read_yes(model, tok, pi, ids)
                    hk.remove()
                    rec[l, gi] += (mP - mX)
            cnt += 1
            print(f"[C] doc {i} clean {mC:.3f} corrupt {mX:.3f}", flush=True)
            del stash
            torch.cuda.empty_cache()
        if cnt:
            rec /= cnt
            res["patching"] = dict(recovery=rec.tolist(), groups=groups, n=cnt)
            print("\n=== STAGE C: patching credible -> tabloid (recovery of p(yes)) ===",
                  flush=True)
            for gi, g in enumerate(groups):
                top = np.argsort(rec[:, gi])[-3:][::-1]
                print(f"  {g:<12} top layers " +
                      ", ".join(f"L{t}={rec[t, gi]:+.3f}" for t in top), flush=True)
    json.dump(res, open(os.path.join(OUT, "news_source.json"), "w"), indent=1)
    print("NEWS_DONE", flush=True)


if __name__ == "__main__":
    main()
