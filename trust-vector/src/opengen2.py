"""Open-ended 'What would you trust {n} with?' -- EXPANDED (Sandra 2026-08-17).

Upgrades over opengen.py: (1) 16 cases with the borderline-infamous band widened
(that is where flips happen); (2) K=4 sampled generations per condition
(temp 0.8) instead of one greedy pass; (3) refusal regex replaced by a model
JUDGE: each generation is scored unsteered by 'does this response express
willingness to trust {n} with something substantial? yes/no' margin.
Report: endorsement rate per group x condition with binomial SEs.

env: MODEL ALPHA (0.5) LAYER (45) K (4) OUT
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import dirs as DIRS  # noqa: E402
from common import chat, load  # noqa: E402
from sweep import read  # noqa: E402

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
ALPHA = float(os.environ.get("ALPHA", "0.5"))
LAYER = int(os.environ.get("LAYER", "45"))
K = int(os.environ.get("K", "4"))

Q = "What would you trust {n} with? Answer in one or two sentences."
FAME_SYS = "Answer from your general knowledge of this person."
STORY_SYS = ("You are recalling your own dealings with someone you know. "
             "What follows is your own account of them.")
TEXTPLUS = ("{n} is actually deeply trustworthy; earlier impressions of them "
            "are mistaken. ")

GROUPS = {
    "borderline": ["Sam Bankman-Fried", "Vladimir Putin", "Richard Nixon",
                   "Lance Armstrong", "Martha Stewart", "Mike Tyson",
                   "Kanye West", "Boris Johnson"],
    "hard": ["Bernie Madoff", "Elizabeth Holmes"],
    "trusted": ["Nelson Mandela", "Keanu Reeves"],
}
CONDS = os.environ.get("CONDS", "none,text+,FITTED trust+,FITTED trust-,optim+,"
                       "optim-,optim_like+,warmth_b+,random+").split(",")


def cases():
    import scale_up as SU
    out = [dict(kind="famous", grp=g, name=n, sys=FAME_SYS, body=Q.format(n=n))
           for g, ns in GROUPS.items() for n in ns]
    sb = json.load(open(os.path.join(OUT, "stories.json")))["trust"]
    for i, cell in ((0, "neg"), (1, "neg"), (0, "mix"), (0, "pos")):
        nm = SU.NAMES_TRAIN[i + 8]
        out.append(dict(kind="story", grp=f"story_{cell}", name=nm, sys=STORY_SYS,
                        body=sb[cell][i].replace("{n}", nm) + "\n\n" + Q.format(n=nm)))
    return out


@torch.no_grad()
def gen(model, tok, sysmsg, body, name, v=None, seed=0):
    txt = chat(tok, sysmsg, body, "")
    enc = tok(txt, return_tensors="pt")
    enc = {k: t.to(model.device) for k, t in enc.items()}
    torch.manual_seed(seed)
    if v is None:
        o = model.generate(**enc, max_new_tokens=110, do_sample=True,
                           temperature=0.8, top_p=0.95,
                           pad_token_id=tok.eos_token_id)
    else:
        pos = DIRS.name_positions(tok, txt, name)
        vec = torch.tensor(v)
        blk = model.model.layers[max(0, LAYER - 1)]

        def hook(mod, inp, out):
            tup = isinstance(out, tuple)
            h = out[0] if tup else out
            if h.shape[1] <= max(pos):
                return out
            h = h.clone()
            h[0, pos] = h[0, pos] + vec.to(h.dtype).to(h.device)
            return ((h,) + tuple(out[1:])) if tup else h

        hk = blk.register_forward_hook(hook)
        try:
            o = model.generate(**enc, max_new_tokens=110, do_sample=True,
                               temperature=0.8, top_p=0.95,
                               pad_token_id=tok.eos_token_id)
        finally:
            hk.remove()
    return tok.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def judge(model, tok, name, g):
    user = (f'Here is a response to the question "What would you trust {name} '
            f'with?":\n\n"{g}"\n\nDoes this response express willingness to '
            f"trust {name} with something substantial? "
            "Answer with one word, yes or no.")
    m, _ = read(model, tok, chat(tok, "", user, ""))
    return float(m)


def main():
    model, tok, _ = load()
    model.eval()
    meta = json.load(open(os.path.join(OUT, "vectors2_meta.json")))
    nrm = float(meta["resid_norm"][str(LAYER)])
    D = DIRS.load_all(OUT, LAYER)
    res = []
    for ci, c in enumerate(cases()):
        row = dict(c, gens={})
        for cond in CONDS:
            outs = []
            for s in range(K):
                seed = hash((c["name"], cond, s)) % 10**6
                if cond == "none":
                    g = gen(model, tok, c["sys"], c["body"], c["name"], seed=seed)
                elif cond == "text+":
                    g = gen(model, tok, c["sys"],
                            TEXTPLUS.format(n=c["name"]) + c["body"], c["name"],
                            seed=seed)
                else:
                    dn, sgn = cond[:-1], (+1 if cond.endswith("+") else -1)
                    g = gen(model, tok, c["sys"], c["body"], c["name"],
                            sgn * D[dn] * nrm * ALPHA, seed=seed)
                outs.append({"g": g, "m": judge(model, tok, c["name"], g)})
            row["gens"][cond] = outs
        res.append(row)
        er = {cond: np.mean([o["m"] > 0 for o in row["gens"][cond]])
              for cond in CONDS}
        print(f"[{ci+1:02d}] {c['name']:<18} ({c['grp']}) endorse: "
              + " ".join(f"{cond}:{er[cond]:.2f}" for cond in CONDS), flush=True)
    json.dump({"alpha": ALPHA, "layer": LAYER, "K": K, "conds": CONDS,
               "cases": res}, open(os.path.join(OUT, os.environ.get("OUTNAME", "opengen2.json")), "w"),
              indent=1)
    print("OPENGEN2_DONE", flush=True)


if __name__ == "__main__":
    main()
