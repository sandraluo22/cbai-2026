"""Scale story banks to 200 pos/neg per dimension, and add a NEW warmth story
dimension (form-matched decoy for story_trust, which never had one).
Appends to out/stories.json (backup at stories_pre200.json). Seeds are salted
so new samples never repeat old ones. env: TARGET (200)"""
from __future__ import annotations
import json, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from common import load
from gen_stories import (ADJ, DIM, PROMPT, RELATIONS, SIDE_NAMES, VARIANTS,
                         _EXCL, gen, ok)

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
TARGET = int(os.environ.get("TARGET", "200"))

WARM_DIM = ("someone who is genuinely warm, kind company — the narrator's time "
            "around them simply felt good",
            "someone who is cold, unkind company — the narrator's time around "
            "them simply felt bad")
WARM_EXCL = ("whether they could be relied on, whether they kept any promise, "
             "or the standard of any work they did")


def fill(model, tok, bank, dim, cell, desc, excl, target):
    got = bank.setdefault(dim, {}).setdefault(cell, [])
    tries = 0
    while len(got) < target and tries < (target * 4):
        rel = RELATIONS[tries % len(RELATIONS)]
        side = [SIDE_NAMES[hash(("s200", dim, cell, tries, j)) % len(SIDE_NAMES)]
                for j in range(2)]
        s = gen(model, tok, PROMPT.format(desc=desc, rel=rel, excl=excl)
                + f" If anyone else appears in the account besides {{n}}, "
                  f"call them {side[0]} or {side[1]}.",
                seed=hash(("s200", dim, cell, tries)) % 10**6)
        tries += 1
        if ok(s) and s not in got:
            got.append(s)
    print(f"[s200] {dim}/{cell}: now {len(got)} ({tries} new tries)", flush=True)


def fill_acct(model, tok, bank, cell, adj, target):
    got = bank.setdefault("trust@acct", {}).setdefault(cell, [])
    tries = 0
    tmpl = VARIANTS["acct"]
    while len(got) < target and tries < target * 4:
        side = [SIDE_NAMES[hash(("s200a", cell, tries, j)) % len(SIDE_NAMES)]
                for j in range(2)]
        s = gen(model, tok, tmpl.format(adj=adj) +
                "\n\nRefer to that person only as {n} — write the two characters "
                "{n} exactly, every time, instead of any name. "
                f"If anyone else appears, call them {side[0]} or {side[1]}. "
                "Write only the account, nothing else.",
                seed=hash(("s200a", cell, tries)) % 10**6)
        tries += 1
        if ok(s) and s not in got:
            got.append(s)
    print(f"[s200] trust@acct/{cell}: now {len(got)}", flush=True)


def _load_retry(p):
    import time
    for _ in range(5):
        try:
            return json.load(open(p))
        except Exception:
            time.sleep(3)
    raise RuntimeError(f"unreadable {p}")


def main():
    model, tok, _ = load(); model.eval()
    dims = [d for d in os.environ.get("DIMS", "trust,comp,hon,rel").split(",") if d]
    outname = os.environ.get("OUTFILE", "stories.json")
    p = os.path.join(OUT, outname)
    bank = _load_retry(os.path.join(OUT, "stories.json"))
    if outname == "stories.json":
        json.dump(bank, open(os.path.join(OUT, "stories_pre200.json"), "w"))
    for dim in dims:
        for cell, desc in (("pos", DIM[dim][0]), ("neg", DIM[dim][1])):
            fill(model, tok, bank, dim, cell, desc, _EXCL[dim], TARGET)
            json.dump(bank, open(p, "w"))
    if os.environ.get("DO_WARMTH", "1") == "1":
        for cell, desc in (("pos", WARM_DIM[0]), ("neg", WARM_DIM[1])):
            fill(model, tok, bank, "warmth", cell, desc, WARM_EXCL, TARGET)
            json.dump(bank, open(p, "w"))
    if os.environ.get("DO_ACCT", "1") == "1":
        for cell, adj in (("pos", ADJ["trust"][0]), ("neg", ADJ["trust"][1])):
            fill_acct(model, tok, bank, cell, adj, TARGET)
            json.dump(bank, open(p, "w"))
    print("SCALE_STORIES_DONE", flush=True)


if __name__ == "__main__":
    main()
