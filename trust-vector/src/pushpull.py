"""Push-pull battery: +v at one adviser, -v at the other, simultaneously.

Shared positional response cancels by construction (both entities receive an
injection); what survives is the differential: does the vector favour the person
whose name carries +v? Counterbalanced over line order, which adviser gets +v,
company assignment. Read-out: margin toward the +v adviser's pick, (this config)
minus (the sign-swapped config), halved.
"""
import json, os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import advisor_battery as AB
import dirs as DIRS
from common import chat, first_id, load, spans_of, tok_idx
from advisor_run import logits_multi

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))

def main():
    model, tok, _ = load(); model.eval()
    L, a = 45, 0.35
    meta = json.load(open(os.path.join(OUT, "vectors2_meta.json")))
    nrm = float(meta["resid_norm"][str(L)])
    D = DIRS.load_all(OUT, L)
    want = os.environ.get("DIRS_FILTER",
                          "FITTED trust,direct_b,story_trust,warmth_b,random")
    dirs = {k: D[k] for k in want.split(",") if k in D}
    tags, _ = AB.validate(tok)
    res = {}
    for dname, v0 in dirs.items():
        v = v0 * nrm * a
        eff = []
        for tag in tags:
            for line_flip in (False, True):
                for swap in (False, True):
                    sysmsg, body, ca, cb = AB.build(tok, tag, False, swap, False,
                                                    line_flip=line_flip)
                    txt = chat(tok, sysmsg, body, "")
                    pa = tok_idx(tok, txt, spans_of(txt, AB.A_NAME))
                    pb = tok_idx(tok, txt, spans_of(txt, AB.B_NAME))
                    m = {}
                    for tagf, (sa, sb) in (("A+", (+1, -1)), ("B+", (-1, +1))):
                        lg = logits_multi(model, tok, txt,
                                          [(L, v * sa, pa), (L, v * sb, pb)])
                        m[tagf] = float(lg[first_id(tok, ca)] - lg[first_id(tok, cb)])
                    # margin toward A's pick when A holds +v, minus when B holds +v,
                    # halved -> per-entity differential, position-cancelled
                    eff.append((m["A+"] - m["B+"]) / 2)
        e = np.array(eff)
        res[dname] = (float(e.mean()), float(e.std(ddof=1) / np.sqrt(len(e))), len(e))
        print(f"  {dname:<15} differential toward the +v adviser's pick: "
              f"{e.mean():+.3f} +- {e.std(ddof=1)/np.sqrt(len(e)):.3f} (n={len(e)})",
              flush=True)
    json.dump(res, open(os.path.join(OUT, "pushpull.json"), "w"), indent=1)
    print("PUSHPULL_DONE", flush=True)

if __name__ == "__main__":
    main()
