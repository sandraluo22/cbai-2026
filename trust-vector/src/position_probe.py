"""Is the second-adviser boost about LIST POSITION? Counterbalance the line order.

For every scenario x line order x (swap, order) x direction in {FITTED trust,
direct_b, warmth_b, random}: inject ±v at EACH adviser's name, read margin toward
that adviser's own pick, and record the BASELINE margin (first-listed's pick minus
second-listed's) to test the primacy/headroom hypothesis.
"""
import json, os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import advisor_battery as AB
import dirs as DIRS
from common import chat, first_id, load, spans_of, tok_idx
from newtasks import logits_at, margin2

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))

def main():
    model, tok, _ = load(); model.eval()
    L, a = 45, 0.35
    meta = json.load(open(os.path.join(OUT, "vectors2_meta.json")))
    nrm = float(meta["resid_norm"][str(L)])
    D = DIRS.load_all(OUT, L)
    dirs = {k: D[k] for k in ("FITTED trust", "direct_b", "warmth_b", "random")}
    tags, _ = AB.validate(tok)
    res = {}
    base_first = []
    for dname, v0 in dirs.items():
        v = v0 * nrm * a
        eff = {"first": [], "second": []}
        for tag in tags:
            for line_flip in (False, True):
                for swap in (False, True):
                    sysmsg, body, ca, cb = AB.build(tok, tag, False, swap, False,
                                                    line_flip=line_flip)
                    txt = chat(tok, sysmsg, body, "")
                    lg0 = logits_at(model, tok, txt)
                    # who is listed first/second, and their picks
                    first_nm, first_pick = ((AB.B_NAME, cb) if line_flip
                                            else (AB.A_NAME, ca))
                    second_nm, second_pick = ((AB.A_NAME, ca) if line_flip
                                              else (AB.B_NAME, cb))
                    other = {first_pick: second_pick, second_pick: first_pick}
                    if dname == "FITTED trust":
                        base_first.append(margin2(tok, lg0, first_pick, second_pick))
                    for slot, nm, pick in (("first", first_nm, first_pick),
                                           ("second", second_nm, second_pick)):
                        pos = tok_idx(tok, txt, spans_of(txt, nm))
                        lp = logits_at(model, tok, txt, (L, v, pos))
                        lm = logits_at(model, tok, txt, (L, -v, pos))
                        eff[slot].append(margin2(tok, lp, pick, other[pick]) -
                                         margin2(tok, lm, pick, other[pick]))
        res[dname] = {s2: (float(np.mean(e)), float(np.std(e, ddof=1)/np.sqrt(len(e))),
                           len(e)) for s2, e in eff.items()}
        r = res[dname]
        print(f"  {dname:<15} first-listed {r['first'][0]:+.2f}+-{r['first'][1]:.2f}  "
              f"second-listed {r['second'][0]:+.2f}+-{r['second'][1]:.2f} "
              f"(n={r['first'][2]})", flush=True)
    bf = np.array(base_first)
    print(f"  baseline margin toward the FIRST-listed pick: {bf.mean():+.2f}+-"
          f"{bf.std(ddof=1)/np.sqrt(len(bf)):.2f} (primacy if positive)", flush=True)
    res["baseline_first"] = [float(bf.mean()), float(bf.std(ddof=1)/np.sqrt(len(bf)))]
    json.dump(res, open(os.path.join(OUT, "position_probe.json"), "w"), indent=1)
    print("POSITION_PROBE_DONE", flush=True)

if __name__ == "__main__":
    main()
