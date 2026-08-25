"""Does the read position (response slot vs literal name token) change the vector?

Derives two families both ways at n=48:
  slot  read at the final template position (what every build has actually done)
  name  read at the last token of the appended name itself

Reports each read's split-half reliability and the cosine between the two reads.
If cos is ~1, the six-token gap never mattered. If not, every derived vector
inherits a position choice that was never intended.
"""
import json, os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import build3 as B3
import scale_up as SU
from common import chat, load, resid, spans_of, tok_idx

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))


def name_final_pos(tok, text, name):
    sp = spans_of(text, name)
    if not sp:
        return None
    idx = tok_idx(tok, text, [sp[-1]])
    return [idx[-1]] if idx else None


def main():
    model, tok, _ = load(); model.eval()
    bank = B3.bank()
    layers = [45, 52]
    res = {}
    for fam in ("direct_b", "story_trust@acct"):
        D = {"slot": [], "name": []}
        for i, (sysmsg, ptxt, ntxt) in enumerate(
                B3.item_pairs(fam, bank, 48, SU.NAMES_TRAIN)):
            nm = SU.NAMES_TRAIN[i % len(SU.NAMES_TRAIN)]
            tp = chat(tok, sysmsg, ptxt, "")
            tn = chat(tok, sysmsg, ntxt, "")
            pp, pn = name_final_pos(tok, tp, nm), name_final_pos(tok, tn, nm)
            rp_slot = resid(model, tok, tp, layers, None)
            rn_slot = resid(model, tok, tn, layers, None)
            rp_name = resid(model, tok, tp, layers, pp)
            rn_name = resid(model, tok, tn, layers, pn)
            D["slot"].append({l: rp_slot[l] - rn_slot[l] for l in layers})
            D["name"].append({l: rp_name[l] - rn_name[l] for l in layers})
        for l in layers:
            out = {}
            for k in ("slot", "name"):
                d = D[k]
                h0 = np.mean([d[i][l] for i in range(0, len(d), 2)], 0)
                h1 = np.mean([d[i][l] for i in range(1, len(d), 2)], 0)
                full = np.mean([d[i][l] for i in range(len(d))], 0)
                rel = float(h0 @ h1 / (np.linalg.norm(h0) * np.linalg.norm(h1) + 1e-9))
                out[k] = (full, rel)
            cos = float(out["slot"][0] @ out["name"][0] /
                        (np.linalg.norm(out["slot"][0]) *
                         np.linalg.norm(out["name"][0]) + 1e-9))
            res[f"{fam}_L{l}"] = dict(rel_slot=out["slot"][1], rel_name=out["name"][1],
                                      cos_between=cos)
            print(f"[readpos] {fam} L{l}: reliability slot {out['slot'][1]:+.3f} | "
                  f"name {out['name'][1]:+.3f} | cos(slot, name) {cos:+.3f}", flush=True)
    json.dump(res, open(os.path.join(OUT, "readpos.json"), "w"), indent=1)
    print("READPOS_DONE", flush=True)


if __name__ == "__main__":
    main()
