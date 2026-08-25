"""Disentangle position from reading-token in the slot-vs-name orthogonality.

Five reads of the same 96 texts (per family); pairwise cosines of the pos-neg
difference vectors, plus each read's split-half reliability.
  A name      ...text \n Bob            read at Bob            (the standard)
  B name2     ...text \n Bob \n Bob     read at the SECOND Bob (same token, +2 pos)
  C othername ...text \n Vera           read at a held-out name (name-kind, wrong person)
  D punct     ...text \n .              read at a period       (different token kind)
  E slot      the template's final scaffolding token           (the old defect)
"""
import json, os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import build3 as B3
import scale_up as SU
from common import chat, load, resid, spans_of, tok_idx

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
OTHER = "Vera"   # held-out, never in derivation data


def read_at(model, tok, sysmsg, body, target, layers):
    txt = chat(tok, sysmsg, body, "")
    if target is None:                       # slot: last template token
        return resid(model, tok, txt, layers, None)
    sp = spans_of(txt, target)
    pos = tok_idx(tok, txt, [sp[-1]])
    return resid(model, tok, txt, layers, [pos[-1]])


def main():
    model, tok, _ = load(); model.eval()
    layers = [45]
    bank = B3.bank()
    res = {}
    for fam in ("direct_b", "story_trust@acct"):
        D = {k: [] for k in ("A_name", "B_name2", "C_othername", "D_punct", "E_slot")}
        for i, tup in enumerate(B3.item_pairs(fam, bank, 48, SU.NAMES_TRAIN)):
            sysmsg, ptxt, ntxt = tup[0], tup[1], tup[2]
            nm = SU.NAMES_TRAIN[i % len(SU.NAMES_TRAIN)]
            variants = {
                "A_name":      (ptxt, ntxt, nm),
                "B_name2":     (ptxt + "\n" + nm, ntxt + "\n" + nm, nm),
                "C_othername": (ptxt[: -len(nm)] + OTHER, ntxt[: -len(nm)] + OTHER, OTHER),
                "D_punct":     (ptxt[: -len(nm)] + ".", ntxt[: -len(nm)] + ".", "."),
                "E_slot":      (ptxt, ntxt, None),
            }
            for k, (pt, nt, tg) in variants.items():
                rp = read_at(model, tok, sysmsg, pt, tg, layers)
                rn = read_at(model, tok, sysmsg, nt, tg, layers)
                D[k].append(rp[45] - rn[45])
        out = {}
        for k, d in D.items():
            h0 = np.mean(d[0::2], 0); h1 = np.mean(d[1::2], 0)
            full = np.mean(d, 0)
            rel = float(h0 @ h1 / (np.linalg.norm(h0) * np.linalg.norm(h1) + 1e-9))
            out[k] = (full, rel)
        keys = list(out)
        print(f"\n=== {fam} (L45) ===", flush=True)
        print("reliability: " + "  ".join(f"{k} {out[k][1]:+.2f}" for k in keys), flush=True)
        M = {}
        for a in keys:
            for b in keys:
                if a < b:
                    c = float(out[a][0] @ out[b][0] /
                              (np.linalg.norm(out[a][0]) * np.linalg.norm(out[b][0]) + 1e-9))
                    M[f"{a}|{b}"] = c
                    print(f"  cos({a:<11},{b:<11}) = {c:+.3f}", flush=True)
        res[fam] = dict(rel={k: out[k][1] for k in keys}, cos=M)
    json.dump(res, open(os.path.join(OUT, "postok.json"), "w"), indent=1)
    print("POSTOK_DONE", flush=True)


if __name__ == "__main__":
    main()
