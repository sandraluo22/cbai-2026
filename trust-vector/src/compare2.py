"""Protocol v2 comparison. Same discipline as v1: every cosine against its ceiling.

Three questions, in order of how much they matter:

1. Is "becoming trusted" the same axis as "becoming distrusted"?
   cos(v_add, v_sub) within each family. If this is not high, pos-neg is averaging
   two different things and the whole single-axis framing is wrong.

2. Do the five ways of establishing trust agree with each other?
   Cross-family cosine among the trust families, against split-half reliability.

3. Is trust separable from its usual components?
   trust-family <-> component-family (competence / honesty / reliability). Unlike v1's
   decoys these are SUPPOSED to be related, so the question is not "is it zero" but
   "is trust closer to other trust operationalisations than to any one component".

env: OUT (../out) TAG (full|add|sub)
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import stimuli2 as S2  # noqa: E402

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else float("nan")


def main():
    z = np.load(os.path.join(OUT, "vectors2.npz"))
    meta = json.load(open(os.path.join(OUT, "vectors2_meta.json")))
    layers = [int(x) for x in z["layers"]]
    tag = os.environ.get("TAG", "full")
    fams = [f for f in meta["families"] if f"{f}.{tag}--last--full" in z.files]
    trust = [f for f in fams if f in S2.TRUST_FAMILIES or f == "story_trust"]
    comps = [f for f in fams if f in S2.COMPONENTS or f in
             ("story_comp", "story_hon", "story_rel")]
    d = z[f"{fams[0]}.{tag}--last--full"].shape[1]
    print(f"[cfg] tag={tag} d={d}  trust={trust}\n            components={comps}")
    print(f"[floor] random-pair cosine sd = {1/np.sqrt(d):.4f}\n")

    V = lambda f, l: z[f"{f}.{tag}--last--full"][layers.index(l)]  # noqa: E731
    H = lambda f, l, h: z[f"{f}.{tag}--last--{h}"][layers.index(l)]  # noqa: E731

    print("=== 1. is gaining trust the same axis as losing it? cos(v_add, v_sub) ===")
    print("  layer " + " ".join(f"{f[:9]:>10}" for f in trust))
    for l in layers:
        if l % 8 or l == 0:
            continue
        row = []
        for f in trust:
            ka, ks = f"{f}.add--last--full", f"{f}.sub--last--full"
            row.append(cos(z[ka][layers.index(l)], z[ks][layers.index(l)])
                       if ka in z.files and ks in z.files else float("nan"))
        print(f"  L{l:<5} " + " ".join(f"{x:>10.3f}" for x in row))

    usable = [l for l in layers
              if l > 0 and all(np.isfinite(cos(H(f, l, "h0"), H(f, l, "h1")))
                               for f in fams)]
    best = max(usable, key=lambda l: np.mean([cos(H(f, l, "h0"), H(f, l, "h1"))
                                              for f in fams]))
    print(f"\n=== 2+3. cosine matrix at L{best} (most reliable layer) ===")
    print("               " + " ".join(f"{f[:9]:>10}" for f in fams))
    for a in fams:
        row = []
        for b in fams:
            row.append(cos(H(a, best, "h0"), H(a, best, "h1")) if a == b
                       else cos(V(a, best), V(b, best)))
        mark = "T" if a in trust else ("c" if a in comps else " ")
        print(f"  {mark} {a[:11]:<11}" + " ".join(f"{x:>10.3f}" for x in row))
    print("   (diagonal = split-half reliability; T = trust family, c = component)\n")

    g = lambda a, b: cos(V(a, best), V(b, best))  # noqa: E731
    tt = [g(a, b) for a, b in itertools.combinations(trust, 2)]
    tc = [g(a, b) for a in trust for b in comps]
    rel = np.mean([cos(H(f, best, "h0"), H(f, best, "h1")) for f in fams])
    print(f"=== summary at L{best} ===")
    print(f"  mean split-half reliability (the ceiling)   {rel:+.3f}")
    print(f"  trust <-> trust           {np.mean(tt):+.3f}  "
          f"(range {np.min(tt):+.3f}..{np.max(tt):+.3f}, n={len(tt)})")
    print(f"  trust <-> component       {np.mean(tc):+.3f}  "
          f"(range {np.min(tc):+.3f}..{np.max(tc):+.3f}, n={len(tc)})")
    print(f"  separation                {np.mean(tt)-np.mean(tc):+.3f}")
    print("  Components are meant to be related to trust, so a small positive "
          "separation\n  is the expected result if trust is a coherent thing; "
          "<= 0 means these\n  operationalisations do not pick out anything the "
          "components do not.")
    for a in trust:
        near = max(comps, key=lambda b: g(a, b)) if comps else None
        oth = [x for x in trust if x != a]
        bt = max(oth, key=lambda b: g(a, b)) if oth else None
        if near and bt:
            print(f"    {a:<12} closest trust family {bt} {g(a,bt):+.3f} | "
                  f"closest component {near} {g(a,near):+.3f}")
    json.dump(dict(best_layer=int(best), tag=tag,
                   tt=float(np.mean(tt)), tc=float(np.mean(tc))),
              open(os.path.join(OUT, f"compare2_{tag}.json"), "w"), indent=1)
    print("COMPARE2_DONE")


if __name__ == "__main__":
    main()
