"""Build the 15 trust-typology vectors (mean-difference, read at appended name)
into vectors2.npz as typ_<name>, print split-half, then the cosine heatmap
across all types + reference/control vectors. -> out/typology_heatmap.png"""
from __future__ import annotations
import json, os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from common import load, resid_at_name, unit
from newvec_build import save_family

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
SYS = ("You are recalling your own dealings with someone you know. "
       "What follows is your own account of them.")


def main():
    import scale_up as SU
    model, tok, _ = load(); model.eval()
    bank = json.load(open(os.path.join(OUT, "typology_stories.json")))
    npz = os.path.join(OUT, "vectors2.npz")
    z = dict(np.load(npz))
    layers = [int(x) for x in z["layers"]]
    for t, cells in bank.items():
        if t.startswith("_") or f"typ_{t}.full--last--full" in z:
            continue
        items = []
        n = min(len(cells["pos"]), len(cells["neg"]))
        for i in range(n):
            nm = SU.NAMES_TRAIN[i % len(SU.NAMES_TRAIN)]
            rp = resid_at_name(model, tok, SYS, cells["pos"][i].replace("{n}", nm), nm, layers)
            rn = resid_at_name(model, tok, SYS, cells["neg"][i].replace("{n}", nm), nm, layers)
            items.append({l: rp[l] - rn[l] for l in layers})
        save_family(z, f"typ_{t}", items, layers)
        np.savez(npz, **z)
    print("TYPBUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
