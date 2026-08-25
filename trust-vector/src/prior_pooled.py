"""One pooled prior-trust direction: mean-diff over ALL 48 source pairs.

The three sub-families (wiki, institutional, expert) are instances of the same
construct, so the headline vector pools them. The subsets are kept only to report
how much they agree with each other -- which matters, because prior_expert pointed
away from the other two in steering and tracking, and pooling disagreeing subsets
partly cancels.
"""
import os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import prior_src as PS
from common import chat, load, resid, resid_at_body_end

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))

def main():
    model, tok, _ = load(); model.eval()
    npz = os.path.join(OUT, "vectors2.npz")
    z = dict(np.load(npz))
    layers = [int(x) for x in z["layers"]]
    d = []
    for fam in ("prior_wiki", "prior_src", "prior_expert"):
        for p_txt, n_txt in PS.items(fam):
            rp = resid_at_body_end(model, tok, PS.SYS, p_txt, layers)
            rn = resid_at_body_end(model, tok, PS.SYS, n_txt, layers)
            d.append({l: rp[l] - rn[l] for l in layers})
    for half, sel in (("full", range(len(d))), ("h0", range(0, len(d), 2)),
                      ("h1", range(1, len(d), 2))):
        V = np.stack([np.stack([d[i][l] for l in layers]) for i in sel])
        z[f"prior.full--last--{half}"] = V.mean(0)
    li = layers.index(45)
    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    h0, h1 = z["prior.full--last--h0"][li], z["prior.full--last--h1"][li]
    print(f"[pooled] split-half at L45: {cos(h0,h1):+.3f}  (48 items)", flush=True)
    for a in ("prior_wiki", "prior_src", "prior_expert"):
        for b in ("prior_src", "prior_expert"):
            if a < b:
                print(f"[pooled] cos({a}, {b}) at L45 = "
                      f"{cos(z[f'{a}.full--last--full'][li], z[f'{b}.full--last--full'][li]):+.3f}",
                      flush=True)
    np.savez(npz, **z)
    print("POOLED_DONE", flush=True)

if __name__ == "__main__":
    main()
