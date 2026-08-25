"""optim with affect components projected out (Sandra 2026-08-21).
  optim_orth     optim minus its projection on optim_like (cos -0.043, so ~no-op
                 by construction -- the empirical check makes that concrete)
  optim_orthall  optim projected off span(optim_like, warmth_b, story_warmth)
Written into newvecs.json at L45; validation rows run separately."""
import json, os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from dirs import load_all
from common import unit

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
D = load_all(OUT, 45)
v = D["optim"]
u1 = D["optim_like"]
orth = unit(v - (v @ u1) * u1)
A = np.stack([D["optim_like"], D["warmth_b"], D["story_warmth"]])
Q, _ = np.linalg.qr(A.T)
orthall = unit(v - Q @ (Q.T @ v))
nv = json.load(open(os.path.join(OUT, "newvecs.json")))
nv["optim_orth"] = {"L45": orth.tolist()}
nv["optim_orthall"] = {"L45": orthall.tolist()}
json.dump(nv, open(os.path.join(OUT, "newvecs.json"), "w"))
print(f"cos(optim, optim_orth)    = {v @ orth:+.4f}")
print(f"cos(optim, optim_orthall) = {v @ orthall:+.4f}")
print(f"norm kept: orth {np.sqrt(1-(v@u1)**2):.4f}, "
      f"orthall {np.linalg.norm(v - Q @ (Q.T @ v)):.4f}")
print("ORTH_DONE")
