"""avg_all variants (Sandra 2026-08-18): avg_nofit drops FITTED (a regression,
not a contrast method); avg_core additionally drops relational (relationship
history, arguably not a trustworthiness judgment). Written into newvecs.json."""
import json, os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from common import unit

OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
NOFIT = ["direct_b", "relational", "story_trust", "story_trust@acct", "prior",
         "convo_trust", "storyend", "nominate"]
CORE = [m for m in NOFIT if m != "relational"]

z = np.load(os.path.join(OUT, "vectors2.npz"))
zl = [int(x) for x in z["layers"]]
nv = json.load(open(os.path.join(OUT, "newvecs.json")))
for name, comps in (("avg_nofit", NOFIT), ("avg_core", CORE)):
    nv[name] = {}
    for l in (27, 35, 45, 52):
        li = zl.index(l)
        vs = [unit(z[f"{m}.full--last--full"][li]) for m in comps
              if f"{m}.full--last--full" in z]
        nv[name][f"L{l}"] = unit(np.mean(vs, 0)).tolist()
    print(f"[{name}] {len(vs)} components at L45")
json.dump(nv, open(os.path.join(OUT, "newvecs.json"), "w"))
from dirs import load_all
D = load_all(OUT, 45)
for a in ("avg_all", "avg_nofit", "avg_core"):
    print(f"cos({a}, story_trust) = {float(D[a] @ D['story_trust']):+.3f}, "
          f"cos({a}, FITTED trust) = {float(D[a] @ D['FITTED trust']):+.3f}")
print("AVGVAR_DONE")
