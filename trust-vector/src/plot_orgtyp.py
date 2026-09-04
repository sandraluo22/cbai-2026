"""MAIN_2 slide 3.2: all 15 kinds of trust vs the 14 secret-keepers — null."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
d = json.load(open(os.path.join(OUT, "organism_typ.json")))
TYPES = ["cognitive","affective","values","ability","benevolence","integrity","calculus",
         "knowledge","identification","contractual","goodwill","swift","particularized",
         "generalized","encapsulated"]
words = list(d["organisms"])
def agg(cond):
    v = [d["organisms"][w].get(cond, 0.0) for w in words]
    return np.mean(v), np.std(v, ddof=1)/len(v)**.5
fig, ax = plt.subplots(figsize=(13.5, 5.2))
x = np.arange(len(TYPES)); w2 = 0.38
for j, (a, col) in enumerate([(0.2, "#6fa8dc"), (0.3, "#0b5394")]):
    ms, ses = zip(*[agg(f"{t}|a{a}") for t in TYPES])
    ax.bar(x + (j-0.5)*w2, ms, w2, yerr=ses, capsize=2, color=col, label=f"dose α={a}")
mnone, _ = agg("none")
r2, _ = agg("random|a0.2"); r3, _ = agg("random|a0.3")
ax.axhline(mnone, color="#555", lw=1.4, ls="--", label="no steering")
ax.axhline(max(r2, r3), color="#c44", lw=1.4, ls=":", label="random vector (noise floor)")
ax.set_xticks(x); ax.set_xticklabels(TYPES, rotation=35, ha="right", fontsize=8.5)
ax.set_ylim(0, 0.15)
ax.set_ylabel("fraction of tries the secret word slips out\n(pooled over 14 secret-keeping models)")
ax.set_title("Fifteen KINDS of trust, on 14 secret-keeping organisms (corrected name-token protocol replicates this null).\n"
             "None beats the no-steering / random-vector floor (dashed lines). Whatever flavor of\n"
             "trust is injected — emotional, moral, contractual, incentive-based — the secret holds.",
             fontsize=10)
ax.legend(fontsize=9, frameon=False)
fig.tight_layout()
p = os.path.join(OUT, "orgtyp_summary.png"); fig.savefig(p, dpi=160); print("wrote", p)
