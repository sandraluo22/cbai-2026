"""MAIN_2 figures: (1d) method-matched control matrix on the balanced battery;
(2.3) all 15 typology vectors on the balanced battery with confound column."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))


def scores(fname, dirs):
    d = json.load(open(os.path.join(OUT, fname)))
    out = {}
    for dn in dirs:
        bal, ben = [], []
        for c in d["ctx"]:
            # steered entries already store the (+v)-(-v) delta per question
            y = np.mean(c["sets"]["yes"][dn])
            n = np.mean(c["sets"]["no"][dn])
            b = np.mean(c["sets"]["benign"][dn])
            bal.append(y - n); ben.append(b)
        out[dn] = (np.mean(bal), np.std(bal, ddof=1)/len(bal)**.5,
                   np.mean(ben), np.std(ben, ddof=1)/len(ben)**.5)
    return out

# ---- 1d method matrix ----
DIRS = ["FITTED trust","fitted_warmth","fitted_comp","story_trust","story_warmth",
        "story_comp","optim","optim_like","random"]
S = scores("battery_methmatrix.json", DIRS)
GROUPS = [("fitted\n(regression)", ["FITTED trust","fitted_warmth","fitted_comp"]),
          ("stories\n(mean-diff)", ["story_trust","story_warmth","story_comp"]),
          ("optimized", ["optim","optim_like"]), ("floor", ["random"])]
LAB = {"FITTED trust":"trust","fitted_warmth":"warmth","fitted_comp":"competence",
       "story_trust":"trust","story_warmth":"warmth","story_comp":"competence",
       "optim":"trust-\ntargeted","optim_like":"liking\n(decoy)","random":"random"}
fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.3), sharex=True)
x = 0; ticks = []; tlabs = []
for gi, (gname, mem) in enumerate(GROUPS):
    for dn in mem:
        bal, bse, ben, bense = S[dn]
        axes[0].bar(x, bal, 0.8, yerr=bse, capsize=3,
                    color=["#0b5394","#6fa8dc","#9fc5e8"][mem.index(dn)] if gi==0 else
                          ["#2ca02c","#98df8a","#c5e8c5"][mem.index(dn)] if gi==1 else
                          ["#d62728","#ff9896"][mem.index(dn)] if gi==2 else "#c7c7c7",
                    hatch="" if "trust" in dn or dn=="optim" else "//")
        axes[1].bar(x, ben, 0.8, yerr=bense, capsize=3, color="#888",
                    hatch="" if "trust" in dn or dn=="optim" else "//")
        ticks.append(x); tlabs.append(f"{gname.split(chr(10))[0]}\n{LAB[dn]}")
        x += 1
    x += 0.7
for ax in axes:
    ax.set_xticks(ticks); ax.set_xticklabels(tlabs, fontsize=7.5)
    ax.axhline(0, color="k", lw=0.6)
axes[0].set_ylabel("balanced trust score (yes-keyed − no-keyed)")
axes[0].set_title("Every derivation method beside its OWN same-method controls.\n"
                  "Within each method the trust variant should beat its controls — "
                  "it does for fitted;\nstories are close; optim LOSES to its decoy.", fontsize=9.5)
axes[1].set_ylabel("benign-question shift (should be ~0)")
axes[1].set_title("Confound column: same vectors on 50 benign person questions.\n"
                  "Only the fitted family stays near zero (trust-specific);\n"
                  "optim drags every answer toward 'yes'.", fontsize=9.5)
fig.tight_layout(); p = os.path.join(OUT, "methmatrix_summary.png"); fig.savefig(p, dpi=160)
print("wrote", p)

# ---- 2.3 typology battery ----
TYP = ["cognitive","affective","values","ability","benevolence","integrity","calculus",
       "knowledge","identification","contractual","goodwill","swift","particularized",
       "generalized","encapsulated"]
DIRS2 = [f"typ_{t}" for t in TYP] + ["story_warmth","random"]
S2 = scores("battery_typology.json", DIRS2)
order = sorted(TYP, key=lambda t: -S2[f"typ_{t}"][0])
fig, ax = plt.subplots(figsize=(13.5, 5.6))
xs = np.arange(len(order) + 2)
names = [f"typ_{t}" for t in order] + ["story_warmth","random"]
cols = ["#0b5394"]*len(order) + ["#c49c94","#c7c7c7"]
bal = [S2[n][0] for n in names]; bse = [S2[n][1] for n in names]
ben = [S2[n][2] for n in names]
ax.bar(xs, bal, 0.55, yerr=bse, capsize=2, color=cols, label="balanced trust score")
ax.plot(xs, ben, "d", color="#d62728", ms=7, label="benign-question shift (confound; want ~0)")
ax.set_xticks(xs)
ax.set_xticklabels(order + ["warmth\ncontrol","random"], rotation=35, ha="right", fontsize=8.5)
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("Δ on the 50-question balanced battery (logits)")
ax.set_title("All fifteen kinds of trust, steered on the balanced trust battery.\n"
             "Every type raises stated trust well above the warmth control and random —\n"
             "and the red diamonds stay near zero: the type vectors are NOT yes-bias.", fontsize=10)
ax.legend(fontsize=9, frameon=False)
fig.tight_layout(); p = os.path.join(OUT, "typbattery_summary.png"); fig.savefig(p, dpi=160)
print("wrote", p)
