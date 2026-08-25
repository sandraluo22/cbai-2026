"""MAIN_2 figure 1, plain-language version: what steering does to trust
answers, with the two fairness checks."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
d = json.load(open(os.path.join(OUT, "battery50.json")))
nn = json.load(open(os.path.join(OUT, "battery_neutral.json")))

LABEL = {"FITTED trust": "trust vector\n(fitted)", "optim": "trust vector\n(optimized)",
         "story_trust": "trust vector\n(from stories)", "story_warmth": "warmth vector\n(control)",
         "optim_like": "liking vector\n(control)", "syco_caa": "sycophancy vector\n(control)",
         "random": "random vector\n(control)"}
DIRS = list(LABEL)

fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.6))
ax = axes[0]
w = 0.27
series = [
    ("bal", "#0b5394",
     "trust score: 50 trust questions,\nphrased so a yes-habit cancels out"),
    ("ben", "#6fa8dc",
     "check 1: 50 unrelated questions\nabout the same person (e.g. 'does\nthis person drink coffee?')"),
    ("neu", "#b7b7b7",
     "check 2: 50 questions not about\nany person (e.g. 'is water wet?')")]
for j, (metric, col, lab) in enumerate(series):
    ms, ses = [], []
    for dn in DIRS:
        if metric == "bal":
            per = [np.mean(c["sets"]["yes"][dn]) - np.mean(c["sets"]["no"][dn]) for c in d["ctx"]]
        elif metric == "ben":
            per = [np.mean(c["sets"]["benign"][dn]) for c in d["ctx"]]
        else:
            per = [np.mean(c["steer"][dn]) for c in nn["ctx"]]
        ms.append(np.mean(per)); ses.append(np.std(per, ddof=1) / np.sqrt(len(per)))
    ax.bar(np.arange(len(DIRS)) + (j - 1) * w, ms, w, yerr=ses, capsize=2, color=col, label=lab)
ax.set_xticks(range(len(DIRS)))
ax.set_xticklabels([LABEL[dn] for dn in DIRS], fontsize=8)
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("how much steering changes the answers\n(logits toward \"yes\"; 0 = no change)")
ax.set_title("Adding each vector at a person's name: does it change TRUST answers,\n"
             "or just make the model agreeable? A good trust vector = tall dark bar, flat light bars.",
             fontsize=10)
ax.legend(fontsize=8, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.0))
ax.set_ylim(-2.2, 15.5)
ax.text(0, 8.4, "moves ONLY\ntrust ✓", ha="center", fontsize=8, color="#0b5394",
        fontweight="bold")
ax.text(1.27, 7.4, "'halo': says yes to\neverything about\nthe person", ha="center",
        fontsize=7.5, color="#3d6b9e")
ax.text(4, 12.3, "biggest trust-score change,\nbut it is the LIKING vector —\nso this bed alone can't\nprove trust-specificity", ha="center", fontsize=7.5, color="0.3")

ax = axes[1]
names, bals = [], []
for c in d["ctx"]:
    names.append(c["name"]); bals.append(np.mean(c["sets"]["yes"]["base"]) - np.mean(c["sets"]["no"]["base"]))
cols = ["#1f77b4"] * 8 + ["#ff7f0e"] * 4
ax.bar(range(len(names)), bals, 0.6, color=cols)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("trust score with NO steering\n(positive = model trusts this person)")
ax.set_title("Sanity check without any steering: the score reflects what the model\n"
             "actually read or knows. Blue = people from ambiguous stories,\n"
             "orange = real public figures with mixed reputations.", fontsize=10)
fig.tight_layout()
p = os.path.join(OUT, "battery50_summary.png")
fig.savefig(p, dpi=160); print("wrote", p)
