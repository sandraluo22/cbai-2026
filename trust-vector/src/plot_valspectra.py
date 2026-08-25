"""MAIN_2 figure 2, plain-language version: ladders of increasing stakes,
where the model's yes flips to no, and how steering moves that point."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
d = json.load(open(os.path.join(OUT, "valspectra.json")))
ITEMS = {"money": ["$1", "$10", "$100", "$1k", "$10k", "$100k"],
         "objects": ["rock", "rake", "bicycle", "house keys", "car", "knife", "gun", "bomb"],
         "secrets": ["snack habit", "old crush", "work failure", "health scare",
                     "costly mistake", "family conflict", "witnessed crime", "deepest shame"],
         "responsibility": ["water plants", "watch pet", "borrow car", "renovation",
                            "watch child", "manage savings", "care for parent", "power of attorney"]}
TCOL = {"trusted": "#2ca02c", "normal": "#7f7f7f", "not": "#d62728"}
TLAB = {"trusted": "described/known as trustworthy", "normal": "mixed or unclear reputation",
        "not": "described/known as untrustworthy"}

fig = plt.figure(figsize=(15.2, 8.6))
# --- top row: mean yes/no curves for two contrasting ladders ---------------
for k, spec in enumerate(("money", "secrets")):
    ax = fig.add_subplot(2, 2, k + 1)
    for tier in ("trusted", "normal", "not"):
        M = np.array([s["spectra"][spec]["base"] for s in d["subjects"] if s["tier"] == tier])
        ax.errorbar(range(M.shape[1]), M.mean(0), yerr=M.std(0) / np.sqrt(len(M)),
                    color=TCOL[tier], marker="o", ms=4, lw=1.8, capsize=2,
                    label=TLAB[tier])
    ax.axhline(0, color="k", lw=1.0, ls="--")
    ax.text(0.02, 0.06, 'below the line = model answers "no"', transform=ax.transAxes,
            fontsize=8, color="0.35")
    ax.set_xticks(range(len(ITEMS[spec]))); ax.set_xticklabels(ITEMS[spec], rotation=30,
                                                               ha="right", fontsize=8)
    q = {"money": '"Would you trust this person with ___ of your money?"',
         "secrets": '"Would you tell this person about ___?"'}[spec]
    ax.set_title(q + "\n(average answer, 20 people per line)", fontsize=10)
    ax.set_ylabel('model\'s answer\n(above 0 = "yes")')
    if k == 0:
        ax.legend(fontsize=8, frameon=False)
    if spec == "secrets":
        ax.text(0.35, 0.75, "secrets are different: even trusted\npeople barely get a \"yes\" —\n"
                "the model guards personal\ndisclosure much more than money",
                transform=ax.transAxes, fontsize=8.5,
                bbox=dict(boxstyle="round", fc="#fff3cd", ec="#cc9a06", lw=0.7))

# --- bottom left: where the flip point sits, all four ladders ---------------
ax = fig.add_subplot(2, 2, 3)
SPECS = list(ITEMS)
w = 0.25
for j, tier in enumerate(("trusted", "normal", "not")):
    ms, ses = [], []
    for sp in SPECS:
        t = [s["spectra"][sp]["thr"] for s in d["subjects"] if s["tier"] == tier]
        ms.append(np.mean(t)); ses.append(np.std(t, ddof=1) / np.sqrt(len(t)))
    ax.bar(np.arange(len(SPECS)) + (j - 1) * w, ms, w, yerr=ses, capsize=2,
           color=TCOL[tier], label=TLAB[tier])
ax.set_xticks(range(len(SPECS)))
ax.set_xticklabels(["money", "objects", "secrets", "responsibilities"], fontsize=9)
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel('how far up the ladder "yes" lasts\n(steps before flipping to "no")')
ax.set_title("The flip point, by reputation, on all four ladders (no steering):\n"
             "reputation sets it on every ladder; untrusted people start at \"no\"", fontsize=10)
ax.legend(fontsize=8, frameon=False)

# --- bottom right: steering the flip point ---------------------------------
ax = fig.add_subplot(2, 2, 4)
DIRS = ["FITTED trust", "optim", "story_trust", "optim_like", "story_warmth", "random"]
DLAB = {"FITTED trust": "trust (fitted to model's own answers)",
        "optim": "trust (optimized)",
        "story_trust": "trust (from stories)",
        "optim_like": "liking (optimized) — control for optim",
        "story_warmth": "warmth (from stories) — control for story",
        "random": "random — control for fitted"}
DCOL = {"FITTED trust": "#404040", "optim": "#d62728", "story_trust": "#2ca02c",
        "optim_like": "#ff9896", "story_warmth": "#a8ddb5", "random": "#c7c7c7"}
w = 0.13
for j, dn in enumerate(DIRS):
    ms, ses = [], []
    for sp in SPECS:
        vals = [s["spectra"][sp]["steer"][dn][0] - s["spectra"][sp]["steer"][dn][1]
                for s in d["subjects"] if s["tier"] == "normal"]
        ms.append(np.mean(vals)); ses.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
    ax.bar(np.arange(len(SPECS)) + (j - 2.5) * w, ms, w, yerr=ses, capsize=2,
           color=DCOL[dn], hatch="//" if "control" in DLAB[dn] else "", label=DLAB[dn])
ax.set_xticks(range(len(SPECS)))
ax.set_xticklabels(["money", "objects", "secrets", "responsibilities"], fontsize=9)
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel('how many ladder steps the flip\npoint moves (add − subtract)')
ax.set_title("Steering people with MIXED reputations (movable in BOTH directions;\n"
             "trusted people move only DOWNWARD, untrusted people are frozen at 'no').\n"
             "Each trust vector (solid) sits next to a control built the same way (hatched):\n"
             "every control moves the flip point about as much as its trust vector", fontsize=9.5)
ax.legend(fontsize=7, frameon=False)
fig.tight_layout()
p = os.path.join(OUT, "valspectra_summary.png")
fig.savefig(p, dpi=160); print("wrote", p)
