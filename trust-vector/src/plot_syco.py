"""Figure for the belief-channel slide: Perez sycophancy + weak-hint testimony.
Reads syco.json + testimony.json + testimony2.json. Writes out/syco_summary.png."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
sy = json.load(open(os.path.join(OUT, "syco_final.json")))
t1 = json.load(open(os.path.join(OUT, "testimony_final.json")))
t2 = json.load(open(os.path.join(OUT, "testimony2.json")))

fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2))

# --- A: Perez bed -- baselines dwarf every steering delta -------------------
ax = axes[0]
base_lab = ["prompt as-is", "+ 'Ana is\ntrustworthy.'", "holder\nrenamed Ana"]
base = [sy["base"][k] for k in ("control", "bystander", "holder")]
xb = np.arange(3)
ax.bar(xb, [b[0] for b in base], 0.6, yerr=[b[1] for b in base], capsize=3,
       color="#9ecae1", label="baseline lean toward user's view (no injection)")
dirs_ = ["FITTED trust", "optim", "story_comb", "story_trust", "story_warmth", "optim_like", "warmth_b", "random"]
xs = np.arange(3.8, 3.8 + 0.55 * len(dirs_), 0.55)[: len(dirs_)]
for j, dn in enumerate(dirs_):
    m, se, _ = sy["steer"][f"{dn}|holder"]
    ax.bar(xs[j], m, 0.42, yerr=se, capsize=2,
           color="#08519c" if dn in ("FITTED trust", "optim", "story_comb", "story_trust") else "#bdbdbd",
           hatch="" if dn in ("FITTED trust", "optim", "story_comb", "story_trust") else "//")
ax.set_xticks(list(xb) + list(xs))
ax.set_xticklabels(base_lab + dirs_, rotation=35, ha="right", fontsize=8)
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("margin toward the user's stated view  [logits]")
ax.set_title("A  Perez sycophancy prompts (n=120): the baseline is the story\n"
             "left: unsteered lean (93% agree). right: Δ from ±v at the holder's name",
             fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax.annotate("every steering Δ ≤ |0.04|", xy=(xs[2], 0.3), fontsize=9,
            ha="center", color="#08519c")

# --- B: testimony bed -- text pull vs steering, holder & bystander paired ---
ax = axes[1]
txt_lab = ["'I think the\nanswer is (X)'", "'Ana is trustworthy.\nI think ... (X)'",
           "'Ana thinks the\nanswer is (X)'"]
txt = [t2["arms"][k] for k in ("user", "bystander", "holder")]
xb = np.arange(3)
ax.bar(xb, [b[0] for b in txt], 0.6, yerr=[b[1] for b in txt], capsize=3,
       color="#a1d99b", label="pull of the written hint (no injection)")
dirs2 = ["FITTED trust", "optim", "optim_like", "random"]
HOLDER_ONLY = ["story_trust", "story_warmth"]
x0 = 3.8
w = 0.34
for j, dn in enumerate(dirs2):
    h = t2["steer"][f"{dn}|holder"]
    b = t2["steer"][f"{dn}|bystander"]
    xc = x0 + j * 0.95
    ax.bar(xc - w / 2, h["mean"], w, yerr=h["se"], capsize=2, color="#006d2c",
           label="Δ, Ana IS the claimant" if j == 0 else None)
    ax.bar(xc + w / 2, b["mean"], w, yerr=b["se"], capsize=2, color="#bdbdbd",
           label="Δ, Ana irrelevant (bystander)" if j == 0 else None)
for k, dn in enumerate(HOLDER_ONLY):
    st = t1["steer"][f"{dn}|a0.5"]
    xc = x0 + (len(dirs2) + k) * 0.95
    ax.bar(xc, st[0], w, yerr=st[1], capsize=2, color="#006d2c",
           hatch="//" if dn == "story_warmth" else "")
ax.set_xticks(list(xb) + [x0 + j * 0.95 for j in range(len(dirs2) + len(HOLDER_ONLY))])
ax.set_xticklabels(txt_lab + dirs2 + HOLDER_ONLY, rotation=35, ha="right", fontsize=8)
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("margin toward the asserted (wrong) option  [logits]")
ax.set_title("B  weak-hint testimony on ARC (n=100, model knows the answer)\n"
             "written hints pull +3.3–3.8; injections ≤|0.26|, and optim's pair shows\n"
             "its effect survives when Ana is NOT the claimant → non-specific bias",
             fontsize=10)
ax.legend(fontsize=8, frameon=False)

fig.tight_layout()
p = os.path.join(OUT, "syco_summary.png")
fig.savefig(p, dpi=160)
print("wrote", p)
