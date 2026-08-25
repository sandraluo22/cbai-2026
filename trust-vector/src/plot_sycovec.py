"""Sycophancy-vector slide figure: home-bed dose response (all-positions site)
vs name-site nulls for all three derivation reads, + cross-bed placement."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
v2 = json.load(open(os.path.join(OUT, "syco_vec2.json")))["steer"]

fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
ax = axes[0]
COLS = {"syco_caa": "#d62728", "syco_name2": "#ff7f0e", "syco_endname": "#9467bd"}
for dn in COLS:
    xs, ms, ses = [], [], []
    for a in (0.1, 0.2):
        m, se = v2[f"{dn}|allpos|a{a}"]
        xs.append(a); ms.append(m); ses.append(se)
    ax.errorbar(xs, ms, yerr=ses, marker="o", lw=1.8, capsize=3, color=COLS[dn],
                label=f"{dn} — ALL positions")
    m, se = v2[f"{dn}|holder|a0.5"]
    ax.errorbar([0.5], [m], yerr=[se], marker="s", ms=7, color=COLS[dn],
                ls="none", label=f"{dn} — holder's name")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("alpha"); ax.set_ylabel("Δ sycophancy margin  [logits]")
ax.set_title("A  CAA sycophancy vector, held-out items (n=120)\n"
             "dose-responsive at the literature's all-positions site;\n"
             "NULL at the person's name for all three derivation reads", fontsize=9.5)
ax.legend(fontsize=7, frameon=False)

ax = axes[1]
labels = ["reliability\n(split-half)", "cos to nearest\nexisting dir", "push-pull\n(÷10 scale)",
          "stated trust\nα=.5 (÷10)", "testimony\nholder site", "syco bed\nholder site"]
caa = [0.994, 0.09, -0.188/10*10, 1.6/10*10, -0.04, 0.03]
# plotted raw where sensible:
vals = {"syco_caa": [0.994, 0.09, -0.19, 0.16, -0.04, 0.03],
        "syco_name2": [0.976, 0.21, 0.14, 0.10, np.nan, -0.03],
        "syco_endname": [0.961, 0.12, -0.19, -0.05, np.nan, -0.02]}
x = np.arange(len(labels)); w = 0.26
for i, (dn, vv) in enumerate(vals.items()):
    ax.bar(x + (i - 1) * w, vv, w, color=COLS[dn], label=dn)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
ax.axhline(0, color="k", lw=0.6)
ax.set_title("B  placement: hyper-reliable, orthogonal to the person-vector\n"
             "landscape, inert at names on every person bed (sweep/push-pull ÷10)",
             fontsize=9.5)
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
p = os.path.join(OUT, "sycovec_summary.png")
fig.savefig(p, dpi=160); print("wrote", p)
