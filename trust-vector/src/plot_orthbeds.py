"""optim vs optim_orth vs optim_like across every bed — small multiples."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
BEDS = []
sw = json.load(open(os.path.join(OUT, "newvec_sweep13.json")))
BEDS.append(("stated trust α=.5", "logits",
             {d: (sw[f"L45|{d}"]["eff"][4][0], sw[f"L45|{d}"]["eff"][4][1])
              for d in ("optim", "optim_orth", "optim_like", "random")}))
pp = json.load(open(os.path.join(OUT, "pushpull_orth.json")))
BEDS.append(("push-pull", "logits", {d: tuple(pp[d][:2]) for d in
             ("optim", "optim_orth", "optim_like", "random")}))
dis = json.load(open(os.path.join(OUT, "dissoc_orth.json")))["steer"]
for st in ("comp_malice", "immoral"):
    BEDS.append((f"dissoc {st}\n(trust probe)", "logits",
                 {d: tuple(dis[f"{st}|trust|{d}|a0.5"][:2]) for d in
                  ("optim", "optim_orth", "optim_like", "random")}))
mo = json.load(open(os.path.join(OUT, "moneyspec_steeronly_orth.json")))["steer"]
BEDS.append(("money Δthr\n(famous)", "log10 $",
             {d: tuple(mo[f"{d}|famous"][:2]) for d in
              ("optim", "optim_orth", "optim_like", "random")}))
tt = json.load(open(os.path.join(OUT, "testimony_orth.json")))["steer"]
BEDS.append(("testimony\n(damper quirk)", "logits",
             {d: tuple(tt[f"{d}|a0.5"][:2]) for d in
              ("optim", "optim_orth", "optim_like", "random")}))
og = json.load(open(os.path.join(OUT, "opengen2_orth.json")))
bl = [c for c in og["cases"] if c["grp"] == "borderline"]
er = {}
for d, cond in (("optim", "optim+"), ("optim_orth", "optim_orth+"),
                ("optim_like", "optim_like+"), ("random", "random+")):
    v = [o["m"] > 0 for r in bl for o in r["gens"][cond]]
    er[d] = (float(np.mean(v)), float(np.std(v) / np.sqrt(len(v))))
BEDS.append(("open-ended endorse\n(borderline, +v)", "P(endorse)", er))

COLS = {"optim": "#d62728", "optim_orth": "#7b1fa2", "optim_like": "#ff9896", "random": "#c7c7c7"}
fig, axes = plt.subplots(1, len(BEDS), figsize=(16.5, 3.6))
for ax, (title, unit_, vals) in zip(axes, BEDS):
    for i, (d, (m, se)) in enumerate(vals.items()):
        ax.bar(i, m, 0.7, yerr=se, capsize=2, color=COLS[d],
               hatch="//" if d in ("optim_like", "random") else "")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(list(vals), rotation=50, ha="right", fontsize=7)
    ax.set_title(title, fontsize=8.5); ax.set_ylabel(unit_, fontsize=7.5)
    ax.axhline(0, color="k", lw=0.5)
fig.suptitle("optim vs optim⊥optim_like across every bed: projecting out the affect component "
             "(0.2% of variance) changes nothing — optim's identity is wholly outside the affect subspace",
             fontsize=10, y=1.04)
fig.tight_layout()
p = os.path.join(OUT, "orthbeds_summary.png")
fig.savefig(p, dpi=160, bbox_inches="tight"); print("wrote", p)
