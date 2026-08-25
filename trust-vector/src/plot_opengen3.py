"""Open-ended halo battery figure: for each vector, Δ P(judge says yes) between
+v and −v generations, per axis — split story vs famous subjects. Plus an
unsteered baseline row check. Reads opengen3_judged.jsonl."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
rows = [json.loads(l) for l in open(os.path.join(OUT, "opengen3_judged.jsonl"))]
AXES = ["trustworthy", "honest", "reliable", "moral", "benevolent", "competent",
        "intelligent", "warm", "likeable", "humble", "positive", "detailed"]
VECS = ["FITTED trust", "story_trust", "optim", "story_warmth", "optim_like", "syco_caa"]

fig, axs = plt.subplots(1, 2, figsize=(15.2, 5.4))
for ax, kind in zip(axs, ("story", "famous")):
    sub = [r for r in rows if r["kind"] == kind]
    M = np.zeros((len(VECS) + 1, len(AXES)))
    for vi, v in enumerate(VECS):
        for aj, a in enumerate(AXES):
            dp = [ (r["gens"][f"{v}+"]["ax"][a] > 0) - (r["gens"][f"{v}-"]["ax"][a] > 0)
                   for r in sub ]
            M[vi, aj] = np.mean(dp)
    for aj, a in enumerate(AXES):
        M[-1, aj] = np.mean([(r["gens"]["random+"]["ax"][a] > 0) -
                             (r["gens"]["none"]["ax"][a] > 0) for r in sub])
    im = ax.imshow(M, cmap="RdBu_r", vmin=-0.75, vmax=0.75, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]:+.2f}".replace("0.", "."), ha="center",
                    va="center", fontsize=7,
                    color="white" if abs(M[i, j]) > 0.45 else "black")
    ax.set_xticks(range(len(AXES))); ax.set_xticklabels(AXES, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(VECS) + 1))
    ax.set_yticklabels(VECS + ["random (vs none)"], fontsize=8.5)
    base = {a: np.mean([r["gens"]["none"]["ax"][a] > 0 for r in sub]) for a in AXES}
    ax.set_title(f"{kind} subjects (n={len(sub)})  —  unsteered P(yes): " +
                 " ".join(f"{a[:4]} {base[a]:.2f}" for a in AXES[:6]) + " …", fontsize=8.5)
cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.01)
cb.set_label("Δ P(judge: yes), (+v) − (−v) generations")
fig.suptitle("Open-ended 'What do you think about {n}?' — judge-scored halo battery "
             "(α=0.5 at name tokens, greedy, 12 axes)", fontsize=11, y=1.02)
p = os.path.join(OUT, "opengen3_summary.png")
fig.savefig(p, dpi=160, bbox_inches="tight")
print("wrote", p)
