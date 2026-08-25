"""MAIN_2 slide 2.2: cross-generalization. Does each trust-type vector move
ONLY its own kind of trust situation, or all of them?"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", os.path.join(_HERE, "..", "out"))
d = json.load(open(os.path.join(OUT, "typology_beds.json")))
BEDS = ["cognitive", "affective", "ability", "values"]
VECS = ["cognitive", "affective", "ability", "values", "control:warmth", "control:random"]
VLAB = {"control:warmth": "warmth (control)", "control:random": "random (control)"}
M = np.array([[d["E"][f"{v}|{b}"] for b in BEDS] for v in VECS])

fig, ax = plt.subplots(figsize=(8.6, 7.2))
im = ax.imshow(M, cmap="Reds", vmin=0, vmax=5, aspect="auto")
for i in range(len(VECS)):
    for j in range(len(BEDS)):
        ax.text(j, i, f"{M[i,j]:.1f}", ha="center", va="center", fontsize=12,
                color="white" if M[i, j] > 3.2 else "black",
                fontweight="bold" if (i < 4 and BEDS[j] == VECS[i]) else "normal")
ax.set_xticks(range(len(BEDS)))
ax.set_xticklabels([f"{b}\nsituation" for b in BEDS], fontsize=9)
ax.set_yticks(range(len(VECS)))
ax.set_yticklabels([VLAB.get(v, f"{v}\ntrust vector") for v in VECS], fontsize=9)
ax.set_xlabel("which kind of trust SITUATION is being steered")
ax.set_ylabel("which trust vector is added")
for i in range(4):
    ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False, ec="#111", lw=2.5))
cb = fig.colorbar(im, fraction=0.046, pad=0.04)
cb.set_label("how much steering moves the answer (logits)")
ax.set_title("Each trust vector steers EVERY kind of trust situation about equally.\n"
             "Boxed = vector matched to its own situation — it is not the standout.\n"
             "The types look different inside the model but act as one trust lever.",
             fontsize=10)
fig.tight_layout()
p = os.path.join(OUT, "crossgen_summary.png")
fig.savefig(p, dpi=160); print("wrote", p)
