"""Do in-context cycles of different node lengths use the SAME circuit heads? From cycle_head_circuit_<model>.json:
 (a) per-size head-importance heatmaps (layer x head) — small multiples;
 (b) cross-size similarity: Pearson r of full head-importance vectors + top-K Jaccard;
 (c) consensus heads — mean importance across sizes, top heads labelled.
High off-diagonal r / Jaccard = one shared head circuit across sizes.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

DIR = os.environ.get("DIR", "runs/axes/5_cyclic"); MODEL = os.environ.get("MODEL", "Llama"); TOPK = int(os.environ.get("TOPK", "15"))
d = json.load(open(f"{DIR}/cycle_head_circuit_{MODEL}.json"))
SIZES = d["sizes"]; nL = d["nL"]; nH = d["nH"]
HM = {s: np.array(d["head_map"][str(s)] if str(s) in d["head_map"] else d["head_map"][s]) for s in SIZES}
vecs = {s: HM[s].flatten() for s in SIZES}
vmax = max(HM[s].max() for s in SIZES)

# cross-size similarity
nS = len(SIZES); R = np.zeros((nS, nS)); J = np.zeros((nS, nS))
topsets = {s: set(map(tuple, np.argwhere(HM[s] >= np.sort(HM[s].flatten())[-TOPK]))) for s in SIZES}
for i, a in enumerate(SIZES):
    for j, b in enumerate(SIZES):
        R[i, j] = np.corrcoef(vecs[a], vecs[b])[0, 1]
        A, B = topsets[a], topsets[b]; J[i, j] = len(A & B) / len(A | B)

consensus = np.mean([HM[s] for s in SIZES], 0)
cflat = sorted([(consensus[L, h], L, h) for L in range(nL) for h in range(nH)], reverse=True)

fig = plt.figure(figsize=(17, 8.8))
gs = fig.add_gridspec(2, nS, height_ratios=[1.25, 1], left=0.05, right=0.9, top=0.9, bottom=0.08, hspace=0.32, wspace=0.5)
# (a) per-size heatmaps
for j, s in enumerate(SIZES):
    ax = fig.add_subplot(gs[0, j])
    im = ax.imshow(HM[s], aspect="auto", cmap="magma", vmin=0, vmax=vmax)
    ax.set_title(f"ring-{s}  (base {d['base'][str(s)] if str(s) in d['base'] else d['base'][s]:.2f})", fontsize=9)
    ax.set_xlabel("head"); ax.set_ylabel("layer" if j == 0 else "")
    for (v, L, h) in cflat[:3]:
        ax.add_patch(plt.Rectangle((h - .5, L - .5), 1, 1, ec="#39FF14", fc="none", lw=1.1))
cax = fig.add_axes([0.915, 0.55, 0.01, 0.3])
fig.colorbar(im, cax=cax, label="validity drop when head ablated")

# (b) similarity matrices
axR = fig.add_subplot(gs[1, 0]); axJ = fig.add_subplot(gs[1, 1])
for ax, Mm, ttl, cm in [(axR, R, "head-map correlation (Pearson r)", "viridis"), (axJ, J, f"top-{TOPK} head Jaccard", "cividis")]:
    im = ax.imshow(Mm, cmap=cm, vmin=0, vmax=1)
    ax.set_xticks(range(nS)); ax.set_xticklabels(SIZES); ax.set_yticks(range(nS)); ax.set_yticklabels(SIZES)
    ax.set_title(ttl, fontsize=9); ax.set_xlabel("ring size"); ax.set_ylabel("ring size")
    for i in range(nS):
        for k in range(nS):
            ax.text(k, i, f"{Mm[i,k]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if Mm[i, k] < 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.12)

# (c) consensus top heads
axC = fig.add_subplot(gs[1, 2:])
lab = [f"L{L}H{h}" for (_, L, h) in cflat[:TOPK]]; val = [v for (v, _, _) in cflat[:TOPK]]
axC.barh(range(TOPK), val, color="#7C3AED"); axC.set_yticks(range(TOPK)); axC.set_yticklabels(lab, fontsize=7)
axC.invert_yaxis(); axC.set_xlabel("mean validity drop across sizes"); axC.set_title(f"consensus circuit heads (top {TOPK}, mean over sizes)", fontsize=9)
axC.spines[["top", "right"]].set_visible(False)

fig.suptitle(f"Circuit heads for in-context cycles across node lengths ({MODEL}): shared heads => one circuit", fontsize=12, y=0.965)
out = f"{DIR}/cycle_head_circuit_{MODEL}.pdf"
fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
print("consensus top heads:", ", ".join(lab[:8]))
print("mean off-diag r:", round((R.sum()-nS)/(nS*nS-nS), 3), " mean off-diag Jaccard:", round((J.sum()-nS)/(nS*nS-nS), 3))
