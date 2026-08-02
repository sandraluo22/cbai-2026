"""Do in-context rings n=3..16 share one circular circuit? Two heatmaps:
 (L) CAUSAL sharing: drop in ring-m next-node validity (rows) when ablating ring-n's fundamental direction
     (cols), as a fraction of ring-m's own baseline. Diagonal = self-ablation. Broad off-diagonal darkening
     = one shared circuit across sizes; a sharp diagonal = size-specific circuits.
 (R) REPRESENTATIONAL: subspace alignment (mean squared cosine of principal angles) between ring-n and
     ring-m fundamental-direction pairs at a late layer.
Reads ring_circuit_sharing_<model>.json.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

DIR = os.environ.get("DIR", "runs/axes/5_cyclic"); MODEL = os.environ.get("MODEL", "Llama")
d = json.load(open(f"{DIR}/ring_circuit_sharing_{MODEL}.json"))
R = d["rings"]; base = d["baseline"]; Mtx = d["ablate_matrix"]; sim = d["subspace_sim"]

# drop fraction: (base_m - validity_m|ablate n) / base_m, clipped at 0
drop = np.zeros((len(R), len(R)))
for i, m in enumerate(R):
    b = base[str(m)] if str(m) in base else base[m]
    for j, n in enumerate(R):
        v = Mtx[str(m)][str(n)] if str(m) in Mtx else Mtx[m][n]
        drop[i, j] = max(0.0, (b - v) / b) if b > 1e-6 else 0.0
S = np.array([[ (sim[str(a)][str(b)] if str(a) in sim else sim[a][b]) for b in R] for a in R], float)

fig, ax = plt.subplots(1, 2, figsize=(15, 6.6))
for a, Mm, ttl, cmap, vmx in [
    (ax[0], drop, f"CAUSAL circuit sharing — fractional drop in ring-m validity\nwhen ablating ring-n's fundamental direction", "magma", None),
    (ax[1], S, f"REPRESENTATIONAL alignment — direction-subspace similarity\n(mean cos² principal angles, layer {d['sim_layer']})", "viridis", 1.0)]:
    im = a.imshow(Mm, cmap=cmap, vmin=0, vmax=vmx, aspect="equal")
    a.set_xticks(range(len(R))); a.set_xticklabels(R); a.set_yticks(range(len(R))); a.set_yticklabels(R)
    a.set_xlabel("ablated ring n  (direction source)"); a.set_ylabel("task ring m  (validity measured)")
    a.set_title(ttl, fontsize=9.5)
    for i in range(len(R)):
        for j in range(len(R)):
            v = Mm[i, j]
            a.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.2,
                   color="white" if (cmap == "magma" and v < (0.6*(vmx or Mm.max()))) or (cmap == "viridis" and v < 0.6) else "black")
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    a.plot([-.5, len(R)-.5], [-.5, len(R)-.5], color="#39FF14", lw=.8, alpha=.5)  # diagonal guide

fig.suptitle(f"In-context rings 3-16: are the circular circuits shared across sizes? ({MODEL})", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = f"{DIR}/ring_circuit_sharing_{MODEL}.pdf"
fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
