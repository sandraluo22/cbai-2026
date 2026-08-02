"""Cross-model RSA layer x layer heatmaps: Qwen-32B vs each 8B model on the grid. Cell (La,Lb) =
Spearman of node-mean RDMs between Qwen-32B layer La and the other model's layer Lb. Reads grid
node-means (divider_basis). Star marks the argmax; dashed line = the relative-depth diagonal.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = "runs/axes/1_decomposition/divider_basis"
REF = os.environ.get("REF", "Qwen32")
OTHERS = os.environ.get("OTHERS", "Llama,Qwen,Gemma").split(",")
LAB = {"Llama": "Llama-8B", "Qwen": "Qwen-8B", "Gemma": "Gemma-9B", "Qwen32": "Qwen-32B"}


def rdms(tag):
    z = np.load(f"{DIR}/nodemeans_{tag}_square_grid.npz")
    nL = sum(k.startswith("layer_") for k in z.files); iu = np.triu_indices(16, 1)
    out = []
    for L in range(nL):
        H = z[f"layer_{L}"].astype(np.float64); H = H - H.mean(0)
        out.append(np.linalg.norm(H[:, None] - H[None], axis=2)[iu])
    return out


def sp(a, b): return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])

R = {REF: rdms(REF), **{o: rdms(o) for o in OTHERS}}
fig, axes = plt.subplots(1, len(OTHERS), figsize=(5.2 * len(OTHERS), 5.0))
for ax, o in zip(np.atleast_1d(axes), OTHERS):
    A = np.array([[sp(a, b) for b in R[o]] for a in R[REF]])       # (nL_ref, nL_other)
    im = ax.imshow(A, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=max(0.3, A.max()),
                   extent=[-0.5, len(R[o]) - 0.5, -0.5, len(R[REF]) - 0.5])
    bi, bj = np.unravel_index(A.argmax(), A.shape)
    ax.plot(bj, bi, "c*", ms=14)
    ax.plot([0, len(R[o]) - 1], [0, len(R[REF]) - 1], "w--", lw=0.8, alpha=0.5)   # rel-depth diagonal
    ax.set_xlabel(f"{LAB[o]} layer"); ax.set_ylabel(f"{LAB[REF]} layer")
    ax.set_title(f"{LAB[REF]} vs {LAB[o]}   (max RSA {A.max():.2f} @ L{bi}/L{bj})", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, label="node-mean RDM Spearman")
fig.suptitle("Cross-model RSA layer × layer (grid) — Qwen-32B vs 8B", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
os.makedirs("runs/axes/1_decomposition/qwen32", exist_ok=True)
for ext in ("pdf", "png"): fig.savefig(f"runs/axes/1_decomposition/qwen32/crossrsa_heatmap_Qwen32.{ext}", dpi=140, bbox_inches="tight")
print("wrote crossrsa_heatmap_Qwen32.png")
