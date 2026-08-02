"""family_spectra extended to ALL layers: for each (family, model), a heatmap of eigenmode power
fraction (x = eigenmode low->high) vs relative depth (y). Rows = families, cols = models.
Unnormalized Laplacian (matches family_spectra). Reads markov_families node-means.
Env: MFDIR MODELS FAMS OUT
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MFDIR = os.environ.get("MFDIR", "runs/axes/1_decomposition/markov_families")
MODELS = os.environ.get("MODELS", "Llama,Qwen,Qwen32,Gemma").split(",")
FAMS = os.environ.get("FAMS", "grid,ring,tree,smallworld,sbm4,sbm2,er_random").split(",")
OUT = os.environ.get("OUT", "runs/axes/1_decomposition/qwen32/family_spectra_alllayers.pdf")
LAB = {"Llama": "Llama-8B", "Qwen": "Qwen-8B", "Gemma": "Gemma-9B", "Qwen32": "Qwen-32B"}


def layer_spectra(model, fam):
    z = np.load(f"{MFDIR}/nodemeans_{model}_{fam}.npz", allow_pickle=True)
    A = np.array(z["adjacency"], float); L = np.diag(A.sum(1)) - A; w, V = np.linalg.eigh(L)
    nL = sum(k.startswith("layer_") for k in z.files)
    M = np.zeros((nL, len(w) - 1))
    for l in range(nL):
        H = z[f"layer_{l}"].astype(float); Hc = H - H.mean(0)
        c = V.T @ Hc; p = (c ** 2).sum(1); p[0] = 0; p /= p.sum() + 1e-12
        M[l] = p[1:]
    return M                                                      # (nL, 15) power fraction


nR, nC = len(FAMS), len(MODELS)
fig, axes = plt.subplots(nR, nC, figsize=(2.7 * nC, 2.3 * nR), squeeze=False)
for i, fam in enumerate(FAMS):
    for j, m in enumerate(MODELS):
        ax = axes[i][j]
        try:
            M = layer_spectra(m, fam)
        except FileNotFoundError:
            ax.axis("off"); continue
        k = M.shape[1]
        im = ax.imshow(M, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=0.30,
                       extent=[0.5, k + 0.5, 0, 1])
        if j == 0: ax.set_ylabel(f"{fam}\nrel depth", fontsize=8)
        else: ax.set_yticks([])
        if i == 0: ax.set_title(LAB.get(m, m), fontsize=10)
        if i == nR - 1: ax.set_xlabel("eigenmode (low→high)", fontsize=8)
        else: ax.set_xticks([])
        ax.set_xticks(range(2, k + 1, 3))
fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01, label="eigenmode power fraction")
fig.suptitle("Eigenmode power vs depth (ALL layers), per family × model", fontsize=13, y=0.995)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, bbox_inches="tight"); fig.savefig(OUT.replace(".pdf", ".png"), dpi=130, bbox_inches="tight")
print("wrote", OUT)
