"""family_spectra first-slide barchart (node-mean variance vs Laplacian eigenmode, per family),
replicated for EVERY layer -> one multipage PDF per model (page L = the 7-family small-multiples at
layer L). Same convention as family_spectra.py (unnormalized Laplacian, P[0]:=0).
Env: TAG(Llama) MFDIR OUTDIR FAMS
Out: <OUTDIR>/family_spectra_bylayer_<TAG>.pdf
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

TAG = os.environ.get("TAG", "Llama")
MFDIR = os.environ.get("MFDIR", "runs/axes/1_decomposition/markov_families")
OUTDIR = os.environ.get("OUTDIR", "runs/axes/1_decomposition")
FAMS = os.environ.get("FAMS", "grid,ring,tree,smallworld,sbm4,er_random,sbm2").split(",")


def load(fam):
    z = np.load(f"{MFDIR}/nodemeans_{TAG}_{fam}.npz", allow_pickle=True)
    A = np.array(z["adjacency"], float); w, V = np.linalg.eigh(np.diag(A.sum(1)) - A)
    nL = sum(k.startswith("layer_") for k in z.files)
    return w, V, {l: z[f"layer_{l}"].astype(float) for l in range(nL)}, nL


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    data = {f: load(f) for f in FAMS}
    nL = max(data[f][3] for f in FAMS)
    out = f"{OUTDIR}/family_spectra_bylayer_{TAG}.pdf"
    with PdfPages(out) as pdf:
        for l in range(nL):
            fig, axes = plt.subplots(2, 4, figsize=(14, 6))
            for ax, f in zip(np.array(axes).flat, FAMS):
                w, V, layers, nlf = data[f]
                if l >= nlf:
                    ax.axis("off"); continue
                H = layers[l]; Hc = H - H.mean(0)
                c = V.T @ Hc; p = (c ** 2).sum(1); p[0] = 0; p /= p.sum() + 1e-12
                k = np.arange(1, len(w))
                ax.bar(k, p[1:], color="tab:purple")
                ax.set_title(f, fontsize=9); ax.set_xlabel("eigenmode (low→high freq)", fontsize=8)
                ax.set_ylabel("variance frac", fontsize=8); ax.set_ylim(0, 0.35); ax.set_xticks(k[::2])
                ax.tick_params(labelsize=7)
            for ax in np.array(axes).flat[len(FAMS):]:
                ax.axis("off")
            fig.suptitle(f"{TAG}: node-mean variance vs Laplacian eigenmode, per family — "
                         f"LAYER {l} / {nL - 1}  (rel depth {l/(nL-1):.2f})", fontsize=12)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(f"wrote {out}  ({nL} pages)")


if __name__ == "__main__":
    main()
