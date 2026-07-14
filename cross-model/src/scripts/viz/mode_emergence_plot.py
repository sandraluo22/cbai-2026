"""Per-eigenmode emergence + head-cascade figure from a cached head_mode_bylayer damage tensor.
Fixes the earlier off-by-one: mode k's eigenvalue is eigenvalues[k] (both arrays length N, index 0 =
constant mode). Env: FAM(grid) TAG(Llama) LC(dir with <TAG>_<FAM>_damage.npz) OUT."""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

FAM = os.environ.get("FAM", "grid"); TAG = os.environ.get("TAG", "Llama")
LC = os.environ.get("LC", "runs/axes/4_circuits/head_mode_bylayer")
OUT = os.environ.get("OUT", LC)


def main():
    d = np.load(f"{LC}/{TAG}_{FAM}_damage.npz")
    D = d["damage"]; w = d["eigenvalues"]; cpow = d["clean_pow"]; readout = int(d["readout"])
    N, nL, _, nH = D.shape                                    # [mode, meas_layer, abl_layer, head]
    layers = np.arange(nL)
    # emergence = ONSET: first layer reaching 50% of the mode's peak power (when it first appears)
    com = np.array([ int(np.argmax(cpow[:, k] >= 0.5 * cpow[:, k].max())) if cpow[:, k].max() > 0 else nL for k in range(N) ], float)
    print(f"[{TAG}/{FAM}] readout L{readout}.  mode  eigenvalue  peakpow  onset(50%-of-peak layer)")
    for k in np.argsort(cpow[readout])[::-1][:6]:
        print(f"    m{k:<2} λ={w[k]:>5.2f}  peak={cpow[:,k].max():.3f}  onset≈L{com[k]:.0f}")
    # frequency-ordering correlation
    strong = np.where(cpow[readout] > 0.03)[0]; strong = strong[strong > 0]
    r = np.corrcoef(w[strong], com[strong])[0, 1]
    print(f"    corr(eigenvalue, emergence layer) over strong modes = {r:+.2f}  (positive => higher freq emerges later)")

    with PdfPages(f"{OUT}/mode_emergence_{TAG}_{FAM}.pdf") as pdf:
        fig, ax = plt.subplots(1, 3, figsize=(17, 5))
        order = np.argsort(cpow[readout])[::-1][:6]
        cmap = plt.get_cmap("coolwarm"); wmax = w.max() + 1e-9
        for k in order:
            c = cmap(w[k] / wmax)
            ax[0].plot(layers, cpow[:, k], "-o", ms=2, color=c, label=f"m{k} λ{w[k]:.1f}")
        ax[0].set_title(f"{FAM}: clean power vs layer per eigenmode\n(colour = frequency; low→early, high→late)", fontsize=9)
        ax[0].set_xlabel("layer"); ax[0].set_ylabel("power"); ax[0].legend(fontsize=7)
        for k in order:
            Dk = D[k]; ro = Dk[readout]; l, h = np.unravel_index(ro.argmax(), ro.shape)
            ax[1].plot(layers, Dk[:, l, h], "-o", ms=2, color=cmap(w[k] / wmax), label=f"m{k} (L{l}H{h})")
        ax[1].set_title("damage vs MEASUREMENT layer, each mode's top head", fontsize=9)
        ax[1].set_xlabel("measurement layer"); ax[1].set_ylabel("damage"); ax[1].legend(fontsize=7)
        # emergence-layer vs frequency scatter (the key result)
        ax[2].scatter(w[strong], com[strong], c=w[strong], cmap="coolwarm", s=60, edgecolors="k", lw=.4)
        for k in strong:
            if cpow[readout][k] > 0.06: ax[2].annotate(f"m{k}", (w[k], com[k]), fontsize=7)
        ax[2].set_title(f"emergence layer vs frequency  (r={r:+.2f})", fontsize=9)
        ax[2].set_xlabel("eigenvalue (frequency)"); ax[2].set_ylabel("onset layer (50% of peak)")
        fig.suptitle(f"{TAG} {FAM} — eigenmode emergence across depth (the DOMINANT mode is assembled last, near readout)", fontsize=11)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(f"wrote {OUT}/mode_emergence_{TAG}_{FAM}.pdf")


if __name__ == "__main__":
    main()
